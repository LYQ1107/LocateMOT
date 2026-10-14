"""Legal fit RMOT: broader teacher regions versus actual native own histories."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data2/user/LocateMOT/data')
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.decision import overlap
    from locatemot.groundjev.feature_client import NativeFeatureClient
    from locatemot.groundjev.perception import GroundAnythingAdapter, GroundingProtocolError
    from locatemot.groundjev.tracker import OnlineTracker
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_video

    torch.set_num_threads(4)
    plan = json.loads((ROOT/'configs/groundjev/rmot_native72_20261014.json').read_text())
    out = ROOT/'outputs/groundjev/compact/rmot_native72_v1'
    out.mkdir(parents=True, exist_ok=False)
    base = torch.load(ROOT/'outputs/groundjev/compact/joint_pack_v3/examples.pt', weights_only=True)
    actor = CompactDecision(ROOT/plan['actor_checkpoint'])
    client = NativeFeatureClient('http://127.0.0.1:8111')
    frontend = GroundAnythingAdapter(parser_policy='official_scoring', decoder_policy='tracking_dense')
    captured = []
    def capture(module, args):
        captured.append({k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                         for k, v in args[0].items() if k != 'class_text'})
    hook = actor.model.register_forward_pre_hook(capture)
    teacher, native, frame_scope = [], [], []
    counts = Counter()
    state = {'status': 'BUILDING_BROAD_FIT_AND_NATIVE_OWN_HISTORY_PACK', 'frames': 0,
             'registered_frames': 72, 'actual_native_candidates': 0,
             'actor_GT_corrected': False, 'validation_or_test_labels_read': False,
             'counter_expression_is_supervised_snapshot_not_actor_rollout': True}
    def save():
        state.update(counts=dict(counts), teacher_groups=len(teacher), native_groups=len(native))
        (out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({k: v for k, v in state.items() if k != 'counts'}), flush=True)
    save()
    try:
        with (out/'native_rollout.jsonl').open('w') as records:
            for clip in plan['clips']:
                video = assert_legal_video(clip['video'], dataset='refer_kitti_v2')
                assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
                queries = [json.loads(Path(clip[k]).read_text()) for k in
                           ('primary_expression_file', 'counter_expression_file')]
                specs = [{'task': 'rmot', 'vocabulary': ['person', 'car'], 'expression': q['sentence']} for q in queries]
                text = [actor.text_features.specification(s) for s in specs]
                tracker = OnlineTracker(actor)
                anchors, last_observation, memory = {}, {}, {}
                previous_time = None
                for number in clip['frames']:
                    timestamp = number/clip['fps']
                    path = DATA/'kitti_tracking_training/image_02'/video/f'{number:06d}.png'
                    with Image.open(path) as raw:
                        image = raw.convert('RGB')
                    label = DATA/'refer_kitti_v2/labels_with_ids/image_02'/video/f'{number:06d}.txt'
                    targets = []
                    for line in label.read_text().splitlines() if label.exists() else []:
                        f = line.split(); x, y, w, h = map(float, f[2:6])
                        targets.append({'id': int(f[1]), 'box': [x*image.width, y*image.height,
                            (x+w)*image.width, (y+h)*image.height]})
                    query_ids = [set(q['label'].get(str(number), [])) for q in queries]
                    assert set.union(*query_ids) <= {g['id'] for g in targets}
                    # GT is read only by the supervision producer. It never
                    # supplies native candidates or modifies the actor memory.
                    try:
                        candidates, perception = frontend(image, 'rmot', ['person', 'car'],
                            expression=specs[0]['expression'], image_path=path)
                    except GroundingProtocolError as error:
                        candidates = []
                        perception = dict(error.metadata, model_response_rejected=True,
                            rejected_output_policy='empty_and_record; no GT proposal substitution')
                        counts['rejected_native_frames'] += 1
                    candidates, feature_meta = client(image, candidates, image_path=path)
                    matched, qualities = {}, {}
                    if candidates and targets:
                        similarity = np.asarray([[overlap(c['box'], g['box']) for g in targets] for c in candidates])
                        rr, cc = linear_sum_assignment(-similarity)
                        for i, j in zip(rr, cc):
                            if similarity[i, j] >= .5:
                                matched[int(i)] = targets[int(j)]['id']
                                qualities[int(i)] = float(similarity[i, j])
                    prior = {t.track_id: copy.copy(t) for t in tracker.tracks.values()}
                    before = dict(anchors)
                    captured.clear()
                    outputs, runtime = tracker.step(image, candidates, timestamp, specs[0])
                    assert runtime['decision_metadata']['GT_read'] is False
                    records.write(json.dumps({'sequence': video, 'frame': number, 'image_path': str(path),
                        'specification': specs[0], 'candidates': candidates, 'outputs': outputs,
                        'state': runtime, 'perception': perception})+'\n'); records.flush()
                    events = {e['candidate']: e for e in runtime['events'] if 'candidate' in e}
                    offset = 0
                    for batch, e in enumerate(captured):
                        n, m = len(e['candidates']), len(e['history'])
                        # Empty-candidate batches still teach lifecycle. A
                        # completely empty frame remains in frame_scope.
                        ranked = events[offset]['decision']['shortlisted_ids'] if n else sorted(
                            prior, key=lambda t: (prior[t].lost_age, t))[:actor.shortlist]
                        assert len(ranked) == m
                        present = set(matched.values())
                        identity_targets, updates, reliability = [], [], []
                        for i in range(n):
                            gid = matched.get(offset+i)
                            correct = [j for j, t in enumerate(ranked) if gid is not None and before.get(t) == gid]
                            known = any(gid is not None and before.get(t) == gid for t in prior)
                            target = m+1 if gid is None else correct[0] if correct else m+2 if known else m
                            quality = qualities.get(offset+i, 0.)
                            identity_targets.append(target)
                            updates.append(float(quality >= .7 and target != m+2))
                            reliability.append(0 if quality < .5 else 1 if quality < .7 else 2 if quality < .85 else 3)
                        life = [3 if before.get(t) is None else 2 if before[t] in present and prior[t].lost_age else
                                0 if before[t] in present else 1 for t in ranked]
                        e.update(target=torch.tensor(identity_targets, dtype=torch.long),
                            update_target=torch.tensor(updates), reliability_target=torch.tensor(reliability, dtype=torch.long),
                            life_target=torch.tensor(life, dtype=torch.long), class_target=torch.full((n,), -100),
                            native_teacher=False, task_index=1, source_sequence=video, frame=number,
                            proposal_source='actual_native_GroundAnything_DLM', history_source='uncorrected Compact own history',
                            update_reliability_source='GT alignment IoU proxy, not native confidence')
                        for query_index in range(2):
                            member = [float(any(g['id'] in query_ids[query_index] and overlap(c['box'], g['box']) >= .5
                                               for g in targets)) for c in candidates[offset:offset+n]]
                            native.append(dict(e, id=f'native_rmot_{video}_{number}_{batch}_query{query_index}',
                                spec_text=text[query_index], membership_target=torch.tensor(member),
                                derived_counter_expression=bool(query_index)))
                            counts['native_membership_positive' if sum(member) else 'native_group_without_query_positive'] += int(sum(member)) if sum(member) else 1
                        offset += n
                    assert offset == len(candidates)
                    for i, event in events.items():
                        if 'track_id' not in event:
                            continue
                        tid, gid = event['track_id'], matched.get(i)
                        if event['action'] == 'NEW':
                            anchors[tid] = gid
                        elif last_observation.get(tid) != before.get(tid):
                            counts['contaminated_history_observations'] += 1
                        last_observation[tid] = gid
                    # The matched-frame GT teacher comparison uses the same
                    # legal images, but is explicitly a separate proposal stage.
                    enriched, _ = client(image, [{'box': g['box'], 'label': 'object', 'native_score': None}
                                                   for g in targets], image_path=path)
                    current = torch.tensor([c['native_region_feature'] for c in enriched]).reshape(len(targets), 2560)
                    predicted = {t: [x+v*(timestamp-memory[t]['time']) for x, v in zip(memory[t]['box'], memory[t]['velocity'])] for t in memory}
                    for start in range(0, len(targets), 16):
                        group = targets[start:start+16]; n = len(group)
                        ranked = sorted(memory, key=lambda t: (-max(overlap(c['box'], predicted[t]) for c in group), t))[:32]
                        m = len(ranked)
                        h = torch.stack([torch.stack([memory[t][k] for k in ('anchor', 'recent', 'long')]) for t in ranked]) if m else torch.empty(0, 3, 2560)
                        geometry = torch.tensor([[[overlap(c['box'], predicted[t]),
                            ((c['box'][0]+c['box'][2]-predicted[t][0]-predicted[t][2])**2+
                             (c['box'][1]+c['box'][3]-predicted[t][1]-predicted[t][3])**2)**.5/(2*(image.width**2+image.height**2)**.5)]
                            for t in ranked] for c in group]).reshape(n, m, 2)
                        present = {g['id'] for g in targets}
                        e = {'candidates': current[start:start+n], 'history': h,
                            'candidate_geo': torch.tensor([[c['box'][0]/image.width, c['box'][1]/image.height,
                                c['box'][2]/image.width, c['box'][3]/image.height] for c in group]),
                            'history_geo': torch.tensor([[*(x/s for x, s in zip(predicted[t], (image.width, image.height, image.width, image.height))),
                                memory[t]['age']/100, (previous_time-memory[t]['time'])/3,
                                (timestamp-memory[t]['time'])/3, 1.] for t in ranked]).reshape(m, 8),
                            'pair_geometry': geometry, 'task_index': 1,
                            'target': torch.tensor([ranked.index(c['id']) if c['id'] in ranked else m+2 if c['id'] in memory else m for c in group]),
                            'update_target': torch.ones(n), 'reliability_target': torch.full((n,), 3),
                            'life_target': torch.tensor([2 if t in present and previous_time > memory[t]['time'] else 0 if t in present else 1 for t in ranked], dtype=torch.long),
                            'class_target': torch.full((n,), -100), 'native_teacher': False,
                            'source_sequence': video, 'frame': number, 'proposal_source': 'teacherGT',
                            'history_source': 'causal teacherGT', 'update_reliability_source': 'ideal teacher proxy, not native confidence'}
                        for query_index in range(2):
                            teacher.append(dict(e, id=f'broad_teacher_{video}_{number}_{start}_query{query_index}',
                                spec_text=text[query_index], membership_target=torch.tensor([float(c['id'] in query_ids[query_index]) for c in group])))
                    for g, f in zip(targets, current):
                        tid = g['id']
                        if tid not in memory:
                            memory[tid] = dict(anchor=f, age=0, velocity=[0.]*4)
                        else:
                            memory[tid]['velocity'] = [(x-y)/(timestamp-memory[tid]['time']) for x, y in zip(g['box'], memory[tid]['box'])]
                        memory[tid].update(recent=f, long=f, box=g['box'], time=timestamp, age=memory[tid]['age']+1)
                    memory = {t: v for t, v in memory.items() if timestamp-v['time'] <= 3}
                    previous_time = timestamp
                    frame_scope.append({'video': video, 'frame': number, 'image_path': str(path),
                        'native_candidates': len(candidates), 'native_matched_regions': len(matched),
                        'teacher_regions': len(targets), 'native_response_rejected': bool(perception.get('model_response_rejected'))})
                    counts['native_matched_regions'] += len(matched)
                    state.update(frames=len(frame_scope), actual_native_candidates=state['actual_native_candidates']+len(candidates))
                    if state['frames'] % 8 == 0:
                        save()
        hook.remove()
        assert state['frames'] == 72 and native and teacher
        # Conservatively count existing features/crops/readable results. The
        # model and optimizer checkpoints are explicitly outside this budget.
        existing_bytes = sum(p.stat().st_size for p in (ROOT/'outputs/groundjev').rglob('*')
            if p.is_file() and p.suffix in {'.json', '.jsonl', '.pt', '.png', '.jpg'}
            and p.name not in {'checkpoint.pt', 'projector_delta.pt', 'latest_training_state.pt'}
            and not any(part.startswith('ckpt-') or part == 'final' for part in p.parts))
        # Counting repeated tensor references separately gives an upper bound
        # on serialized tensor data for both comparison packs.
        planned_tensor_bytes = sum(v.numel()*v.element_size()
            for examples in (base['examples']+teacher, base['examples']+teacher+native)
            for e in examples for v in e.values() if torch.is_tensor(v))
        planned_tensor_bytes += 2*base['class_text'].numel()*base['class_text'].element_size()
        assert existing_bytes+planned_tensor_bytes+2**20 < 25*2**30
        state['feature_budget_existing_conservative_bytes'] = existing_bytes
        state['feature_budget_new_tensor_upper_bound_bytes'] = planned_tensor_bytes
        sizes = {}
        for variant, new in [('broad_teacher', teacher), ('broad_native', teacher+native)]:
            directory = out/variant; directory.mkdir()
            payload = dict(base, examples=base['examples']+new)
            torch.save(payload, directory/'examples.pt')
            sizes[variant] = (directory/'examples.pt').stat().st_size
            receipt = {'status': 'DECLARED_JOINT_SEMANTIC_TRAIN_PACK_COMPLETE', 'examples': len(payload['examples']),
                'registered_new_frames': 72, 'variant': variant, 'feature_file_bytes': sizes[variant],
                'teacher_GT_proposals_declared': True, 'actual_native_own_history_groups': len(native) if variant == 'broad_native' else 0,
                'GT_state_correction': False, 'Novel_GT_supervised': False}
            (directory/'owner_state.json').write_text(json.dumps(receipt, indent=2)+'\n')
        assert sum(sizes.values()) < 25*2**30
        (out/'frame_scope.json').write_text(json.dumps(frame_scope, indent=2)+'\n')
        state.update(status='BROAD_FIT_NATIVE_OWN_HISTORY_PACK_COMPLETE', feature_files_bytes=sizes,
            GT_IDs_in_actor_input=False, future_images_read_by_actor=False, original_teacher_groups_retained=256)
        save()
    except Exception as error:
        state.update(status='BROAD_FIT_NATIVE_PACK_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
