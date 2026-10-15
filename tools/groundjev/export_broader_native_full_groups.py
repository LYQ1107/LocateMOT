"""Watch completed native TRAIN clips and export exact causal Full-v3 questions."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import copy, deepcopy
import gzip
from itertools import zip_longest
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def export_clip(name, source, destination, plan, actor, stateful):
    import numpy as np
    import torch
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.assigned_identity_update import build_assigned_update_request
    from locatemot.groundjev.decision import OneJevDecision, overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.group_reference_rows import REFERENCE_KEY
    from tools.groundjev.prepare_tracking_decisions import PROTOCOL, annotations
    from tools.groundjev.streamed_tracking_media import CANDIDATE_KEY, HISTORY_KEY
    from tools.groundjev.run_video import ReplayFrames
    from train import common, mm, sft
    from qev.prompt import render_question

    manifest = json.loads((source/'manifests'/(name+'.json')).read_text())
    source_plan = json.loads((ROOT/plan['source_collector_plan']).read_text())
    seq = next(s for s in json.loads((PROTOCOL/source_plan['protocols'][manifest['dataset']]).read_text())['sequences']
               if s['sequence'] == name)
    allowed = {int(Path(f).stem) for f in seq['train_frames']}
    assert set(manifest['frame_numbers']) <= allowed
    truth = annotations(seq, set(manifest['frame_numbers']))
    runtime = json.loads((source/'clips'/name/'result.json').read_text())
    original = source/'clips'/name
    destination.mkdir(parents=True, exist_ok=False)
    groups_path = destination/'groups.jsonl.gz'
    rows_path = destination/'train_refs.jsonl.gz'
    group_reference = str(groups_path.relative_to(ROOT))
    replay = ReplayFrames(ROOT/runtime['native_candidate_archive'])
    builder = OneJevDecision.__new__(OneJevDecision); builder.evidence_format = plan['evidence_format']
    tracker = OnlineTracker(actor)
    anchors, aliases, provenance = {}, {}, {}
    pool = list(range(1, 100001))
    random.Random(sorted(p.stem for p in (source/'manifests').glob('*.json')).index(name)*1009).shuffle(pool)
    counts = Counter()
    specification = {'task': 'mot', 'vocabulary': ['person'], 'expression': None}
    mm.set_root(ROOT)

    with gzip.open(groups_path, 'wt', encoding='utf-8') as groups, \
         gzip.open(rows_path, 'wt', encoding='utf-8') as refs, \
         gzip.open(destination/'candidate_registry.jsonl.gz', 'wt', encoding='utf-8') as registry, \
         (original/'frame_decisions.jsonl').open() as traces:

        def write_group(identifier, request, media, targets):
            groups.write(json.dumps({'id': identifier, 'state': request.state, 'media': media,
                'questions': {k: q.model_dump(exclude_none=True) for k, q in request.questions.items()}}, ensure_ascii=False)+'\n')
            counts['request_groups'] += 1
            for key, (answer, family) in targets.items():
                question = request.questions[key]
                common.target_vector({str(answer): 1.}, render_question('q', question).labels)
                row = {'id': identifier+'_'+key, 'split': 'train', 'source': name, 'task': family,
                    'target': {str(answer): 1.}, REFERENCE_KEY: {'path': group_reference, 'group': identifier, 'question': key},
                    'supervision': {'tracking_task_bucket': 'mot', 'actor': 'actual fixed sharedCompact CPU own history',
                        'GT_labels_external_only_no_state_correction': True}}
                refs.write(json.dumps(row, ensure_ascii=False)+'\n')
                counts['rows'] += 1; counts['family:'+family] += 1

        for expected, line in zip_longest(zip(manifest['frame_numbers'], manifest['images']), traces):
            assert expected is not None and line is not None
            frame, path = expected; saved = json.loads(line)
            raw = replay.frame(frame); candidates = raw['candidates']
            assert saved['frame'] == frame and saved['image_path'] == path == raw['image_path']
            assert saved['candidates'] == [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')} for c in candidates]
            before, prior_ids = dict(anchors), set(tracker.tracks)
            actor.captured = []
            with Image.open(path) as image_source:
                image = image_source.convert('RGB')
            timestamp = frame/manifest['fps']
            outputs, state = tracker.step(image, candidates, timestamp, specification)
            assert outputs == saved['outputs'] and state == saved['state']
            events = {e['candidate']: e for e in state['events'] if 'candidate' in e}
            assert set(events) == set(range(len(candidates)))
            gt, matched = truth[frame], {}
            if candidates and gt:
                matrix = np.asarray([[overlap(c['box'], g['box']) for g in gt] for c in candidates])
                rr, cc = linear_sum_assignment(-matrix)
                matched = {int(i): gt[int(j)] for i, j in zip(rr, cc) if matrix[i, j] >= .5}
            present = {g['gt_id'] for g in matched.values()}
            captured = actor.captured or [([], [])]
            offset = 0
            for batch, (group, ranked_source) in enumerate(captured):
                ranked = []
                for track in ranked_source:
                    aliases.setdefault(track.track_id, pool[len(aliases)])
                    shadow = copy(track); shadow.track_id = aliases[track.track_id]; ranked.append(shadow)
                base, native = builder.build_request(image, group, ranked, timestamp, specification)
                media = [{'type': 'image', 'path': path, 'w': image.width, 'h': image.height}]
                if group:
                    sheet = native[len(media)]['image']
                    media.append({'type': 'image', 'path': path, 'w': sheet.width, 'h': sheet.height,
                        CANDIDATE_KEY: [c['box'] for c in group]})
                if ranked:
                    history = [dict(id=aliases[t.track_id], **deepcopy(provenance[t.track_id])) for t in ranked_source]
                    assert all(t[k]['sequence'] == name and t[k]['frame'] <= frame
                               for t in history for k in ('anchor', 'long', 'recent'))
                    sheet = native[len(media)]['image']
                    media.append({'type': 'image', 'path': path, 'w': sheet.width, 'h': sheet.height, HISTORY_KEY: history})
                rebuilt, videos, metadata = mm.load_media(media)
                assert not videos and not metadata and len(rebuilt) == len(native)
                assert all(a.size == b['image'].size and a.tobytes() == b['image'].tobytes() for a, b in zip(rebuilt, native))
                counts['native_pixel_groups_exact'] += 1
                counts['native_crop_sheets_exact'] += len(native)-1
                targets = {}
                if 'empty' in base.questions:
                    targets['empty'] = ('yes' if gt else 'no', 'empty_presence')
                for local in range(len(group)):
                    target = matched.get(offset+local); gid = target['gt_id'] if target else None
                    correct = [t for t in ranked_source if gid is not None and before.get(t.track_id) == gid]
                    known = any(gid is not None and before.get(tid) == gid for tid in prior_ids)
                    answer = 'IGNORE' if gid is None else str(aliases[min(correct, key=lambda t: t.track_id).track_id]) if correct else 'NO_MATCH' if known else 'NEW'
                    reliable = target is not None and target['visibility'] >= .6 and answer != 'NO_MATCH'
                    targets[f'match_{local}'] = (answer, 'identity')
                    targets[f'update_{local}'] = ('yes' if reliable else 'no', 'memory_update')
                    targets[f'reliability_{local}'] = ('0' if target is None else '1' if not reliable else '3' if target['visibility'] >= .85 else '2', 'reliability')
                for track in ranked_source:
                    gid = before.get(track.track_id)
                    action = 'TERMINATE' if gid is None else 'REACTIVATE' if gid in present and track.lost_age else 'ACTIVE' if gid in present else 'LOST'
                    targets[f'life_{aliases[track.track_id]}'] = (action, 'lifecycle')
                identifier = f'{name}_{frame}_batch{batch}'
                write_group(identifier, base, media, targets)
                selected = {local: aliases[e['track_id']] for local in range(len(group))
                    if (e := events[offset+local]).get('track_id') is not None and e['action'] != 'NEW'}
                assigned = build_assigned_update_request(base.state, selected) if selected else None
                assigned_targets = {}
                for local in range(len(group)):
                    target = matched.get(offset+local); gid = target['gt_id'] if target else None
                    event = events[offset+local]; tid = event.get('track_id')
                    answer = None
                    if local in selected and gid is not None and before.get(tid) is not None:
                        answer = 'yes' if before[tid] == gid and target['visibility'] >= .6 else 'no'
                        assigned_targets[f'assigned_update_{local}'] = (answer, 'assigned_identity_memory_update')
                    registry.write(json.dumps({'sequence': name, 'frame': frame, 'candidate': offset+local,
                        'ordinary_group': identifier, 'actual_actor_action': event['action'], 'actual_actor_identity': tid,
                        'selected_input_alias': selected.get(local), 'posthoc_GT_identity': gid,
                        'posthoc_immutable_birth_GT_identity': before.get(tid),
                        'assigned_update_target': answer, 'GT_outside_input': True})+'\n')
                    counts['candidate_registry_rows'] += 1
                    if local in selected:
                        counts['existing_assignments_in_input'] += 1
                        counts['assigned_target:'+str(answer)] += 1
                if assigned is not None:
                    write_group(identifier+'_assigned', assigned, media, assigned_targets)
                offset += len(group)
            assert offset == len(candidates)
            for i, event in events.items():
                tid = event.get('track_id')
                if tid is None:
                    continue
                crop = {'path': path, 'box': candidates[i]['box'], 'sequence': name, 'frame': frame}
                if event['action'] == 'NEW':
                    anchors[tid] = matched[i]['gt_id'] if i in matched else None
                    provenance[tid] = {k: deepcopy(crop) for k in ('anchor', 'long', 'recent')}
                else:
                    provenance[tid]['recent'] = deepcopy(crop)
                    if event['memory_action'] == 'UPDATE': provenance[tid]['long'] = deepcopy(crop)
            counts['frames'] += 1; counts['candidates'] += len(candidates)
            counts['rejected_frames'] += bool(saved['perception'].get('model_response_rejected'))
    replay.close()
    assert counts['frames'] == runtime['frames'] and counts['candidates'] == counts['candidate_registry_rows']
    config = json.loads((ROOT/'configs/groundjev/onejev_unified600_native_v2.json').read_text())
    config.update(model=str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
        train=str(rows_path), device='cpu', plan_only=True, max_steps=0, name='native_CPU_plan', runs_dir=str(destination/'CPU_plan'))
    config_path = destination/'CPU_plan_config.json'; config_path.write_text(json.dumps(config, indent=2)+'\n')
    official, epoch = sft.plan_only, sft.plan_epoch
    summary = {'sequence': name, 'counts': dict(counts), 'row_references': str(rows_path),
        'all_actor_outputs_states_exact': True, 'new_OneJev_training_updates': 0,
        'question_evidence_format': plan['evidence_format'], 'full_goal_complete': False}

    def capture(bundle, data, settings, world):
        started = time.monotonic(); cached = epoch(bundle, data, settings, 0, 0, 1)
        elapsed = time.monotonic()-started
        batches, dropped, lengths = cached
        assert len(data) == counts['rows'] and dropped == 0 and len(lengths) == len(data)
        os.environ['PLAN_WORLD'] = str(world)
        assert stateful.protocol(settings)['input_io_protocol'] == plan['input_io_protocol']
        sft.plan_epoch = lambda *unused, **kwargs: cached
        try:
            with (destination/'native_CPU_plan_report.log').open('w') as report, redirect_stdout(report):
                code = official(bundle, data, settings, world)
        finally:
            sft.plan_epoch = epoch
        assert code == 0
        summary.update(status='REAL_CLIP_FULL_V3_GROUP_REFS_PIXELS_AND_ALL_NATIVE_CPU_ROWS_VALIDATED',
            max_tokens=max(lengths), total_planned_tokens=sum(lengths), dropped_rows=dropped,
            actual_plan_epoch_seconds=elapsed, actual_world_size=world,
            hypothetical_4row_epoch_updates=len(sft.plan_steps(batches, lengths, 1, settings.accum_rows)))
        return code

    sft.plan_only = capture
    try:
        assert sft.main(['--config', str(config_path), '--plan-only']) == 0
    finally:
        sft.plan_only = official
    assert not torch.cuda.is_initialized()
    (destination/'validated_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    import torch
    from locatemot.groundjev.compact import CompactDecision
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
    from tools.groundjev.run_unified600_controls import alive
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    source, out = ROOT/plan['source'], ROOT/plan['output']; out.mkdir(parents=True, exist_ok=args.resume)
    previous = json.loads((out/'owner_state.json').read_text()) if args.resume else None
    state = {'status': 'WAITING_COMPLETED_NATIVE_TRAIN_CLIPS', 'owner_pid': os.getpid(),
        'owner_starttick': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'completed': {}, 'new_OneJev_training_updates': 0, 'new_GPUs': 0, 'full_goal_complete': False}
    if previous is not None:
        assert all(row['question_evidence_format'] == plan['evidence_format'] for row in previous['completed'].values())
        state['completed'] = previous['completed']
        state['resumed_completed_clips'] = len(previous['completed'])
        state['previous_owner'] = [previous['owner_pid'], previous['owner_starttick']]

    def save():
        temporary = out/'owner_state.tmp'; temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(out/'owner_state.json')
        print(json.dumps({'status': state['status'], 'completed_clips': len(state['completed']),
            'frames': sum(v['counts']['frames'] for v in state['completed'].values()),
            'rows': sum(v['counts']['rows'] for v in state['completed'].values())}), flush=True)

    class Recorder(CompactDecision):
        def accepted_batch(self, group, ranked, inputs):
            self.captured.append((group, [copy(t) for t in ranked]))

    save()
    try:
        torch.set_num_threads(plan['actor_threads']); torch.manual_seed(0)
        stateful = configure_group_reference_sft()
        actor = Recorder(ROOT/plan['actor_checkpoint'], device='cpu')
        while True:
            producer = json.loads((source/'owner_state.json').read_text())
            for name in producer['completed']:
                if name in state['completed']:
                    continue
                state.update(status='EXPORTING_AND_PLANNING_'+name); save()
                destination = out/'clips'/name
                attempt = 1
                while destination.exists():
                    destination = out/'clips'/(name+f'_resume{attempt}'); attempt += 1
                result = export_clip(name, source, destination, plan, actor, stateful)
                assert result['status'] == 'REAL_CLIP_FULL_V3_GROUP_REFS_PIXELS_AND_ALL_NATIVE_CPU_ROWS_VALIDATED'
                state['completed'][name] = result
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < plan['new_export_budget_bytes']
                save()
            # Exporting a clip can outlast the producer. Re-read its terminal
            # receipt before interpreting the producer's liveness.
            producer = json.loads((source/'owner_state.json').read_text())
            if set(producer['completed'])-state['completed'].keys():
                continue
            if producer['status'] == 'BROADER_NATIVE_TRAIN43_VIDEO1056_FRAME_COLLECTION_COMPLETE':
                assert len(state['completed']) == 43 and sum(r['counts']['frames'] for r in state['completed'].values()) == 1056
                break
            assert producer['status'] != 'BROADER_NATIVE_TRAIN_COLLECTION_FAILED', producer.get('error')
            assert alive([producer['owner_pid'], producer['owner_starttick']]), 'Native source owner exited before complete receipt'
            state['status'] = 'WAITING_MORE_COMPLETED_NATIVE_TRAIN_CLIPS'; save(); time.sleep(20)
        state['status'] = 'ALL43_NATIVE_TRAIN_CLIPS_FULL_V3_GROUP_REFS_AND_CPU_PLANS_VALIDATED'; save()
    except Exception as error:
        state.update(status='BROADER_NATIVE_FULL_GROUP_EXPORT_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
