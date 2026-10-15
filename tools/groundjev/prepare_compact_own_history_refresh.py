"""Paired physical-proposal packs: old Full history versus actual Compact history."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.prepare_tracking_decisions import PROTOCOL, annotations

    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    torch.set_num_threads(plan['actor_cpu_threads']); torch.manual_seed(0)
    diagnosis = json.loads((ROOT/plan['diagnosis_plan']).read_text())
    source = ROOT/plan['diagnosis']
    rows = [json.loads(line) for line in (ROOT/diagnosis['source']).open()]
    references = [json.loads(line) for line in (source/'actor_frames.jsonl').open()]
    labels = {(r['sequence'], r['frame'], r['candidate']): r for r in
              map(json.loads, (source/'posthoc_candidate_labels.jsonl').open())}
    base = torch.load(ROOT/plan['source_pack']/'examples.pt', map_location='cpu', weights_only=True)
    old = [dict(e, native_teacher=False) if e['task_index'] == 0 else e for e in base['examples']
           if not (e['task_index'] == 0 and e['derived_withholding'])]
    physical = {e['id']: e for e in old if e['task_index'] == 0}
    assert len(physical) == 80 and sum(len(e['candidates']) for e in physical.values()) == 1184
    feature_groups = defaultdict(list)
    for example in physical.values():
        feature_groups[example['source_sequence'], example['frame']].append(example)
    for row in rows:
        offset = 0
        for example in sorted(feature_groups[row['sequence'], row['frame']],
                              key=lambda e: int(re.search(r'_batch(\d+)$', e['id'])[1])):
            group = row['candidates'][offset:offset+len(example['candidates'])]
            assert [c['box'] for c in group] == example['candidate_boxes']
            for candidate, vector in zip(group, example['candidates']):
                candidate['native_region_feature'] = vector.tolist()
            offset += len(group)
        assert offset == len(row['candidates'])
    sequences = {}
    for name, file in diagnosis['sequence_protocols'].items():
        seq = next(s for s in json.loads((PROTOCOL/file).read_text())['sequences'] if s['sequence'] == name)
        frames = {r['frame'] for r in rows if r['sequence'] == name}
        assert frames <= {int(Path(f).stem) for f in seq['train_frames']}
        sequences[name] = seq, annotations(seq, frames)

    class Recorder(CompactDecision):
        def accepted_batch(self, group, ranked, inputs):
            self.captured.append({'inputs': {k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                for k, v in inputs.items() if k != 'class_text'},
                'track_ids': [t.track_id for t in ranked], 'lost_ages': [t.lost_age for t in ranked]})

        def decide(self, *args):
            result = super().decide(*args)
            self.actual_lifecycle = result[1]
            return result

    actor = Recorder(ROOT/plan['checkpoint'], device='cpu', batch_candidates=16, shortlist=32)
    refreshed, previous, anchors, tracker = {}, None, {}, None
    target_counts = Counter(); history_values = []
    for row, reference in zip(rows, references):
        name, frame, candidates = row['sequence'], row['frame'], row['candidates']
        seq, gt = sequences[name]
        if name != previous:
            previous, anchors = name, {}
            tracker = OnlineTracker(actor)
        before, prior_ids = dict(anchors), set(tracker.tracks)
        truth = {t['gt_id']: t for t in gt[frame]}
        gids = {i: labels[name, frame, i]['candidate_GT_identity'] for i in range(len(candidates))}
        present = {gid for gid in gids.values() if gid is not None}
        actor.captured = []
        with Image.open(row['image_path']) as raw:
            image = raw.convert('RGB')
        outputs, state = tracker.step(image, candidates, frame/seq['fps'],
            {'task': 'mot', 'vocabulary': ['person'], 'expression': None})
        assert outputs == reference['outputs'] and state == reference['state']
        events = {e['candidate']: e for e in state['events'] if 'candidate' in e}
        offset = 0
        for batch, captured in enumerate(actor.captured):
            inputs, ids = captured['inputs'], captured['track_ids']
            n, m = len(inputs['candidates']), len(ids)
            identifier = f'{name}_{frame}_batch{batch}'
            original = physical[identifier]
            assert torch.equal(inputs['candidates'], original['candidates'])
            assert torch.equal(inputs['candidate_geo'], original['candidate_geo'])
            assert torch.equal(inputs['spec_text'], original['spec_text'])
            target, update, reliability = [], [], []
            for i in range(n):
                gid = gids[offset+i]
                correct = [tid for tid in ids if gid is not None and before.get(tid) == gid]
                known = any(gid is not None and before.get(tid) == gid for tid in prior_ids)
                value = m+1 if gid is None else ids.index(min(correct)) if correct else m+2 if known else m
                target.append(value)
                reliable = gid is not None and truth[gid]['visibility'] >= .6 and value != m+2
                update.append(float(reliable))
                reliability.append(0 if gid is None else 1 if not reliable else 3 if truth[gid]['visibility'] >= .85 else 2)
                target_counts['EXISTING' if value < m else ('NEW', 'IGNORE', 'NO_MATCH')[value-m]] += 1
            life = [3 if before.get(tid) is None else 2 if before[tid] in present and lost else
                    0 if before[tid] in present else 1 for tid, lost in zip(ids, captured['lost_ages'])]
            names = [str(tid) for tid in ids]+['NEW', 'IGNORE', 'NO_MATCH']
            teacher = torch.tensor([[events[offset+i]['decision']['probabilities'][k] for k in names] for i in range(n)])
            value = dict(original, **inputs, target=torch.tensor(target), update_target=torch.tensor(update),
                reliability_target=torch.tensor(reliability), life_target=torch.tensor(life, dtype=torch.long),
                teacher=teacher/teacher.sum(-1, keepdim=True),
                update_teacher=torch.tensor([events[offset+i]['decision']['update_probability'] for i in range(n)]),
                reliability_teacher=torch.tensor([[events[offset+i]['decision']['reliability_probabilities'][str(k)] for k in range(4)] for i in range(n)]),
                life_teacher=torch.tensor([[actor.actual_lifecycle[tid]['probabilities'][k] for k in
                    ('ACTIVE', 'LOST', 'REACTIVATE', 'TERMINATE')] for tid in ids]).reshape(m, 4),
                life_teacher_mask=torch.ones(m, dtype=torch.bool), native_teacher=False,
                history_source='actual fixed sharedCompact CPU own history; GT only outside input',
                original_physical_group_id=identifier)
            refreshed[identifier] = value
            history_values.extend(inputs['history_geo'][:, 4:8].tolist())
            offset += n
        assert offset == len(candidates)
        for i, event in events.items():
            if event['action'] == 'NEW': anchors[event['track_id']] = gids[i]
    assert len(rows) == len(references) == 48 and refreshed.keys() == physical.keys()
    new = [refreshed[e['id']] if e['task_index'] == 0 else e for e in old]
    assert [e['id'] for e in old] == [e['id'] for e in new]
    assert Counter(e['task_index'] for e in old) == {0: 80, 1: 325, 2: 62}
    assert not torch.cuda.is_initialized()
    out = ROOT/plan['pack_output']; out.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, examples in [('old', old), ('refreshed', new)]:
        directory = out/name; directory.mkdir()
        torch.save(dict(base, examples=examples), directory/'examples.pt')
        record = {'status': 'DECLARED_JOINT_SEMANTIC_TRAIN_PACK_COMPLETE', 'arm': name,
            'examples': len(examples), 'task_groups': {'mot': 80, 'rmot': 325, 'ovmot': 62},
            'MOT_frames': 48, 'MOT_candidates': 1184, 'derived_withholding_aug_omitted': 69,
            'MOT_distillation_disabled_in_both_new_arms': True, 'new_GPU_or_ROI_archive': False}
        (directory/'owner_state.json').write_text(json.dumps(record, indent=2)+'\n')
        paths[name] = str(directory)
    from tools.groundjev.diagnose_compact_action_temporal_domain import distribution
    summary = {'status': 'PAIRED_COMPACT_PHYSICAL_HISTORY_PACKS_AND_REAL_INPUT_CAPTURE_VALIDATED',
        'packs': paths, 'physical_MOT_groups_each': 80, 'physical_candidates_each': 1184,
        'all48_default_outputs_states_and_instrumented_actor_exact': True,
        'actual_native_forward_inputs_captured_before_memory_write': True,
        'same_every_row_ID_and_nonMOT_examples_and_class_table': True,
        'refreshed_target_action_counts': dict(target_counts),
        'refreshed_history_scalar_domain': distribution(history_values),
        'new_training_updates': 0, 'feature_pack_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        'combined_registered_feature_upper_bytes': plan['combined_registered_feature_upper_bytes'],
        'physical_coverage_or_large_foundation_training_claim': False, 'full_goal_complete': False}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < plan['new_pack_feature_budget_bytes']
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
