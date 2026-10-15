"""Measure actual CPU Compact fit-history support, with labels outside the actor."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import numpy as np
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    import torch
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.decision import overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.prepare_tracking_decisions import PROTOCOL, annotations

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['device'] == 'cpu'
    assert plan['combined_registered_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    source, out = ROOT/plan['source'], ROOT/plan['output']
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'COMPACT_ACTUAL_FIT48_CPU_SUPPORT_RUNNING', 'plan': str(args.plan)}

    def save():
        (out/'owner_state.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)

    save()
    try:
        torch.set_num_threads(plan['torch_threads'])
        torch.manual_seed(0)
        rows = [json.loads(line) for line in source.open()]
        assert len(rows) == plan['expected_frames']
        assert sum(len(row['candidates']) for row in rows) == plan['expected_candidates']
        frames = defaultdict(set)
        for row in rows:
            assert row['frame'] not in frames[row['sequence']]
            frames[row['sequence']].add(row['frame'])
        pack = torch.load(ROOT/plan['existing_native_ROI_pack'], map_location='cpu', weights_only=True)
        features = defaultdict(list)
        for example in pack:
            # Withholding augments identity options, not physical detector candidates.
            if not example['derived_withholding']:
                features[example['source_sequence'], example['frame']].append(example)
        assert set(features) == {(r['sequence'], r['frame']) for r in rows}
        for row in rows:
            offset = 0
            for example in sorted(features[row['sequence'], row['frame']],
                                  key=lambda e: int(re.search(r'_batch(\d+)$', e['id'])[1])):
                vectors = example['candidates']
                group = row['candidates'][offset:offset+len(vectors)]
                assert [c['box'] for c in group] == example['candidate_boxes']
                assert vectors.shape == (len(group), 2560) and bool(torch.isfinite(vectors).all())
                for candidate, vector in zip(group, vectors):
                    candidate['native_region_feature'] = vector.tolist()
                offset += len(group)
            assert offset == len(row['candidates'])
        protocols = {}
        for name, file in plan['sequence_protocols'].items():
            protocol = json.loads((PROTOCOL/file).read_text())
            seq = next(seq for seq in protocol['sequences'] if seq['sequence'] == name)
            allowed = {int(Path(frame).stem): frame for frame in seq['train_frames']}
            assert frames[name] <= allowed.keys()
            protocols[name] = seq, allowed, annotations(seq, frames[name])
        actor = CompactDecision(ROOT/plan['checkpoint'], device='cpu',
            batch_candidates=plan['candidate_group_size'], shortlist=plan['shortlist_limit'],
            shortlist_policy=plan['shortlist_policy'])
        assert all(p.device.type == 'cpu' for p in actor.model.parameters())
        assert not actor.model.training
        counts, per_sequence = Counter(), defaultdict(Counter)
        previous, anchors, tracker = None, {}, None
        started = time.monotonic()
        with (out/'actor_frames.jsonl').open('w') as trace, (out/'posthoc_candidate_labels.jsonl').open('w') as labels:
            for row in rows:
                name, frame, candidates = row['sequence'], row['frame'], row['candidates']
                seq, allowed, truth = protocols[name]
                path = Path(seq['image_dir'])/allowed[frame]
                assert path.resolve() == Path(row['image_path']).resolve()
                assert all(len(c['native_region_feature']) == 2560 for c in candidates)
                if name != previous:
                    previous, anchors = name, {}
                    tracker = OnlineTracker(actor, lost_seconds=plan['lost_seconds'],
                        update_threshold=plan['update_threshold'], memory_policy=plan['memory_policy'])
                before = dict(anchors)
                # Only causal image/proposals/state enter inference. Ground truth is used below.
                with Image.open(path) as raw:
                    image = raw.convert('RGB')
                outputs, state = tracker.step(image, candidates, frame/seq['fps'],
                    {'task': 'mot', 'vocabulary': ['person'], 'expression': None})
                trace.write(json.dumps({'sequence': name, 'frame': frame, 'image_path': str(path),
                    'candidate_source': str(source), 'candidate_count': len(candidates),
                    'source_perception': row['perception'], 'outputs': outputs, 'state': state})+'\n')
                events = {e['candidate']: e for e in state['events'] if 'candidate' in e}
                assert set(events) == set(range(len(candidates)))
                targets = truth[frame]
                matched = {}
                if candidates and targets:
                    matrix = np.asarray([[overlap(c['box'], t['box']) for t in targets] for c in candidates])
                    rr, cc = linear_sum_assignment(-matrix)
                    matched = {int(i): targets[int(j)] for i, j in zip(rr, cc) if matrix[i, j] >= .5}
                for i, event in events.items():
                    target, assigned = matched.get(i), event.get('track_id')
                    gid = target['gt_id'] if target else None
                    answer = None
                    if assigned is None:
                        category = 'no_final_identity'
                    elif event['action'] == 'NEW':
                        category = 'NEW_initialization'
                    elif gid is None:
                        category = 'unmatched_not_universal_negative_truth'
                    elif before.get(assigned) is None:
                        category = 'unknown_birth_anchor'
                    elif before[assigned] == gid:
                        category = 'known_consistent_existing_identity'
                        answer = 'yes' if target['visibility'] >= .6 else 'no'
                    else:
                        category = 'known_conflicting_existing_identity'
                        answer = 'no'
                    counts[category] += 1
                    per_sequence[name][category] += 1
                    if answer is not None:
                        counts['potential_supervised:'+answer] += 1
                        per_sequence[name]['potential_supervised:'+answer] += 1
                    labels.write(json.dumps({'sequence': name, 'frame': frame, 'candidate': i,
                        'actual_final_action': event['action'], 'actual_actor_identity': assigned,
                        'immutable_birth_GT_identity': before.get(assigned), 'candidate_GT_identity': gid,
                        'eligibility': category, 'potential_assigned_ID_target': answer,
                        'GT_label_fields_outside_model_input': True})+'\n')
                for i, event in events.items():
                    if event['action'] == 'NEW':
                        anchors[event['track_id']] = matched[i]['gt_id'] if i in matched else None
                counts['frames'] += 1
                counts['candidates'] += len(candidates)
                counts['source_rejected_frames'] += bool(row['perception'].get('model_response_rejected'))
                per_sequence[name]['frames'] += 1
                per_sequence[name]['candidates'] += len(candidates)
        assert counts['frames'] == plan['expected_frames'] and counts['candidates'] == plan['expected_candidates']
        categories = ['no_final_identity', 'NEW_initialization', 'unmatched_not_universal_negative_truth',
            'unknown_birth_anchor', 'known_consistent_existing_identity', 'known_conflicting_existing_identity']
        assert sum(counts[c] for c in categories) == counts['candidates']
        assert not torch.cuda.is_initialized()
        summary = {'status': 'ACTUAL_COMPACT_CPU_FIT48_IDENTITY_SUPPORT_DIAGNOSIS_COMPLETE',
            'checkpoint': str(ROOT/plan['checkpoint']), 'candidate_source': str(source),
            'native_ROI_source': str(ROOT/plan['existing_native_ROI_pack']),
            'existing_ROI_groups_joined_by_exact_sequence_frame_batch_and_boxes': sum(map(len, features.values())),
            'counts': dict(counts), 'per_sequence': {k: dict(v) for k, v in per_sequence.items()},
            'actual_existing_assignments': sum(counts[c] for c in categories[2:]),
            'device': 'cpu', 'dtype': 'float32', 'torch_threads': plan['torch_threads'],
            'causal_rollout_elapsed_s_excluding_model_and_annotation_loading': time.monotonic()-started,
            'source_candidate_boxes_labels_and_rejects_unchanged': True,
            'ROI_features_reused_exactly_from_existing_native_pack_not_synthesized': True,
            'GT_only_external_posthoc_labels_no_state_corrections': True,
            'unknown_or_unmatched_not_negative_targets': True,
            'actor': 'existing trained sharedCompact, not a Full native actor or teacher-corrected state',
            'new_training_rows_exported': 0, 'new_GPU_jobs': 0, 'new_image_or_feature_artifacts': 0,
            'fit_support_not_held_out_tracking_accuracy_or_Full_training_gain': True,
            'combined_registered_upper_bytes': plan['combined_registered_upper_bytes'],
            'full_goal_complete': False}
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < plan['output_byte_budget']
        status.update(status=summary['status'], counts=summary['counts'], summary=str(out/'summary.json'))
        save()
    except Exception as error:
        status.update(status='COMPACT_ACTUAL_FIT48_CPU_SUPPORT_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
