"""Explain completed RMOT output losses at the unchanged membership threshold."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    results, common_alignment = {}, None
    for run in args.runs:
        result = json.loads((run/'result.json').read_text())
        assert result['task'] == 'rmot' and result['all_input_frames_processed']
        official = json.loads((run/'official_metrics.json').read_text())
        diagnostics = json.loads((run/'association_events.json').read_text())
        manifest = json.loads(Path(result['video_manifest']).read_text())
        query = json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())['label']
        alignment = {(e['frame'],e['candidate']): e['matched_gt_identity'] for e in diagnostics['events']}
        if common_alignment is None:
            common_alignment = alignment
        assert alignment == common_alignment, 'Runs must use the same candidates and posthoc physical alignment'
        counts = {key: Counter() for key in ('query_positive', 'query_negative', 'unmatched_unknown')}
        head = Counter()
        frames, observations, actual_outputs = 0, 0, 0
        with (run/'frame_decisions.jsonl').open() as trace:
            for line in trace:
                row = json.loads(line)
                frames += 1
                assert row['frame'] == manifest['frame_numbers'][frames-1]
                expected_outputs = []
                targets = set(query.get(str(row['frame']), []))
                for event in row['state']['events']:
                    if 'candidate' not in event:
                        continue
                    index = event['candidate']
                    identity = alignment[(row['frame'],index)]
                    group = 'unmatched_unknown' if identity is None else 'query_positive' if identity in targets else 'query_negative'
                    probability = event['decision']['query_membership_probability']
                    accepts = probability >= .5
                    assigned = event.get('track_id') is not None
                    counts[group]['total'] += 1
                    counts[group]['head_accepts' if accepts else 'head_rejects'] += 1
                    if not assigned:
                        counts[group]['no_physical_assignment'] += 1
                    elif not accepts:
                        counts[group]['assigned_then_membership_rejected'] += 1
                    else:
                        counts[group]['emitted'] += 1
                        expected_outputs.append((event['track_id'], tuple(row['candidates'][index]['box'])))
                    if identity is not None:
                        positive = group == 'query_positive'
                        head['TP' if accepts and positive else 'FP' if accepts else 'FN' if positive else 'TN'] += 1
                    observations += 1
                actual = [(o['track_id'],tuple(o['box'])) for o in row['outputs']]
                assert expected_outputs == actual
                actual_outputs += len(actual)
        assert frames == result['frames'] == official['frames'] == diagnostics['frames']
        assert observations == len(alignment)
        registered_head = diagnostics['posthoc_query_membership_matched_physical_GT_only']
        assert registered_head['head_was_present'] and dict(head) == registered_head['counts']
        for counter in counts.values():
            assert counter['total'] == sum(counter[k] for k in
                ('no_physical_assignment','assigned_then_membership_rejected','emitted'))
        results[str(run)] = {
            'official_metrics': official['metrics'], 'frames': frames, 'candidates': observations,
            'actual_emitted_detections': actual_outputs,
            'registered_target_proposal_coverage': official['posthoc_proposal_coverage_IoU_05'],
            'matched_candidate_output_funnel': {k:dict(v) for k,v in counts.items()},
            'query_membership_matched_physical_GT_only': registered_head,
            'existing_identity_events': diagnostics['counts'].get('existing_identity_events'),
            'correct_option_coverage': diagnostics['correct_option_coverage'],
            'existing_association_accuracy': diagnostics['existing_association_accuracy'],
            'actual_outputs_exactly_reconstructed_from_saved_actions_and_membership': True,
        }
    report = {'status': 'ACTUAL_MATCHED_RMOT_OUTPUT_FUNNELS_VALIDATED', 'results': results,
        'threshold': .5, 'thresholds_or_predictions_or_states_changed': False,
        'new_model_invocations': 0, 'GT_used_posthoc_only': True, 'full_goal_complete': False,
        'limits': 'Disjoint funnel counts describe GT-aligned candidate observations, not official FN '
            'decomposition across HOTA thresholds. Proposal coverage uses per-target overlap and may differ '
            'from one-to-one physical alignment. Unknown/unmatched candidates are not labeled negatives. '
            'Birth-anchor association event sets differ with each causal history; no oracle correction or metric replacement.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    for run, row in results.items():
        print(Path(run).name, json.dumps(row['matched_candidate_output_funnel']['query_positive']))


if __name__ == '__main__':
    main()
