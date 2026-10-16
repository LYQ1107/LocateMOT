"""Freeze completed native shared-model controls and actual assigned-write effects."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.groundjev.run_history_refresh_controls import validate_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    owner = json.loads((args.owner/'owner_state.json').read_text())
    plan = json.loads(args.plan.read_text())
    completed, comparisons = {}, {}
    for scope in ('mot_full1203', 'rmot_single800', 'rmot_multi800'):
        for arm in ('before', 'default', 'assigned_identity'):
            key = arm+'_'+scope
            recorded = owner['completed'][key]
            validated = validate_run(Path(recorded['run']), plan['scopes'][scope], Path(recorded['checkpoint']))
            assert validated == recorded
            runtime = json.loads((Path(recorded['run'])/'result.json').read_text())
            assert bool(runtime.get('assigned_identity_updates', False)) == (arm == 'assigned_identity')
            completed[key] = validated
        first = Path(completed['default_'+scope]['run'])/'frame_decisions.jsonl'
        second = Path(completed['assigned_identity_'+scope]['run'])/'frame_decisions.jsonl'
        counts = dict(frames=0, exact_exported_output_frames=0, exact_event_assignment_frames=0,
                      assigned_identity_calls=0, existing_assignments_requeried=0,
                      changed_same_state_update_probabilities=0, update_gate_flips_at_half=0,
                      observation_hold_to_assigned_update=0, observation_update_to_assigned_hold=0)
        maximum = 0.
        with first.open() as fa, second.open() as fb:
            for line_a, line_b in zip_longest(fa, fb):
                assert line_a is not None and line_b is not None
                a, b = json.loads(line_a), json.loads(line_b)
                assert a['frame'] == b['frame'] and a['candidates'] == b['candidates']
                counts['frames'] += 1
                counts['exact_exported_output_frames'] += a['outputs'] == b['outputs']
                event_key = lambda row: [(e.get('candidate'),e['action'],e.get('track_id')) for e in row['state']['events']]
                counts['exact_event_assignment_frames'] += event_key(a) == event_key(b)
                report = b['state']['decision_metadata']['assigned_identity_update']
                counts['assigned_identity_calls'] += report['native_second_pass_calls']
                counts['existing_assignments_requeried'] += report['existing_assignments']
                queried = 0
                for event in b['state']['events']:
                    if 'candidate' not in event:
                        assert 'lifecycle_decision' in event
                        continue
                    decision = event['decision']
                    if 'observation_update_probability' not in decision:
                        continue
                    assert event['action'] in ('ACTIVE','REACTIVATE')
                    queried += 1
                    old, new = decision['observation_update_probability'], decision['update_probability']
                    counts['changed_same_state_update_probabilities'] += old != new
                    counts['update_gate_flips_at_half'] += (old >= .5) != (new >= .5)
                    counts['observation_hold_to_assigned_update'] += old < .5 <= new
                    counts['observation_update_to_assigned_hold'] += new < .5 <= old
                    maximum = max(maximum, abs(old-new))
                assert queried == report['existing_assignments']
        counts['max_absolute_same_state_update_probability_change'] = maximum
        counts['paired_official_metrics_equal'] = completed['default_'+scope]['official_metrics'] == completed['assigned_identity_'+scope]['official_metrics']
        comparisons[scope] = counts
    result = dict(status='NATIVE_SHARED6000_FIRST_THREE_FULL_SCOPES_ALL9_UNITS_VALIDATED',
        completed=completed, comparisons=comparisons,
        same_state_update_audit='Compare observation versus assigned-identity probability within the same assigned arm pre-write causal input; future histories may differ from default.',
        scope='Original registered DT1203 and RMOTsingle800/multi800, same final6000 shared checkpoint; TAO remains pending. No task-specific checkpoint/threshold selection, no full dataset or ordinary-model parity claim.',
        new_GPU_or_optimizer_updates=0,original_owners_or_protocol_changed=False,full_goal_complete=False)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons)))


if __name__ == '__main__':
    main()
