"""Audit frozen completed native RMOT labels before selecting a larger curriculum."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--completed-units', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = ROOT/'outputs/groundjev/training_data/rmot_allfit1814_query18_full_v3_groups_v2_shared_gpu7'
    state = json.loads((base/'owner_state.json').read_text())
    completed = dict(list(state['completed'].items())[:args.completed_units])
    assert len(completed) == args.completed_units
    totals, reports, physical = Counter(), {}, set()
    for name, summary in completed.items():
        directory = Path(summary['row_references']).parent
        assert json.loads((directory/'validated_summary.json').read_text()) == summary
        assert summary['dropped_rows'] == 0 and summary['all_actor_outputs_states_exact']
        alias_birth, counts = {}, Counter()
        for record in read_rows(directory/'candidate_registry.jsonl.gz'):
            counts['candidates'] += 1
            alias = record['selected_input_alias']
            if alias is not None:
                gid = record['posthoc_immutable_birth_GT_identity']
                if alias in alias_birth:
                    assert alias_birth[alias] == gid
                alias_birth[alias] = gid
                if gid is None:
                    assert record['assigned_update_target'] is None
                    counts['unknown_birth_assigned_inputs_correctly_without_forced_update_target'] += 1
                    if record['posthoc_GT_identity'] is not None:
                        counts['unknown_birth_assigned_inputs_with_current_released_GT_match'] += 1
        for row in read_rows(directory/'train_refs.jsonl.gz'):
            assert len(row['target']) == 1
            answer = next(iter(row['target']))
            family = row['task']; counts['rows'] += 1; counts[family+':'+answer] += 1
            if family in ('query_membership', 'empty_query_presence'):
                counts['next_semantic_curriculum_eligible_rows'] += 1
            if family == 'lifecycle':
                alias = int(row['groundjev_group_question_ref']['question'].removeprefix('life_'))
                if alias not in alias_birth:
                    counts['lifecycle_rows_alias_birth_not_auditable_from_selected_registry'] += 1
                elif alias_birth[alias] is None:
                    assert answer == 'TERMINATE'
                    counts['lifecycle_TERMINATE_rows_with_verified_unknown_birth'] += 1
                else:
                    assert answer != 'TERMINATE'
                    counts['lifecycle_rows_with_verified_known_birth'] += 1
        assert counts['rows'] == summary['counts']['rows']
        assert counts['candidates'] == summary['counts']['candidates']
        for answer in ('yes','no'):
            assert counts['query_membership:'+answer] == summary['counts'].get('query_membership:'+answer,0)
        manifest = json.loads((ROOT/'outputs/groundjev/training_data/rmot_allfit1814_query18_scope_v1/manifests'/f'{name}.json').read_text())
        physical.update((summary['video'],f) for f in manifest['frame_numbers'])
        totals.update(counts); reports[name] = dict(counts)
    result = dict(status='COMPLETED_NATIVE_RMOT_LABEL_DISTRIBUTION_AND_UNKNOWN_BIRTH_AUDITED',
        completed_units=len(completed), unit_counts=reports, totals=dict(totals),
        physical_frames=len(physical),
        finding='TERMINATE for an unknown-birth track is an exporter proxy; missing GT correspondence alone does not establish permanent exit.',
        future_curriculum='Only query_membership and empty_query_presence from this component initially; preserve all auxiliary rows for later controlled supervision work.',
        membership_limit='Released-query IoU>=.5 labels; not visibility, exhaustive physical-object validity or calibrated uncertainty.',
        no_original_targets_deleted_or_modified=True, no_unknown_birth_target_fabrication=True,
        no_GT_or_future_state_in_runtime=True, new_GPU_or_model_training_calls=0,
        current6000_curriculum_changed=False,
        diagnostic_uses_existing_reserved_future_curriculum16MiB=True, full_goal_complete=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    assert not args.out.exists()
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], completed_units=len(completed), physical_frames=len(physical), totals=dict(totals))),flush=True)


if __name__ == '__main__':
    main()
