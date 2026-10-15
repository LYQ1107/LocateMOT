"""Collect completed official studies, checking against their actual run artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    studies = {
        'fixed_affinity_controls_v1': (25, 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE'),
        'task_subsets1200_cpu_v2_feature_sources':
            (24, 'ALL_FOUR_TASK_SUBSETS_AND_REGISTERED_OFFICIAL_CONTROLS_COMPLETE'),
        'context_sizes_v1': (12, 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE'),
        'hybrid_shortlist32_v1': (12, 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE'),
    }
    for name, (expected, terminal) in studies.items():
        directory = ROOT/'outputs/groundjev/compact'/name
        owner = json.loads((directory/'owner_state.json').read_text())
        assert owner['status'] == terminal and not owner.get('failed')
        completed = owner.get('completed_controls', owner.get('completed'))
        assert len(completed) == expected
        rows = {}
        for key, entry in completed.items():
            run = ROOT/entry['run']
            official = json.loads((run/'official_metrics.json').read_text())
            result = json.loads((run/'result.json').read_text())
            assert entry['metrics'] == official['metrics']
            assert entry['decoded_frames'] == official['frames'] == result['frames']
            assert result['all_input_frames_processed']
            assert not result['gt_read_by_runtime'] and not result['future_frames_read_by_runtime']
            assert result['end_to_end_fps'] is None
            assert entry.get('all_exact_candidate_geometry_order_and_frames_retained',
                             entry.get('all_native_geometry_labels_scores_order_and_frames_retained'))
            # Older JSON runs predate the explicit final-state path metadata.
            final_state = Path(result['final_identity_memory_path']) if 'final_identity_memory_path' in result else run/'final_identity_memory.json'
            assert final_state.is_file()
            assert (run/'tracks.mp4').is_file()
            rows[key] = {**entry, 'official_metrics_file': str(run/'official_metrics.json'),
                         'metrics_equal_actual_official_report': True,
                         'runtime_all_frames_causal_and_final_state_present': True}
        report = {
            'status': 'COMPLETED_STUDY_OFFICIAL_ARTIFACTS_VALIDATED',
            'study': name, 'completed_units': expected, 'results': rows,
            'validation_scope': 'Official scores and runtime frame counts agree with terminal owner; '
                'candidate ordering and decoded-video counts use already completed per-unit owner checks.',
            'metric_scales': 'HOTA/IDF1 fractions; TETA/AssocA/ClsA percentages; IDSW/FP/FN counts',
            'limits': 'One-seed registered validation scopes, frozen policy interventions or '
                'fixed-total-update training subsets; not complete datasets, isolated JEV proof or E2E speed.',
            'full_goal_complete': False,
        }
        (directory/'validated_summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, expected, report['status'])


if __name__ == '__main__':
    main()
