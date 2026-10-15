"""Collect the terminal two800frame study's actual official and diagnostic artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT/'outputs/groundjev/multitask_runs/full_rmot_shared600_v1'
    owner = json.loads((directory/'owner_state.json').read_text())
    assert owner['status'] == 'FULL_TWO800_RMOT_FRESH_COMPACT_AND_ALL_MATCHED_FULL_CONTROLS_COMPLETE'
    assert owner['owned_native_service_groups_stopped'] and len(owner['completed']) == 12
    rows = {}
    for name, entry in owner['completed'].items():
        run = ROOT/entry['run']
        official = json.loads((run/'official_metrics.json').read_text())
        runtime = json.loads((run/'result.json').read_text())
        events = json.loads((run/'association_events.json').read_text())
        assert entry['official'] == official
        assert entry['decoded_frames'] == official['frames'] == runtime['frames'] == events['frames'] == 800
        assert runtime['all_input_frames_processed']
        assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        assert runtime['resumed_completed_frames'] == 0
        assert entry['all_native_candidates_and_rejected_frames_retained']
        assert runtime['perception_rejected_frames'] == official['perception_rejected_frames']
        fresh = name.endswith('_shared_compact_fresh')
        assert (runtime['end_to_end_fps'] is not None) == fresh
        if runtime['backend'] == 'full':
            assert runtime['evidence_format'] == 'v3_crop_motion'
            membership = events['posthoc_query_membership_matched_physical_GT_only']
            assert membership['head_was_present'] and membership['threshold'] == .5
            first = json.loads((run/'frame_decisions.jsonl').open().readline())
            assert first['state']['decision_metadata']['query_output_policy'] == 'native Noul membership>=.5; physical identity memory retained before output filtering'
        final_state = Path(runtime.get('final_identity_memory_path', str(run/'final_identity_memory.json')))
        assert final_state.is_file() and (run/'tracks.mp4').is_file()
        rows[name] = {
            'run': str(run), 'frames': 800, 'metrics': official['metrics'],
            'same_native_candidates_and_all_parser_rejections_validated_by_terminal_owner': True,
            'query_GT_boxes': official['release_query_GT_boxes'],
            'empty_target_frames': official['empty_target_frames'],
            'empty_target_detection_count': official['empty_target_frame_predictions'],
            'native_parser_rejected_frames': len(official['perception_rejected_frames']),
            'native_membership': events['posthoc_query_membership_matched_physical_GT_only'],
            'correct_option_coverage': events['correct_option_coverage'],
            'existing_association_accuracy': events['existing_association_accuracy'],
            'identity_recovery': events['posthoc_identity_recovery'],
            'actual_live_pipeline_FPS': runtime['end_to_end_fps'],
            'perception_excluded_replay_FPS': runtime['association_replay_fps'],
            'core_Torch_GPU_allocator_peak_bytes': runtime.get('peak_decision_gpu_memory_bytes'),
            'core_process_RSS_peak_bytes': runtime.get('peak_decision_process_rss_bytes'),
            'GT_and_future_frames_not_read_by_runtime': True,
        }
    report = {'status': 'ALL12_TWO800_RMOT_OFFICIAL_ARTIFACTS_VALIDATED', 'results': rows,
        'total_feature_archive_bytes': owner['new_feature_cache_bytes'],
        'full_goal_complete': False,
        'limits': 'Two expressions on one legal validation video. Simple/Full replay the same live Compact '
            'candidate source; only fresh Compact is perception-inclusive here. Paired same-video fresh '
            'Full/Compact is a separate running study. Core allocator/RSS are not external-service/total '
            'VRAM peaks. Birth-anchor diagnostic sets evolve with each model. No model promotion.'}
    (directory/'all12_validated_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])


if __name__ == '__main__':
    main()
