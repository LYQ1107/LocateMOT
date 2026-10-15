"""Curate completed full800 native tracking units without polling the live owner."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    owner = json.loads((args.root/'owner_state.json').read_text())
    result = {}
    archive_bytes = 0
    for scope in ('single800', 'multi800'):
        name = scope+'_shared_compact_fresh'
        completed = owner['completed'][name]
        output = ROOT/completed['run']
        runtime = json.loads((output/'result.json').read_text())
        official = json.loads((output/'official_metrics.json').read_text())
        diagnostic = json.loads((output/'association_events.json').read_text())
        manifest = json.loads(Path(runtime['video_manifest']).read_text())
        query = json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())
        assert query['sentence'] == runtime['specification']['expression'] == manifest['expression']
        assert runtime['specification']['vocabulary'] == manifest['vocabulary']
        assert runtime['frames'] == completed['decoded_frames'] == len(manifest['images']) == 800
        assert runtime['resumed_completed_frames'] == 0 and runtime['resume_source'] is None
        assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        assert runtime['grounding_parser_policy'] == 'official_scoring' and runtime['grounding_decoder_policy'] == 'tracking_dense'
        assert runtime['native_feature_url'] and runtime['decision_device'] == 'cuda:0'
        assert runtime['end_to_end_fps'] == 800/runtime['complete_video_wall_seconds']
        assert runtime['final_full_memory_export_in_pipeline_timer']
        assert runtime['memory_checkpoint_format'] == 'json' and runtime['memory_export_interval'] == 1
        assert official == completed['official']
        source = Path(runtime['native_candidate_archive'])
        assert source.stat().st_size == runtime['native_candidate_archive_bytes']
        archive_bytes += source.stat().st_size
        count, candidates, targets, empty, empty_detections, empty_alarm_frames = 0, 0, 0, 0, 0, 0
        rejected = []
        with source.open() as native, (output/'frame_decisions.jsonl').open() as trace:
            for raw, saved in zip_longest(native, trace):
                assert raw is not None and saved is not None
                raw, saved = json.loads(raw), json.loads(saved)
                assert raw['frame'] == saved['frame'] == manifest['frame_numbers'][count]
                assert raw['image_path'] == saved['image_path'] == manifest['images'][count]
                geometry = [{k: v for k, v in c.items() if k not in
                    ('native_region_feature', 'temporal_region_feature')} for c in raw['candidates']]
                assert saved['candidates'] == geometry and saved['perception'] == raw['perception']
                assert all(len(c['native_region_feature']) == 2560 for c in raw['candidates'])
                assert len({r['track_id'] for r in saved['outputs']}) == len(saved['outputs'])
                if saved['perception'].get('model_response_rejected'):
                    rejected.append(saved['frame'])
                target_count = len(query['label'].get(str(saved['frame']), []))
                targets += target_count
                if not target_count:
                    empty += 1
                    empty_detections += len(saved['outputs'])
                    empty_alarm_frames += bool(saved['outputs'])
                count += 1
                candidates += len(raw['candidates'])
        assert count == 800 and rejected == official['perception_rejected_frames'] == runtime['perception_rejected_frames']
        assert targets == official['release_query_GT_boxes'] and empty == official['empty_target_frames']
        assert empty_detections == official['empty_target_frame_predictions']
        assert runtime['trainable_compact_core_parameters'] == 1146511
        assert (output/'tracks.mp4').is_file() and Path(runtime['final_identity_memory_path']).is_file()
        result[scope] = {'run': str(output), 'frames': count, 'native_candidates': candidates,
            'native_parser_rejected_frames': len(rejected), 'strict_wire_invalid_frames': len(runtime['strict_wire_invalid_frames']),
            'strict_wire_diagnostic_not_reclassified_or_discarded': True,
            'metrics': official['metrics'], 'query_GT_boxes': targets,
            'posthoc_proposal_coverage': official['posthoc_proposal_coverage_IoU_05'],
            'empty_target_frames': empty, 'predicted_detections_on_empty_frames': empty_detections,
            'empty_frames_with_any_prediction': empty_alarm_frames,
            'empty_frame_alarm_rate': empty_alarm_frames/empty,
            'actual_unresumed_pipeline_FPS': runtime['end_to_end_fps'],
            'pipeline_wall_seconds': runtime['complete_video_wall_seconds'],
            'model_initialization_seconds_excluded': runtime['model_initialization_seconds'],
            'timing_scope': runtime['timing_scope'],
            'core_GPU_Torch_allocator_peak_bytes': runtime['peak_decision_gpu_memory_bytes'],
            'core_process_RSS_peak_bytes': runtime['peak_decision_process_rss_bytes'],
            'frozen_CPU_One_text_lookup': runtime['frozen_CPU_text_table'],
            'core_memory_does_not_include_external_DLM_and_ROI_services': True,
            'native_archive_bytes': source.stat().st_size,
            'all_frames_candidates_geometry_labels_scores_order_and_video_validated': True,
            'query_membership_matched_physical_GT_only': diagnostic['posthoc_query_membership_matched_physical_GT_only'],
            'physical_birth_anchor_recovery_posthoc': diagnostic['posthoc_identity_recovery'],
            'physical_memory_update_counts': {k: v for k, v in diagnostic['counts'].items() if k.startswith('memory_')},
            'native_invocation_git_commit': runtime['invocation_git_commit'],
            'video': str((output/'tracks.mp4').resolve())}
    assert archive_bytes <= owner['total_new_feature_cache_budget_bytes'] == 2*2**30
    single = owner['completed']['single800_simple']
    result['single800_same_candidates_simple'] = {'run': single['run'], 'frames': single['decoded_frames'],
        'metrics': single['official']['metrics'], 'same_native_candidates_validated': single['all_native_candidates_and_rejected_frames_retained'],
        'pipeline_FPS': None, 'replay_speed_not_comparable_to_live_Compact_FPS': True}
    early_path = ROOT/'outputs/groundjev/multitask_runs/multi800_simple_cpu_v1/owner_state.json'
    if early_path.is_file():
        early = json.loads(early_path.read_text())
        if early['status'] == 'EARLY_FIXED_CANDIDATE_MULTI800_SIMPLE_VALIDATED':
            assert early['decoded_frames'] == 800 and early['native_candidates'] == 2208
            actual = json.loads((ROOT/early['run']/'result.json').read_text())
            assert actual['end_to_end_fps'] is None
            result['multi800_same_candidates_simple'] = {'run': early['run'], 'frames': 800,
                'metrics': early['metrics'], 'same_native_candidates_validated': True,
                'pipeline_FPS': None, 'replay_speed_not_comparable_to_live_Compact_FPS': True}
    for scope in ('single800', 'multi800'):
        key = scope+'_same_candidates_simple'
        if key not in result:
            continue
        output = ROOT/result[key]['run']
        actual = json.loads((output/'result.json').read_text())
        manifest = json.loads(Path(actual['video_manifest']).read_text())
        query = json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())
        rows = [json.loads(line) for line in (output/'frame_decisions.jsonl').open()]
        assert [r['frame'] for r in rows] == manifest['frame_numbers']
        empty = [r for r in rows if not query['label'].get(str(r['frame']), [])]
        alarms = sum(bool(r['outputs']) for r in empty)
        result[key].update(empty_target_frames=len(empty), empty_frames_with_any_prediction=alarms,
            empty_frame_alarm_rate=alarms/len(empty), predicted_detections_on_empty_frames=sum(len(r['outputs']) for r in empty))
    report = {'status': 'BOTH_FULL800_NATIVE_COMPACT_RMOT_OUTPUTS_AND_TIMING_VALIDATED',
        'physical_GPU': owner['physical_gpu'], 'results': result, 'total_new_archive_bytes': archive_bytes,
        'registered_new_archive_cap_bytes': 2*2**30, 'new_feature_files_created_by_validation': False,
        'absolute_pipeline_speed_only': True,
        'paired_Full_Compact_speed_or_total_service_VRAM_claim': False,
        'limits': 'Two expressions on one legal validation video, not complete RMOT dataset. Single target has only18GT/782emptyframes; empty-frame alarm rate and detection counts are distinct. Physical recovery counts concern all annotated objects, not only query targets. Unmatched/unknown birth ownership does not prove every such memory write wrong. No promotion or full-goal completion.',
        'GT_used_for_posthoc_only': True, 'full_goal_complete': False}
    (args.root/'fresh_compact800_validation_and_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: {x: v for x, v in r.items() if x in
        ('metrics', 'native_candidates', 'actual_unresumed_pipeline_FPS', 'empty_frame_alarm_rate', 'empty_frames_with_any_prediction')}
        for k, r in result.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
