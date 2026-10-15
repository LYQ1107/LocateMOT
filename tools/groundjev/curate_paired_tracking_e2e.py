"""Curate the completed real same-video Full/Compact pipeline measurement."""
from itertools import zip_longest
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.run_paired_tracking_e2e import canonical
    from tools.groundjev.completed_gpu_handoff import validate_release
    directory = ROOT/'outputs/groundjev/e2e/full_compact_rmot800_v1'
    owner = json.loads((directory/'owner_state.json').read_text())
    assert owner['status'] == 'SAME800_SAME_GPU_FRESH_FULL_COMPACT_E2E_AND_DETECTOR_PAIRING_VALIDATED'
    assert owner['candidate_comparison']['same_actual_detector_candidates']
    assert owner['candidate_comparison']['different_frames'] == []
    assert set(owner['completed']) == {'full_native600','compact'}
    releases = validate_release(ROOT/'outputs/groundjev/gpu5_followups_v1/gpu_release.json',5)
    manifests, archives, results = {}, {}, {}
    for arm, entry in owner['completed'].items():
        run = ROOT/entry['run']
        actual = json.loads((run/'result.json').read_text())
        official = json.loads((run/'official_metrics.json').read_text())
        manifest = json.loads(Path(actual['video_manifest']).read_text())
        assert official == entry['official']
        assert actual['frames'] == official['frames'] == entry['decoded_frames'] == 800
        assert actual['resumed_completed_frames'] == 0 and actual['all_input_frames_processed']
        assert not actual['gt_read_by_runtime'] and not actual['future_frames_read_by_runtime']
        assert actual['association_replay_fps'] is None
        assert actual['end_to_end_fps'] == entry['actual_end_to_end_fps'] == 800/actual['complete_video_wall_seconds']
        assert actual['final_full_memory_export_in_pipeline_timer']
        assert actual['grounding_parser_policy'] == 'official_scoring'
        assert actual['grounding_decoder_policy'] == 'tracking_dense'
        assert official['perception_rejected_frames'] == actual['perception_rejected_frames']
        assert not entry['sampled_resources']['errors'] and entry['sampled_resources']['sampled_owned_GPU_sum_peak_MiB'] > 0
        assert Path(actual['final_identity_memory_path']).is_file() and (run/'tracks.mp4').is_file()
        query = json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())
        assert query['sentence'] == actual['specification']['expression'] == manifest['expression']
        frames, empty, alarms, empty_outputs = 0, 0, 0, 0
        with (run/'frame_decisions.jsonl').open() as trace:
            for line in trace:
                row = json.loads(line)
                assert row['frame'] == manifest['frame_numbers'][frames]
                assert row['image_path'] == manifest['images'][frames]
                if not query['label'].get(str(row['frame']),[]):
                    empty += 1
                    alarms += bool(row['outputs'])
                    empty_outputs += len(row['outputs'])
                frames += 1
        assert frames == 800 and empty == official['empty_target_frames']
        assert empty_outputs == official['empty_target_frame_predictions']
        reference = ROOT/'outputs/groundjev/multitask_runs/full_rmot_shared600_v1'/('single800_native600' if arm == 'full_native600' else 'single800_shared_compact_fresh')/'official_metrics.json'
        assert official['metrics'] == json.loads(reference.read_text())['metrics']
        latency = sorted(actual['latencies_seconds'])
        results[arm] = {**entry,'actual_invocation_git_commit':actual['invocation_git_commit'],
            'empty_target_frames':empty,'empty_frames_with_any_output':alarms,
            'empty_alarm_frame_rate':alarms/empty,'empty_target_detection_count':empty_outputs,
            'per_frame_inference_latency_seconds':{'mean':statistics.mean(latency),'p50':statistics.median(latency),
                'p95':latency[int(.95*len(latency))],'max':max(latency)},
            'latency_scope_note':'Per-frame inference timer ends after association/memory step; trailing log/state/video writes are covered by full pipeline wall time, not this latency distribution.',
            'matches_prior_same_candidate_official_accuracy':True}
        manifests[arm] = manifest
        archives[arm] = Path(actual['native_candidate_archive'])
    assert manifests['full_native600'] == manifests['compact']
    assert len({r['hardware_at_launch'].split(',')[0] for r in results.values()}) == 1
    frames, candidates = 0, 0
    rejected = []
    with archives['full_native600'].open() as left,archives['compact'].open() as right:
        for a,b in zip_longest(left,right):
            assert a is not None and b is not None
            a,b = json.loads(a),json.loads(b)
            assert canonical(a) == canonical(b)
            assert a['frame'] == manifests['compact']['frame_numbers'][frames]
            assert all(len(c['native_region_feature']) == 2560 for c in b['candidates'])
            if a['perception'].get('model_response_rejected'):rejected.append(a['frame'])
            candidates += len(a['candidates'])
            frames += 1
    assert frames == 800
    assert rejected == results['compact']['official']['perception_rejected_frames']
    assert rejected == results['full_native600']['official']['perception_rejected_frames']
    full,compact = results['full_native600'],results['compact']
    report = {'status':'REAL_PAIRED800_FULL_COMPACT_TIMING_RESOURCES_AND_ACCURACY_VALIDATED',
        'physical_gpu':5,'frames_each':frames,'identical_native_candidates_each':candidates,
        'identical_parser_rejected_frames':rejected,'results':results,
        'observed_pipeline_FPS_ratio_Compact_over_Full':compact['actual_end_to_end_fps']/full['actual_end_to_end_fps'],
        'observed_wall_time_reduction_fraction':1-compact['pipeline_wall_seconds']/full['pipeline_wall_seconds'],
        'sampled_total_owned_VRAM_reduction_fraction':1-compact['sampled_resources']['sampled_owned_GPU_sum_peak_MiB']/full['sampled_resources']['sampled_owned_GPU_sum_peak_MiB'],
        'all_prior_owned_groups_released_before_followup':releases['all_owned_groups_gone'],
        'limits':'One800frame single-expression pair,18GTboxes/782empty frames; poor recall remains. '
            'Fixed Full-then-Compact order, one seed, concurrent other CPU/project workloads; observed '
            'speed ratio is not counterbalanced/statistical evidence or realtime performance. Native '
            'services/core initialization excluded and separately timed. Total VRAM is a5second '
            'sampled lower bound, distinct from core allocator/RSS. Full uses real images; Compact '
            'additionally pays nativeROI service cost. CPU frozen token lookup is extra to the1.146M core.',
        'full_goal_complete':False}
    (directory/'validated_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','identical_native_candidates_each','observed_pipeline_FPS_ratio_Compact_over_Full','observed_wall_time_reduction_fraction','sampled_total_owned_VRAM_reduction_fraction')},indent=2))


if __name__ == '__main__':
    main()
