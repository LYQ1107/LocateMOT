"""Finish the multi800 fixed-candidate CPU baseline independently of slow Full jobs."""
import argparse
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = ROOT/'outputs/groundjev/multitask_runs/full_rmot_shared600_v1'
    proof = json.loads((original/'fresh_compact800_validation_and_summary.json').read_text())
    assert proof['status'] == 'BOTH_FULL800_NATIVE_COMPACT_RMOT_OUTPUTS_AND_TIMING_VALIDATED'
    manifest = ROOT/'outputs/groundjev/multitask_pilot_data/rmot_multi_empty_full.json'
    source = original/'multi800_shared_compact_fresh/frame_candidates.jsonl'
    output = args.out/'run'
    state = {'status': 'STARTING_FIXED_CANDIDATE_MULTI800_SIMPLE_CPU', 'owner_pid': os.getpid(),
        'additional_GPUs': 0, 'new_feature_cache_files': False,
        'existing_GPU5_owner_or_inference_interrupted': False,
        'protocol': 'same800frames/rawsource/Simple/JSONeveryframe as scheduled baseline, separate output for early comparison',
        'full_goal_complete': False}
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''

    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)

    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, check=True)

    save()
    try:
        run('INFERENCE', [str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py',
            '--task', 'rmot', '--video', str(manifest), '--backend', 'simple',
            '--candidate-replay', str(source), '--omit-replay-feature-vectors', '--out', str(output)])
        run('OFFICIAL', ['python3', 'tools/groundjev/evaluate_multitask.py', '--video', str(manifest), '--run', str(output)])
        run('POSTHOC', ['python3', 'tools/groundjev/analyze_tracking_events.py', '--video', str(manifest), '--run', str(output)])
        m = json.loads(manifest.read_text())
        count, candidates = 0, 0
        with source.open() as native, (output/'frame_decisions.jsonl').open() as trace:
            for a, b in zip_longest(native, trace):
                assert a is not None and b is not None
                a, b = json.loads(a), json.loads(b)
                expected = [{k: v for k, v in c.items() if k not in
                    ('native_region_feature', 'temporal_region_feature')} for c in a['candidates']]
                assert b['candidates'] == expected and a['frame'] == b['frame'] == m['frame_numbers'][count]
                assert a['image_path'] == b['image_path'] == m['images'][count]
                count += 1
                candidates += len(expected)
        decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
        assert count == decoded == 800 and candidates == 2208
        runtime = json.loads((output/'result.json').read_text())
        assert runtime['end_to_end_fps'] is None and not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        official = json.loads((output/'official_metrics.json').read_text())
        assert len(official['perception_rejected_frames']) == 19
        state.update(status='EARLY_FIXED_CANDIDATE_MULTI800_SIMPLE_VALIDATED', run=str(output),
            decoded_frames=decoded, native_candidates=candidates, all_native_geometry_order_and_frames_retained=True,
            metrics=official['metrics'], compact_metrics=proof['results']['multi800']['metrics'],
            E2E_or_paired_speed_claim=False,
            later_identical_baseline_not_independent_seed='If the existing owner later produces its scheduled baseline, retain both as duplicate same-source executions, not extra statistical samples.')
        save()
    except Exception as error:
        state.update(status='EARLY_MULTI800_SIMPLE_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
