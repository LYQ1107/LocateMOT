"""Complement fixed-total-budget task mixtures with matched MOT-exposure prefixes."""
import argparse
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    original = ROOT/plan['original_MOT_training']
    before = json.loads((original/'owner_state.json').read_text())
    assert before['steps'] == 1200 and before['selected_tasks'] == ['mot']
    state = {'status': 'STARTING_MATCHED_MOT_EXPOSURE_CONTROLS', 'owner_pid': os.getpid(), 'device': 'cpu',
        'additional_GPUs': 0, 'new_feature_cache_files': False, 'completed': {}, 'full_goal_complete': False,
        'interpretation': 'same MOT exposure, different total updates from multi-task arms; complements fixed-total-budget comparison'}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    python = str(ROOT/'.venv-onejev-py310/bin/python')
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)
    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    save()
    try:
        for steps in plan['steps']:
            name = 'MOT'+str(steps)
            training = args.out/(name+'_training')
            run(name+'_TRAINING', [python, 'tools/groundjev/train_compact_joint.py', '--pack', str(ROOT/plan['pack']),
                '--checkpoint', str(ROOT/plan['initialization']), '--out', str(training), '--steps', str(steps),
                '--device', 'cpu', '--tasks', 'mot', '--per-task-sampling'])
            trained = json.loads((training/'owner_state.json').read_text())
            assert trained['status'] == 'SHARED_COMPACT_TRAINING_COMPLETE' and trained['steps'] == steps
            assert trained['sampled_example_ids']['0'] == before['sampled_example_ids']['0'][:steps]
            prior_log = {r['step']: r for r in before['training_log']}
            differences = {r['step']: abs(r['loss']-prior_log[r['step']]['loss']) for r in trained['training_log']}
            assert max(differences.values()) == 0, differences
            assert trained['parameters'] == before['parameters'] == 1146511
            assert all(v == 0 for v in trained['three_task_checkpoint_reload_max_difference'].values())
            output = args.out/(name+'_full1203')
            manifest = ROOT/plan['manifest']
            source = ROOT/plan['candidates']
            run(name+'_INFERENCE', [python, 'tools/groundjev/run_video.py', '--task', 'mot', '--video', str(manifest),
                '--backend', 'compact', '--compact-device', 'cpu', '--checkpoint', str(training/'checkpoint.pt'),
                '--candidate-replay', str(source), '--omit-replay-feature-vectors', '--out', str(output)])
            run(name+'_OFFICIAL', ['python3', 'tools/groundjev/evaluate_video.py', '--video', str(manifest), '--run', str(output)])
            run(name+'_POSTHOC', ['python3', 'tools/groundjev/analyze_tracking_events.py', '--video', str(manifest), '--run', str(output)])
            m = json.loads(manifest.read_text())
            count = 0
            with source.open() as a, (output/'frame_decisions.jsonl').open() as b:
                for old, new in zip_longest(a, b):
                    assert old is not None and new is not None
                    old, new = json.loads(old), json.loads(new)
                    geometry = [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')}
                                for c in old['candidates']]
                    assert new['candidates'] == geometry and new['frame'] == old['frame'] == m['frame_numbers'][count]
                    assert new['image_path'] == old['image_path'] == m['images'][count]
                    count += 1
            decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
            assert count == decoded == 1203
            result = json.loads((output/'result.json').read_text())
            assert result['end_to_end_fps'] is None and not result['gt_read_by_runtime'] and not result['future_frames_read_by_runtime']
            state['completed'][name] = {'training': str(training), 'steps': steps, 'parameters': trained['parameters'],
                'all_actual_MOT_example_prefixes_exact': True, 'logged_loss_prefix_max_difference': max(differences.values()),
                'decoded_frames': decoded, 'run': str(output), 'all_native_candidates_geometry_order_retained': True,
                'metrics': json.loads((output/'official_metrics.json').read_text())['metrics']}
            save()
        state['status'] = 'MOT400_600_EXPOSURE_MATCHED_FULL1203_CONTROLS_COMPLETE'
        save()
    except Exception as error:
        state.update(status='MOT_EXPOSURE_CONTROL_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
