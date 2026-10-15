"""One detached CPU owner: genuine B2 training, four-frame gate, complete video."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-study', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'B2_ORIGINAL_U2_CORE_STUDY_STARTING', 'owner_pid': os.getpid(),
        'device': 'cpu', 'additional_GPUs': 0, 'old_U2_candidate_queue_restarted': False, 'completed': {}}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    python = str(ROOT/'.venv-onejev-py310/bin/python')
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        training = (args.resume_study or args.out)/'training'
        if not args.resume_study:
            run('TRAIN_B2_300', [python, 'tools/groundjev/train_u2_bridge.py', '--out', str(training)])
        trained = json.loads((training/'owner_state.json').read_text())
        assert trained['status'] == 'GROUND_U2_CORE300_TRAINING_COMPLETE_OFFICIAL_METRICS_PENDING'
        assert trained['completed_steps'] == 300 and trained['reload_max_difference'] == 0
        state['training'] = trained
        manifest = ROOT/'outputs/groundjev/pilot_data/dancetrack.json'
        full = json.loads(manifest.read_text())
        short = dict(full, images=full['images'][:4], frame_numbers=full['frame_numbers'][:4],
                     scope='explicit_B2_four_frame_functional_check')
        short_path = args.out/'functional4_manifest.json'
        short_path.write_text(json.dumps(short, indent=2)+'\n')
        source = ROOT/'outputs/groundjev/compact/dancetrack0004_native300_v2_empty_roi_fixed/compact_distilled/frame_decisions.jsonl'
        jobs = [('functional4', short_path), ('full1203', manifest)]
        if args.resume_study:
            old = json.loads((args.resume_study/'owner_state.json').read_text())
            state['completed']['functional4'] = old['completed']['functional4']
            regression = json.loads((args.resume_study/'numeric_repair_regression.json').read_text())
            assert regression['all198_prefix_decisions_exact'] and regression['failed_frame199_now_finite']
            state['numeric_repair_regression'] = regression
            jobs = [('full1203', manifest)]
        for name, video in jobs:
            output = args.out/name
            command = [python, 'tools/groundjev/run_u2_bridge_video.py', '--checkpoint', str(training/'checkpoint.pt'),
                '--video', str(video), '--candidate-replay', str(source), '--out', str(output)]
            if args.resume_study:
                command += ['--resume-from', str(args.resume_study/'full1203'),
                            '--resume-state', str(args.resume_study/'validated_identity_state198.pt')]
            run(name+'_INFERENCE', command)
            run(name+'_OFFICIAL', ['python3', 'tools/groundjev/evaluate_video.py', '--video', str(video), '--run', str(output)])
            report = json.loads((output/'result.json').read_text())
            if not args.resume_study:
                assert report['causal_memory_save_restore_next_frame_max_difference'] == 0
            else:
                assert report['resumed_completed_frames'] == 198
            decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
            assert decoded == len(json.loads(video.read_text())['images'])
            state['completed'][name] = {'decoded_frames': decoded, 'run': str(output),
                'official': json.loads((output/'official_metrics.json').read_text())}
            save()
        state['status'] = 'B2_U2_CORE300_COMPLETE1203_NATIVE_PROPOSAL_CONTROL_VALIDATED'
        save()
    except Exception as error:
        state.update(status='B2_U2_CORE_STUDY_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
