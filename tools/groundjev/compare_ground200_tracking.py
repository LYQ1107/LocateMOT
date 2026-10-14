"""Fresh same-protocol tracking controls through the two owned retention services."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-port', type=int, default=8101)
    parser.add_argument('--adapted-port', type=int, default=8102)
    parser.add_argument('--reference-name', default='original')
    parser.add_argument('--adapted-name', default='projector200')
    parser.add_argument('--reference-model', type=Path, default=ROOT/'models/GroundAnything-DLM')
    parser.add_argument('--adapted-model', type=Path, default=ROOT/'models/GroundAnything-DLM-TrackProj200')
    parser.add_argument('--launch-adapted-gpu', type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'WAITING_EXISTING_RETENTION_SERVICES',
        'scope': 'registered MOT4/RMOTsingle8/RMOTmulti12 functional controls',
        'original_checkpoint': str(args.reference_model.resolve()),
        'adapted_checkpoint': str(args.adapted_model.resolve()),
        'backend': 'same unchanged Simple geometry associator',
        'parser': 'official_scoring', 'decoder': 'tracking_dense',
        'requested_seed': 42, 'all_registered_empty_and_rejected_frames_retained': True,
        'full_dataset_accuracy_claim': False, 'services_created_or_stopped': args.launch_adapted_gpu is not None,
        'reference_port': args.reference_port, 'adapted_port': args.adapted_port,
        'completed': {}}

    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(name, command):
        state['status'] = name; save()
        with (args.out/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    save()
    server = None
    server_log = None
    try:
        if args.launch_adapted_gpu is not None:
            import socket
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', args.adapted_port))
            server_log = (args.out/'adapted_server.log').open('w')
            server = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py',
                '--gpus', str(args.launch_adapted_gpu), '--',
                str(ROOT/'external/GroundAnything/.venv-serve/bin/python'),
                'tools/groundjev/serve_groundanything.py', '--model', str(args.adapted_model),
                '--port', str(args.adapted_port), '--mem-fraction-static', '.4',
                '--out', str(args.out/'adapted_service')], cwd=ROOT,
                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(owned_adapted_process_group=server.pid, physical_adapted_gpu=args.launch_adapted_gpu)
            save()
        deadline = time.monotonic()+600
        for port in (args.reference_port, args.adapted_port):
            while True:
                if server is not None and server.poll() is not None:
                    raise RuntimeError('Owned adapted service exited before readiness')
                try:
                    with urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as response:
                        models = json.load(response)
                    if any(m['id'] == 'groundinganything' for m in models['data']):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f'Existing retention API {port} unavailable')
                time.sleep(2)
        scopes = [('dense_mot4', 'mot', 'outputs/groundjev/pilot_data/mot17_first4.json'),
            ('refer_single8', 'rmot', 'outputs/groundjev/multitask_pilot_data/rmot_single_empty.json'),
            ('refer_multi12', 'rmot', 'outputs/groundjev/multitask_pilot_data/rmot_multi_empty.json')]
        for name, task, manifest in scopes:
            for variant, port in [(args.reference_name, args.reference_port), (args.adapted_name, args.adapted_port)]:
                output = args.out/name/variant
                run(name+'_'+variant+'_INFERENCE', [str(ROOT/'.venv-onejev-py310/bin/python'),
                    'tools/groundjev/run_video.py', '--task', task, '--video', manifest,
                    '--backend', 'simple', '--grounding-url', f'http://127.0.0.1:{port}/v1',
                    '--grounding-parser', 'official_scoring', '--grounding-decoder-policy', 'tracking_dense',
                    '--perception-failure-policy', 'empty_and_record', '--out', str(output)])
                scorer = 'tools/groundjev/evaluate_video.py' if task == 'mot' else 'tools/groundjev/evaluate_multitask.py'
                run(name+'_'+variant+'_OFFICIAL_EVAL', ['python3', scorer, '--video', manifest, '--run', str(output)])
                expected = json.loads((ROOT/manifest).read_text())['frame_numbers']
                observed = [json.loads(line)['frame'] for line in (output/'frame_decisions.jsonl').read_text().splitlines()]
                assert observed == expected
                state['completed'][name+'_'+variant] = json.loads((output/'official_metrics.json').read_text())
                save()
        state['status'] = 'NATIVE_GROUND_FRESH_TRACKING_CONTROLS_COMPLETE'; save()
    except Exception as error:
        state.update(status='NATIVE_GROUND200_TRACKING_CONTROLS_FAILED', error=repr(error)); save()
        raise
    finally:
        if server is not None:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL); server.wait()
            state['owned_adapted_service_stopped'] = True
            save()
        if server_log is not None:
            server_log.close()


if __name__ == '__main__':
    main()
