"""One blocking owner: fresh ordered native services and unchanged tracking metrics."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-completed', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'STARTING_ISOLATED_ORDERED_GROUND_CONTROLS', 'plan': str(args.plan),
        'physical_gpu': plan['physical_gpu'], 'completed': {}, 'variant_roots': {},
        'original_source_or_seed_or_metric_changes': False,
        'full_dataset_claim': False, 'shared_full9_services_touched': False}
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)

    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    save()
    try:
        for variant in plan['variants']:
            name = variant['name']; directory = args.out/name
            if args.resume_completed:
                prior = json.loads((args.resume_completed/'owner_state.json').read_text())
                keys = [name+'_'+clip['name'] for clip in plan['clips']]
                if all(key in prior['completed'] for key in keys):
                    for clip in plan['clips']:
                        old = args.resume_completed/name/clip['name']
                        frames = [json.loads(x)['frame'] for x in (old/'frame_decisions.jsonl').read_text().splitlines()]
                        assert frames == json.loads((ROOT/clip['manifest']).read_text())['frame_numbers']
                    state['completed'].update({key: prior['completed'][key] for key in keys})
                    state['variant_roots'][name] = str(args.resume_completed/name)
                    state['status'] = 'REUSING_COMPLETE_ISOLATED_'+name; save()
                    continue
            directory.mkdir()
            state['variant_roots'][name] = str(directory)
            port = plan['port']
            deadline = time.monotonic()+30
            while True:
                try:
                    with socket.socket() as probe:
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        probe.bind(('127.0.0.1', port))
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(1)
            server = None
            with (directory/'server.log').open('w') as server_log:
                try:
                    server = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus',
                        str(plan['physical_gpu']), '--', str(ROOT/'external/GroundAnything/.venv-serve/bin/python'),
                        'tools/groundjev/serve_groundanything.py', '--model', str(ROOT/variant['checkpoint']),
                        '--port', str(port), '--mem-fraction-static', '.4', '--out', str(directory/'service')],
                        cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(status='WAITING_FRESH_'+name, owned_server_process_group=server.pid); save()
                    deadline = time.monotonic()+600
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError('Fresh native service exited before readiness')
                        try:
                            with urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as reply:
                                models = json.load(reply)
                            if any(m['id'] == 'groundinganything' for m in models['data']):
                                break
                        except (OSError, ValueError):
                            pass
                        if time.monotonic() > deadline:
                            raise RuntimeError('Owned fresh service readiness timeout')
                        time.sleep(2)
                    for clip in plan['clips']:
                        output = directory/clip['name']
                        state['status'] = name+'_'+clip['name']; save()
                        with (directory/(clip['name']+'.log')).open('w') as log:
                            command = [str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py',
                                '--task', clip['task'], '--video', clip['manifest'], '--backend', 'simple',
                                '--grounding-url', f'http://127.0.0.1:{port}/v1', '--grounding-parser', 'official_scoring',
                                '--grounding-decoder-policy', 'tracking_dense', '--perception-failure-policy',
                                'empty_and_record', '--out', str(output)]
                            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                            scorer = 'evaluate_video.py' if clip['task'] == 'mot' else 'evaluate_multitask.py'
                            subprocess.run(['python3', 'tools/groundjev/'+scorer, '--video', clip['manifest'],
                                '--run', str(output)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                        manifest = json.loads((ROOT/clip['manifest']).read_text())
                        observed = [json.loads(x)['frame'] for x in (output/'frame_decisions.jsonl').read_text().splitlines()]
                        assert observed == manifest['frame_numbers']
                        state['completed'][name+'_'+clip['name']] = json.loads((output/'official_metrics.json').read_text())
                        save()
                finally:
                    if server is not None and server.poll() is None:
                        os.killpg(server.pid, signal.SIGTERM)
                        try:
                            server.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(server.pid, signal.SIGKILL); server.wait()
            state['last_owned_service_stopped'] = name; save()
        repeat_checks = {}
        for clip in plan['clips']:
            directories = [args.out/label/clip['name'] for label in plan['repeat_pair']]
            traces = [[json.loads(x) for x in (p/'frame_decisions.jsonl').read_text().splitlines()] for p in directories]
            for a, b in zip(*traces):
                assert a['frame'] == b['frame'] and a['image_path'] == b['image_path']
                assert a['perception']['result']['raw_output'] == b['perception']['result']['raw_output']
                assert a['candidates'] == b['candidates'] and a['outputs'] == b['outputs']
            repeat_checks[clip['name']] = {'frames': len(traces[0]), 'same_raw_wire_candidates_and_outputs': True}
        state.update(status='ISOLATED_ORDERED_GROUND_CONTROLS_COMPLETE', repeat_checks=repeat_checks); save()
    except Exception as error:
        state.update(status='ISOLATED_ORDERED_GROUND_CONTROLS_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
