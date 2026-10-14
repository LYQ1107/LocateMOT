"""Fresh same-protocol tracking controls through the two owned retention services."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'WAITING_EXISTING_RETENTION_SERVICES',
        'scope': 'registered MOT4/RMOTsingle8/RMOTmulti12 functional controls',
        'original_checkpoint': 'models/GroundAnything-DLM',
        'adapted_checkpoint': 'models/GroundAnything-DLM-TrackProj200',
        'backend': 'same unchanged Simple geometry associator',
        'parser': 'official_scoring', 'decoder': 'tracking_dense',
        'requested_seed': 42, 'all_registered_empty_and_rejected_frames_retained': True,
        'full_dataset_accuracy_claim': False, 'services_created_or_stopped': False,
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
    try:
        deadline = time.monotonic()+600
        for port in (8101, 8102):
            while True:
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
            for variant, port in [('original', 8101), ('projector200', 8102)]:
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
        state['status'] = 'NATIVE_GROUND200_FRESH_TRACKING_CONTROLS_COMPLETE'; save()
    except Exception as error:
        state.update(status='NATIVE_GROUND200_TRACKING_CONTROLS_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
