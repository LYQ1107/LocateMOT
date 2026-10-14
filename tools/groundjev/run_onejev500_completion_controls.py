"""One blocking owner waits for the verified current500 trainer and scores matched controls."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    training = ROOT/plan['training']
    owner = json.loads((training/'owner_state.json').read_text())
    pid = owner['owner_pid']; proc = Path('/proc')/str(pid)
    expected_start = None
    if proc.exists():
        command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert (proc/'cwd').resolve() == ROOT and 'tools/groundjev/run_stateful_onejev500.py' in command
        expected_start = (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]
    state = {'status': 'WAITING_VERIFIED_ORIGINAL500_TRAINER', 'training_owner_pid': pid,
        'training_owner_start_ticks': expected_start, 'physical_eval_gpu': plan['physical_gpu'],
        'training_restarted_or_protocol_modified': False, 'end_to_end_FPS_claim': False,
        'fixed_native_candidate_controls': True, 'completed': {}}

    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def alive():
        try:
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            return fields[0] != 'Z' and fields[19] == expected_start
        except FileNotFoundError:
            return False

    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONPATH'] = str(ROOT/'external/OneJev')

    def run(name, command):
        state['status'] = name; save()
        with (args.out/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    save()
    try:
        while expected_start is not None and alive():
            time.sleep(20)
        owner = json.loads((training/'owner_state.json').read_text())
        assert owner['status'] == 'STATEFUL_ONEJEV500_TRAINING_COMPLETE', owner
        logs = [json.loads(x) for x in (training/'log.jsonl').read_text().splitlines()]
        done = next(x for x in logs if x.get('event') == 'done')
        assert done['steps'] == 500
        assert all(math.isfinite(x['loss']) for x in logs if 'loss' in x)
        state['actual_training_done'] = done
        python = str(ROOT/'.venv-onejev-py310/bin/python')
        prefix = ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(plan['physical_gpu']), '--', python]
        run('VALIDATE_REAL_ONEJEV500_MM_RELOAD', prefix+['tools/groundjev/check_onejev125_checkpoint.py', '--step', '500'])
        for job in plan['jobs']:
            manifest_path = ROOT/job['manifest']; manifest = json.loads(manifest_path.read_text())
            output = args.out/job['name']
            command = prefix+['tools/groundjev/run_video.py', '--task', manifest['task'], '--video', str(manifest_path),
                '--backend', job['backend'], '--checkpoint', str(ROOT/job['checkpoint']),
                '--evidence-format', 'v3_crop_motion', '--adaptive-context-groups',
                '--candidate-replay', str(ROOT/job['candidates']), '--out', str(output)]
            run(job['name']+'_INFERENCE', command)
            scorer = 'evaluate_video.py' if manifest['task'] == 'mot' else 'evaluate_multitask.py'
            run(job['name']+'_OFFICIAL_EVAL', ['python3', 'tools/groundjev/'+scorer,
                '--video', str(manifest_path), '--run', str(output)])
            rows = [json.loads(x) for x in (output/'frame_decisions.jsonl').read_text().splitlines()]
            assert [x['frame'] for x in rows] == manifest['frame_numbers']
            replay = {json.loads(x)['frame']: json.loads(x)['candidates'] for x in (ROOT/job['candidates']).read_text().splitlines()}
            assert all(x['candidates'] == replay[x['frame']] for x in rows)
            state['completed'][job['name']] = {'run': str(output),
                'official_metrics': json.loads((output/'official_metrics.json').read_text()),
                'all_exact_native_candidate_rows_preserved': True}
            save()
        state['status'] = 'ONEJEV500_MATCHED_OFFICIAL_CONTROLS_COMPLETE'; save()
    except Exception as error:
        state.update(status='ONEJEV500_COMPLETION_CONTROLS_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
