"""One blocking owner for real detector-proposal own-state supervision."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = ROOT/'outputs/groundjev/training_data/native_ground_own_history_v1'
    owner = ROOT/'outputs/groundjev/training_data/native_ground_own_history_v1_owner'
    owner.mkdir(parents=True, exist_ok=False)
    state = {'status': 'ACTUAL_NATIVE_CANDIDATE_ROLLOUT_STARTED', 'physical_gpu': 7,
             'native_ground_service_gpu': 6, 'out': str(out), 'formal_training_complete': False}
    def save():
        (owner/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
    save()
    command = ['python3', str(ROOT/'tools/groundjev/run_with_resources.py'), '--gpus', '7', '--',
               str(ROOT/'.venv-onejev-py310/bin/python'), str(ROOT/'tools/groundjev/prepare_student_rollout.py'),
               '--out', str(out), '--candidate-source', 'native_ground']
    with (owner/'rollout.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        state['child_pid'] = child.pid; save()
        for line in child.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        code = child.wait()
    state['exit_code'] = code
    if code:
        state['status'] = 'ACTUAL_NATIVE_CANDIDATE_ROLLOUT_FAILED'; save()
        raise RuntimeError('Preserve failed actor and completed frames; inspect first root cause')
    state['status'] = 'ACTUAL_NATIVE_CANDIDATE_ROLLOUT_COMPLETE'
    state['summary'] = json.loads((out/'summary.json').read_text()); save()


if __name__ == '__main__':
    main()
