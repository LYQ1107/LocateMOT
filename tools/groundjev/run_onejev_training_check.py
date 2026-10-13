"""One blocking owner for the finite official real-image SFT correctness run."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = ROOT/'outputs/groundjev/training/onejev_teacher_check_100'
    out.mkdir(parents=True, exist_ok=True)
    receipt = out/'owner_state.json'
    if receipt.exists():
        raise RuntimeError('Preserve an existing training owner; do not restart this stage implicitly')
    summary = json.loads((ROOT/'outputs/groundjev/training_data/teacher_v1/summary.json').read_text())
    if not summary['official_target_names_verified_all_rows'] or summary['development_frames_used']:
        raise RuntimeError('Legal real-image curriculum validation must pass before training')
    state = {'status':'WAITING_GPU4_MEMORY', 'physical_gpus':[4], 'optimizer_step_budget':100,
             'scope':'teacher_history_training_correctness_check_not_formal_training_completion',
             'training_rows':summary['rows'], 'seed':0, 'owner_pid':os.getpid(),
             'initialization':'real pinned OmniJev/OneJev-0.8B', 'other_projects_preserved':True}
    receipt.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state), flush=True)
    while True:
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free = {int(r[0]):int(r[1]) for r in csv.reader(io.StringIO(raw))}
        if free[4] >= 45*1024:
            break
        time.sleep(30)
    command = ['python3',str(ROOT/'tools/groundjev/run_with_resources.py'),'--gpus','4','--',
               str(ROOT/'.venv-onejev-py310/bin/python'),'-m','train.sft',
               '--config',str(ROOT/'configs/groundjev/onejev_teacher_check.yaml')]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'external/OneJev')
    state.update(status='RUNNING_OFFICIAL_MULTIMODAL_FULL_SFT',command=command)
    receipt.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state), flush=True)
    with (out/'owner_console.log').open('w') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line,end='',flush=True)
        code = process.wait()
    state.update(status='TRAINING_CHECK_TERMINAL',exit_code=code)
    receipt.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    raise SystemExit(code)


if __name__=='__main__':
    main()
