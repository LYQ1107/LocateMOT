"""One blocking owner waits for the verified DLM bundle, then owns its service."""
import csv
import io
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/groundjev/model_setup/ground_service_queue.json'


def main():
    owners = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p/'cmdline').read_bytes().decode().split('\0')[:-1]
            if str((p/'cwd').resolve()) == str(ROOT) and 'tools/groundjev/download_models.py' in argv:
                owners.append(int(p.name))
        except OSError:
            pass
    receipt = ROOT/'outputs/groundjev/model_setup/GroundAnything-DLM_download.json'
    state = {'state':'WAITING_OFFICIAL_DLM_BUNDLE','download_owner_pids':owners,'physical_gpu':6,
             'memory_fraction_static':.4,'other_projects_preserved':True}
    OUT.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    while not receipt.exists():
        alive = False
        for pid in owners:
            try:
                alive |= (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z'
            except FileNotFoundError:
                pass
        if not alive:
            raise RuntimeError('DLM download owner ended without a complete official bundle receipt')
        time.sleep(20)
    if json.loads(receipt.read_text())['status'] != 'DOWNLOAD_COMPLETE_NOT_MODEL_LOAD':
        raise RuntimeError('DLM bundle is not verified complete')
    setup = json.loads((ROOT/'outputs/groundjev/model_setup/ground_setup_resume.json').read_text())
    if setup['status'] != 'OFFICIAL_SERVE_ENVIRONMENT_CHECKS_COMPLETE_NOT_MODEL_LOAD':
        raise RuntimeError('Official serving environment checks must complete before GPU loading')
    # Internal resource wait belongs to this sole owner, not agent-level polling.
    while True:
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free = {int(r[0]):int(r[1]) for r in csv.reader(io.StringIO(raw))}
        if free[6] >= 42*1024:
            break
        if state['state'] != 'WAITING_GPU6_MEMORY':
            state['state'] = 'WAITING_GPU6_MEMORY'
            OUT.write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps(state),flush=True)
        time.sleep(30)
    state['state'] = 'STARTING_OFFICIAL_DLM_SERVICE'
    OUT.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    command = ['python3',str(ROOT/'tools/groundjev/run_with_resources.py'),'--gpus','6','--',
        str(ROOT/'external/GroundAnything/.venv-serve/bin/python'),str(ROOT/'tools/groundjev/serve_groundanything.py'),
        '--mem-fraction-static','0.4']
    result = subprocess.run(command,cwd=ROOT)
    state.update(state='DLM_SERVICE_TERMINAL',exit_code=result.returncode)
    OUT.write_text(json.dumps(state,indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
