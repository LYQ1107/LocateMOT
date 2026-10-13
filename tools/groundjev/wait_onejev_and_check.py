"""One blocking owner gates the real-frame check on verified weights and runtime."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT/'.venv-onejev-py310/bin/python'
OUT = ROOT/'outputs/groundjev/model_setup/onejev_check_queue.json'


def upstream():
    found = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p/'cmdline').read_bytes().decode().split('\0')[:-1]
            cwd = str((p/'cwd').resolve())
            if cwd != str(ROOT) or not argv:
                continue
            if 'tools/groundjev/download_models.py' in argv and 'python' in Path(argv[0]).name:
                found['weights'] = int(p.name)
            if Path(argv[0]).name == 'uv' and '--python' in argv and argv[argv.index('--python')+1] == '.venv-onejev-py310/bin/python':
                found['environment'] = int(p.name)
        except OSError:
            pass
    return found


def alive(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    owners = upstream()
    state = {'state':'WAITING_VERIFIED_WEIGHTS_AND_ENVIRONMENT','upstream_pids':owners,
             'physical_gpus':[6],'training_launched':False}
    OUT.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    receipt = ROOT/'outputs/groundjev/model_setup/OneJev-0.8B_download.json'
    check = [str(PYTHON),'-c','import torch,transformers,accelerate,peft,qev; assert transformers.__version__ == "5.17.0"']
    while True:
        weights_ready = receipt.exists() and json.loads(receipt.read_text())['status'] == 'DOWNLOAD_COMPLETE_NOT_MODEL_LOAD'
        env_pending = 'environment' in owners and alive(owners['environment'])
        if not env_pending:
            result = subprocess.run(check,cwd=ROOT,capture_output=True,text=True)
            if result.returncode:
                state.update(state='ENVIRONMENT_NOT_READY_AFTER_OWNER_EXIT',error=result.stderr)
                OUT.write_text(json.dumps(state,indent=2)+'\n')
                raise RuntimeError(result.stderr)
        if weights_ready and not env_pending:
            break
        if not weights_ready and ('weights' not in owners or not alive(owners['weights'])):
            state['state'] = 'WEIGHT_OWNER_TERMINAL_WITHOUT_VERIFIED_BUNDLE'
            OUT.write_text(json.dumps(state,indent=2)+'\n')
            raise RuntimeError('Inspect the original download owner failure; do not load partial weights')
        time.sleep(20)
    state['state'] = 'RUNNING_REAL_PRETRAINED_FRAME_CHECK'
    OUT.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    manifest = json.loads((ROOT/'outputs/groundjev/pilot_data/mot17.json').read_text())
    command = ['python3',str(ROOT/'tools/groundjev/run_with_resources.py'),'--gpus','6','--',str(PYTHON),
               str(ROOT/'tools/groundjev/check_onejev_real_frame.py'),'--image',manifest['images'][0]]
    subprocess.run(command,cwd=ROOT,check=True)
    state['state'] = 'REAL_PRETRAINED_FRAME_CHECK_COMPLETE'
    OUT.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    main()
