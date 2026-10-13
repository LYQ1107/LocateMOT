"""One blocking owner starts a four-frame native clip after its DLM service is ready."""
import json
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    owner = None
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p/'cmdline').read_bytes().decode().split('\0')[:-1]
            if str((p/'cwd').resolve()) == str(ROOT) and 'tools/groundjev/wait_and_serve_groundanything.py' in argv:
                owner = int(p.name)
        except OSError:
            pass
    if owner is None:
        raise RuntimeError('No verified GroundJEV service owner is live')
    state = {'state':'WAITING_OWN_DLM_SERVICE','service_owner_pid':owner,'scope':'four_frame_functional_check',
             'training_steps':0,'physical_gpu':6}
    receipt = ROOT/'outputs/groundjev/model_setup/first_clip_queue.json'
    receipt.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    while True:
        stat = Path('/proc')/str(owner)/'stat'
        if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z':
            raise RuntimeError('DLM service owner ended before readiness; inspect its first failure')
        queue = json.loads((ROOT/'outputs/groundjev/model_setup/ground_service_queue.json').read_text())
        if queue['state'] == 'STARTING_OFFICIAL_DLM_SERVICE':
            try:
                with urlopen('http://127.0.0.1:8101/v1/models',timeout=3) as response:
                    models = json.load(response)
                if any(m['id']=='groundinganything' for m in models['data']):
                    break
            except (URLError,OSError):
                pass
        time.sleep(10)
    state['state'] = 'RUNNING_REAL_DLM_ONEJEV_CLIP'
    receipt.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    command = ['python3',str(ROOT/'tools/groundjev/run_with_resources.py'),'--gpus','6','--',
        str(ROOT/'.venv-onejev-py310/bin/python'),str(ROOT/'tools/groundjev/run_video.py'),'--task','mot',
        '--video',str(ROOT/'outputs/groundjev/pilot_data/mot17.json'),'--backend','full','--limit','4',
        '--out',str(ROOT/'outputs/groundjev/runs/mot17_first_native_clip')]
    result = subprocess.run(command,cwd=ROOT)
    state.update(state='FIRST_NATIVE_CLIP_TERMINAL',exit_code=result.returncode)
    receipt.write_text(json.dumps(state,indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
