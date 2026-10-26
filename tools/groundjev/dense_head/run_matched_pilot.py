"""One blocking owner for both independent GPU arms; no recovery of old eval."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json


def main():
    registration=json.loads((RUN/'GPU_REGISTRATION.json').read_text())
    pause=Path(registration['old_pause_audit'])
    assert json.loads(pause.read_text())['status']=='PAUSED_BY_USER_FOR_ARCHITECTURE_PIVOT'
    assert len({v['gpu'] for v in registration['arms'].values()})==2
    children={};threads=[];started=time.time()
    for arm,device in registration['arms'].items():
        raw=subprocess.check_output(['nvidia-smi','-i',str(device['gpu']),
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        uuid,free=map(str.strip,raw.strip().split(','));assert uuid==device['uuid'] and int(free)>14000,raw
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(device['gpu']),OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        command=[registration['python'],str(ROOT/'tools/groundjev/dense_head/train_dense_head.py'),
                 '--config',str(ROOT/'configs/groundjev/dense_head'/device['config'])]
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,start_new_session=True)
        children[arm]=process
        def drain(name,proc):
            directory=RUN/name;directory.mkdir(exist_ok=True)
            with (directory/'owner.log').open('a') as log:
                for line in proc.stdout:
                    log.write(line);log.flush();print(name+': '+line.rstrip(),flush=True)
        thread=threading.Thread(target=drain,args=(arm,process));thread.start();threads.append(thread)
    write_json(RUN/'owner_registration.json',{'status':'RUNNING','pid':os.getpid(),
        'children':{a:p.pid for a,p in children.items()},'start_unix':started,
        'maximum_combined_GPU_hours':23.,'smoke_audit_budget_reserved_GPU_hours':1.,
        'old_evaluation_automatic_resume':False})
    # Conservative occupancy budget: elapsed wall time times both registered
    # GPUs, even if one arm finishes early. Leaves one GPU-hour for diagnostics.
    def budget_watch():
        while any(p.poll() is None for p in children.values()):
            if (time.time()-started)*2/3600>23.:
                for proc in children.values():
                    if proc.poll() is None:os.kill(proc.pid,signal.SIGTERM)
                return
            time.sleep(10)
    watchdog=threading.Thread(target=budget_watch,daemon=True);watchdog.start()
    codes={arm:p.wait() for arm,p in children.items()}
    for t in threads:t.join()
    receipt={'status':'ALL_PILOT_OWNERS_EXITED','exit_codes':codes,'elapsed_seconds':time.time()-started,
             'old_evaluation_restarted':False,'gpu_processes_started':len(children)}
    write_json(RUN/'owner_completion.json',receipt);print(json.dumps(receipt),flush=True)
    if any(codes.values()):raise SystemExit(1)


if __name__=='__main__':main()
