"""One blocking owner for both independent GPU arms; no recovery of old eval."""
from __future__ import annotations

import json
import argparse
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
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--shared-budget-root',type=Path)
    args=parser.parse_args();run=args.run_root
    registration=json.loads((run/'GPU_REGISTRATION.json').read_text())
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
            directory=run/name;directory.mkdir(exist_ok=True)
            with (directory/'owner.log').open('a') as log:
                for line in proc.stdout:
                    log.write(line);log.flush();print(name+': '+line.rstrip(),flush=True)
        thread=threading.Thread(target=drain,args=(arm,process));thread.start();threads.append(thread)
    write_json(run/'owner_registration.json',{'status':'RUNNING','pid':os.getpid(),
        'children':{a:p.pid for a,p in children.items()},'start_unix':started,
        'maximum_combined_GPU_hours':23.,'smoke_audit_budget_reserved_GPU_hours':1.,
        'old_evaluation_automatic_resume':False,'shared_budget_root':str(args.shared_budget_root) if args.shared_budget_root else None,
        'whole_task_GPU_hour_budget':24 if args.shared_budget_root else None})
    # Conservative occupancy budget: elapsed wall time times both registered
    # GPUs, even if one arm finishes early. Leaves one GPU-hour for diagnostics.
    budget_finished=threading.Event()
    def budget_watch():
        roots=[args.shared_budget_root,run] if args.shared_budget_root else [run]
        while True:
            active=[];cost=1. if args.shared_budget_root else 0.
            for directory in roots:
                for arm in ('g0','g2'):
                    terminal=directory/arm/'receipt.json';reg=directory/arm/'registration.json'
                    if terminal.exists():cost+=json.loads(terminal.read_text())['GPU_hours'];continue
                    if not reg.exists():continue
                    row=json.loads(reg.read_text());pid=row['pid'];proc=Path('/proc')/str(pid)
                    if not proc.exists():continue
                    try:
                        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                        owned=proc.stat().st_uid==os.getuid() and (proc/'cwd').resolve()==ROOT and 'train_dense_head.py' in command
                    except OSError:continue
                    if owned:active.append(pid);cost+=(time.time()-row['start_unix'])/3600
            limit_reached=(cost>24.) if args.shared_budget_root else (time.time()-started)*2/3600>23.
            if limit_reached:
                write_json(run/'budget_stop_request.json',{'owned_active_pids':active,'total_GPU_hours_upper_estimate':cost,
                    'one_GPU_hour_reserved_for_smokes':bool(args.shared_budget_root),'time':time.time()})
                for pid in active:
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                # Preserve the last valid checkpoints if an evaluator cannot
                # return before the whole-task GPU-hour limit.
                time.sleep(60)
                for pid in active:
                    try:
                        proc=Path('/proc')/str(pid)
                        if proc.exists() and proc.stat().st_uid==os.getuid() and (proc/'cwd').resolve()==ROOT and b'train_dense_head.py' in (proc/'cmdline').read_bytes():os.kill(pid,signal.SIGKILL)
                    except OSError:pass
                break
            if all(p.poll() is not None for p in children.values()) and not active:break
            time.sleep(10)
        if args.shared_budget_root:
            write_json(run/'global_budget_receipt.json',{'GPU_hours_upper_estimate_including_one_hour_smoke_reserve':cost,
                'active_training_pids_remaining':active,'limit_GPU_hours':24,'time':time.time()})
        budget_finished.set()
    watchdog=threading.Thread(target=budget_watch,daemon=True);watchdog.start()
    codes={arm:p.wait() for arm,p in children.items()}
    for t in threads:t.join()
    if args.shared_budget_root:budget_finished.wait()
    receipt={'status':'ALL_PILOT_OWNERS_EXITED','exit_codes':codes,'elapsed_seconds':time.time()-started,
             'old_evaluation_restarted':False,'gpu_processes_started':len(children)}
    write_json(run/'owner_completion.json',receipt);print(json.dumps(receipt),flush=True)
    if any(codes.values()):raise SystemExit(1)


if __name__=='__main__':main()
