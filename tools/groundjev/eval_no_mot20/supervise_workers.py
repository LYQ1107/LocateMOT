"""Recover terminated workers internally, without polling from the agent.

An old worker must have a terminal resource receipt newer than its last service
launch, and its recorded service PID must be gone. Active services and replay
processes are never stopped by this supervisor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import load,save


def service_gone(pid):
    try:os.kill(int(pid),0)
    except ProcessLookupError:return True
    return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=load(args.config);run=Path(config['output_root'])
    children={};attempts=Counter();handled={};started=time.time()
    while True:
        db=sqlite3.connect(run/'queue.sqlite',timeout=120)
        states=dict(db.execute('SELECT state,COUNT(*) FROM units GROUP BY state'))
        remaining=sum(n for state,n in states.items() if state!='COMPLETE')
        for worker in config['workers']:
            name=worker['name'];directory=run/'workers'/name
            if name in children:
                process,handle=children[name]
                if process.poll() is None:continue
                handle.close();del children[name]
                if process.returncode:
                    print(json.dumps({'event':'RECOVERY_OWNER_EXITED','worker':name,
                                      'exit_code':process.returncode,'attempts':attempts[name]}),flush=True)
            terminal=directory/'resource_receipt.json';ready=directory/'service_ready.json'
            if not terminal.exists() or not ready.exists():continue
            timestamp=terminal.stat().st_mtime_ns
            if timestamp<=ready.stat().st_mtime_ns or handled.get(name)==timestamp:continue
            if not load(terminal)['all_owned_service_processes_stopped']:continue
            if not service_gone(load(ready)['pid']):continue
            with (directory/'.recovery.lock').open('a') as recovery_lock:
                try:fcntl.flock(recovery_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
            own=db.execute("SELECT COUNT(*) FROM units WHERE owner=? AND state IN ('RUNNING','FAILED')",(name,)).fetchone()[0]
            if not own and not states.get('PENDING'):continue
            if attempts[name]>=3:
                save(directory/'recovery_attention.json',{'status':'REPEATED_RECOVERY_FAILURE',
                     'attempts':attempts[name],'unfinished_units_preserved':True,'time':time.time()})
                continue
            command=[config['onejev_python'],str(ROOT/'tools/groundjev/eval_no_mot20/dispatch_4gpu.py'),
                     '--config',str(args.config.resolve()),'--stage','detection','--recover-worker',name]
            handle=(directory/'recovery_owner.log').open('a')
            handle.write(json.dumps({'command':command,'time':time.time()})+'\n');handle.flush()
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            children[name]=(process,handle);attempts[name]+=1;handled[name]=timestamp
            print(json.dumps({'event':'RECOVERY_OWNER_STARTED','worker':name,'pid':process.pid}),flush=True)
        db.close()
        save(run/'supervisor_state.json',{'status':'ACTIVE' if remaining else 'DETECTION_COMPLETE',
             'pid':os.getpid(),'queue_states':states,'recovery_attempts':dict(attempts),
             'elapsed_seconds':time.time()-started,'time':time.time()})
        if not remaining and not children:return
        time.sleep(15)


if __name__=='__main__':main()
