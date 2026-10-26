"""Replay the current J1 input protocol on already Simple-scored full videos.

One existing physical GPU is shared with the detector queue. This command owns
and waits for its finite registered backlog; it never launches another detector.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import load,save,arm_directory
from tools.groundjev.eval_no_mot20.dispatch_4gpu import gpu_environment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/groundjev/eval_no_mot20_4gpu_v1.json')
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();config=load(args.config);run=Path(config['output_root'])
    worker=next(w for w in config['workers'] if w['gpu']==args.gpu)
    arm=arm_directory('j1',config);audit=load(run/'model_audit.json')
    directory=run/'engineering_checks'/f'{arm}_ready_backfill';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ready=[j for j in load(run/'jobs.json')['jobs']
               if (Path(j['path'])/'simple/frozen_receipt.json').exists()
               and not (Path(j['path'])/arm/'frozen_receipt.json').exists()]
        families=sorted({j['dataset'] for j in ready})
        grouped={family:sorted([j for j in ready if j['dataset']==family],key=lambda j:(j['frames'],j['id'])) for family in families}
        ordered=[grouped[family][index] for index in range(max([len(v) for v in grouped.values()] or [0]))
                 for family in families if index<len(grouped[family])]
        registration={'status':'REGISTERED_READY_VIDEO_BACKFILL','arm':arm,'jobs':ordered,
            'checkpoint':audit['J1_merged'],'competition_geometry':config['j1_competition_geometry'],
            'candidate_evidence':config['j1_candidate_evidence'],'memory_transport':config['j1_memory_transport'],
            'gpu':args.gpu,'uuid':worker['uuid'],'main_G2_queue_continues':True,'new_training':False}
        save(directory/f'registration_{time.time_ns()}.json',registration)
        results=[];env=gpu_environment(args.gpu)
        for job in ordered:
            path=Path(job['path'])
            if (path/arm/'frozen_receipt.json').exists():continue
            raw=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid,memory.free',
                                         '--format=csv,noheader,nounits'],text=True).strip()
            uuid,free=[part.strip() for part in raw.split(',')]
            if uuid!=worker['uuid'] or int(free)<12288:
                save(directory/'owner.json',{'status':'RESOURCE_WAIT_REQUIRED_BACKLOG_PRESERVED',
                     'gpu_reading':raw,'pending_job':job['id'],'results':results})
                raise RuntimeError(f'Existing registered GPU lacks J1 free-memory requirement: {raw}')
            command=[config['onejev_python'],str(ROOT/'tools/groundjev/eval_no_mot20/replay_j1.py'),
                '--job',str(path),'--checkpoint',audit['J1_merged']['path'],'--arm',arm,
                '--competition-geometry',config['j1_competition_geometry'],
                '--candidate-evidence',config['j1_candidate_evidence'],
                '--memory-transport',config['j1_memory_transport'],'--no-followups']
            began=time.time()
            with (directory/f"{job['id']}.log").open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                save(directory/'owner.json',{'status':'RUNNING_REAL_COMPLETE_VIDEO_BACKFILL',
                    'child_pid':child.pid,'gpu':args.gpu,'arm':arm,'job':job['id'],
                    'free_MiB_at_start':int(free),'command':command,'results':results})
                code=child.wait()
            frozen=path/arm/'frozen_receipt.json'
            result={'job':job['id'],'frames':job['frames'],'exit_code':code,'seconds':time.time()-began,
                'status':'COMPLETE' if code==0 and frozen.exists() else 'UNFINISHED_OR_OTHER_OWNER',
                'log':str(directory/f"{job['id']}.log")}
            if frozen.exists():
                actual=load(frozen)['input'];reference=load(path/'simple/frozen_receipt.json')['input']
                if actual!=reference:raise ValueError('Ready-video backfill changed native candidates')
            results.append(result);save(directory/'progress.json',{'results':results,'registration':registration})
            print(json.dumps(result),flush=True)
        status='READY_VIDEO_BACKFILL_COMPLETE' if all(r['status']=='COMPLETE' for r in results) else 'BACKFILL_HAS_EXPLICIT_UNFINISHED_JOBS'
        save(directory/'receipt.json',{'status':status,'results':results,'registration':registration,
                                     'full_evaluation_goal_complete':False})
        save(directory/'owner.json',{'status':status,'results':results})


if __name__=='__main__':main()
