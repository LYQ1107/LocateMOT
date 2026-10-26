"""Native immutable J1 step1200, following a frozen Simple result."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,load,save,job_manifest,sha_file
from tools.groundjev.eval_no_mot20.replay_simple import audit_input,score_output


def replay(job,checkpoint,python=sys.executable,device='cuda:0'):
    if not (job/'simple/frozen_receipt.json').exists():raise ValueError('Simple must be officially scored and frozen first')
    with (job/'.j1.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'ALREADY_RUNNING'}
        output=job/'j1'
        if (output/'frozen_receipt.json').exists():return load(output/'frozen_receipt.json')
        manifest=job_manifest(job)
        if not (output/'result.json').exists():
            extra=[]
            if output.exists() and any(output.iterdir()):
                previous=job/f'j1_partial_{time.time_ns()}';output.rename(previous)
                if (previous/'frame_decisions.jsonl').is_file():extra=['--resume-from',str(previous)]
            command=[python,str(ROOT/'tools/groundjev/run_video.py'),'--task',manifest['task'],
                '--video',str(job/'manifest.json'),'--backend','trackjev','--checkpoint',str(checkpoint),
                '--trackjev-arm','B4','--trackjev-shortlist','31','--trackjev-device',device,
                '--candidate-replay',str(job/'detector_candidates.jsonl'),
                '--grounding-parser','official_scoring','--grounding-decoder-policy','tracking_dense',
                '--perception-failure-policy','empty_and_record','--no-match-policy','only_if_omitted',
                '--memory-export-interval','0','--memory-checkpoint-format','binary','--no-video',
                '--out',str(output)]+extra
            with (job/'j1_launch.log').open('a') as log:
                log.write(json.dumps({'command':command})+'\n');log.flush()
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=audit_input(job,output);measured=score_output(job,output)
        receipt={'status':'J1_FIXED_CANDIDATE_EVALUATION_COMPLETE','model_step':1200,
                 'checkpoint':str(checkpoint),'model_audit':str(RUN/'model_audit.json'),
                 'input':actual,'metrics':measured,'G2_calls':0,
                 'prediction_sha256':sha_file(output/'predictions.txt'),
                 'OVMOT_semantics':'fixed native G2 labels; J1 identity only; no J1 classification improvement claim'}
        save(output/'frozen_receipt.json',receipt)
        from tools.groundjev.eval_no_mot20.compare_arms import compare
        compare(job,'j1')
        return receipt


def evaluate_jev_queue(config,jobs):
    # Called after detector services have stopped. One GPU slot executes a
    # complete online sequence at a time; a later unit never inherits its IDs.
    from tools.groundjev.eval_no_mot20.dispatch_4gpu import gpu_environment
    checkpoint=load(Path(config['output_root'])/'model_audit.json')['J1_merged']['path']
    ready=[j for j in jobs if (Path(j['path'])/'simple/frozen_receipt.json').exists()]
    def work(worker,selection):
        env=gpu_environment(worker['gpu']);results=[]
        for entry in selection:
            job=Path(entry['path'])
            try:
                command=[config['onejev_python'],str(ROOT/'tools/groundjev/eval_no_mot20/replay_j1.py'),
                         '--job',str(job),'--checkpoint',checkpoint]
                with (job/'j1_worker.log').open('a') as log:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if entry['task']=='mot':
                    command=[config['ground_python'],str(ROOT/'tools/groundjev/eval_no_mot20/replay_coupled_cached_video.py'),
                             '--job',str(job),'--config',str(ROOT/'configs/groundjev/eval_no_mot20_4gpu_v1.json')]
                    with (job/'coupled_worker.log').open('a') as log:
                        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                else:save(job/'coupled/status.json',{'status':'NOT_SUPPORTED_BY_CURRENT_CHECKPOINT',
                    'reason':'existing coupled readout has no trained RMOT expression/OVMOT category decision contract'})
                results.append({'job':entry['id'],'status':'COMPLETE'})
            except Exception as error:
                save(job/'jev_error.json',{'status':'FAILED','error':repr(error),'worker':worker['name']})
                results.append({'job':entry['id'],'status':'FAILED'})
        return results
    with ThreadPoolExecutor(max_workers=len(config['workers'])) as pool:
        futures=[pool.submit(work,w,ready[i::len(config['workers'])]) for i,w in enumerate(config['workers'])]
        return [f.result() for f in futures]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(replay(args.job,args.checkpoint)),flush=True)


if __name__=='__main__':main()
