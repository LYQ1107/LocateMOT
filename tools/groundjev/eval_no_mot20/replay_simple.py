"""Use the existing causal runtime and official scorers on an immutable cache."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import (RUN,canonical,candidate_payload,job_manifest,load,
    merged_eval_manifest,save,sha_file)


def audit_input(job,output):
    import hashlib
    from itertools import zip_longest
    count=0;digest=hashlib.sha256()
    with (job/'detector_candidates.jsonl').open() as left,(output/'frame_decisions.jsonl').open() as right:
        for a,b in zip_longest(left,right):
            if a is None or b is None:raise ValueError('Replay changed registered frame count')
            first,second=json.loads(a),json.loads(b)
            if candidate_payload(first)!=candidate_payload(second):
                raise ValueError(f"Replay changed candidates at {first['frame']}")
            digest.update(canonical(candidate_payload(second))+b'\n');count+=1
    receipt=load(job/'cache_receipt.json')
    if digest.hexdigest()!=receipt['candidate_sha256']:raise ValueError('Replay input digest differs from G2')
    return {'frames':count,'candidate_sha256':digest.hexdigest(),'identical_geometry_order_labels_scores_rejection':True}


def score_output(job,output):
    evaluation=merged_eval_manifest(job);evaluation_path=job/'scoring_manifest.json';save(evaluation_path,evaluation)
    script='evaluate_video.py' if evaluation['task']=='mot' else 'evaluate_multitask.py'
    start=time.perf_counter()
    with (output/'official_scoring.log').open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'tools/groundjev'/script),'--video',str(evaluation_path),
                        '--run',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    measured=load(output/'official_metrics.json');measured['official_scoring_seconds']=time.perf_counter()-start
    # Every job is one complete video or one expression/video. Only the
    # aggregate can establish that all official validation jobs are finished.
    measured['registered_manifest_scope_label']=evaluation['result_scope']
    measured['result_scope']='REGISTERED_COMPLETE_VIDEO';save(output/'official_metrics.json',measured)
    return measured


def replay_and_score(job:Path,score=False):
    with (job/'.simple.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'ALREADY_RUNNING'}
        if (job/'simple/frozen_receipt.json').exists():return load(job/'simple/frozen_receipt.json')
        manifest=job_manifest(job);receipt=load(job/'cache_receipt.json')
        if sha_file(job/'detector_candidates.jsonl')!=receipt['jsonl_sha256']:raise ValueError('G2 cache changed')
        from tools.groundjev.eval_no_mot20.evaluate_g2_detection import measure
        if not (job/'detection_metrics.json').exists():measure(job)
        output=job/'simple'
        if not (output/'result.json').exists():
            if output.exists() and any(output.iterdir()):
                # Preserve unfinished attempts. Fresh replay starts at the legal
                # initial identity state; the expensive G2 detector is not rerun.
                archive=job/f'simple_failed_{time.time_ns()}';output.rename(archive)
            command=[sys.executable,str(ROOT/'tools/groundjev/run_video.py'),'--task',manifest['task'],
                     '--video',str(job/'manifest.json'),'--backend','simple','--candidate-replay',
                     str(job/'detector_candidates.jsonl'),'--grounding-parser','official_scoring',
                     '--grounding-decoder-policy','tracking_dense','--perception-failure-policy','empty_and_record',
                     '--memory-export-interval','0','--memory-checkpoint-format','binary','--no-video','--out',str(output)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            with (job/'simple_launch.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=audit_input(job,output)
        if not score:return actual
        measured=score_output(job,output)
        frozen={'status':'SIMPLE_OFFICIAL_COMPLETE_READY_FOR_JEV','input':actual,
                'prediction_sha256':sha_file(output/'predictions.txt'),'metrics':measured,
                'result_sha256':sha_file(output/'result.json'),'G2_calls':0,
                'candidate_cache':str(job/'detector_candidates.jsonl'),
                'candidate_cache_sha256':receipt['jsonl_sha256']}
        save(output/'frozen_receipt.json',frozen)
        save(job/'status.json',{'status':'READY_FOR_JEV','simple':str(output),'frames':receipt['frames']})
        return frozen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--score',action='store_true');args=parser.parse_args()
    print(json.dumps(replay_and_score(args.job,args.score)),flush=True)


if __name__=='__main__':main()
