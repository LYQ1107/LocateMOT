"""Observe the existing native queue; never launch or stop an inference job."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,load,save


def counters(jobs):
    frames={};requests=0
    for job in jobs:
        database=Path(job['path'])/'detector.sqlite'
        if not database.exists():continue
        db=sqlite3.connect(f'file:{database}?mode=ro',uri=True,timeout=30)
        try:
            frames[job['id']]={int(row[0]) for row in db.execute('SELECT frame FROM frames')}
            requests+=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
        finally:db.close()
    return frames,requests


def resource_window(path,start,end,registered):
    samples=[];previous=None
    for line in path.open():
        row=json.loads(line)
        if row['time']>end:break
        if row['time']<start:previous=row
        else:samples.append(row)
    if previous:samples.insert(0,previous)
    gpu_seconds=defaultdict(float);peaks=defaultdict(int);simultaneous=0;gaps=[];coverage=0.
    for index,row in enumerate(samples):
        left=max(start,row['time'])
        right=min(end,samples[index+1]['time'] if index+1<len(samples) else end)
        duration=max(0.,right-left)
        if not duration:continue
        coverage+=duration;resident=defaultdict(int)
        for process in row['compute_processes']:
            if process['uuid'] in registered and (process['owned_G2_service'] or process.get('owned_evaluation',False)):
                resident[process['uuid']]+=process['memory_MiB']
        simultaneous=max(simultaneous,len(resident))
        for uuid,memory in resident.items():
            gpu_seconds[uuid]+=duration;peaks[uuid]=max(peaks[uuid],memory)
        if index+1<len(samples):gaps.append(samples[index+1]['time']-row['time'])
    return {'sampled_owned_residency_GPU_hours_estimate':sum(gpu_seconds.values())/3600,
        'by_GPU':{registered[uuid]:{'GPU_hours_estimate':seconds/3600,
                                  'sampled_peak_owned_MiB':peaks[uuid]} for uuid,seconds in gpu_seconds.items()},
        'peak_simultaneous_registered_owned_GPUs':simultaneous,
        'telemetry_covered_seconds':coverage,'maximum_sample_gap_seconds':max(gaps,default=None),
        'measurement':'Left-held resident-process samples, including idle model residency; '
                      'not kernel busy time or foreign-job utilization. Sampled memory peaks are lower bounds.'}


def observe(run,out,minimum_new_frames=64,max_seconds=600,interval=5):
    jobs=load(run/'jobs.json')['jobs']
    config=load(ROOT/'configs/groundjev/eval_no_mot20_4gpu_v1.json')
    registered={worker['uuid']:str(worker['gpu']) for worker in config['workers']}
    start=time.time();clock=time.perf_counter();before,requests_before=counters(jobs)
    save(out.with_name(out.stem+'_registration.json'),{
        'status':'OBSERVING_EXISTING_REGISTERED_QUEUE','start_unix':start,
        'minimum_new_frames':minimum_new_frames,'maximum_observation_seconds':max_seconds,
        'baseline_persisted_frames':sum(map(len,before.values())),
        'baseline_persisted_requests':requests_before,'physical_GPUs':registered,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'new_G2_calls':0,'inference_processes_started_or_stopped':0,'GT_read':False})
    while True:
        after,requests_after=counters(jobs)
        new_count=sum(len(frames-before.get(job,set())) for job,frames in after.items())
        if new_count>=minimum_new_frames or time.perf_counter()-clock>=max_seconds:break
        time.sleep(interval)
    end=time.time();elapsed=time.perf_counter()-clock;datasets=defaultdict(lambda:{'frames':0,'rejected':0})
    for job in jobs:
        new_frames=after.get(job['id'],set())-before.get(job['id'],set())
        if not new_frames:continue
        db=sqlite3.connect(f"file:{job['path']}/detector.sqlite?mode=ro",uri=True,timeout=30)
        try:
            for frame in new_frames:
                row=json.loads(gzip.decompress(db.execute('SELECT data FROM frames WHERE frame=?',(frame,)).fetchone()[0]))
                datasets[job['dataset']]['frames']+=1
                datasets[job['dataset']]['rejected']+=int(row['perception']['model_response_rejected'])
        finally:db.close()
    result={'status':'EXISTING_QUEUE_OBSERVATION_COMPLETE','start_unix':start,'end_unix':end,
        'elapsed_seconds':elapsed,'new_persisted_query_frames':new_count,
        'new_persisted_native_requests':requests_after-requests_before,
        'aggregate_persisted_query_frames_per_wall_second':new_count/elapsed,
        'aggregate_persisted_native_requests_per_wall_second':(requests_after-requests_before)/elapsed,
        'frame_target_reached':new_count>=minimum_new_frames,'datasets':dict(datasets),
        'resource_window':resource_window(run/'gpu_telemetry.jsonl',start,end,registered),
        'scope':'Observed production window, including swaps/recovery/association and rejected frames. '
                'Sparse TAO frames and repeated RMOT query-frames are not physical video FPS. '
                'No single-GPU matched control; no fourfold-speedup claim.',
        'new_G2_calls':0,'inference_processes_started_or_stopped':0,'GT_read':False,
        'full_goal_complete':False}
    save(out,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RUN)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--minimum-new-frames',type=int,default=64)
    parser.add_argument('--max-seconds',type=float,default=600)
    args=parser.parse_args()
    print(json.dumps(observe(args.root,args.out,args.minimum_new_frames,args.max_seconds)),flush=True)


if __name__=='__main__':main()
