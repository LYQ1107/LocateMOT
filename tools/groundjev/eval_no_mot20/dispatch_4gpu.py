"""Four persistent, single-GPU services consuming distinct resumable frame units.

SQLite handles atomic queue claims. Association always consumes a complete
video in its original order and is never split into independent ID histories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import load,save,job_manifest


def gpu_environment(gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
             MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
             PYTHONPATH=str(ROOT)+':'+str(ROOT/'external/OneJev'))
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        env.pop(key,None)
    env['NO_PROXY']='127.0.0.1,localhost';env['no_proxy']=env['NO_PROXY']
    return env


def log_run(command,path,env):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps({'command':command,'start':time.time()})+'\n');handle.flush()
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'exit={result.returncode}; see {path}')


def ensure_images(job):
    manifest=job_manifest(job)
    if not manifest.get('raw_video'):return
    frame_root=Path(manifest['images'][0]).parent;frame_root.mkdir(parents=True,exist_ok=True)
    with (frame_root/'.decode.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (frame_root/'decode_receipt.json').exists():return
        began=time.perf_counter()
        command=['ffmpeg','-v','error','-y','-threads','2','-i',manifest['raw_video'],
                 '-vsync','0','-pix_fmt','rgb24','-start_number','1',str(frame_root/'%06d.png')]
        subprocess.run(command,check=True)
        if not all(Path(p).is_file() for p in manifest['images']):
            raise RuntimeError('Decoded physical video is incomplete')
        save(frame_root/'decode_receipt.json',{'frames':len(manifest['images']),'command':command,
              'seconds':time.perf_counter()-began,'lossless_decoded_pixels':True})


class Queue:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=120)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,job TEXT,start INTEGER,
            stop INTEGER,priority INTEGER,state TEXT,owner TEXT,attempts INTEGER,error TEXT)''')

    def initialize(self,jobs,chunk):
        # The first short complete job from every family gets priority, then
        # every remaining registered job. Selection uses size, never accuracy.
        ordered=sorted(jobs,key=lambda j:(j['frames'],j['id']))
        seen=set();first=[];rest=[]
        for job in ordered:
            if job['dataset'] not in seen:first.append(job);seen.add(job['dataset'])
            else:rest.append(job)
        # Interleave entire videos/queries by family. A thousand short TAO
        # units must not starve every later SportsMOT/AnimalTrack video.
        families=sorted({j['dataset'] for j in ordered})
        grouped={name:[j for j in ordered if j['dataset']==name] for name in families}
        balanced=[]
        for index in range(max(map(len,grouped.values()))):
            for name in families:
                if index<len(grouped[name]):balanced.append(grouped[name][index])
        with self.db:
            for rank,job in enumerate(balanced):
                for start in range(0,job['frames'],chunk):
                    stop=min(start+chunk,job['frames']);key=f"{job['id']}:{start}:{stop}"
                    self.db.execute('INSERT OR IGNORE INTO units VALUES(?,?,?,?,?,?,?,?,?)',
                                    (key,job['path'],start,stop,rank,'PENDING',None,0,None))
        return first

    def claim(self,owner):
        self.db.execute('BEGIN IMMEDIATE')
        row=self.db.execute("SELECT id,job,start,stop FROM units WHERE state='PENDING' ORDER BY priority,start LIMIT 1").fetchone()
        if row:self.db.execute("UPDATE units SET state='RUNNING',owner=?,attempts=attempts+1 WHERE id=?",(owner,row[0]))
        self.db.commit();return row

    def finish(self,key,error=None):
        with self.db:self.db.execute('UPDATE units SET state=?,error=? WHERE id=?',('FAILED' if error else 'COMPLETE',error,key))


def start_service(config,worker,directory):
    gpu,port=worker['gpu'],worker['port'];env=gpu_environment(gpu)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',port))
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free',
                                 '--format=csv,noheader,nounits'],text=True).strip()
    uuid,free=[s.strip() for s in raw.split(',')]
    if uuid!=worker['uuid'] or int(free)<config['minimum_free_MiB']:
        raise RuntimeError(f'GPU UUID/free-memory requirement not met: {raw}')
    command=[config['ground_python'],str(ROOT/'tools/groundjev/serve_groundanything.py'),
             '--model',config['g2_model'],'--port',str(port),'--mem-fraction-static',str(config['memory_fraction']),
             '--max-queued-requests','2','--out',str(directory/'service')]
    if config.get('decoder_overlay'):
        command+=['--decoder-overlay',str(ROOT/config['decoder_overlay'])]
    handle=(directory/'service.log').open('a')
    began=time.perf_counter()
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+600
        while True:
            if process.poll() is not None:raise RuntimeError(f'DLM service exited {process.returncode}; {directory}/service.log')
            try:
                with urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:
                    if any(d['id']=='groundinganything' for d in json.load(response)['data']):break
            except (OSError,ValueError):pass
            if time.monotonic()>deadline:raise TimeoutError('DLM service startup exceeded 600s')
            time.sleep(2)
        save(directory/'service_ready.json',{'gpu':gpu,'uuid':uuid,'pid':process.pid,'command':command,
             'model':config['g2_model'],'load_seconds':time.perf_counter()-began,'free_MiB_at_start':int(free)})
        return process,handle
    except BaseException:
        stop_service(process,handle);raise


def stop_service(process,handle):
    if process is not None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
    if handle is not None:handle.close()


def concurrency_gate(config,worker,context):
    from PIL import Image
    from locatemot.groundjev.perception import GroundAnythingAdapter,GroundingProtocolError
    from tools.groundjev.eval_no_mot20.common import candidate_payload
    entry=context['test_job'];manifest=job_manifest(Path(entry['path']));frame=manifest['frame_numbers'][0]
    path=manifest['images'][0]
    def native():
        frontend=GroundAnythingAdapter(f"http://127.0.0.1:{worker['port']}/v1",parser_policy='official_scoring',
            decoder_policy='tracking_dense',allow_vocabulary_fallback=False)
        with Image.open(path) as image:
            try:
                candidates,metadata=frontend(image.convert('RGB'),'mot',['person'],image_path=path)
                rejected=False
            except GroundingProtocolError as error:candidates=[];metadata=error.metadata;rejected=True
        return {'frame':frame,'image_path':path,'candidates':candidates,
                'perception':{'model_response_rejected':rejected},'raw_metadata':metadata}
    barrier=context['barrier'];barrier.wait(timeout=700)
    if worker['name']==config['workers'][0]['name']:context['baseline']=native()
    barrier.wait(timeout=700)
    context['results'][worker['name']]=native()
    barrier.wait(timeout=700)
    equal=all(candidate_payload(value)==candidate_payload(context['baseline']) for value in context['results'].values())
    if worker['name']==config['workers'][0]['name']:
        save(Path(config['output_root'])/'engineering_checks/worker_consistency.json',{
            'status':'FOUR_WORKER_NATIVE_CONSISTENCY_PASSED' if equal else 'FOUR_WORKER_NATIVE_CONSISTENCY_FAILED',
            'baseline':context['baseline'],'workers':context['results'],'seed':42,
            'seed_scope':'request_local_v1','functional_only':True,'new_evaluation_frames':0})
        print(json.dumps({'event':'CONCURRENCY_GATE','passed':equal,'workers':len(context['results'])}),flush=True)
    if not equal:raise RuntimeError('Four-worker native output consistency failed; no formal queue was entered')


def finish_video_pipeline(config,worker,job,directory,env):
    if not (job/'cache_receipt.json').exists():return False
    with (job/'.pipeline.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        command=[config['onejev_python'],str(ROOT/'tools/groundjev/eval_no_mot20/replay_simple.py'),'--job',str(job),'--score']
        log_run(command,directory/'simple.log',gpu_environment(''))
        return (job/'simple/frozen_receipt.json').exists()


def run_jev_for_video(config,worker,job,directory,env):
    audit=load(Path(config['output_root'])/'model_audit.json');manifest=job_manifest(job)
    for arm in ('j1','coupled'):
        if (job/arm/'frozen_receipt.json').exists() or (job/arm/'status.json').exists():continue
        if arm=='coupled' and manifest['task']!='mot':
            save(job/arm/'status.json',{'status':'NOT_SUPPORTED_BY_CURRENT_CHECKPOINT',
                'reason':'no trained referring expression / open-vocabulary semantic head'})
            continue
        command=([config['onejev_python'],str(ROOT/'tools/groundjev/eval_no_mot20/replay_j1.py'),
                  '--job',str(job),'--checkpoint',audit['J1_merged']['path']] if arm=='j1' else
                 [config['ground_python'],str(ROOT/'tools/groundjev/eval_no_mot20/replay_coupled_cached_video.py'),
                  '--job',str(job),'--config',str(ROOT/'configs/groundjev/eval_no_mot20_4gpu_v1.json')])
        try:log_run(command,directory/f'{arm}.log',env)
        except Exception as error:
            save(job/f'{arm}_error.json',{'status':'FAILED','error':repr(error),'worker':worker['name']})
            print(json.dumps({'event':'JEV_ARM_FAILED','arm':arm,'job':job.name,'error':repr(error)}),flush=True)


def detection_worker(config,worker,first,context):
    run=Path(config['output_root']);directory=run/'workers'/worker['name'];directory.mkdir(parents=True,exist_ok=True)
    queue=Queue(run/'queue.sqlite');process=handle=None
    env=gpu_environment(worker['gpu']);began=time.perf_counter();units=0
    try:
        process,handle=start_service(config,worker,directory)
        concurrency_gate(config,worker,context)
        # Each family receives an actual 32-input measurement before it enters
        # the full queue. These exact frames remain valid resumable cache rows.
        if first is not None:
            job=Path(first['path']);ensure_images(job)
            command=[config['ground_python'],str(ROOT/'tools/groundjev/eval_no_mot20/export_g2_candidates.py'),
                     '--job',str(job),'--grounding-url',f"http://127.0.0.1:{worker['port']}/v1",
                     '--start','0','--stop',str(min(32,first['frames'])),'--worker',worker['name'],
                     '--timeout',str(config['request_timeout'])]
            started=time.perf_counter();log_run(command,directory/'preflight.log',env)
            if not (directory/'preflight.json').exists():
                save(directory/'preflight.json',{'status':'REAL_GPU_32_INPUT_PREFLIGHT_COMPLETE',
                    'dataset':first['dataset'],'job':str(job),'inputs':min(32,first['frames']),
                    'wall_seconds':time.perf_counter()-started,'gpu':worker['gpu'],'uuid':worker['uuid'],
                    'functional_only':True})
            key=f"{first['id']}:0:{min(32,first['frames'])}"
            queue.finish(key)
        while True:
            unit=queue.claim(worker['name'])
            if unit is None:break
            key,path,start,stop=unit;job=Path(path)
            try:
                ensure_images(job)
                command=[config['ground_python'],str(ROOT/'tools/groundjev/eval_no_mot20/export_g2_candidates.py'),
                    '--job',path,'--grounding-url',f"http://127.0.0.1:{worker['port']}/v1",'--start',str(start),
                    '--stop',str(stop),'--timeout',str(config['request_timeout']),'--worker',worker['name']]
                log_run(command,directory/'detection.log',env)
                queue.finish(key);units+=1
                save(directory/'status.json',{'status':'DETECTING','last_unit':key,'completed_units':units,
                     'elapsed_seconds':time.perf_counter()-began,'gpu':worker['gpu']})
                if finish_video_pipeline(config,worker,job,directory,env):
                    with (job/'.jev_pipeline.lock').open('w') as pipeline_lock:
                        try:fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:continue
                        # Rotate this physical slot to JEV while the other
                        # workers continue detecting; never allocate a fifth GPU.
                        stop_service(process,handle);process=handle=None
                        run_jev_for_video(config,worker,job,directory,env)
                        process,handle=start_service(config,worker,directory)
            except Exception as error:
                queue.finish(key,repr(error))
                save(directory/'last_error.json',{'unit':key,'error':repr(error),'time':time.time()})
                print(json.dumps({'event':'UNIT_FAILED','worker':worker['name'],'unit':key,'error':repr(error)}),flush=True)
                # Service failures receive one identical restart. Failed units
                # remain explicit for a later resume, other families continue.
                stop_service(process,handle);process=handle=None
                process,handle=start_service(config,worker,directory)
        return {'worker':worker['name'],'status':'DETECTION_QUEUE_DRAINED','completed_units':units}
    finally:
        stop_service(process,handle)
        save(directory/'resource_receipt.json',{'gpu':worker['gpu'],'uuid':worker['uuid'],
            'service_wall_seconds':time.perf_counter()-began,'GPU_hours_allocated':(time.perf_counter()-began)/3600,
            'all_owned_service_processes_stopped':True})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['detection','simple','jev'],required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=load(args.config);run=Path(config['output_root']);registry=load(run/'jobs.json')
    if len(config['workers'])>4 or len({w['gpu'] for w in config['workers']})!=len(config['workers']):
        raise ValueError('At most four distinct physical GPUs')
    with (run/f'.dispatch_{args.stage}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='detection':
            if not (run/'model_audit.json').exists():raise ValueError('Immutable model audit is required')
            queue=Queue(run/'queue.sqlite');first=queue.initialize(registry['jobs'],config['detection_unit_frames'])
            if args.resume:
                with queue.db:queue.db.execute("UPDATE units SET state='PENDING',owner=NULL WHERE state='RUNNING'")
            first_by_dataset={dataset:min((j for j in registry['jobs'] if j['dataset']==dataset and j['frames']>=32),
                                          key=lambda j:(j['frames'],j['id'])) for dataset in registry['summary']}
            with queue.db:
                for worker in config['workers']:
                    job=first_by_dataset[worker['initial_dataset']]
                    key=f"{job['id']}:0:{min(32,job['frames'])}"
                    queue.db.execute("UPDATE units SET state='RUNNING',owner=? WHERE id=? AND state!='COMPLETE'",(worker['name'],key))
            context={'barrier':threading.Barrier(len(config['workers'])),'results':{},
                     'test_job':first_by_dataset['sportsmot']}
            with ThreadPoolExecutor(max_workers=len(config['workers'])) as pool:
                futures=[pool.submit(detection_worker,config,w,first_by_dataset.get(w['initial_dataset']),context) for w in config['workers']]
                results=[]
                for future in as_completed(futures):
                    try:results.append(future.result())
                    except Exception as error:results.append({'status':'WORKER_FAILED','error':repr(error)})
                    print(json.dumps(results[-1]),flush=True)
            pending=queue.db.execute("SELECT state,COUNT(*) FROM units GROUP BY state").fetchall()
            save(run/'detection_stage_receipt.json',{'results':results,'queue_states':dict(pending),
                'status':'COMPLETE' if all(s=='COMPLETE' for s,n in pending) else 'INCOMPLETE'})
        else:
            from tools.groundjev.eval_no_mot20.replay_simple import replay_and_score
            for job in registry['jobs']:
                path=Path(job['path'])
                if not (path/'cache_receipt.json').exists():continue
                if args.stage=='simple':replay_and_score(path,score=True)
                else:
                    from tools.groundjev.eval_no_mot20.replay_j1 import evaluate_jev_queue
                    evaluate_jev_queue(config,registry['jobs']);break


if __name__=='__main__':main()
