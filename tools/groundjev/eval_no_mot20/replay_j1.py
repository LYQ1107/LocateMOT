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
from tools.groundjev.eval_no_mot20.common import RUN,load,save,job_manifest,sha_file,arm_directory
from tools.groundjev.eval_no_mot20.replay_simple import audit_input,score_output


def replay(job,checkpoint,python=sys.executable,device='cuda:0',arm='j1',competition_geometry='legacy_xywh',
           candidate_evidence='full_frame',memory_transport='legacy',empty_history_question='identity',history_evidence='none'):
    if not (job/'simple/frozen_receipt.json').exists():raise ValueError('Simple must be officially scored and frozen first')
    with (job/f'.{arm}.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'ALREADY_RUNNING'}
        output=job/arm
        if (output/'frozen_receipt.json').exists():return load(output/'frozen_receipt.json')
        manifest=job_manifest(job)
        if not (output/'result.json').exists():
            extra=[]
            if output.exists() and any(output.iterdir()):
                previous=job/f'{arm}_partial_{time.time_ns()}';output.rename(previous)
                if (previous/'frame_decisions.jsonl').is_file():extra=['--resume-from',str(previous)]
            command=[python,str(ROOT/'tools/groundjev/run_video.py'),'--task',manifest['task'],
                '--video',str(job/'manifest.json'),'--backend','trackjev','--checkpoint',str(checkpoint),
                '--trackjev-arm','B4','--trackjev-shortlist','31','--trackjev-device',device,
                '--trackjev-competition-geometry',competition_geometry,
                '--trackjev-candidate-evidence',candidate_evidence,
                '--trackjev-memory-transport',memory_transport,
                '--trackjev-empty-history-question',empty_history_question,
                '--trackjev-history-evidence',history_evidence,
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
                 'artifact_arm':arm,'competition_geometry':competition_geometry,
                 'candidate_evidence':candidate_evidence,'memory_transport':memory_transport,
                 'empty_history_question':empty_history_question,
                 'history_evidence':history_evidence,
                 'checkpoint':str(checkpoint),'model_audit':str(RUN/'model_audit.json'),
                 'input':actual,'metrics':measured,'G2_calls':0,
                 'prediction_sha256':sha_file(output/'predictions.txt'),
                 'OVMOT_semantics':'fixed native G2 labels; J1 identity only; no J1 classification improvement claim'}
        save(output/'frozen_receipt.json',receipt)
        from tools.groundjev.eval_no_mot20.compare_arms import compare
        compare(job,arm)
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
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--arm');parser.add_argument('--competition-geometry',choices=['xyxy','legacy_xywh'])
    parser.add_argument('--candidate-evidence',choices=['full_frame','train_current_crops'])
    parser.add_argument('--memory-transport',choices=['legacy','local_xyxy'])
    parser.add_argument('--empty-history-question',choices=['identity','train_birth_v1'])
    parser.add_argument('--history-evidence',choices=['none','causal_crops'])
    parser.add_argument('--no-followups',action='store_true',help='run only this explicitly registered control')
    args=parser.parse_args();config_path=ROOT/'configs/groundjev/eval_no_mot20_4gpu_v1.json';config=load(config_path)
    arm=args.arm or arm_directory('j1',config)
    # Named historical arms must remain reconstructible after the main
    # protocol advances; an explicit legacy arm cannot inherit new images.
    known={
        'j1':('legacy_xywh','full_frame','legacy'),
        'j1_competition_xyxy_v2':('xyxy','full_frame','legacy'),
        'j1_current_crops_v3':('xyxy','train_current_crops','legacy'),
        'j1_current_crops_memory_v4':('xyxy','train_current_crops','local_xyxy'),
        'j1_train_birth_prompt_v5':('xyxy','train_current_crops','local_xyxy'),
        'j1_causal_history_v6':('xyxy','train_current_crops','local_xyxy'),
    }
    defaults=known.get(arm,(config.get('j1_competition_geometry','legacy_xywh'),
                           config.get('j1_candidate_evidence','full_frame'),
                           config.get('j1_memory_transport','legacy')))
    geometry=args.competition_geometry or defaults[0]
    evidence=args.candidate_evidence or defaults[1]
    memory=args.memory_transport or defaults[2]
    birth=args.empty_history_question or ('train_birth_v1' if arm in ('j1_train_birth_prompt_v5','j1_causal_history_v6') else 'identity')
    history=args.history_evidence or ('causal_crops' if arm=='j1_causal_history_v6' else 'none')
    print(json.dumps(replay(args.job,args.checkpoint,arm=arm,competition_geometry=geometry,
                            candidate_evidence=evidence,memory_transport=memory,empty_history_question=birth,
                            history_evidence=history)),flush=True)
    if args.no_followups:return
    # Backfill one already scored Simple scope in the same reserved GPU
    # slot; detection caches and historical J1 predictions stay untouched.
    if (config.get('j1_backfill_ready') or config.get('j1_backfill_legacy')) and arm!='j1':
        jobs=load(Path(config['output_root'])/'jobs.json')['jobs']
        pending=[j for j in jobs if (Path(j['path'])/'simple/frozen_receipt.json').exists()
                 and not (Path(j['path'])/arm/'frozen_receipt.json').exists()]
        if pending:
            entry=min(pending,key=lambda j:(j['dataset']!='sportsmot',j['frames'],j['id']))
            print(json.dumps(replay(Path(entry['path']),args.checkpoint,arm=arm,competition_geometry=geometry,
                                    candidate_evidence=evidence,memory_transport=memory,empty_history_question=birth,
                                    history_evidence=history)),flush=True)
    for probe_mode in config.get('execution_probe_modes',['fast_eager']) if config.get('execution_probe_after_j1') else []:
        receipt=Path(config['output_root'])/'engineering_checks'/probe_mode/'receipt.json'
        if receipt.exists():continue
        # The dispatcher has stopped this slot's detector and is waiting for
        # this child. Native J1 inference has exited before the execution probe.
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        worker=next((w for w in config['workers'] if str(w['gpu'])==visible),None)
        if worker is not None:
            with (args.job/'.jev_pipeline.lock').open('a') as slot_lock:
                try:fcntl.flock(slot_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    log=Path(config['output_root'])/'engineering_checks/execution_profile_owner.log'
                    with log.open('a') as handle:
                        result=subprocess.run([config['ground_python'],str(ROOT/'tools/groundjev/eval_no_mot20/profile_execution.py'),
                            '--config',str(config_path),'--worker',worker['name'],'--mode',probe_mode],cwd=ROOT,
                            stdout=handle,stderr=subprocess.STDOUT)
                    if result.returncode:save(log.with_suffix('.error.json'),{
                        'status':'EXECUTION_PROFILE_FAILED_TRACKING_RESULT_PRESERVED',
                        'exit_code':result.returncode,'log':str(log)})


if __name__=='__main__':main()
