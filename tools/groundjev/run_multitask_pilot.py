"""One blocking owner for legal native multilingual-task loops and fixed-candidate controls."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--names',nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=5)
    parser.add_argument('--resume-from',type=Path,help='reuse completed native/replay units; preserve original failed owner')
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()): parser.error('preserve the existing owner and predictions')
    args.out.mkdir(parents=True,exist_ok=True)
    state={'status':'REGISTERED_MULTITASK_NATIVE_LOOPS_STARTING','names':args.names,
        'decision_physical_gpu':args.gpu,'DLM_service_physical_gpu':6,
        'checkpoint':str(ROOT/'models/OneJev-0.8B'),'vocabulary_chunk_size':96,
        'evidence_format':'v1','grounding_decoder_policy':'tracking_dense',
        'candidate_control':'exact native candidate replay by Simple',
        'resume_from':str(args.resume_from) if args.resume_from else None,
        'trained_on_evaluation_GT':False,'pilot_completion':False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    peaks,devices={},defaultdict(int)
    errors=[]; stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                current=defaultdict(int)
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'],text=True)
                for uuid,pid,amount in csv.reader(io.StringIO(raw)):
                    proc=Path('/proc')/pid.strip()
                    try:
                        if not (proc/'cwd').resolve().is_relative_to(ROOT): continue
                        amount=int(amount)*2**20;key=uuid.strip()+':'+pid.strip()
                        peaks[key]=max(peaks.get(key,0),amount);current[uuid.strip()]+=amount
                    except (OSError,ValueError):continue
                for uuid,amount in current.items():devices[uuid]=max(devices[uuid],amount)
            except Exception as error:
                errors.append(str(error));return
            stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def run(stage,command):
        state['status']=stage;save()
        with (args.out/f'{stage.lower()}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        if code:
            state.update(status='MULTITASK_STAGE_FAILED',failed_stage=stage,exit_code=code);save()
            raise RuntimeError('First failed native/evaluation stage retained; inspect before resuming')
    save()
    try:
        for name in args.names:
            video=ROOT/'outputs/groundjev/multitask_pilot_data'/f'{name}.json'
            manifest=json.loads(video.read_text());task=manifest['task']
            full,simple=args.out/name/'full',args.out/name/'simple'
            common=['--task',task,'--video',str(video),'--grounding-parser','official_scoring',
                '--grounding-decoder-policy','tracking_dense','--perception-failure-policy','empty_and_record',
                '--grounding-vocabulary-chunk-size','96']
            reused={}
            if args.resume_from:
                from tools.groundjev.evaluate_multitask import check_scope
                for backend in ('full','simple'):
                    prior=args.resume_from/name/backend
                    if (prior/'result.json').is_file():
                        prior_result=json.loads((prior/'result.json').read_text())
                        prior_rows=[json.loads(line) for line in (prior/'frame_decisions.jsonl').read_text().splitlines()]
                        check_scope(manifest,prior_result,prior_rows)
                        reused[backend]=prior
            if 'full' in reused:
                full=reused['full'];state.setdefault('reused_completed_units',[]).append(str(full));save()
            else:
                run(name+'_NATIVE_FULL',['python3',str(ROOT/'tools/groundjev/run_with_resources.py'),
                    '--gpus',str(args.gpu),'--',str(ROOT/'.venv-onejev-py310/bin/python'),
                    str(ROOT/'tools/groundjev/run_video.py'),*common,'--backend','full','--out',str(full)])
            if 'simple' in reused:
                simple=reused['simple'];state.setdefault('reused_completed_units',[]).append(str(simple));save()
            else:
                run(name+'_FIXED_CANDIDATE_SIMPLE',[str(ROOT/'.venv-onejev-py310/bin/python'),
                    str(ROOT/'tools/groundjev/run_video.py'),*common,'--backend','simple','--out',str(simple),
                    '--candidate-replay',str(full/'frame_decisions.jsonl')])
            for backend,directory in [('full',full),('simple',simple)]:
                run(name+'_EVALUATE_'+backend,['python3',str(ROOT/'tools/groundjev/evaluate_multitask.py'),
                    '--video',str(video),'--run',str(directory)])
            state.setdefault('completed_scopes',{})[name]={backend:json.loads((directory/'official_metrics.json').read_text())
                for backend,directory in [('full',full),('simple',simple)]}
            save()
        state['status']='REGISTERED_MULTITASK_NATIVE_LOOPS_AND_CONTROLS_COMPLETE';save()
    finally:
        stop.set();thread.join(timeout=10)
        (args.out/'gpu_process_peaks.json').write_text(json.dumps({'sampling_interval_seconds':5,
            'process_peaks_bytes':peaks,'simultaneous_project_device_peaks_bytes':dict(devices),
            'scope':'all observed GroundJEV processes including concurrent MOT/SFT; foreign projects excluded',
            'monitor_errors':errors},indent=2)+'\n')


if __name__=='__main__':main()
