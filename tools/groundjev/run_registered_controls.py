"""One blocking owner for finite, preregistered fixed-native-candidate controls."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume-completed',type=Path)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Preserve the existing control owner')
    args.out.mkdir(parents=True)
    state={'status':'REGISTERED_CONTROL_OWNER_STARTED','plan':str(args.plan),'physical_gpu':plan['physical_gpu'],
        'native_candidates_fixed':True,'end_to_end_FPS_claim':False,'Pilot_complete':False,'completed':{}}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'status':state['status'],'completed':list(state['completed'])}),flush=True)
    peaks,device_peaks={},defaultdict(int);errors=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
                current=defaultdict(int)
                for uuid,pid,amount in csv.reader(io.StringIO(raw)):
                    try:
                        if not (Path('/proc')/pid.strip()/'cwd').resolve().is_relative_to(ROOT):continue
                        value=int(amount)*2**20;key=uuid.strip()+':'+pid.strip()
                        peaks[key]=max(peaks.get(key,0),value);current[uuid.strip()]+=value
                    except (OSError,ValueError):continue
                for uuid,value in current.items():device_peaks[uuid]=max(device_peaks[uuid],value)
            except Exception as error:errors.append(str(error));return
            stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def run(stage,command):
        state['status']=stage;save()
        with (args.out/f'{stage}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        if code:
            state.update(status='REGISTERED_CONTROL_FAILED',failed_stage=stage,exit_code=code);save()
            raise RuntimeError('First failed control preserved; resume only after root-cause repair')
    save()
    try:
        from tools.groundjev.evaluate_multitask import check_scope
        for job in plan['jobs']:
            name=job['name'];video=ROOT/job['video'];manifest=json.loads(video.read_text())
            out=args.out/name;prior=args.resume_completed/name if args.resume_completed else None
            if prior is not None and (prior/'result.json').is_file():
                result=json.loads((prior/'result.json').read_text())
                rows=[json.loads(s) for s in (prior/'frame_decisions.jsonl').read_text().splitlines()]
                check_scope(manifest,result,rows);out=prior
            else:
                command=['python3',str(ROOT/'tools/groundjev/run_with_resources.py'),'--gpus',str(plan['physical_gpu']),'--',
                    str(ROOT/'.venv-onejev-py310/bin/python'),str(ROOT/'tools/groundjev/run_video.py'),
                    '--task',manifest['task'],'--video',str(video),'--candidate-replay',str(ROOT/job['candidates']),
                    '--backend',job['backend'],'--evidence-format',job['evidence_format'],'--out',str(out)]
                if 'checkpoint' in job:command+=['--checkpoint',str(ROOT/job['checkpoint'])]
                if job.get('adaptive_context'):command+=['--adaptive-context-groups']
                if 'category_policy' in job:command+=['--category-policy',job['category_policy']]
                run(name+'_INFERENCE',command)
            run(name+'_OFFICIAL_EVAL',['python3',str(ROOT/'tools/groundjev/evaluate_multitask.py'),'--video',str(video),'--run',str(out)])
            state['completed'][name]={'run':str(out),'official_metrics':json.loads((out/'official_metrics.json').read_text())};save()
        state['status']='REGISTERED_FINITE_CONTROLS_COMPLETE';save()
    finally:
        stop.set();thread.join(timeout=10)
        (args.out/'gpu_process_peaks.json').write_text(json.dumps({'sampling_interval_seconds':5,
            'process_peaks_bytes':peaks,'simultaneous_project_device_peaks_bytes':dict(device_peaks),
            'scope':'all GroundJEV processes observed, including concurrent SFT/MOT; foreign projects excluded',
            'errors':errors},indent=2)+'\n')


if __name__=='__main__':main()
