"""Blocking, internal telemetry/report loop; no agent-level process polling.

Reads only the registered four GPUs and task artifacts. Optional publishing
stages only the seven small evaluation reports, never model/data/cache files.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,load,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--publish',action='store_true');parser.add_argument('--until-first-simple',action='store_true')
    parser.add_argument('--until-first-jev',action='store_true')
    args=parser.parse_args();config=load(args.config);run=Path(config['output_root'])
    gpus=','.join(str(w['gpu']) for w in config['workers']);began=time.time();last_report=last_push=0;last_signature=None
    fields='index,uuid,memory.used,utilization.gpu,power.draw'
    while True:
        now=time.time()
        try:
            raw=subprocess.check_output(['nvidia-smi','-i',gpus,'--query-gpu='+fields,
                                         '--format=csv,noheader,nounits'],text=True)
            rows=list(csv.reader(io.StringIO(raw)))
            app_raw=subprocess.check_output(['nvidia-smi','-i',gpus,
                '--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
            owned_groups=set()
            for path in (run/'workers').glob('*/service_ready.json'):
                ready=load(path);owned_groups.add(int(ready['pid']))
            processes=[]
            for uuid,pid,memory in csv.reader(io.StringIO(app_raw)):
                try:
                    owned=os.getpgid(int(pid)) in owned_groups
                    owned_eval=(Path('/proc')/pid.strip()/'cwd').resolve()==ROOT
                except (ProcessLookupError,OSError):owned=owned_eval=False
                processes.append({'uuid':uuid.strip(),'pid':int(pid),'memory_MiB':int(memory),
                                  'owned_G2_service':owned,'owned_evaluation':owned_eval})
            with (run/'gpu_telemetry.jsonl').open('a') as handle:
                handle.write(json.dumps({'time':now,'devices':[dict(zip(fields.split(','),[v.strip() for v in row])) for row in rows],
                                        'memory_and_utilization_include_shared_foreign_jobs':True,
                                        'compute_processes':processes,
                                        'owned_G2_memory_MiB':sum(p['memory_MiB'] for p in processes if p['owned_G2_service']),
                                        'owned_evaluation_memory_MiB':sum(p['memory_MiB'] for p in processes if p['owned_evaluation'])})+'\n')
        except Exception as error:
            with (run/'monitor_errors.jsonl').open('a') as handle:handle.write(json.dumps({'time':now,'error':repr(error)})+'\n')
        if now-last_report>=60:
            command=[config['onejev_python'],str(ROOT/'tools/groundjev/eval_no_mot20/aggregate_metrics.py'),
                     '--write-docs','--with-teta']
            proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            if proc.returncode:
                print(json.dumps({'event':'REPORT_FAILED','error':proc.stderr[-3000:]}),flush=True)
            else:
                signature=proc.stdout.strip()
                if signature!=last_signature:print(signature,flush=True);last_signature=signature
                report=load(run/'aggregate_metrics.json')
                simple=sum(d['tracking'].get('simple',{}).get('complete_jobs',0) for d in report['datasets'].values())
                if args.publish and now-last_push>=3600:
                    names=[str(p.relative_to(ROOT)) for p in (ROOT/'docs/GROUNDJEV').glob('EVAL_NO_MOT20_*.md')]
                    if config.get('publication_mode')=='connected_github':
                        save(run/'publication_request.json',{'status':'AWAITING_CONNECTED_GITHUB_UPLOAD',
                            'paths':names,'time':now,'repository':'LYQ1107/LocateMOT',
                            'branch':'codex/g2-four-gpu-no-mot20-eval-20261026'})
                        last_push=now
                        continue
                    subprocess.run(['git','add','--',*names],cwd=ROOT,check=True)
                    diff=subprocess.run(['git','diff','--cached','--quiet','--',*names],cwd=ROOT)
                    if diff.returncode==1:
                        subprocess.run(['git','commit','-m','research: update measured non-MOT20 evaluation progress','--',*names],cwd=ROOT,check=True)
                        push=subprocess.run(['git','push','origin','HEAD'],cwd=ROOT,capture_output=True,text=True)
                        if push.returncode:
                            save(run/'publication_error.json',{'status':'GIT_PUSH_FAILED_RESEARCH_CONTINUES',
                                'error':push.stderr[-2000:],'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
                    last_push=now
                if args.until_first_simple and simple:
                    print(json.dumps({'event':'FIRST_COMPLETE_SIMPLE_OFFICIAL_RESULT','jobs':simple}),flush=True)
                    return
                jev=sum(d['tracking'].get(arm,{}).get('complete_jobs',0)
                        for d in report['datasets'].values() for arm in ('j1','coupled'))
                if args.until_first_jev and jev:
                    print(json.dumps({'event':'FIRST_COMPLETE_JEV_OFFICIAL_RESULT','jobs':jev}),flush=True)
                    return
            last_report=now
        time.sleep(5)


if __name__=='__main__':main()
