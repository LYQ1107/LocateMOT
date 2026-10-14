"""One blocking recovery owner gates unchanged full native evaluations on queue regression."""
import json
import os
from pathlib import Path
import subprocess
import time
import argparse

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--regression-receipt',type=Path,required=True);args=parser.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=False)
    upstream=None
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]
            if len(argv)>1 and 'python' in Path(argv[0]).name and argv[1]=='tools/groundjev/check_ground_queue_capacity.py':
                assert (proc/'cwd').resolve()==ROOT
                upstream=(int(proc.name),(proc/'stat').read_text().rsplit(')',1)[1].split()[19])
        except (FileNotFoundError,PermissionError):pass
    state={'status':'WAITING_ACTUAL18_CLIENT_QUEUE_REGRESSION','regression_owner':upstream,
        'max_queued_requests':32,'max_running_requests':1,'unchanged_client_concurrency_per_task':2,
        'nine_canonical_tasks_not_all30':True,'completed':{},'children':{}}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    save()
    try:
        if upstream:
            while True:
                try:
                    fields=(Path('/proc')/str(upstream[0])/'stat').read_text().rsplit(')',1)[1].split()
                    if fields[0]=='Z' or fields[19]!=upstream[1]:break
                except FileNotFoundError:break
                time.sleep(10)
        receipt=json.loads(args.regression_receipt.read_text())
        assert receipt['status']=='ACTUAL18_CONCURRENT_REQUESTS_QUEUE32_PASS' and receipt['owned_regression_service_stopped']
        state['status']='RESUMING_VERIFIED_UNFINISHED_CANONICAL_UNITS';save()
        jobs=[('original',7,8101,'models/GroundAnything-DLM','configs/eval/groundjev_original_dense_ref9.local.yaml',
            'groundjev_original_dense_ref9_queue32_v3_20261014','original_dense_ref9_v3_queue32',
            'groundjev_original_dense_ref9_direct_v2_20261014'),
            ('projector200',6,8102,'models/GroundAnything-DLM-TrackProj200','configs/eval/groundjev_trackproj200_dense_ref9.local.yaml',
            'groundjev_trackproj200_dense_ref9_queue32_v2_20261014','trackproj200_dense_ref9_v2_queue32',
            'groundjev_trackproj200_dense_ref9_20261014')]
        children=[]
        for name,gpu,port,model,recipe,run_id,folder,prior in jobs:
            child_out=ROOT/'outputs/groundjev/ground_retention'/folder
            log=(out/(name+'.log')).open('w')
            command=['python3','tools/groundjev/run_original_ground_dense_ref9.py','--out',str(child_out),
                '--run-id',run_id,'--gpu',str(gpu),'--port',str(port),'--model',str(ROOT/model),
                '--recipe',recipe,'--max-queued-requests','32','--resume-result-root',
                str(ROOT/'external/GroundAnything/outputs/groundjev_retention'/prior)]
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((name,child,log,child_out));state['children'][name]={'pid':child.pid,'out':str(child_out)};save()
        failures=[]
        for name,child,log,child_out in children:
            code=child.wait();log.close()
            final=json.loads((child_out/'owner_state.json').read_text())
            if code:failures.append({'unit':name,'exit_code':code,'state':final})
            else:state['completed'][name]=final
            save()
        assert not failures,failures
        state['status']='BOTH_NATIVE_CANONICAL_FULL9_RESUMPTIONS_COMPLETE';save()
    except Exception as error:
        state.update(status='FULL9_QUEUE_RECOVERY_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
