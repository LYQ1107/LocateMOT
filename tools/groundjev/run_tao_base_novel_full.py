"""One blocking owner for genuine Base/Novel full-video perception and shared-core controls."""
import argparse
import json
from pathlib import Path
import os
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.groundjev.service_process_group import stop_owned_service


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    state={'status':'WAITING_REGISTERED_GPU5_MEMORY_OWNER','physical_gpu':5,'manifest':'outputs/groundjev/multitask_pilot_data/ovmot_tao_base_novel63_full.json',
        'all40annotation_frames':True,'fixed1203vocabulary':True,'Novel_GT_for_training':False,'completed':{}}
    upstream=None
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]
            if len(argv)>1 and 'python' in Path(argv[0]).name and argv[1]=='tools/groundjev/run_memory_controls.py':
                assert (proc/'cwd').resolve()==ROOT
                upstream=(int(proc.name),(proc/'stat').read_text().rsplit(')',1)[1].split()[19])
        except (FileNotFoundError,PermissionError):pass
    state['verified_memory_owner']=upstream
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'status':state['status'],'completed':list(state['completed'])}),flush=True)
    def run(stage,command):
        state['status']=stage;save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    save();services=[];logs=[]
    try:
        if upstream:
            while True:
                try:
                    fields=(Path('/proc')/str(upstream[0])/'stat').read_text().rsplit(')',1)[1].split()
                    if fields[0]=='Z' or fields[19]!=upstream[1]:break
                except FileNotFoundError:break
                time.sleep(10)
        for port in (8104,8112):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind(('127.0.0.1',port))
        commands=[('ground',['tools/groundjev/serve_groundanything.py','--port','8104','--mem-fraction-static','.4',
                    '--out',str(args.out/'ground_service')]),
                  ('features',['tools/groundjev/serve_native_features.py','--port','8112',
                    '--receipt',str(args.out/'feature_service.json')])]
        for name,command in commands:
            log=(args.out/(name+'_server.log')).open('w');logs.append(log)
            child=subprocess.Popen(['python3','tools/groundjev/run_with_resources.py','--gpus','5','--',
                str(ROOT/'external/GroundAnything/.venv-serve/bin/python'),*command],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True);services.append(child)
        state['owned_service_groups']=[s.pid for s in services];state['status']='WAITING_OWNED_NATIVE_SERVICES';save()
        deadline=time.monotonic()+600
        while True:
            if any(s.poll() is not None for s in services):raise RuntimeError('Owned TAO native service failed')
            try:
                with urlopen('http://127.0.0.1:8104/v1/models',timeout=5) as response:
                    ground=any(x['id']=='groundinganything' for x in json.load(response)['data'])
                feature=(args.out/'feature_service.json').exists()
                if ground and feature:break
            except OSError:pass
            if time.monotonic()>deadline:raise RuntimeError('TAO native service readiness timeout')
            time.sleep(2)
        candidates=args.out/'native_candidates'
        run('NATIVE_ALL_CLASS_PERCEPTION',['.venv-onejev-py310/bin/python','tools/groundjev/stream_tao_native_candidates.py',
            '--manifest',state['manifest'],'--out',str(candidates)])
        for service in services:stop_owned_service(service)
        services.clear();state['native_services_stopped_before_core_controls']=True;save()
        for name,backend in [('simple','simple'),('shared_compact','compact')]:
            output=args.out/name
            command=['python3','tools/groundjev/run_with_resources.py','--gpus','5','--',
                str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/run_video.py','--task','ovmot',
                '--video',state['manifest'],'--backend',backend,'--candidate-replay',str(candidates/'frame_candidates.jsonl'),
                '--checkpoint','outputs/groundjev/compact/rmot_native72_stage_v1/broad_native_training/checkpoint.pt',
                '--omit-replay-feature-vectors','--out',str(output)]
            run(name+'_INFERENCE',command)
            run(name+'_OFFICIAL_TETA',['python3','tools/groundjev/evaluate_multitask.py','--video',state['manifest'],'--run',str(output)])
            state['completed'][name]=json.loads((output/'official_metrics.json').read_text());save()
        state['status']='FULL_BASE_NOVEL_VIDEO_NATIVE_AND_SHARED_CORE_CONTROLS_COMPLETE';save()
    except Exception as error:
        state.update(status='FULL_BASE_NOVEL_VIDEO_FAILED',error=repr(error));save();raise
    finally:
        for service in services:stop_owned_service(service)
        for log in logs:log.close()
        state['owned_native_service_groups_stopped']=True;save()


if __name__=='__main__':main()
