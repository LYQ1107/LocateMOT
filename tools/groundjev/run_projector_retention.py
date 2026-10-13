"""Single blocking owner: adapted native DLM replica and same-protocol dense/referring controls."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Preserve existing retention/control results')
    args.out.mkdir(parents=True)
    state={'status':'STARTING_ADAPTED_NATIVE_DLM_REPLICA','scope':'legal4frame dense and8/12frame referring functional controls',
        'adapted_checkpoint':str(ROOT/'models/GroundAnything-DLM-TrackProj3'),
        'adapted_physical_gpu':7,'original_server_gpu':6,'same_parser':'official_scoring',
        'same_decoder':'tracking_dense','requested_seed':42,'associator':'unchanged geometry Simple',
        'original_server_restarted':False,'original_model_overwritten':False,'formal_original_dense_ref_benchmark_claim':False,
        'formal_adaptation_complete':False,'completed':{}}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'status':state['status'],'completed':list(state['completed'])}),flush=True)
    save()
    def run(stage,command):
        state['status']=stage;save()
        with (args.out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            if child.wait():raise RuntimeError(f'Preserve first failed retention stage: {stage}')
    with (args.out/'adapted_server.log').open('w') as server_log:
        server=subprocess.Popen(['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',
            str(ROOT/'external/GroundAnything/.venv-serve/bin/python'),'tools/groundjev/serve_groundanything.py',
            '--model',state['adapted_checkpoint'],'--port','8102','--mem-fraction-static','.4',
            '--out',str(args.out/'adapted_service')],cwd=ROOT,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
        state['owned_replica_process_group']=server.pid;save()
        try:
            deadline=time.monotonic()+600
            while True:
                if server.poll() is not None:raise RuntimeError('Adapted native service failed before readiness; inspect server log')
                try:
                    with urlopen('http://127.0.0.1:8102/v1/models',timeout=5) as response:
                        data=json.load(response)
                    if any(m['id']=='groundinganything' for m in data['data']):break
                except (OSError,ValueError):pass
                if time.monotonic()>deadline:raise RuntimeError('Adapted service readiness deadline exceeded; owner remains accountable for cleanup')
                time.sleep(2)
            state['status']='ADAPTED_REPLICA_GPU_READY';save()
            scopes=[('dense_mot4','mot',ROOT/'outputs/groundjev/pilot_data/mot17_first4.json'),
                ('refer_single8','rmot',ROOT/'outputs/groundjev/multitask_pilot_data/rmot_single_empty.json'),
                ('refer_multi12','rmot',ROOT/'outputs/groundjev/multitask_pilot_data/rmot_multi_empty.json')]
            for name,task,video in scopes:
                for variant,port in [('original',8101),('projector3',8102)]:
                    out=args.out/name/variant
                    run(name+'_'+variant+'_INFERENCE',[str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/run_video.py',
                        '--task',task,'--video',str(video),'--backend','simple','--grounding-url',f'http://127.0.0.1:{port}/v1',
                        '--grounding-parser','official_scoring','--grounding-decoder-policy','tracking_dense',
                        '--perception-failure-policy','empty_and_record','--out',str(out)])
                    entry='tools/groundjev/evaluate_video.py' if task=='mot' else 'tools/groundjev/evaluate_multitask.py'
                    run(name+'_'+variant+'_OFFICIAL_EVAL',['python3',entry,'--video',str(video),'--run',str(out)])
                    state['completed'][name+'_'+variant]={'run':str(out),'metrics':json.loads((out/'official_metrics.json').read_text())};save()
            state['status']='NATIVE_PROJECTOR_FUNCTIONAL_DENSE_REF_CONTROLS_COMPLETE';save()
        except Exception as error:
            state.update(status='NATIVE_PROJECTOR_RETENTION_CONTROL_FAILED',error=str(error));save();raise
        finally:
            # This owner created an isolated process group. Stop only its
            # adapted replica and workers; original service/UDMT untouched.
            try:os.killpg(server.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL);server.wait()
            state['owned_adapted_replica_stopped']=True;save()


if __name__=='__main__':main()
