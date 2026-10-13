"""One owner: trained Compact/controls, full same-candidate native ROI video."""
from collections import defaultdict
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/compact/dancetrack0004_native300_v1')
    parser.add_argument('--resume-first',type=Path)
    args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    state={'status':'STARTING_FULL_TRAINED_COMPACT_NATIVE_CONTROLS','physical_core_gpu':7,
        'native_feature_service_gpu':5,'registered_frames_per_model':1203,'scope':'one full legal validation video',
        'candidate_source':'original complete native Ground DLM V1 candidates; no new detector queries',
        'native_ROI_policy':'stream original frozen DLM ROI once, reuse identical frame features for controls',
        'temporary_feature_cache_budget_bytes':25*2**30,'end_to_end_FPS_claim':False,
        'timing':'feature extraction and association replay; detector excluded', 'completed':{}}
    peaks,device_peaks={},defaultdict(int);stop=threading.Event();errors=[]
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'status':state['status'],'completed':list(state['completed'])}),flush=True)
    def monitor():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
                current=defaultdict(int)
                for uuid,pid,amount in csv.reader(io.StringIO(raw)):
                    try:
                        if not (Path('/proc')/pid.strip()/'cwd').resolve().is_relative_to(ROOT):continue
                        value=int(amount)*2**20;key=uuid.strip()+':'+pid.strip();peaks[key]=max(peaks.get(key,0),value);current[uuid.strip()]+=value
                    except OSError:continue
                for uuid,value in current.items():device_peaks[uuid]=max(device_peaks[uuid],value)
            except Exception as error:errors.append(str(error));return
            stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def run(stage,command):
        state['status']=stage;save()
        with (out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        if code:raise RuntimeError(f'First failed full Compact stage retained: {stage}, exit{code}')
    save()
    try:
        candidates=ROOT/'outputs/groundjev/pilot_runs/dancetrack0004_pretrained_dense_v3_failure_inclusive/full/frame_decisions.jsonl'
        for index,name in enumerate(('compact_distilled','set_gt','mlp_distilled')):
            directory=out/name
            command=['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',
                str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/run_video.py','--task','mot',
                '--video','outputs/groundjev/pilot_data/dancetrack.json','--backend','compact','--checkpoint',
                str(ROOT/'outputs/groundjev/compact/native300_v1'/(name+'.pt')),
                '--candidate-replay',str(candidates),'--out',str(directory)]
            if index==0:command+=['--native-feature-url','http://127.0.0.1:8111']
            if index==0 and args.resume_first:command+=['--resume-from',str(args.resume_first)]
            run(name+'_INFERENCE',command)
            run(name+'_OFFICIAL_EVAL',['python3','tools/groundjev/evaluate_video.py',
                '--video','outputs/groundjev/pilot_data/dancetrack.json','--run',str(directory)])
            state['completed'][name]={'run':str(directory),'metrics':json.loads((directory/'official_metrics.json').read_text())};save()
            candidates=out/'compact_distilled/frame_decisions.jsonl'
            feature_bytes=sum(p.stat().st_size for p in out.rglob('*.jsonl'))
            feature_bytes+=sum(p.stat().st_size for p in out.rglob('*identity_memory.json'))
            assert feature_bytes<25*2**30
            state['observed_feature_and_memory_JSON_bytes']=feature_bytes;save()
        state['status']='FULL_TRAINED_COMPACT_MATCHED_CONTROLS_COMPLETE';save()
    except Exception as error:
        state.update(status='FULL_TRAINED_COMPACT_CONTROL_FAILED',error=str(error));save();raise
    finally:
        stop.set();thread.join(timeout=10)
        (out/'gpu_process_peaks.json').write_text(json.dumps({'sampling_interval_seconds':5,
            'process_peaks_bytes':peaks,'simultaneous_all_project_device_peaks_bytes':dict(device_peaks),
            'foreign_projects_excluded':True,'scope':'includes concurrent original OneJev500 and freshFullV3',
            'monitor_errors':errors},indent=2)+'\n')


if __name__=='__main__':main()
