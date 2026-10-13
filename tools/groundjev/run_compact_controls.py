"""Single blocking owner for trained compact exact-native-candidate controls."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'outputs/groundjev/compact/native300_controls_v1';out.mkdir(parents=True,exist_ok=False)
    state={'status':'STARTING_TRAINED_COMPACT_CONTROLS','physical_gpu':7,'frames_per_model':4,
        'scope':'legal MOT17 short control, not full video/dataset','native_feature_replay':True,
        'GT_read_by_runtime':False,'end_to_end_FPS_claim':False,'completed':{}}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    def run(stage,command):
        state['status']=stage;save()
        with (out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        if code:raise RuntimeError(f'First failed Compact stage retained: {stage}, exit{code}')
    save()
    try:
        for name in ('compact_distilled','set_gt','mlp_distilled'):
            checkpoint=ROOT/'outputs/groundjev/compact/native300_v1'/(name+'.pt');directory=out/name
            run(name+'_INFERENCE',['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',
                str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/run_video.py','--task','mot',
                '--video','outputs/groundjev/pilot_data/mot17_first4.json','--backend','compact',
                '--checkpoint',str(checkpoint),'--candidate-replay',
                'outputs/groundjev/runs/mot17_first4_pretrained_v6_native_adaptive_resume/frame_decisions.jsonl',
                '--out',str(directory)])
            run(name+'_OFFICIAL_EVAL',['python3','tools/groundjev/evaluate_video.py',
                '--video','outputs/groundjev/pilot_data/mot17_first4.json','--run',str(directory)])
            state['completed'][name]={'run':str(directory),'metrics':json.loads((directory/'official_metrics.json').read_text())};save()
        state['status']='TRAINED_COMPACT_MATCHED_CONTROLS_COMPLETE';save()
    except Exception as error:
        state.update(status='TRAINED_COMPACT_CONTROL_FAILED',error=str(error));save();raise


if __name__=='__main__':main()
