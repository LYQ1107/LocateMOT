"""One blocking owner for native-proposal joint-task before/after metrics."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/compact/joint1200_controls_v1')
    parser.add_argument('--training',type=Path,default=ROOT/'outputs/groundjev/compact/joint1200_v1')
    parser.add_argument('--reuse-before',type=Path)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    train=json.loads((args.training/'owner_state.json').read_text())
    assert train['status']=='SHARED_COMPACT_JOINT1200_TRAINING_COMPLETE'
    state={'status':'JOINT_TASK_NATIVE_PROPOSAL_CONTROLS_STARTED','physical_core_gpu':7,
        'native_ROI_service_gpu':5,'same_shared_core_and_memory_for_three_tasks':True,
        'scope':'registered short legal validation clips, not full datasets',
        'GT_or_parser_or_metric_changes':False,'completed':{}}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'status':state['status'],'completed':list(state['completed'])}),flush=True)
    def run(stage,command):
        state['status']=stage;save()
        with (out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        if code:raise RuntimeError(f'Joint control failed at{stage}; retain first root cause and completed units')
    scopes=[('mot4','mot','outputs/groundjev/pilot_data/mot17_first4.json',
             'outputs/groundjev/runs/mot17_first4_pretrained_v6_native_adaptive_resume/frame_decisions.jsonl'),
            ('rmot_single8','rmot','outputs/groundjev/multitask_pilot_data/rmot_single_empty.json',
             'outputs/groundjev/task_controls/registered_v1/rmot_single_empty_pretrained_v3/frame_decisions.jsonl'),
            ('rmot_multi12','rmot','outputs/groundjev/multitask_pilot_data/rmot_multi_empty.json',
             'outputs/groundjev/task_controls/registered_v1/rmot_multi_empty_pretrained_v3/frame_decisions.jsonl'),
            ('ovmot4','ovmot','outputs/groundjev/multitask_pilot_data/ovmot_tao_first4.json',
             'outputs/groundjev/multitask_candidate_controls/tao_exact_geometry_v1/frame_candidates.jsonl')]
    save()
    try:
        for name,task,video,candidates in scopes:
            if task=='mot':
                state['completed'][name+'_before']={'run':'outputs/groundjev/compact/native300_controls_v1/compact_distilled',
                    'metrics':json.loads((ROOT/'outputs/groundjev/compact/native300_controls_v1/compact_distilled/official_metrics.json').read_text())}
                variants=['after']
            elif args.reuse_before:
                previous=json.loads((args.reuse_before/'owner_state.json').read_text())
                assert previous['status']=='JOINT_THREE_TASK_NATIVE_PROPOSAL_CONTROLS_COMPLETE'
                state['completed'][name+'_before']=previous['completed'][name+'_before']
                variants=['after']
            else:variants=['before','after']
            for variant in variants:
                directory=out/(name+'_'+variant)
                checkpoint=ROOT/'outputs/groundjev/compact/native300_v1/compact_distilled.pt' if variant=='before' else args.training/'checkpoint.pt'
                source=Path(state['completed'][name+'_before']['run'])/'frame_decisions.jsonl' if variant=='after' and task!='mot' else ROOT/candidates
                command=['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',
                    str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/run_video.py','--task',task,
                    '--video',video,'--backend','compact','--checkpoint',str(checkpoint),
                    '--candidate-replay',str(source),'--out',str(directory)]
                if variant=='before' and task!='mot':command+=['--native-feature-url','http://127.0.0.1:8111']
                run(name+'_'+variant+'_INFERENCE',command)
                evaluator='tools/groundjev/evaluate_video.py' if task=='mot' else 'tools/groundjev/evaluate_multitask.py'
                run(name+'_'+variant+'_OFFICIAL_EVAL',['python3',evaluator,'--video',video,'--run',str(directory)])
                state['completed'][name+'_'+variant]={'run':str(directory),'metrics':json.loads((directory/'official_metrics.json').read_text())};save()
        state['status']='JOINT_THREE_TASK_NATIVE_PROPOSAL_CONTROLS_COMPLETE';save()
    except Exception as error:
        state.update(status='JOINT_TASK_CONTROL_FAILED',error=str(error));save();raise


if __name__=='__main__':main()
