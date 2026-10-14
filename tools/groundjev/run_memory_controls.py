"""One blocking owner for fixed-model full-video memory policy interventions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from itertools import zip_longest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());args.out.mkdir(parents=True,exist_ok=False)
    state={'status':'STARTING_FIXED_NATIVE_FULL_VIDEO_MEMORY_CONTROLS','physical_gpu':plan['physical_gpu'],
        'checkpoint':plan['checkpoint'],'candidate_feature_source':plan['candidates'],
        'GT_or_threshold_or_metric_or_model_weights_changed':False,'completed':{},'end_to_end_FPS_claim':False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'status':state['status'],'completed':list(state['completed'])}),flush=True)
    def run(name,command):
        state['status']=name;save()
        with (args.out/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        for variant in plan['variants']:
            name=variant['name'];directory=args.out/name
            command=['python3','tools/groundjev/run_with_resources.py','--gpus',str(plan['physical_gpu']),'--',
                str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/run_video.py','--task','mot',
                '--video',plan['manifest'],'--backend','compact','--checkpoint',plan['checkpoint'],
                '--candidate-replay',plan['candidates'],'--memory-policy',variant['memory_policy'],
                '--memory-feature-view',variant['feature_view'],'--lost-seconds',str(variant['lost_seconds']),
                '--ema-alpha','.1','--omit-replay-feature-vectors','--out',str(directory)]
            run(name+'_INFERENCE',command)
            run(name+'_OFFICIAL_EVAL',['python3','tools/groundjev/evaluate_video.py','--video',plan['manifest'],'--run',str(directory)])
            run(name+'_POSTHOC_EVENTS',['python3','tools/groundjev/analyze_tracking_events.py','--video',plan['manifest'],'--run',str(directory)])
            manifest=json.loads((ROOT/plan['manifest']).read_text())
            count=0;rejected=[]
            with (ROOT/plan['candidates']).open() as a,(directory/'frame_decisions.jsonl').open() as b:
                for old,new in zip_longest(a,b):
                    assert old is not None and new is not None
                    old,new=json.loads(old),json.loads(new)
                    expected=[{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in old['candidates']]
                    assert new['frame']==old['frame'] and new['image_path']==old['image_path'] and new['candidates']==expected
                    assert new['state']['decision_metadata']['GT_read'] is False
                    if new['perception'].get('model_response_rejected'):rejected.append(new['frame'])
                    count+=1
            assert count==len(manifest['frame_numbers'])==1203
            decoded=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=nb_read_frames','-of','default=noprint_wrappers=1:nokey=1',str(directory/'tracks.mp4')],text=True).strip())
            assert decoded==1203 and len(rejected)==27
            diagnostic=json.loads((directory/'association_events.json').read_text());diagnostic.pop('events')
            comparison={}
            if name=='reference_gated3':
                old=ROOT/plan['prior_reference']
                different=[];maximum=0.
                with (old/'frame_decisions.jsonl').open() as a,(directory/'frame_decisions.jsonl').open() as b:
                    for left,right in zip_longest(a,b):
                        assert left is not None and right is not None
                        left,right=json.loads(left),json.loads(right)
                        geometry=lambda outputs:[{k:v for k,v in o.items() if k!='association_probability'} for o in outputs]
                        if geometry(left['outputs'])!=geometry(right['outputs']):different.append(right['frame'])
                        elif left['outputs']:
                            maximum=max(maximum,max(abs(x['association_probability']-y['association_probability']) for x,y in zip(left['outputs'],right['outputs'])))
                comparison={'prior_reference_changed_prediction_frames':different,'maximum_native_probability_difference_on_same_predictions':maximum,
                    'comparison_scope':'prior GPU7/resumed run; current GPU5 fresh baseline is common to every intervention'}
            result=json.loads((directory/'result.json').read_text())
            state['completed'][name]={'variant':variant,'run':str(directory),
                'metrics':json.loads((directory/'official_metrics.json').read_text())['metrics'],
                'posthoc':diagnostic,'fixed_candidate_and_feature_source_rows':count,'rejected_frames':rejected,
                'decoded_video_frames':decoded,'peak_decision_gpu_allocated_bytes':result['peak_decision_gpu_memory_bytes'],
                'association_replay_fps':result['association_replay_fps'],'reference_retention':comparison}
            save()
        state['status']='FIXED_NATIVE_FULL_VIDEO_MEMORY_CONTROLS_COMPLETE';save()
        (args.out/'aggregate_results.json').write_text(json.dumps(state,indent=2)+'\n')
    except Exception as error:
        state.update(status='MEMORY_CONTROL_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
