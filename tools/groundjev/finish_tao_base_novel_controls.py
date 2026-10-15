"""Resume valid TAO native decisions and finish unchanged official controls on CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    original=ROOT/'outputs/groundjev/multitask_runs/tao_base_novel63_full_v2_expression_none'
    source=original/'native_candidates/frame_candidates.jsonl'
    receipt=json.loads((source.parent/'receipt.json').read_text())
    assert receipt['frames']==40 and receipt['native_query_requests']==520 and receipt['candidates']==35228
    manifest=ROOT/'outputs/groundjev/multitask_pilot_data/ovmot_tao_base_novel63_full.json'
    state={'status':'RESUMING_NATIVE_TAO_CONTROLS_CPU','owner_pid':os.getpid(),'completed':{},
        'native_perception_repeated':False,'every_native_box_retained':True,'CPU_core_not_GPU_or_E2E_speed':True,
        'native_source':str(source),'official_scorer_or_GT_changed':False}
    def save():(args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
    def run(stage,command):
        state['status']=stage;save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'external/OneJev'))
    save()
    try:
        for name,backend in [('simple','simple'),('shared_compact','compact')]:
            output=args.out/name
            command=[str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/run_video.py',
                '--task','ovmot','--video',str(manifest),'--backend',backend,'--candidate-replay',str(source),
                '--checkpoint','outputs/groundjev/compact/rmot_native72_stage_v1/broad_native_training/checkpoint.pt',
                '--omit-replay-feature-vectors','--out',str(output)]
            if name=='simple':command+=['--resume-from',str(original/'simple')]
            else:command+=['--compact-device','cpu']
            run(name+'_INFERENCE',command)
            run(name+'_OFFICIAL_TETA',['python3','tools/groundjev/evaluate_multitask.py','--video',str(manifest),'--run',str(output)])
            run(name+'_POSTHOC',['python3','tools/groundjev/analyze_tracking_events.py','--video',str(manifest),'--run',str(output)])
            m=json.loads(manifest.read_text());rows=[json.loads(x) for x in (output/'frame_decisions.jsonl').open()]
            assert [r['frame'] for r in rows]==m['frame_numbers'] and len(rows)==40
            with source.open() as raw:
                for old,new in zip(map(json.loads,raw),rows):
                    geometry=[{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in old['candidates']]
                    assert new['candidates']==geometry
            decoded=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=nb_read_frames','-of','default=noprint_wrappers=1:nokey=1',str(output/'tracks.mp4')],text=True))
            assert decoded==40
            state['completed'][name]={'official':json.loads((output/'official_metrics.json').read_text()),
                'decoded_frames':decoded,'all35228native_proposals_retained':True};save()
        state['status']='FULL40_BASE_NOVEL_NATIVE_SOURCE_CONTROLS_COMPLETE';save()
    except Exception as error:
        state.update(status='TAO_CPU_CONTROL_RECOVERY_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
