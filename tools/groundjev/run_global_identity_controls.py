"""One CPU owner: fixed-final A1/A2 development video, then all five scopes."""
import argparse
import gzip
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def validate(run,job,source,checkpoint):
    from tools.groundjev.run_video import ReplayFrames
    runtime=json.loads((run/'result.json').read_text());official=json.loads((run/'official_metrics.json').read_text())
    manifest=json.loads((source/job['manifest']).read_text())
    assert runtime['frames']==official['frames']==job['frames']
    assert runtime['all_input_frames_processed'] and not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
    assert runtime['global_identity'] and Path(runtime['checkpoint']).resolve()==checkpoint.resolve()
    assert runtime['decision_device']=='cpu' and runtime['peak_decision_gpu_memory_bytes']==0
    assert runtime['end_to_end_fps'] is None
    replay=ReplayFrames(source/job['candidates']);count=0;frames=0
    try:
        with (run/'frame_decisions.jsonl').open() as f:
            for expected,line in zip_longest(zip(manifest['frame_numbers'],manifest['images']),f):
                assert expected is not None and line is not None
                number,image=expected;row=json.loads(line);original=replay.frame(number)
                assert row['frame']==number and row['image_path']==image==original['image_path']
                candidates=[{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in original['candidates']]
                assert row['candidates']==candidates
                assert row['perception'].get('model_response_rejected',False)==original['perception'].get('model_response_rejected',False)
                events=[e for e in row['state']['events'] if 'candidate' in e]
                assert sorted(e['candidate'] for e in events)==list(range(len(candidates)))
                ids=[e['track_id'] for e in events if 'track_id' in e];assert len(set(ids))==len(ids)
                count+=len(candidates);frames+=1
    finally:replay.close()
    assert frames==job['frames'] and count==job['candidates_count']
    decoded=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=nb_read_frames','-of','default=noprint_wrappers=1:nokey=1',str(run/'tracks.mp4')],text=True))
    assert decoded==frames and Path(runtime['final_identity_memory_path']).is_file()
    return dict(run=str(run),checkpoint=str(checkpoint),frames=frames,native_candidates=count,
        all_original_candidates_frames_and_rejected_frames_exact=True,unique_current_identity_allocations=True,
        official_metrics=official['metrics'],base_novel=official.get('base_novel'),
        actual_end_to_end_FPS=None,association_replay_fps=runtime['association_replay_fps'],
        peak_decision_gpu_memory_bytes=0,peak_RSS_bytes=runtime['peak_decision_process_rss_bytes'],
        physical_parameters=runtime['trainable_compact_core_parameters'],frozen_semantic_parameters=runtime['frozen_semantic_parameters'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=Path(plan['immutable_source_root']);out=ROOT/plan['output'];owner=out/'R2_owner';owner.mkdir(exist_ok=False)
    registered=json.loads((source/plan['evaluation_plan']).read_text())
    state=dict(status='GLOBAL_IDENTITY_R2_FIXED_CONTROLS_RUNNING',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        completed={},new_GPU_processes=0,full_goal_complete=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(ROOT)+':/data2/user/LocateMOT/third_party/TrackEval')
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    def save():
        path=owner/'owner_state.tmp';path.write_text(json.dumps(state,indent=2)+'\n');path.replace(owner/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']))),flush=True)
    def run(stage,command):
        state['status']=stage;save()
        with gzip.open(owner/(stage+'.log.gz'),'wt') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:log.write(line)
            code=child.wait()
        state.pop('child_pid',None)
        assert code==0,f'{stage} exited{code}; complete compressed log and unfinished source prefix retained'
    save()
    try:
        for scope,job in registered['scopes'].items():
            manifest=json.loads((source/job['manifest']).read_text())
            for arm in ('A1','A2'):
                checkpoint=Path(plan['model_output'])/arm/'checkpoint.pt';assert checkpoint.is_file()
                destination=out/'controls'/(arm+'_'+scope)
                if not (destination/'result.json').exists():
                    assert not destination.exists(),f'Unfinished unit {destination}; resume its retained causal prefix explicitly'
                    run(arm+'_'+scope+'_INFERENCE',[python,'tools/groundjev/run_video.py','--task',manifest['task'],
                        '--video',str(source/job['manifest']),'--backend','compact','--compact-device','cpu',
                        '--global-identity','--checkpoint',str(checkpoint),'--candidate-replay',str(source/job['candidates']),
                        '--omit-replay-feature-vectors','--memory-checkpoint-format','binary','--memory-export-interval','0','--out',str(destination)])
                if not (destination/'official_metrics.json').exists():
                    scorer='evaluate_video.py' if manifest['task']=='mot' else 'evaluate_multitask.py'
                    run(arm+'_'+scope+'_OFFICIAL',[python,'tools/groundjev/'+scorer,'--video',str(source/job['manifest']),'--run',str(destination)])
                run(arm+'_'+scope+'_EVENTS',[python,'tools/groundjev/analyze_tracking_events.py',
                    '--video',str(source/job['manifest']),'--run',str(destination)])
                state['completed'][arm+'_'+scope]=validate(destination,job,source,checkpoint);save()
        state['status']='GLOBAL_IDENTITY_R2_A1_A2_ALL_FIVE_REGISTERED_SCOPES_COMPLETE';save()
        (out/'R2_all_five_scopes.json').write_text(json.dumps(state,indent=2)+'\n')
    except Exception as error:
        state.update(status='GLOBAL_IDENTITY_R2_OWNER_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
