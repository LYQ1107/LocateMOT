"""One blocking per-device owner for preregistered shared Full tracking controls."""
import argparse
from itertools import zip_longest
import json
import os
from pathlib import Path
import select
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.groundjev.run_video import ReplayFrames


def wait_dependency(dependency):
    pid,tick=dependency['identity'];proc=Path('/proc')/str(pid)
    try:
        fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if fields[0]!='Z' and fields[19]==tick:
            assert proc.joinpath('cwd').resolve()==ROOT
            assert dependency['script'] in proc.joinpath('cmdline').read_bytes().decode().split('\0')
            descriptor=os.pidfd_open(pid)
            try:
                events=select.poll();events.register(descriptor,select.POLLIN);events.poll()
            finally:
                os.close(descriptor)
    except (FileNotFoundError,ProcessLookupError):
        pass
    state=json.loads((ROOT/dependency['directory']/'owner_state.json').read_text())
    assert state['status']==dependency['required_terminal_status'],state['status']
    return state


def validate_run(run,job,checkpoint):
    manifest=json.loads((ROOT/job['manifest']).read_text())
    runtime=json.loads((run/'result.json').read_text())
    official=json.loads((run/'official_metrics.json').read_text())
    assert runtime['frames']==official['frames']==job['frames']
    assert runtime['all_input_frames_processed'] and runtime['backend']=='full'
    assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
    assert runtime['evidence_format']=='v3_crop_motion' and runtime['adaptive_context_groups']
    assert Path(runtime['checkpoint']).resolve()==checkpoint.resolve()
    assert runtime['perception_rejected_frames']==official['perception_rejected_frames']
    assert runtime['end_to_end_fps'] is None
    assert Path(runtime['shared_input_features_source']).resolve()==(ROOT/job['candidates']).resolve()
    if manifest['task']=='ovmot':
        assert runtime['category_policy']=='native_hierarchical'
        assert len(runtime['specification']['vocabulary'])==1203
    source=ReplayFrames(ROOT/job['candidates'])
    count,frames,rejected=0,[],[]
    try:
        with (run/'frame_decisions.jsonl').open() as trace:
            for expected,line in zip_longest(zip(manifest['frame_numbers'],manifest['images']),trace):
                assert expected is not None and line is not None
                number,image=expected;row=json.loads(line);original=source.frame(number)
                assert row['frame']==number and row['image_path']==image==original['image_path']
                if original['perception'].get('model_response_rejected'):
                    rejected.append(number)
                candidates=[{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in original['candidates']]
                assert row['candidates']==candidates
                if manifest['task']=='rmot':
                    assert row['state']['decision_metadata']['query_output_policy']=='native Noul membership>=.5; physical identity memory retained before output filtering'
                count+=len(candidates);frames.append(number)
    finally:
        source.close()
    assert len(frames)==job['frames'] and count==job['candidates_count']
    assert rejected==runtime['perception_rejected_frames']
    assert Path(runtime['final_identity_memory_path']).is_file()
    decoded=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=nb_read_frames','-of','default=noprint_wrappers=1:nokey=1',str(run/'tracks.mp4')],text=True))
    assert decoded==job['frames']
    return {'run':str(run),'checkpoint':str(checkpoint),'frames':decoded,
        'native_candidates':count,'same_exact_all_candidates':True,
        'parser_rejected_frame_count':len(runtime['perception_rejected_frames']),
        'official_metrics':official['metrics'],'base_novel':official.get('base_novel'),
        'GT_and_future_frames_not_read_by_runtime':True,'actual_end_to_end_FPS':None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--worker',choices=['old','refreshed'],required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    proof=json.loads((ROOT/plan['registration_validation']).read_text())
    assert proof['status']=='HISTORY_REFRESH_FOUR_SCOPE_SOURCES_AND_DEPENDENCIES_VALIDATED'
    gpu=plan['workers'][args.worker]['physical_gpu']
    assert gpu=={'old':4,'refreshed':5}[args.worker]
    root=ROOT/plan['output_root'];root.mkdir(parents=True,exist_ok=True)
    directory=root/(args.worker+'_worker');directory.mkdir(exist_ok=False)
    state={'status':'REGISTERED_WAITING_DEPENDENCIES','owner_pid':os.getpid(),
        'owner_starttick':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'worker':args.worker,'physical_gpu':gpu,'training_dependency':plan['training'],
        'GPU5_dependency':plan['gpu5_release'] if gpu==5 else None,
        'owner_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'completed':{},'new_model_calls_started':False,'new_DLM_or_feature_archive':False,
        'full_goal_complete':False,'ordinary_model_performance_achieved':False}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'external/OneJev'))
    def save():
        temp=directory/'owner_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(directory/'owner_state.json')
        print(json.dumps({'status':state['status'],'worker':args.worker,'physical_gpu':gpu,'completed':list(state['completed'])}),flush=True)
    def run(stage,command):
        state['status']=stage;save()
        with (directory/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['current_child_pid']=child.pid;save()
            for line in child.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        state.pop('current_child_pid',None)
        if code:
            raise RuntimeError(f'{stage} exited{code}; retain complete units and unfinished causal prefix')
    def measure(arm,scope):
        job=plan['scopes'][scope];manifest=json.loads((ROOT/job['manifest']).read_text())
        checkpoint=ROOT/plan['checkpoints'][arm];output=root/(arm+'_'+scope)
        command=['python3','tools/groundjev/run_with_resources.py','--gpus',str(gpu),'--',
            str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/run_video.py',
            '--task',manifest['task'],'--video',str(ROOT/job['manifest']),'--backend','full',
            '--checkpoint',str(checkpoint),'--evidence-format','v3_crop_motion','--adaptive-context-groups',
            '--candidate-replay',str(ROOT/job['candidates']),'--omit-replay-feature-vectors',
            '--memory-checkpoint-format','binary','--memory-export-interval','0','--out',str(output)]
        if manifest['task']=='rmot':
            command+=['--native-rmot-membership']
        if manifest['task']=='ovmot':
            command+=['--category-policy','native_hierarchical']
        name=arm+'_'+scope;state['new_model_calls_started']=True
        run(name+'_INFERENCE',command)
        scorer='evaluate_video.py' if manifest['task']=='mot' else 'evaluate_multitask.py'
        run(name+'_OFFICIAL',['python3','tools/groundjev/'+scorer,'--video',str(ROOT/job['manifest']),'--run',str(output)])
        run(name+'_EVENTS',['python3','tools/groundjev/analyze_tracking_events.py',
            '--video',str(ROOT/job['manifest']),'--run',str(output)])
        state['completed'][name]=validate_run(output,job,checkpoint);save()
    save()
    try:
        if gpu==5:
            state['status']='WAITING_ORIGINAL_GPU5_NATURAL_OWNER_EXIT';save()
            wait_dependency(plan['gpu5_release'])
            state['original_GPU5_queue_completed_before_any_model']=True
            measure('before','tao_raw_full40')
        state['status']='WAITING_PAIRED_TRAINING_AND_BOTH_NATIVE_MM_RELOADS';save()
        trained=wait_dependency(plan['training'])
        assert set(trained['completed'])=={'old','refreshed'}
        for name,entry in trained['completed'].items():
            assert entry['done']['steps']==600 and entry['done']['examples']==2400
            assert Path(entry['native_MM_reload']['checkpoint']).resolve()==(ROOT/plan['checkpoints'][name]).resolve()
        state['both_training_arms_and_native_MM_reloads_completed']=True
        if args.worker=='old':
            for scope,entry in proof['scopes'].items():
                if entry['before_baseline'] is not None:
                    state['completed']['before_'+scope]=dict(entry['before_baseline'],
                        reused_unchanged_completed_unit=True,frames=entry['frames'],native_candidates=entry['native_candidates'])
        for scope in plan['scopes']:
            measure(args.worker,scope)
        state['status']='HISTORY_REFRESH_WORKER_ALL_REGISTERED_UNITS_VALIDATED';save()
    except Exception as error:
        state.update(status='HISTORY_REFRESH_CONTROLS_WORKER_FAILED',error=repr(error));save()
        raise


if __name__=='__main__':
    main()
