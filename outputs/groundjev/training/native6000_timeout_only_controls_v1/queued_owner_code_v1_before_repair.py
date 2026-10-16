"""One queued native6000 lifecycle control owner, after original controls finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.run_history_refresh_controls import wait_dependency,validate_run
    from tools.groundjev.gpu7_training_window import acquire_training_window,FD_ENV
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert plan['physical_gpu']==7 and plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    proof=json.loads((ROOT/plan['termination_regression']).read_text())
    assert proof['status']=='REAL_NATIVE_AND_COMPACT_DEFAULT_PREFIX_EARLY_DEFERRAL_AND_EXPIRY_VALIDATED'
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='WAITING_ORIGINAL_NATIVE6000_ALL_CONTROLS_SUCCESS',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        completed={},physical_gpu=7,new_optimizer_updates=0,full_goal_complete=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    python=str(ROOT/'.venv-onejev-py310/bin/python');gpu=['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',python]
    window=None
    def save():
        temp=out/'owner_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']))),flush=True)
    def run(stage,command):
        state.update(status=stage,command=command);save()
        with (out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,pass_fds=() if window is None else (window.fileno(),),
                                   stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save();code=child.wait()
        state.pop('child_pid',None);state['last_exit_code']=code
        assert code==0,f'{stage} exited{code}; retain failure and unfinished causal prefix'
    save()
    try:
        original=wait_dependency(plan['dependency'])
        controls=json.loads((ROOT/plan['original_control_plan']).read_text())
        checkpoint=ROOT/plan['checkpoint']
        for scope,job in controls['scopes'].items():
            baseline=original['completed']['default_'+scope]
            assert Path(baseline['checkpoint']).resolve()==checkpoint.resolve()
            assert baseline['frames']==job['frames'] and baseline['same_exact_all_candidates']
            state['completed']['before_'+scope]=baseline
        state.update(status='WAITING_EXCLUSIVE_GPU7_WINDOW_AFTER_ORIGINAL_CONTROLS');save()
        window=acquire_training_window();env[FD_ENV]=str(window.fileno())
        for scope,job in controls['scopes'].items():
            state.update(status='WAITING_GPU7_MEMORY_FOR_'+scope);save()
            while True:
                free=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=plan['minimum_free_MiB']:break
                time.sleep(20)
            state['launch_free_MiB']=free
            manifest=json.loads((ROOT/job['manifest']).read_text());destination=out/('timeout_only_'+scope)
            command=gpu+['tools/groundjev/run_video.py','--task',manifest['task'],'--video',str(ROOT/job['manifest']),
                '--backend','full','--checkpoint',str(checkpoint),'--evidence-format','v3_crop_motion','--adaptive-context-groups',
                '--candidate-replay',str(ROOT/job['candidates']),'--omit-replay-feature-vectors',
                '--memory-checkpoint-format','binary','--memory-export-interval','0','--lifecycle-termination-policy','timeout_only',
                '--out',str(destination)]
            if manifest['task']=='rmot':command+=['--native-rmot-membership']
            if manifest['task']=='ovmot':
                command+=['--category-policy','native_hierarchical','--category-replay',state['completed']['before_'+scope]['run']]
            run(scope+'_TIMEOUT_ONLY_INFERENCE',command)
            scorer='evaluate_video.py' if manifest['task']=='mot' else 'evaluate_multitask.py'
            run(scope+'_OFFICIAL',['python3','tools/groundjev/'+scorer,'--video',str(ROOT/job['manifest']),'--run',str(destination)])
            run(scope+'_EVENTS',['python3','tools/groundjev/analyze_tracking_events.py','--video',str(ROOT/job['manifest']),'--run',str(destination)])
            validated=validate_run(destination,job,checkpoint)
            runtime=json.loads((destination/'result.json').read_text())
            assert runtime['lifecycle_termination_policy']=='timeout_only' and runtime['lost_seconds']==3.
            if manifest['task']=='ovmot':
                categories=runtime['native_category_replay']
                assert categories['frames_reused']==job['frames'] and categories['candidates_reused']==job['candidates_count']
                assert categories['actual_native_category_calls']==0 and categories['all_recorded_root_and_conditional_leaf_probabilities_retained']
            validated['lifecycle_termination_policy']='timeout_only';state['completed']['timeout_only_'+scope]=validated;save()
        assert len(state['completed'])==8
        state.update(status='NATIVE6000_TIMEOUT_ONLY_ALL_FOUR_FIXED_SCOPES_VALIDATED');save()
        (out/'all_four_completed.json').write_text(json.dumps(state,indent=2)+'\n')
    except Exception as error:
        state.update(status='NATIVE_TIMEOUT_ONLY_OWNER_FAILED',error=repr(error));save();raise
    finally:
        if window is not None:window.close()


if __name__=='__main__':main()
