"""One blocking owner for matched native physical/semantic SFT and fixed controls."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.run_history_refresh_controls import validate_run,wait_dependency
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and plan['physical_gpu']==4
    assert plan['steps_each']==6000 and plan['questions_each']==24000
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    inputs=ROOT/plan['input_output'];proof=json.loads((inputs/'completion_validation.json').read_text())
    assert proof['status']=='BOTH_COMPLETE24000_NATIVE_MM_COURSES_VALIDATED_ZERO_DROPS_NOT_TRAINED'
    for report in proof['CPU_plans'].values():
        assert report['rows']==24000 and report['dropped_rows']==report['actual_optimizer_steps']==0
        assert report['planned_optimizer_steps']==6000
    snapshot=json.loads((ROOT/plan['initial_state_validation']).read_text())
    assert snapshot['step']==6000 and snapshot['all320_FP32_parameters_and_both_full_Adam_moments_finite']
    assert Path(snapshot['saved_full_state']['path']).resolve()==(ROOT/plan['initial_training_state']).resolve()
    out=ROOT/plan['owner_output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='REGISTERED_PAIRED_RMOT_PHYSICAL_SEMANTIC_NATIVE_STUDY',
        owner_pid=os.getpid(),owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        owner_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_gpu=4,completed={},actual_completed_stage_optimizer_steps=0,
        original_four_GPU_and_temporal_courses_unchanged=True,full_goal_complete=False)
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(ROOT/'external/OneJev'),PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    gpu=['python3','tools/groundjev/run_with_resources.py','--gpus','4','--',python]
    def save():
        temp=out/'owner_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']))),flush=True)
    def run(stage,command):
        state.update(status=stage,command=command);save()
        with (out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        state.pop('child_pid',None);state['last_exit_code']=code
        assert code==0,f'{stage} exited{code}; keep the first failure, completed units and unfinished native state'
    def wait_memory():
        state['status']='WAITING_REGISTERED_GPU4_STAGE_MEMORY';save()
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).strip())
            if free>=plan['minimum_launch_free_MiB']:return free
            time.sleep(20)
    save()
    try:
        release=wait_dependency(plan['GPU4_release_dependency'])
        state['completed']['old_GPU4_owner_success']=dict(status=release['status'],owner_exited=True)
        baseline_source=json.loads((ROOT/plan['baseline_dependency']['directory']/'owner_state.json').read_text())
        assert baseline_source['completed']['training']['done']['steps']==6000
        assert baseline_source['completed']['native_MM_reload']['status']=='REAL_ONEJEV6000_NATIVE_MM_RELOAD_VALIDATED'
        checkpoints={}
        for arm in ('physical','semantic'):
            assert shutil.disk_usage(plan['runs_dir']).free>=plan['minimum_checkpoint_free_GiB']*2**30
            state['launch_free_MiB']=wait_memory()
            config=json.loads((inputs/arm/'CPU_config.json').read_text())
            config.update(device='cuda:0',max_steps=6000,name=plan['training_names'][arm],
                runs_dir=plan['runs_dir'],save_every=1000)
            assert config['model']==str(ROOT/plan['initial_checkpoint'])
            training=Path(plan['runs_dir'])/config['name'];assert not training.exists()
            config_path=out/(arm+'_training_config.json');config_path.write_text(json.dumps(config,indent=2)+'\n')
            run(arm+'_SHARED_NATIVE6000_TRAINING',gpu+['tools/groundjev/run_memory_bounded_native_sft.py',
                '--physical-gpu','4','--memory-fraction',str(plan['torch_allocator_fraction']),'--',
                'tools/groundjev/run_source_manifest_sft.py','--source-training-backend','fla',
                '--config',str(config_path),'--paired-index-shuffle','--initial-training-state',
                str(ROOT/plan['initial_training_state']),'--expected-initial-step','6000',
                '--deterministic-execution','--seed-global-rngs'])
            records=[json.loads(line) for line in (training/'log.jsonl').open()]
            done=next(row for row in records if row.get('event')=='done')
            assert done['steps']==6000 and done['examples']==24000
            assert all(math.isfinite(r[k]) for r in records if 'loss' in r for k in ('loss','grad_norm'))
            initial=json.loads((training/'initial_fp32_state_receipt.json').read_text())
            assert initial['source_step']==6000 and initial['initialization_max_difference']==0
            state['actual_completed_stage_optimizer_steps']+=6000
            state['completed'][arm+'_training']=dict(done=done,initial_FP32=initial);save()
            run(arm+'_FULL_STATE_NUMERIC_GATE',[python,'tools/groundjev/freeze_shared_training_checkpoint.py',
                '--step','6000','--training',str(training),'--initial-step','6000','--out',str(out/(arm+'_full_state')),
                '--registered-feature-upper',str(plan['combined_registered_feature_upper_bytes'])])
            checkpoint=training/'ckpt-step6000';checkpoints[arm]=checkpoint
            run(arm+'_NATIVE_MM_RELOAD',gpu+['tools/groundjev/check_onejev125_checkpoint.py',
                '--step','6000','--checkpoint',str(checkpoint),'--training-root',str(training),
                '--reference-checkpoint',str(ROOT/plan['initial_checkpoint'])])
            state['completed'][arm+'_native_MM_reload']=json.loads((training/'checkpoint6000_validation.json').read_text());save()
        state['status']='BOTH_NATIVE_TRAININGS_COMPLETE_CPU_WAIT_ORIGINAL_ASSIGNED_BASELINES';save()
        baseline=wait_dependency(plan['baseline_dependency'])
        controls=json.loads((ROOT/'configs/groundjev/full_history_refresh600_controls_v1.json').read_text())
        results=out/'controls';results.mkdir()
        for scope,job in controls['scopes'].items():
            source=baseline['completed']['assigned_identity_'+scope];source_run=Path(source['run'])
            assert json.loads((source_run/'result.json').read_text())['assigned_identity_updates']
            state['completed']['before_'+scope]=validate_run(source_run,job,ROOT/plan['initial_checkpoint']);save()
            manifest=json.loads((ROOT/job['manifest']).read_text())
            for arm in ('physical','semantic'):
                state['launch_free_MiB']=wait_memory();destination=results/(arm+'_'+scope)
                command=gpu+['tools/groundjev/run_video.py','--task',manifest['task'],'--video',str(ROOT/job['manifest']),
                    '--backend','full','--checkpoint',str(checkpoints[arm]),'--evidence-format','v3_crop_motion',
                    '--adaptive-context-groups','--assigned-identity-updates','--candidate-replay',str(ROOT/job['candidates']),
                    '--omit-replay-feature-vectors','--memory-checkpoint-format','binary','--memory-export-interval','0',
                    '--out',str(destination)]
                if manifest['task']=='rmot':command+=['--native-rmot-membership']
                if manifest['task']=='ovmot':command+=['--category-policy','native_hierarchical']
                run(arm+'_'+scope+'_INFERENCE',command)
                scorer='evaluate_video.py' if manifest['task']=='mot' else 'evaluate_multitask.py'
                run(arm+'_'+scope+'_OFFICIAL',['python3','tools/groundjev/'+scorer,
                    '--video',str(ROOT/job['manifest']),'--run',str(destination)])
                run(arm+'_'+scope+'_EVENTS',['python3','tools/groundjev/analyze_tracking_events.py',
                    '--video',str(ROOT/job['manifest']),'--run',str(destination)])
                runtime=json.loads((destination/'result.json').read_text());assert runtime['assigned_identity_updates']
                state['completed'][arm+'_'+scope]=validate_run(destination,job,checkpoints[arm]);save()
        state['status']='PAIRED_RMOT_NATIVE6000_PHYSICAL_SEMANTIC_AND_ALL_FIXED_CONTROLS_VALIDATED';save()
    except Exception as error:
        state.update(status='PAIRED_RMOT_NATIVE_OWNER_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
