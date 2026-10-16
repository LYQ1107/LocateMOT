"""One blocking owner: paired native CPU plans, queued shared SFT and controls."""
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
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    from tools.groundjev.run_history_refresh_controls import wait_dependency,validate_run
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and plan['physical_gpu']==7
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    out=ROOT/plan['owner_output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='REGISTERED_PAIRED_TEMPORAL_TEACHER_FOUNDATION_STUDY',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        completed={},physical_gpu_only_after_verified_dependency=7,actual_production_optimizer_steps=0,full_goal_complete=False)
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTHONPATH=str(ROOT/'external/OneJev'),PYTHONUNBUFFERED='1')
    gpu=['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',python]
    def save():
        temp=out/'owner_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']),actual_production_optimizer_steps=state['actual_production_optimizer_steps'])),flush=True)
    def run(stage,command):
        state.update(status=stage,command=command);save()
        with (out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        state.pop('child_pid',None);state['last_exit_code']=code
        assert code==0,f'{stage} exited{code}; preserve first failure and all completed units'
    def wait_memory():
        state.update(status='WAITING_GPU7_MEMORY_FOR_REGISTERED_NEXT_STAGE');save()
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=plan['minimum_launch_free_MiB']:return free
            time.sleep(20)
    save()
    try:
        run('INDEXED_SOURCE_ROW_REGRESSION',[python,'tools/groundjev/validate_indexed_native_sources.py',
            '--out',str(out/'indexed_input_validation')])
        run('PREPARE_PAIRED_LEGAL_TEMPORAL_TEACHERS',[python,'tools/groundjev/prepare_paired_temporal_teacher.py','--plan',str(args.plan)])
        data=ROOT/plan['output'];summary=json.loads((data/'summary.json').read_text())
        assert summary['paired_all_current_images_boxes_aliases_questions_targets_exact'] and summary['source_units']==43
        state['completed']['curriculum']=summary;save()
        for arm in ('short','long'):
            config=json.loads((data/arm/'training_config.json').read_text())
            cpu=dict(config,name='paired_temporal_'+arm+'_CPU_plan',runs_dir=str(out/(arm+'_CPU_plan')),device='cpu')
            path=out/(arm+'_CPU_plan_config.json');path.write_text(json.dumps(cpu,indent=2)+'\n')
            report=out/(arm+'_CPU_plan_validation.json')
            run(arm+'_ALL24000_NATIVE_CPU_INPUTS',[python,'tools/groundjev/run_broader_shared_onejev.py',
                '--cpu-plan',str(path),'--report',str(report)])
            state['completed'][arm+'_CPU_plan']=json.loads(report.read_text());save()
        proof=dict(status='PAIRED_SHORT_LONG_NATIVE24000_INPUTS_AND6000_UPDATES_EACH_VALIDATED_NOT_TRAINED',
            curriculum=summary,cpu_plans={a:state['completed'][a+'_CPU_plan'] for a in ('short','long')},
            new_optimizer_updates=0,dependency=plan['training_dependency'],full_goal_complete=False)
        (out/'prepared_and_CPU_validated.json').write_text(json.dumps(proof,indent=2)+'\n')
        state.update(status='WAITING_ORIGINAL_NATIVE6000_OWNER_SUCCESSFUL_EXIT',dependency=plan['training_dependency']);save()
        original=wait_dependency(plan['training_dependency'])
        state['completed']['original6000_owner_success']=dict(status=original['status'],owner_exited=True);save()
        assert shutil.disk_usage(plan['runs_dir']).free>=plan['minimum_checkpoint_free_GiB']*2**30
        configurations={};checkpoints={}
        for arm in ('short','long'):
            free=wait_memory();config=json.loads((data/arm/'training_config.json').read_text());configurations[arm]=config
            training=Path(config['runs_dir'])/config['name'];assert not training.exists()
            state['launch_free_MiB']=free;save()
            run(arm+'_SHARED_ONEJEV6000_TRAINING',gpu+['tools/groundjev/run_source_manifest_sft.py','--source-training-backend','fla',
                '--config',str(data/arm/'training_config.json'),'--paired-index-shuffle',
                '--initial-training-state',str(ROOT/plan['initial_training_state']),'--expected-initial-step',str(plan['initial_expected_step']),
                '--deterministic-execution','--seed-global-rngs'])
            records=[json.loads(line) for line in (training/'log.jsonl').open()]
            done=next(r for r in records if r.get('event')=='done')
            assert done['steps']==6000 and done['examples']==24000
            assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in records if 'loss' in r)
            latest=json.loads((training/'latest_training_state.json').read_text())
            assert latest['step']==6000 and latest['seen']==24000
            state['actual_production_optimizer_steps']+=6000
            state['completed'][arm+'_training']=dict(done=done,latest_full_state=latest);save()
            checkpoint=training/'ckpt-step6000';checkpoints[arm]=checkpoint
            run(arm+'_NATIVE_MM_RELOAD',gpu+['tools/groundjev/check_onejev125_checkpoint.py','--step','6000',
                '--checkpoint',str(checkpoint),'--training-root',str(training),'--reference-checkpoint',config['model']])
            state['completed'][arm+'_native_reload']=json.loads((training/'checkpoint6000_validation.json').read_text());save()
        controls=json.loads((ROOT/'configs/groundjev/full_history_refresh600_controls_v1.json').read_text())
        results=out/'controls';results.mkdir()
        for scope,job in controls['scopes'].items():
            baseline=original['completed']['default_'+scope]
            state['completed']['before_'+scope]=validate_run(Path(baseline['run']),job,ROOT/plan['initial_checkpoint']);save()
            manifest=json.loads((ROOT/job['manifest']).read_text())
            for arm in ('short','long'):
                wait_memory();destination=results/(arm+'_'+scope);checkpoint=checkpoints[arm]
                command=gpu+['tools/groundjev/run_video.py','--task',manifest['task'],'--video',str(ROOT/job['manifest']),
                    '--backend','full','--checkpoint',str(checkpoint),'--evidence-format','v3_crop_motion','--adaptive-context-groups',
                    '--candidate-replay',str(ROOT/job['candidates']),'--omit-replay-feature-vectors',
                    '--memory-checkpoint-format','binary','--memory-export-interval','0','--out',str(destination)]
                if manifest['task']=='rmot':command+=['--native-rmot-membership']
                if manifest['task']=='ovmot':command+=['--category-policy','native_hierarchical']
                run(arm+'_'+scope+'_INFERENCE',command)
                scorer='evaluate_video.py' if manifest['task']=='mot' else 'evaluate_multitask.py'
                run(arm+'_'+scope+'_OFFICIAL',['python3','tools/groundjev/'+scorer,'--video',str(ROOT/job['manifest']),'--run',str(destination)])
                run(arm+'_'+scope+'_EVENTS',['python3','tools/groundjev/analyze_tracking_events.py','--video',str(ROOT/job['manifest']),'--run',str(destination)])
                state['completed'][arm+'_'+scope]=validate_run(destination,job,checkpoint);save()
        state.update(status='PAIRED_TEMPORAL_TEACHER_SHARED_FOUNDATION_TRAINING_AND_ALL_CONTROLS_COMPLETE');save()
    except Exception as error:
        state.update(status='PAIRED_TEMPORAL_TEACHER_OWNER_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
