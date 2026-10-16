"""One blocking owner for full-source planning, shared four-GPU SFT and controls."""
import argparse
from collections import Counter, defaultdict
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


def cpu_plan(config, report):
    import torch
    from tools.groundjev.run_source_manifest_sft import configure_source_manifest_sft
    from train import sft
    torch.set_num_threads(4)
    stateful=configure_source_manifest_sft('fla')
    def capture(bundle, rows, args, world):
        started=time.perf_counter()
        batches,dropped,lengths=sft.plan_epoch(bundle,rows,args,0,0,1)
        steps=sft.plan_steps(batches,lengths,4,args.accum_rows)
        assert world==1 and dropped==0 and len(rows)>=456062
        assert sorted(i for batch in batches for i in batch)==list(range(len(rows)))
        assert sum(len(batch) for step in steps for per in step for batch in per)==len(rows)
        manifest=json.loads(Path(args.train_file).read_text())
        assert len(rows)==manifest['rows']
        images=defaultdict(set);families=Counter();cursor=0
        for source in manifest['sources']:
            for row in rows[cursor:cursor+source['rows']]:
                families[source['bucket']+':'+row['task']]+=1
                if row.get('media'):
                    images[source['bucket']].add(str(Path(row['media'][0]['path']).resolve()))
            cursor+=source['rows']
        assert cursor==len(rows) and not torch.cuda.is_initialized()
        result=dict(status='FULL_SOURCE_ALL_NATIVE_MM_ROWS_AND_FOUR_RANK_GLOBAL16_PLAN_VALIDATED',
            rows=len(rows),rows_per_task=manifest['rows_per_task'],dropped_rows=0,
            maximum_tokens=max(lengths),planned_tokens=sum(lengths),planned_updates=len(steps),
            planned_world_size=4,global_rows_per_update=args.accum_rows,
            question_families=dict(families),
            planned_unique_first_media_source_paths={k:len(v) for k,v in images.items()},
            media_path_counts_are_filtered_row_exposure_not_raw_scope_counts=True,
            CPU_plan_elapsed_s=time.perf_counter()-started,actual_optimizer_updates=0,
            full_goal_complete=False)
        report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        return 0
    sft.plan_only=capture
    sys.argv=[sys.argv[0],'--config',str(config),'--plan-only','--paired-index-shuffle',
        '--deterministic-execution','--seed-global-rngs']
    return stateful.main()


def main():
    from tools.groundjev.run_history_refresh_controls import wait_dependency,validate_run
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--cpu-plan',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.cpu_plan:return cpu_plan(args.cpu_plan,args.report)
    plan=json.loads(args.plan.read_text());args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    proof=json.loads((ROOT/plan['distributed_validation']/'result.json').read_text())
    assert proof['status']=='REAL_NATIVE_ONE_AND_FOUR_GPU_GLOBAL_MEAN_FULL_STATE_AND_EXACT_RESUME_VALIDATED'
    assert proof['continuous_vs_restored_models_moments_scheduler_all_rank_RNG_exact']
    native=json.loads((ROOT/plan['distributed_validation']/'native_MM_reload.json').read_text())
    assert native['status']=='REAL_ONEJEV2_NATIVE_MM_RELOAD_VALIDATED'
    assert native['repeated_reload_probability_max_difference']<=1e-6
    final_save=json.loads((ROOT/plan['distributed_validation']/'final_nonboundary_validation.json').read_text())
    assert final_save['status']=='FINAL_NONBOUNDARY_FULL_STATE_SAME_ACTUAL_MODEL_AND_MOMENTS_VALIDATED'
    state=dict(status='WAITING_FULL_SOURCE_MANIFEST_BEFORE_LARGE_JOINT_TRAINING',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        owner_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan=str(args.plan),completed={},actual_training_updates=0,full_goal_complete=False)
    def save():
        tmp=args.out/'owner_state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.out/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']))),flush=True)
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'external/OneJev'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(stage,command):
        state.update(status=stage,command=command);save()
        with (args.out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        state.pop('child_pid',None);state['last_exit_code']=code
        assert code==0,f'{stage} exited{code}; retain completed units and latest full state'
    save()
    try:
        wait_dependency(plan['source_dependency'])
        manifest_path=ROOT/plan['source_manifest'];manifest=json.loads(manifest_path.read_text())
        assert manifest['rows']>=456062 and manifest['all_source_files_immutable_completed']
        model_root=Path(plan['runs_dir']);model_root.mkdir(parents=True,exist_ok=True)
        training=model_root/plan['training_name'];assert not training.exists()
        assert shutil.disk_usage(model_root).free>=150*2**30
        config=json.loads((ROOT/'configs/groundjev/onejev_unified600_native_v2.json').read_text())
        config.update(model=str(ROOT/plan['initial_checkpoint']),train=str(manifest_path),
            runs_dir=str(model_root),name=plan['training_name'],epochs=1,max_steps=0,
            accum_rows=16,save_every=1000,log_every=25,seed=0,lr=2e-6)
        cpu_config=dict(config,device='cpu',name='full_joint_CPU_plan',runs_dir=str(args.out/'CPU_plan'))
        cpu_path=args.out/'CPU_plan_config.json';cpu_path.write_text(json.dumps(cpu_config,indent=2)+'\n')
        report_path=args.out/'CPU_plan_validation.json'
        run('FULL_NATIVE_CPU_PLAN',[python,str(Path(__file__).resolve()),'--cpu-plan',str(cpu_path),'--report',str(report_path)])
        measured=json.loads(report_path.read_text());config['max_steps']=measured['planned_updates']
        config_path=args.out/'training_config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
        state['completed']['full_CPU_plan']=measured;save()
        state['status']='WAITING_SHARED_FOUR_GPU_FREE_MEMORY';save()
        while True:
            free=list(map(int,subprocess.check_output(['nvidia-smi','-i','4,5,6,7',
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).split()))
            if min(free)>=plan['launch_min_free_MiB_each']:break
            time.sleep(20)
        state['launch_free_memory_MiB']=free
        run('FULL_JOINT_FOUR_GPU_TRAINING',['python3','tools/groundjev/run_with_resources.py','--gpus','4','5','6','7','--',
            python,'-m','torch.distributed.run','--standalone','--nproc-per-node','4',
            'tools/groundjev/run_source_manifest_sft.py','--source-training-backend','fla','--config',str(config_path),
            '--paired-index-shuffle','--distributed-sft','--deterministic-execution','--seed-global-rngs',
            '--initial-training-state',str(ROOT/plan['initial_training_state']),'--expected-initial-step','600'])
        records=[json.loads(l) for l in (training/'log.jsonl').open()]
        done=next(r for r in records if r.get('event')=='done')
        assert done['steps']==config['max_steps'] and done['examples']==manifest['rows']
        assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in records if 'loss'in r)
        latest=json.loads((training/'latest_training_state.json').read_text())
        assert latest['step']==done['steps'] and latest['seen']==done['examples']
        state['actual_training_updates']=done['steps'];state['completed']['training']=dict(done=done,latest=latest);save()
        checkpoint=training/'final'
        run('FINAL_NATIVE_MM_RELOAD',['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',python,
            'tools/groundjev/check_onejev125_checkpoint.py','--step',str(done['steps']),'--checkpoint',str(checkpoint),
            '--training-root',str(training),'--reference-checkpoint',str(ROOT/plan['initial_checkpoint'])])
        controls=json.loads((ROOT/'configs/groundjev/full_history_refresh600_controls_v1.json').read_text())
        results=args.out/'controls';results.mkdir()
        for scope,job in controls['scopes'].items():
            video=json.loads((ROOT/job['manifest']).read_text())
            before=ROOT/job['baseline_run'] if job['baseline_run'] else ROOT/controls['output_root']/('before_'+scope)
            if not (before/'official_metrics.json').exists():
                state.update(status='WAITING_ORIGINAL_COMPLETE_TAO_BASELINE');save()
                while not (before/'official_metrics.json').exists():
                    original=Path('/proc/2467924/stat').read_text().rsplit(')',1)[1].split()
                    assert original[0]!='Z' and original[19]=='45243513'
                    time.sleep(20)
            state['completed']['before_'+scope]=validate_run(before,job,ROOT/plan['initial_checkpoint']);save()
            for arm in ('default','assigned_identity'):
                output=results/(arm+'_'+scope)
                command=['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',python,
                    'tools/groundjev/run_video.py','--task',video['task'],'--video',str(ROOT/job['manifest']),
                    '--backend','full','--checkpoint',str(checkpoint),'--evidence-format','v3_crop_motion',
                    '--adaptive-context-groups','--candidate-replay',str(ROOT/job['candidates']),
                    '--omit-replay-feature-vectors','--memory-checkpoint-format','binary','--memory-export-interval','0','--out',str(output)]
                if video['task']=='rmot':command+=['--native-rmot-membership']
                if video['task']=='ovmot':
                    command+=['--category-policy','native_hierarchical']
                    if arm=='assigned_identity':command+=['--category-replay',str(results/('default_'+scope))]
                if arm=='assigned_identity':command+=['--assigned-identity-updates']
                name=arm+'_'+scope;run(name+'_INFERENCE',command)
                scorer='evaluate_video.py' if video['task']=='mot' else 'evaluate_multitask.py'
                run(name+'_OFFICIAL',['python3','tools/groundjev/'+scorer,'--video',str(ROOT/job['manifest']),'--run',str(output)])
                run(name+'_EVENTS',['python3','tools/groundjev/analyze_tracking_events.py','--video',str(ROOT/job['manifest']),'--run',str(output)])
                runtime=json.loads((output/'result.json').read_text())
                assert bool(runtime.get('assigned_identity_updates',False))==(arm=='assigned_identity')
                state['completed'][name]=validate_run(output,job,checkpoint);save()
        state['status']='FULL_SOURCE_JOINT_EPOCH_AND_ALL_FIXED_BEFORE_DEFAULT_FINAL_ID_CONTROLS_VALIDATED';save()
    except Exception as error:
        state.update(status='FULL_SOURCE_JOINT_OWNER_FAILED',error=repr(error));save();raise


if __name__=='__main__':raise SystemExit(main())
