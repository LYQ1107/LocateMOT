"""One blocking owner for native one/four-GPU and complete resume validation."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    import torch
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    output = ROOT/plan['output']; output.mkdir(parents=True, exist_ok=False)
    models = Path(plan['model_outputs']); models.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(models).free > 65*2**30
    (output/'registration.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = dict(status='PREPARING_REAL_MULTITASK_DISTRIBUTED_SFT_PROBE', owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        plan=str(args.plan), completed={}, full_goal_complete=False)
    def save():
        (output/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(dict(status=state['status'], completed=list(state['completed']))), flush=True)
    save()
    try:
        with gzip.open(ROOT/plan['source_curriculum'], 'rt') as source:
            source_rows = [json.loads(l) for l in source]
        selections = [('mot', 'identity', None), ('mot', 'assigned_identity_memory_update', None),
            ('mot', 'lifecycle', None), ('mot', 'reliability', None),
            ('rmot', 'query_membership', 'yes'), ('rmot', 'query_membership', 'no'),
            ('ovmot', 'category_group', None), ('ovmot', 'category_leaf', None)]
        rows = []
        for bucket, family, target in selections:
            rows.append(next(r for r in source_rows if r['supervision']['tracking_task_bucket'] == bucket
                and r['task'] == family and (target is None or r['target'] == {target: 1.})))
        assert len({r['id'] for r in rows}) == 8
        train = output/'train.jsonl.gz'
        with gzip.open(train, 'wt') as file:
            for row in rows: file.write(json.dumps(row)+'\n')
        config_base = json.loads((ROOT/'outputs/groundjev/training_data/broader_shared_onejev6000_v2_fla/training_config.json').read_text())
        config_base.update(train=str(train), runs_dir=str(models), max_steps=2, accum_rows=4, save_every=1, log_every=1)
        python = str(ROOT/'.venv-onejev-py310/bin/python')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        if plan.get('deterministic_execution'):
            env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        for label, name, gpus, stop, resume in [
                ('single', 'single', [4], None, False),
                ('four_continuous', 'four_continuous', [4,5,6,7], None, False),
                ('four_partial', 'four_resumed', [4,5,6,7], 1, False),
                ('four_restore', 'four_resumed', [4,5,6,7], None, True)]:
            state.update(status='WAITING_FREE_MEMORY_'+label); save()
            while True:
                raw = subprocess.check_output(['nvidia-smi','-i',','.join(map(str,gpus)),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                free = list(map(int, raw.split()))
                if min(free) >= plan['launch_min_free_MiB_each']: break
                time.sleep(20)
            config = dict(config_base, name=name)
            config_path = output/(name+'_config.json')
            if config_path.exists(): assert json.loads(config_path.read_text()) == config
            else: config_path.write_text(json.dumps(config, indent=2)+'\n')
            launch = ['python3','tools/groundjev/run_with_resources.py','--gpus',*map(str,gpus),'--',python]
            if len(gpus)>1: launch += ['-m','torch.distributed.run','--standalone','--nproc-per-node','4']
            launch += ['tools/groundjev/probe_distributed_native_sft.py','--config',str(config_path),
                '--receipts',str(output/'rank_receipts'),'--label',label,
                '--backend','fla' if plan['training_backend']=='fla052_training_only' else 'reference']
            if plan.get('deterministic_execution'):launch += ['--deterministic-execution']
            if plan.get('seed_global_rngs'):launch += ['--seed-global-rngs']
            if resume: launch += ['--resume-state',str(models/name/'latest_training_state.pt')]
            else: launch += ['--initial-training-state',str(ROOT/plan['initial_training_state'])]
            if stop: launch += ['--stop-after-saved-step',str(stop)]
            state.update(status='RUNNING_'+label, command=launch, launch_free_memory_MiB=free); save()
            with (output/(label+'.log')).open('w') as log:
                child = subprocess.Popen(launch, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                state['child_pid'] = child.pid; save()
                for line in child.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                code = child.wait()
            state.pop('child_pid',None)
            assert code == 0, f'{label} exited{code}; preserve completed units and saved full states'
            latest = json.loads((models/name/'latest_training_state.json').read_text())
            assert latest['step'] == (1 if stop else 2) and latest['seen'] == (4 if stop else 8)
            for rank in range(len(gpus)):
                receipt = json.loads((output/'rank_receipts'/f'{label}_rank{rank}.json').read_text())
                assert receipt['world'] == len(gpus) and receipt['rank'] == rank
                assert receipt['optimizer_steps_this_invocation'] == (1 if stop or resume else 2)
                assert all(s['all_rank_max_difference'] == 0 for s in receipt['steps'])
            state['completed'][label] = dict(checkpoint=latest, exit_code=code, world=len(gpus)); save()

        receipts = [json.loads((output/'rank_receipts'/f'{name}_rank0.json').read_text())
            for name in ('single','four_continuous','four_restore')]
        assert receipts[0]['plans'][0]['global_rows'] == receipts[1]['plans'][0]['global_rows'] == receipts[2]['plans'][0]['global_rows']
        assert sorted(i for step in receipts[0]['plans'][0]['global_rows'] for i in step) == list(range(8))
        assert all(r['plans'][0]['lengths'] == receipts[0]['plans'][0]['lengths'] for r in receipts)
        single, continuous, restored = [torch.load(models/name/'latest_training_state.pt',
            map_location='cpu', weights_only=True, mmap=True) for name in ('single','four_continuous','four_resumed')]
        assert all(s['step'] == 2 and s['seen'] == 8 for s in (single, continuous, restored))
        assert len(continuous['rank_rngs']) == len(restored['rank_rngs']) == 4
        assert continuous['scheduler'] == restored['scheduler']
        max_diff = 0.; norms = {'exp_avg':[0.,0.], 'exp_avg_sq':[0.,0.]}; checked=0
        for name, a in single['fp32_trainable_parameters'].items():
            b = continuous['fp32_trainable_parameters'][name]; c = restored['fp32_trainable_parameters'][name]
            assert a.dtype == b.dtype == c.dtype == torch.float32 and torch.isfinite(a).all() and torch.isfinite(b).all()
            max_diff = max(max_diff, float((a-b).abs().max()))
            assert torch.equal(b,c), ('resume model mismatch',name)
            checked += a.numel()
        assert checked == 752393024 and max_diff <= plan['same_FP32_master_max_difference_gate'], max_diff
        for index, a in single['optimizer']['state'].items():
            b, c = continuous['optimizer']['state'][index], restored['optimizer']['state'][index]
            assert int(a['step']) == int(b['step']) == int(c['step']) == 2
            for key in norms:
                assert torch.isfinite(a[key]).all() and torch.isfinite(b[key]).all()
                assert torch.equal(b[key],c[key]), ('resume moment mismatch', index,key)
                x,y=a[key].double(),b[key].double()
                norms[key][0] += float((x-y).square().sum()); norms[key][1] += float(x.square().sum())
        moment_relative = {k: math.sqrt(n/max(d,1e-100)) for k,(n,d) in norms.items()}
        assert all(v <= plan['full_Adam_moment_relative_L2_gate'] for v in moment_relative.values()), moment_relative
        for a,b in zip(continuous['rank_rngs'],restored['rank_rngs']):
            assert a['rank']==b['rank'] and a['python_rng']==b['python_rng'] and a['numpy_rng']==b['numpy_rng']
            assert torch.equal(a['torch_rng'],b['torch_rng']) and torch.equal(a['cuda_rng'],b['cuda_rng'])
        result = dict(status='REAL_NATIVE_ONE_AND_FOUR_GPU_GLOBAL_MEAN_FULL_STATE_AND_EXACT_RESUME_VALIDATED',
            source_plan=str(args.plan), actual_unique_rows=8, actual_updates_each=2,
            FP32_parameters_compared=checked, all320_model_master_max_difference=max_diff,
            full_Adam_moment_relative_L2=moment_relative, all_four_rank_model_and_moment_difference=0.,
            continuous_vs_restored_models_moments_scheduler_all_rank_RNG_exact=True,
            CPU_global_row_plans_equal_no_duplicates_or_drops=True, all_three_task_rows_present=True,
            model_and_optimizer_checkpoints_retained_separately=str(models),
            current_production_training_or_evaluation_changed=False,
            large_scale_training_started=False, speed_or_tracking_gain_claim=False, full_goal_complete=False)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state.update(status=result['status'], result=result); save()
        assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) < plan['diagnostic_cap_bytes']
    except Exception as error:
        state.update(status='DISTRIBUTED_NATIVE_SFT_PROBE_FAILED', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
