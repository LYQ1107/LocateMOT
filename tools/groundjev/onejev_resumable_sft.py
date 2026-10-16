"""Native SFT math/plans with full FP32/Adam/RNG, opt-in distributed state.

The upstream BF16 weight export is suitable for inference, but cannot restore
FP32 masters or Adam moments. Keep one atomic full training-state checkpoint.
"""
import argparse
import inspect
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'external/OneJev'))


def collect_rank_rng_state():
    import torch
    import torch.distributed as dist
    import numpy as np
    n=np.random.get_state()
    local=dict(rank=dist.get_rank(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(),
        numpy_rng=(n[0],n[1].tolist(),n[2],n[3],n[4]))
    gathered=[None]*dist.get_world_size()
    dist.all_gather_object(gathered,local)
    return gathered


def write_state(model,opt,sched,args,step,seen,epoch,next_step,run_dir,rank_rngs=None):
    import torch
    import numpy as np
    state={'format':'groundjev-full-fp32-adam-rng-v1','step':step,'seen':seen,'epoch':epoch,
        'next_step_in_epoch':next_step,'protocol':protocol(args),
        'fp32_trainable_parameters':{k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad},
        'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if rank_rngs is None else [torch.cuda.get_rng_state()],
        'python_rng':random.getstate()}
    n=np.random.get_state();state['numpy_rng']=(n[0],n[1].tolist(),n[2],n[3],n[4])
    if rank_rngs is not None:
        state['rank_rngs']=rank_rngs
    target=Path(run_dir)/'latest_training_state.pt';temporary=target.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(target)
    receipt={'format':state['format'],'step':step,'seen':seen,'epoch':epoch,'next_step_in_epoch':next_step,
        'path':str(target),'bytes':target.stat().st_size,'FP32_masters_and_Adam_moments_saved':True,
        'scheduler_and_all_RNG_saved':True,'single_latest_training_state':True,'feature_cache':False}
    (Path(run_dir)/'latest_training_state.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


PROTOCOL_KEYS=('model','train_file','seed','lr','max_steps','epochs','max_len','batch_tokens',
    'accum_rows','max_rows','brier_weight','augment','train_vision','full_finetune','weight_decay','warmup_frac')


def protocol(args):
    result = {k: getattr(args, k) for k in PROTOCOL_KEYS}
    if getattr(args, 'paired_index_shuffle', False):
        result['sampling_protocol'] = 'paired-index-shuffle-v1'
    if getattr(args, 'distributed_sft', False):
        result['distributed_training_protocol'] = 'native-summed-global-row-mean-allreduce-v1'
        result['world_size'] = int(__import__('os').environ.get('WORLD_SIZE','1'))
    if getattr(args, 'deterministic_execution', False):
        result['deterministic_execution'] = 'torch-deterministic-algorithms-cudnn-deterministic-cublas4096x8-v1'
    if getattr(args, 'seed_global_rngs', False):
        result['global_rng_initialization'] = 'registered-seed-applied-to-Torch-CUDA-Python-NumPy-before-loading-v1'
    return result


def restore_state(path,model,opt,sched,args):
    import torch
    import numpy as np
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['format']=='groundjev-full-fp32-adam-rng-v1'
    assert state['protocol']==protocol(args)
    parameters={k:v for k,v in model.named_parameters() if v.requires_grad}
    assert parameters.keys()==state['fp32_trainable_parameters'].keys()
    with torch.no_grad():
        for k,v in parameters.items():
            target=state['fp32_trainable_parameters'][k].to(v.device)
            assert v.dtype==target.dtype==torch.float32
            v.copy_(target)
            assert torch.equal(v,target)
    opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
    if 'rank_rngs' in state:
        import torch.distributed as dist
        local=state['rank_rngs'][dist.get_rank()]
        assert local['rank']==dist.get_rank() and len(state['rank_rngs'])==dist.get_world_size()
        torch.set_rng_state(local['torch_rng']);torch.cuda.set_rng_state(local['cuda_rng']);random.setstate(local['python_rng'])
        n=local['numpy_rng']
    else:
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
        n=state['numpy_rng']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    receipt={'status':'FULL_FP32_ADAM_SCHEDULER_RNG_CURSOR_RESTORED','source':str(path),
        'step':state['step'],'seen':state['seen'],'next_step_in_epoch':state['next_step_in_epoch'],
        'FP32_master_max_difference':0.,'trainable_parameter_entries':len(parameters),
        'protocol_exact':True,'sampling_protocol':state['protocol'].get('sampling_protocol','native-length-buckets')}
    if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
        (Path(args.runs_dir)/args.name/'resumed_training_state_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return state


def initialize_fp32_stage(path,model,args,expected_step):
    """New explicit curriculum: preserve real masters, reset optimizer/scheduler/RNG by its recipe."""
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['format']=='groundjev-full-fp32-adam-rng-v1' and state['step']==expected_step
    parameters={name:value for name,value in model.named_parameters() if value.requires_grad}
    assert parameters.keys()==state['fp32_trainable_parameters'].keys()
    changed=0
    with torch.no_grad():
        for name,param in parameters.items():
            master=state['fp32_trainable_parameters'][name]
            assert param.dtype==master.dtype==torch.float32
            target=master.to(param.device)
            changed+=int((param!=target).sum())
            param.copy_(target)
            assert torch.equal(param,target)
    receipt={'status':'NEW_STAGE_REAL_FP32_MASTERS_INITIALIZED','source':str(path),'source_step':expected_step,
        'FP32_parameter_entries':len(parameters),'parameters':sum(p.numel() for p in parameters.values()),
        'changed_vs_BF16_inference_export_initialization':changed,'initialization_max_difference':0.,
        'new_optimizer_scheduler_and_RNG':'explicit newrecipe reset, not originaltrajectory resume',
        'old_training_state_or_inference_weights_modified':False}
    out=Path(args.runs_dir)/args.name/'initial_fp32_state_receipt.json'
    if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
        out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--resume-state',type=Path)
    parser.add_argument('--initial-training-state',type=Path)
    parser.add_argument('--expected-initial-step',type=int,default=500)
    parser.add_argument('--paired-index-shuffle',action='store_true')
    parser.add_argument('--distributed-sft',action='store_true',help='explicit native distributed global-row-mean training and per-rank RNG checkpoints')
    parser.add_argument('--stop-after-saved-step',type=int,help='explicit checkpoint/resume probe; returns before completion without a done event')
    parser.add_argument('--deterministic-execution',action='store_true',help='explicit deterministic Torch/CUDA execution protocol for repeatable distributed resumes')
    parser.add_argument('--seed-global-rngs',action='store_true',help='apply the registered recipe seed to all global RNGs before initialization')
    options,argv=parser.parse_known_args()
    assert not (options.resume_state and options.initial_training_state)
    if options.deterministic_execution:
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    from train import sft
    if options.paired_index_shuffle:
        from paired_sft_sampler import index_shuffle_plan
        sft.plan_epoch = index_shuffle_plan(sft.plan_epoch)
        native_parse_args = sft.parse_args
        def parse_paired_args(argv=None):
            args = native_parse_args(argv)
            args.paired_index_shuffle = True
            return args
        sft.parse_args = parse_paired_args
    if options.distributed_sft:
        native_distributed_parse=sft.parse_args
        def parse_distributed_args(argv=None):
            args=native_distributed_parse(argv)
            args.distributed_sft=True
            return args
        sft.parse_args=parse_distributed_args
    if options.deterministic_execution:
        native_deterministic_parse=sft.parse_args
        def parse_deterministic_args(argv=None):
            args=native_deterministic_parse(argv)
            args.deterministic_execution=True
            return args
        sft.parse_args=parse_deterministic_args
    if options.seed_global_rngs:
        native_seeded_parse=sft.parse_args
        def parse_seeded_args(argv=None):
            args=native_seeded_parse(argv)
            args.seed_global_rngs=True
            return args
        sft.parse_args=parse_seeded_args
    code=inspect.getsource(sft.main)
    replacements=[
        ('    params = [p for p in model.parameters() if p.requires_grad]\n    if world > 1 and not args.full_finetune:',
         '    if _initial_training_state:\n'
         '        _initialize_fp32_stage(_initial_training_state,model,args,_expected_initial_step)\n'
         '    params = [p for p in model.parameters() if p.requires_grad]\n    if world > 1 and not args.full_finetune:'),
        ('    step = 0\n    seen = 0',
         '    resume = _restore_state(_resume_path,model,opt,sched,args) if _resume_path else None\n'
         '    step = resume["step"] if resume else 0\n    seen = resume["seen"] if resume else 0'),
        ('        for step_plan in steps_plan:',
         '        for step_index, step_plan in enumerate(steps_plan):\n'
         '            if resume and (epoch < resume["epoch"] or (epoch == resume["epoch"] and step_index < resume["next_step_in_epoch"])):\n'
         '                continue'),
        ('                log.info("saved %s to %s", "model" if args.full_finetune else "adapter", ckpt)',
         '                log.info("saved %s to %s", "model" if args.full_finetune else "adapter", ckpt)\n'
         '                _write_state(model,opt,sched,args,step,seen,epoch,step_index+1,run_dir,_rank_rngs)'),
        ('            if (is_main or args.fsdp) and args.save_every and step % args.save_every == 0:',
         '            _saving = args.save_every and step % args.save_every == 0\n'
         '            _rank_rngs = _collect_rank_rng_state() if _distributed_sft and world > 1 and _saving else None\n'
         '            if (is_main or args.fsdp) and _saving:'),
        ('            if eval_sets and args.eval_every and step % args.eval_every == 0 and step != total_steps:',
         '            if _distributed_sft and world > 1 and _saving:\n'
         '                dist.barrier()\n'
         '            if _stop_after_saved_step and step == _stop_after_saved_step:\n'
         '                assert _saving\n'
         '                if is_main:\n'
         '                    print("PROBE_STOP_AFTER_SAVED_STEP_NOT_TRAINING_COMPLETE", step, flush=True)\n'
         '                    logf.close()\n'
         '                if world > 1:\n'
         '                    dist.destroy_process_group()\n'
         '                return 0\n'
         '            if eval_sets and args.eval_every and step % args.eval_every == 0 and step != total_steps:'),
        ('    elapsed = time.perf_counter() - t_start',
         '    if _distributed_sft and (not args.save_every or step % args.save_every):\n'
         '        _rank_rngs = _collect_rank_rng_state() if world > 1 else None\n'
         '        if is_main:\n'
         '            _write_state(model,opt,sched,args,step,seen,epoch,step_index+1,run_dir,_rank_rngs)\n'
         '        if world > 1:\n'
         '            dist.barrier()\n'
         '    elapsed = time.perf_counter() - t_start'),
    ]
    for old,new in replacements:
        assert code.count(old)==1, 'Upstream SFT source changed; inspect before applying checkpoint-only extension'
        code=code.replace(old,new)
    args=sft.parse_args(argv)
    if options.seed_global_rngs:
        import numpy as np
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    world=int(__import__('os').environ.get('WORLD_SIZE','1'))
    assert not args.fsdp and (world==1 or options.distributed_sft)
    namespace=dict(sft.__dict__,_write_state=write_state,_restore_state=restore_state,_resume_path=options.resume_state,
        _initialize_fp32_stage=initialize_fp32_stage,_initial_training_state=options.initial_training_state,
        _expected_initial_step=options.expected_initial_step,
        _collect_rank_rng_state=collect_rank_rng_state,_distributed_sft=options.distributed_sft,
        _stop_after_saved_step=options.stop_after_saved_step)
    exec(compile(code,str(Path(__file__).resolve()),'exec'),namespace)
    return namespace['main'](argv)


if __name__=='__main__':raise SystemExit(main())
