"""Official single-GPU SFT math/plans with complete FP32/Adam/RNG checkpoints.

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


def write_state(model,opt,sched,args,step,seen,epoch,next_step,run_dir):
    import torch
    import numpy as np
    state={'format':'groundjev-full-fp32-adam-rng-v1','step':step,'seen':seen,'epoch':epoch,
        'next_step_in_epoch':next_step,'protocol':{k:getattr(args,k) for k in PROTOCOL_KEYS},
        'fp32_trainable_parameters':{k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad},
        'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()}
    n=np.random.get_state();state['numpy_rng']=(n[0],n[1].tolist(),n[2],n[3],n[4])
    target=Path(run_dir)/'latest_training_state.pt';temporary=target.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(target)
    receipt={'format':state['format'],'step':step,'seen':seen,'epoch':epoch,'next_step_in_epoch':next_step,
        'path':str(target),'bytes':target.stat().st_size,'FP32_masters_and_Adam_moments_saved':True,
        'scheduler_and_all_RNG_saved':True,'single_latest_training_state':True,'feature_cache':False}
    (Path(run_dir)/'latest_training_state.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


PROTOCOL_KEYS=('model','train_file','seed','lr','max_steps','epochs','max_len','batch_tokens',
    'accum_rows','max_rows','brier_weight','augment','train_vision','full_finetune','weight_decay','warmup_frac')


def restore_state(path,model,opt,sched,args):
    import torch
    import numpy as np
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['format']=='groundjev-full-fp32-adam-rng-v1'
    assert state['protocol']=={k:getattr(args,k) for k in PROTOCOL_KEYS}
    parameters={k:v for k,v in model.named_parameters() if v.requires_grad}
    assert parameters.keys()==state['fp32_trainable_parameters'].keys()
    with torch.no_grad():
        for k,v in parameters.items():v.copy_(state['fp32_trainable_parameters'][k].to(v.device))
    opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
    n=state['numpy_rng'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    return state


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--resume-state',type=Path)
    options,argv=parser.parse_known_args()
    from train import sft
    code=inspect.getsource(sft.main)
    replacements=[
        ('    step = 0\n    seen = 0',
         '    resume = _restore_state(_resume_path,model,opt,sched,args) if _resume_path else None\n'
         '    step = resume["step"] if resume else 0\n    seen = resume["seen"] if resume else 0'),
        ('        for step_plan in steps_plan:',
         '        for step_index, step_plan in enumerate(steps_plan):\n'
         '            if resume and (epoch < resume["epoch"] or (epoch == resume["epoch"] and step_index < resume["next_step_in_epoch"])):\n'
         '                continue'),
        ('                log.info("saved %s to %s", "model" if args.full_finetune else "adapter", ckpt)',
         '                log.info("saved %s to %s", "model" if args.full_finetune else "adapter", ckpt)\n'
         '                _write_state(model,opt,sched,args,step,seen,epoch,step_index+1,run_dir)'),
    ]
    for old,new in replacements:
        assert code.count(old)==1, 'Upstream SFT source changed; inspect before applying checkpoint-only extension'
        code=code.replace(old,new)
    args=sft.parse_args(argv)
    assert not args.fsdp and int(__import__('os').environ.get('WORLD_SIZE','1'))==1
    namespace=dict(sft.__dict__,_write_state=write_state,_restore_state=restore_state,_resume_path=options.resume_state)
    exec(compile(code,str(Path(__file__).resolve()),'exec'),namespace)
    return namespace['main'](argv)


if __name__=='__main__':raise SystemExit(main())
