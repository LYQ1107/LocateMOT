"""CPU cross-frame supervised contrastive native appearance adaptation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from locatemot.groundjev.temporal_appearance import TemporalAppearance
    from tools.groundjev.temporal_identity_data import TemporalIdentityData
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(plan['cpu_threads']);torch.manual_seed(plan['seed']);torch.use_deterministic_algorithms(True)
    rng=random.Random(plan['seed']);data=TemporalIdentityData(plan)
    out=args.out or Path(plan['training_output'])
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    model=TemporalAppearance();initial={k:v.clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    step=seen=0;elapsed_before=0.;exposure=Counter()
    if args.resume:
        saved=torch.load(out/'latest_training_state.pt',map_location='cpu',weights_only=False)
        assert saved['protocol']==plan
        model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);rng.setstate(saved['sampler_rng'])
        step=saved['step'];seen=saved['observations'];elapsed_before=saved['training_seconds'];exposure.update(saved['unit_batch_exposure'])
    else:
        torch.save(dict(architecture=plan['architecture'],state_dict=initial,step=0),out/'initial_projection.pt')
    stop=args.stop_after or plan['optimizer_steps'];assert step<stop<=plan['optimizer_steps']
    state=dict(status='TEMPORAL_APPEARANCE_CONTRASTIVE_TRAINING',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],step=step,
        target_steps=plan['optimizer_steps'],observations=seen,CPU_only=True,new_GPU_processes=0,full_goal_complete=False)
    def publish_state():
        temp=out/'owner_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'owner_state.json')
        print(json.dumps(state),flush=True)
    publish_state();start=time.perf_counter()
    try:
        with (out/'log.jsonl').open('a') as log:
            for step in range(step+1,stop+1):
                features,labels,unit=data.batch(rng);opt.zero_grad()
                z=model(features);scores=(z@z.T)/plan['temperature']
                diagonal=torch.eye(len(z),dtype=torch.bool)
                scores=scores.masked_fill(diagonal,-1e4)
                positive=labels[:,None].eq(labels[None,:])&~diagonal
                assert positive.any(1).all()
                log_probs=scores-torch.logsumexp(scores,dim=1,keepdim=True)
                loss=-(log_probs*positive).sum(1).div(positive.sum(1)).mean()
                assert torch.isfinite(loss)
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']))
                assert norm>0 and torch.isfinite(torch.tensor(norm))
                opt.step();seen+=len(z);exposure[unit]+=1
                elapsed=elapsed_before+time.perf_counter()-start
                if step==1 or step%100==0 or step==stop:
                    record=dict(step=step,actual_ROI_exposures=seen,loss=float(loss.detach()),gradient_norm=norm,
                        sampled_unit=unit,actual_batch_observations=len(z),training_seconds=elapsed)
                    log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                if step%plan['save_every']==0 or step==stop:
                    saved=dict(protocol=plan,step=step,observations=seen,state_dict=model.state_dict(),optimizer=opt.state_dict(),
                        torch_rng=torch.get_rng_state(),sampler_rng=rng.getstate(),training_seconds=elapsed,unit_batch_exposure=dict(exposure))
                    temp=out/'latest_training_state.tmp';torch.save(saved,temp);temp.replace(out/'latest_training_state.pt')
                    state.update(step=step,observations=seen);publish_state()
        if stop!=plan['optimizer_steps']:
            state.update(status='EXPLICIT_TEMPORAL_APPEARANCE_PREFIX_COMPLETE_FULL_TRAINING_INCOMPLETE');publish_state();return
        model.eval();checkpoint=dict(architecture=plan['architecture'],state_dict=model.state_dict(),step=step,
            supervision='TRAIN immutable candidate GT identity cross-frame supervised contrastive',
            no_query_or_GT_runtime_input=True,source_manifest=plan['output'])
        torch.save(checkpoint,out/'projection.pt')
        fresh=TemporalAppearance();fresh.load_state_dict(torch.load(out/'projection.pt',weights_only=True)['state_dict']);fresh.eval()
        with torch.no_grad():assert torch.equal(model(features),fresh(features))
        for name,tensor in model.state_dict().items():assert torch.isfinite(tensor).all() and not torch.equal(tensor,initial[name])
        for entry in opt.state_dict()['state'].values():
            assert int(entry['step'])==stop and torch.isfinite(entry['exp_avg']).all() and torch.isfinite(entry['exp_avg_sq']).all()
        assert not torch.cuda.is_initialized()
        state.update(status='TEMPORAL_APPEARANCE_CONTRASTIVE_TRAINING_COMPLETE',training_seconds=elapsed,
            parameters=sum(p.numel() for p in model.parameters()),all_parameters_changed_and_finite=True,
            all_Adam_moments_finite_and_steps_exact=True,projection_reload_output_exact=True,
            unit_batch_exposure=dict(exposure),production_projection=str(out/'projection.pt'),
            official_tracking_metrics='pending',native_OneJev_decoder_or_joint_foundation_training=False)
        publish_state()
    except Exception as error:
        state.update(status='TEMPORAL_APPEARANCE_TRAINING_FAILED',error=repr(error));publish_state();raise


if __name__=='__main__':main()
