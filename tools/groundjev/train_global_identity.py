"""Actual deterministic CPU gradients for A1/A2, followed by fixed final export."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from locatemot.groundjev.compact import CompactCore
    from locatemot.groundjev.global_identity_core import GlobalIdentityCore,physical_inputs,physical_loss
    from tools.groundjev.global_identity_data import training_frames
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--arm',choices=['A1','A2'],required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=Path(plan['immutable_source_root']);proof=json.loads((ROOT/plan['output']/'label_audit.json').read_text())
    assert proof['totals']['cross_group_complete_competitor_set_retrieved']==4804
    torch.set_num_threads(plan['cpu_threads']);torch.manual_seed(plan['seed']);torch.use_deterministic_algorithms(True)
    initial=torch.load(source/plan['initial_checkpoint'],map_location='cpu',weights_only=True)
    base=CompactCore(initial['architecture'],semantic=initial['semantic']);base.load_state_dict(initial['state_dict'])
    model=GlobalIdentityCore(base,competition=args.arm=='A2');model.train()
    canonical=torch.load('/data3/user/LocateMOT-checkpoints/compact_canonical_physical_initial_v1.pt',map_location='cpu',weights_only=True)
    with torch.no_grad():
        model.physical_specification.copy_(base.task.weight[0]+base.text(canonical['state_dict']['physical_spec_text']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    out=Path(plan['model_output'])/args.arm;out.mkdir(parents=True,exist_ok=False)
    initial_state={k:v.clone() for k,v in model.state_dict().items()}
    steps=0;counts=Counter();started=time.perf_counter();first=[]
    with (out/'log.jsonl').open('w') as log:
        for epoch in range(plan['epochs']):
            frames=0
            for frame in training_frames(plan):
                e=physical_inputs(frame['image_size'],frame['candidates'],frame['tracks'],frame['timestamp'],plan['competition_neighbors_each_metric'])
                optimizer.zero_grad();prediction=model(e);loss,metrics=physical_loss(prediction,e,frame['labels'])
                assert torch.isfinite(loss)
                loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']))
                assert math.isfinite(gradient)
                if metrics['supervised_actions'] or metrics['duplicate']:
                    assert gradient>0;optimizer.step();steps+=1
                counts['frames']+=1;frames+=1;counts['candidates']+=len(frame['candidates'])
                counts['supervised_action_exposures']+=metrics['supervised_actions'];counts['unknown_masked_exposures']+=metrics['masked_actions']
                record=dict(epoch=epoch,step=steps,unit=frame['unit'],frame=frame['frame'],loss=float(loss.detach()),
                    gradient_norm=gradient,metrics=metrics,elapsed_seconds=time.perf_counter()-started)
                if len(first)<5:first.append(record)
                if frames==1 or frames%100==0:
                    log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
            assert frames==1056
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,steps=steps,
                torch_rng=torch.get_rng_state(),plan=plan,arm=args.arm,counts=dict(counts)),out/'latest_training_state.pt')
    model.eval();checkpoint=dict(format='global-identity-compact-v1',state_dict=model.state_dict(),
        arm=args.arm,competition=model.competition,initial_checkpoint=str(source/plan['initial_checkpoint']),
        width=model.width,plan=plan,actual_steps=steps,epochs=plan['epochs'],same_physical_parameters_all_tasks=True,
        physical_inputs_exclude_query_and_GT=True)
    torch.save(checkpoint,out/'checkpoint.pt')
    changed={k:int((v!=initial_state[k]).sum()) for k,v in model.state_dict().items()}
    assert changed['pair.0.weight']>0 and changed['actions.0.weight']>0
    if args.arm=='A2':assert changed['edge.0.weight']>0 and changed['message.weight']>0
    report=dict(status='GLOBAL_IDENTITY_COMPACT_TRAINING_COMPLETE',arm=args.arm,actual_steps=steps,
        epochs=plan['epochs'],counts=dict(counts),first5=first,parameters=sum(p.numel() for p in model.parameters()),
        changed_elements=changed,training_seconds=time.perf_counter()-started,CPU_only=True,new_GPU_processes=0,
        discrete_Hungarian_in_gradient_path=False,losses='Masked multi-positive ID/NEW/REJECT CE, duplicate BCE, GT-quality pairwise ranking.',
        final_checkpoint=str(out/'checkpoint.pt'),official_results='pending',full_goal_complete=False)
    (out/'training_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
