"""Finite shared-core semantic adaptation, native MOT plus fit/Base teachers."""
import json
import argparse
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    import torch.nn.functional as F
    from locatemot.groundjev.compact import CompactCore
    torch.manual_seed(0);torch.set_num_threads(4);rng=random.Random(0)
    parser=argparse.ArgumentParser();parser.add_argument('--pack',type=Path,default=ROOT/'outputs/groundjev/compact/joint_pack_v3')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/compact/joint1200_v1')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/groundjev/compact/native300_v1/compact_distilled.pt')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--device', choices=['cpu','cuda:0'], default='cuda:0')
    parser.add_argument('--tasks', nargs='+', choices=['mot','rmot','ovmot'], default=['mot','rmot','ovmot'])
    parser.add_argument('--per-task-sampling', action='store_true', help='independent seed0+task sampling streams for shared task prefixes across subset controls')
    parser.add_argument('--resume-state', type=Path)
    args=parser.parse_args();source=args.pack
    assert json.loads((source/'owner_state.json').read_text())['status']=='DECLARED_JOINT_SEMANTIC_TRAIN_PACK_COMPLETE'
    pack=torch.load(source/'examples.pt',weights_only=True);by_task={i:[e for e in pack['examples'] if e['task_index']==i] for i in range(3)}
    assert all(by_task.values())
    task_indices=[('mot','rmot','ovmot').index(task) for task in args.tasks]
    assert len(set(task_indices))==len(task_indices)
    streams={task:random.Random(task) for task in task_indices}
    sampled={task:[] for task in task_indices}
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    state={'status':'TRAINING_SHARED_COMPACT_MOT_RMOT_BASE_OVMOT','steps':0,'target_steps':args.steps,'seed':0,
        'task_examples':{str(i):len(v) for i,v in by_task.items()},'Novel_GT_supervised':False,
        'additional_teacher_proposal_stage_not_native_detector_rollout':True,
        'source_pack':str(source),'initialization':str(args.checkpoint),
        'device':args.device,'selected_tasks':args.tasks,'per_task_independent_sampling':args.per_task_sampling,
        'actual_native_RMOT_groups':sum(e.get('proposal_source')=='actual_native_GroundAnything_DLM' for e in pack['examples'])}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    save();model=CompactCore(semantic=True).to(args.device)
    previous=torch.load(args.checkpoint,weights_only=True)
    missing,unexpected=model.load_state_dict(previous['state_dict'],strict=False)
    assert not unexpected and all(k.startswith(('text.','category.','membership.','class_scale')) for k in missing)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    classes=pack['class_text'].to(args.device);start=time.perf_counter();logs=[]
    protocol={'pack':str(source.resolve()),'initialization':str(args.checkpoint.resolve()),
        'tasks':args.tasks,'device':args.device,'steps':args.steps,'per_task_sampling':args.per_task_sampling}
    start_step=0
    if args.resume_state:
        saved=torch.load(args.resume_state,map_location=args.device,weights_only=False)
        assert saved['protocol']==protocol
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'].cpu());rng.setstate(saved['legacy_rng'])
        for task,stream in streams.items():stream.setstate(saved['sampling_streams'][task])
        sampled=saved['sampled_example_ids'];start_step=saved['step']
        state.update(resume_state=str(args.resume_state),restored_step=start_step)
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for step in range(start_step+1,args.steps+1):
        task=task_indices[(step-1)%len(task_indices)]
        selected=(streams[task] if args.per_task_sampling else rng).choice(by_task[task])
        sampled[task].append(selected['id'])
        e={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in selected.items()};e['class_text']=classes
        y=model(e);n=len(e['candidates'])
        loss=F.cross_entropy(y['match'],e['target']) if n else y['match'].sum()*0
        if n:
            loss+=.3*F.binary_cross_entropy_with_logits(y['update'],e['update_target'])
            loss+=.1*F.cross_entropy(y['reliability'],e['reliability_target'])
        if len(e['history']):loss+=.3*F.cross_entropy(y['life'],e['life_target'])
        valid=e['class_target']>=0
        class_loss=F.cross_entropy(y['category'][valid],e['class_target'][valid]) if valid.any() else y['category'].sum()*0
        member_loss=F.binary_cross_entropy_with_logits(y['membership'],e['membership_target']) if n else y['membership'].sum()*0
        loss+=class_loss+.5*member_loss
        if e['native_teacher'] and n:
            loss+=.2*-(e['teacher']*F.log_softmax(y['match'],-1)).sum(-1).mean()
        optimizer.zero_grad();loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert torch.isfinite(loss) and gradient>0
        optimizer.step()
        if args.device=='cpu' and (step%100==0 or step==args.steps):
            temporary=out/'latest_training_state.tmp'
            torch.save({'step':step,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'sampling_streams':{k:v.getstate() for k,v in streams.items()},
                'legacy_rng':rng.getstate(),'selected_tasks':args.tasks,'sampled_example_ids':sampled,
                'protocol':protocol},temporary)
            temporary.replace(out/'latest_training_state.pt')
        if step==1 or step%100==0:
            record={'step':step,'task':task,'loss':float(loss),'class_loss':float(class_loss),
                'membership_loss':float(member_loss),'gradient_norm':gradient,'elapsed_seconds':time.perf_counter()-start};logs.append(record);print(json.dumps(record),flush=True)
    model.eval();checkpoint={'architecture':'set','semantic':True,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'steps':args.steps,'seed':0,'tasks':args.tasks,'Base_only_OV_supervision':True,'text_features':'real original OneJev frozen token mean; not language decoder reasoning'}
    torch.save(checkpoint,out/'checkpoint.pt');changed={k:int((initial[k]!=v.detach().cpu()).sum()) for k,v in model.state_dict().items()}
    fresh=CompactCore(semantic=True).to(args.device);fresh.load_state_dict(torch.load(out/'checkpoint.pt',weights_only=True)['state_dict']);fresh.eval();deltas={}
    for task in range(3):
        e={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in by_task[task][0].items()};e['class_text']=classes
        with torch.no_grad():a=model(e);b=fresh(e)
        delta=max(float((a[k]-b[k]).abs().max()) for k in a if a[k].numel());assert delta==0;deltas[str(task)]=delta
    assert all(changed[k]>0 for k in ('category.weight','membership.0.weight','text.1.weight','visual.1.weight'))
    state.update(status='SHARED_COMPACT_JOINT1200_TRAINING_COMPLETE' if args.steps==1200 else 'SHARED_COMPACT_TRAINING_COMPLETE',steps=args.steps,
        parameters=sum(p.numel() for p in model.parameters()),changed_elements=sum(changed.values()),
        changed_semantic_head_elements={k:v for k,v in changed.items() if k.startswith(('text.','category.','membership.'))},
        three_task_checkpoint_reload_max_difference=deltas,peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
        sampled_example_ids={str(k):v for k,v in sampled.items()},
        checkpoint=str(out/'checkpoint.pt'),checkpoint_bytes=(out/'checkpoint.pt').stat().st_size,
        training_log=logs,official_tracking_class_referring_results='pending same native proposals and official evaluation');save()


if __name__=='__main__':main()
