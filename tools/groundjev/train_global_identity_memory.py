"""A3 true assigned-ID write gradients, source-actor replay and TRAIN calibration."""
import argparse
from collections import Counter
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from locatemot.groundjev.global_identity_runtime import GlobalIdentityDecision
    from locatemot.groundjev.global_identity_core import physical_inputs
    from locatemot.groundjev.global_competition import sparse_competition
    from tools.groundjev.global_identity_data import training_frames
    from tools.groundjev.compact_reference_data import CompactReferenceInputs,geometry
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    settings=json.loads(args.plan.read_text());plan=json.loads((ROOT/settings['physical_plan']).read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4);torch.manual_seed(settings['seed']);torch.use_deterministic_algorithms(True)
    source=Path(plan['immutable_source_root']);out=Path(settings['output'])
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    decision=GlobalIdentityDecision(settings['physical_checkpoint']);model=decision.model
    initial={k:v.clone() for k,v in model.state_dict().items()}
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.write.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.write.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    index=json.loads((source/plan['source_index']).read_text())
    calibration=set(sorted(index['units'])[::5]);counts=Counter();steps=0;first=[]
    class References(CompactReferenceInputs):
        @lru_cache(maxsize=4)
        def frame_features(self,unit,frame):
            info=self.units[unit]
            with np.load(source/info['frames'][str(frame)]['feature_path'],allow_pickle=False) as file:features=file['regions']
            assert features.dtype==np.float32;return features
    references=References(source/Path(plan['source_index']).parent);references.text=decision.text_features
    component=source/settings['RMOT_component']
    with gzip.open(component/'physical_refs.jsonl.gz','rt') as f:rmot=list(map(json.loads,f))
    with gzip.open(component/'groups.jsonl.gz','rt') as f:groups={r['id']:r for r in map(json.loads,f)}
    assert len(rmot)==2882 and sum(r['task']=='assigned_identity_memory_update' for r in rmot)==1454
    registry={}
    def write_logits(prediction,e,rows,columns,old_update,history_geo):
        confidences=[];margins=[];extra=[]
        for r,c,update in zip(rows,columns,old_update):
            logits=prediction['logits'][r];rivals=torch.cat([logits[:c],logits[c+1:]])
            confidences.append(logits.softmax(0)[c]);margins.append(logits[c]-rivals.max())
            cosine=F.cosine_similarity(e['candidates'][r],e['history'][c,2],dim=0)
            extra.append(torch.stack([torch.as_tensor(update),history_geo[c,4],history_geo[c,5],cosine]))
        return model.assigned_write(prediction,torch.tensor(rows),torch.tensor(columns),
            torch.stack(confidences),torch.stack(margins),torch.stack(extra))
    def batches():
        for frame in training_frames(plan):
            unit=frame['unit']
            if unit not in registry:
                path=Path(index['units'][unit]['full_source_rows']).parent/'candidate_registry.jsonl.gz'
                with gzip.open(path,'rt') as f:registry[unit]={(r['frame'],r['candidate']):r for r in map(json.loads,f)}
            by_id={t.track_id:i for i,t in enumerate(frame['tracks'])};rows=[];columns=[];targets=[]
            for i in range(len(frame['candidates'])):
                r=registry[unit][frame['frame'],i];tid=r['actual_actor_identity']
                if r['assigned_update_target'] is None or tid not in by_id or frame['labels']['unique_GT'][i]<0:continue
                if frame['labels']['birth_anchors'].get(tid) is None:continue
                rows.append(i);columns.append(by_id[tid]);targets.append(float(r['assigned_update_target']=='yes'))
            if not rows:
                yield unit,None,None,dict(source='mot',frames=1,masked_frame=True);continue
            e=physical_inputs(frame['image_size'],frame['candidates'],frame['tracks'],frame['timestamp'],32)
            with torch.no_grad():
                prediction=model(e)
                baseline,_,_=decision.semantic.decide(SimpleNamespace(size=frame['image_size']),frame['candidates'],
                    frame['tracks'],frame['timestamp'],dict(task='mot',vocabulary=['person'],expression=None))
            logits=write_logits(prediction,e,rows,columns,[baseline[i]['update_probability'] for i in rows],e['history_geo'])
            yield unit,logits,torch.tensor(targets),dict(source='mot',frames=1,questions=len(rows))
        for row in rmot:
            unit='rmot/'+row['source'];group=groups[row['groundjev_group_question_ref']['group']]
            state=json.loads(group['state']) if isinstance(group['state'],str) else group['state']
            e=references.inputs(unit,group['id']);w,h=group['media'][0]['w'],group['media'][0]['h']
            check=geometry(state,w,h)
            assert all(torch.equal(e[k],check[k]) for k in check)
            edges,edge_geo=sparse_competition([c['box'] for c in state['candidates']],e['candidates'].numpy(),32)
            e.update(edges=torch.from_numpy(edges),edge_geometry=torch.from_numpy(edge_geo))
            with torch.no_grad():prediction=model(e);old=decision.semantic.model(e)['update'].sigmoid()
            if row['task']=='identity':
                target=next(iter(row['target']));identities=[str(t['id']) for t in state['historical_tracks']]
                assert target in identities
                yield unit,None,None,dict(source='rmot',identity_reference=1);continue
            local=int(row['groundjev_group_question_ref']['question'].rsplit('_',1)[1])
            selected=state['final_assignment']['selected_existing_identity_by_candidate'][str(local)]
            column=[t['id'] for t in state['historical_tracks']].index(selected)
            logits=write_logits(prediction,e,[local],[column],[old[local]],e['history_geo'])
            target=next(iter(row['target']));assert target in ('yes','no')
            yield unit,logits,torch.tensor([float(target=='yes')]),dict(source='rmot',questions=1,paired_native_reference=True)
    start_epoch=0;completed_batch=-1
    if args.resume:
        saved=torch.load(out/'latest_write_training_state.pt',map_location='cpu',weights_only=True)
        assert saved['settings']==settings
        model.write.load_state_dict(saved['write']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);steps=saved['steps'];counts.update(saved['counts']);first=saved['first5']
        start_epoch=saved['epoch'];completed_batch=saved['completed_batch']
    def save_partial(epoch,batch):
        path=out/'latest_write_training_state.tmp'
        torch.save(dict(write=model.write.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            steps=steps,counts=dict(counts),first5=first,settings=settings,epoch=epoch,completed_batch=batch),path)
        path.replace(out/'latest_write_training_state.pt')
    with (out/'log.jsonl').open('a' if args.resume else 'w') as log:
        for epoch in range(start_epoch,settings['epochs']):
            for batch_number,(unit,logits,target,meta) in enumerate(batches()):
                if epoch==start_epoch and batch_number<=completed_batch:continue
                for k,v in meta.items():
                    if isinstance(v,(int,bool)):counts[meta['source']+':'+k]+=int(v)
                if logits is None or unit in calibration:continue
                optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(logits,target);assert torch.isfinite(loss)
                loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.write.parameters(),settings['gradient_clip']))
                assert np.isfinite(gradient)
                # Saturated correct BCE batches can have exactly zero gradient.
                # A real path is proved by the first updates and changed weights.
                if steps<5:assert gradient>0
                optimizer.step();steps+=1
                counts['finite_zero_gradient_fitted_batches']+=gradient==0
                counts['fitted_questions']+=len(target);counts['yes']+=int(target.sum());counts['no']+=len(target)-int(target.sum())
                if steps<=5 or steps%250==0:
                    record=dict(epoch=epoch,step=steps,unit=unit,loss=float(loss.detach()),gradient=gradient,questions=len(target))
                    if len(first)<5:first.append(record)
                    log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                if steps%250==0:save_partial(epoch,batch_number)
            save_partial(epoch,batch_number)
    held_logits=[];held_targets=[]
    for unit,logits,target,meta in batches():
        if logits is not None and unit in calibration:held_logits.append(logits.detach());held_targets.append(target)
    x=torch.cat(held_logits);y=torch.cat(held_targets);assert len(x)>0 and 0<y.sum()<len(y)
    raw_scale=torch.tensor(0.,requires_grad=True);bias=torch.tensor(0.,requires_grad=True)
    calibrator=torch.optim.Adam([raw_scale,bias],lr=settings['calibration_learning_rate'])
    for _ in range(settings['calibration_optimizer_steps']):
        calibrator.zero_grad();calibrated=x*raw_scale.exp()+bias
        loss=F.binary_cross_entropy_with_logits(calibrated,y);loss.backward();calibrator.step()
    scale=float(raw_scale.exp().detach());offset=float(bias.detach());assert scale>0
    with torch.no_grad():model.write[-1].weight.mul_(scale);model.write[-1].bias.mul_(scale).add_(offset)
    assert all(torch.isfinite(t).all() for t in model.write.state_dict().values())
    assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items() if not k.startswith('write.'))
    payload=torch.load(settings['physical_checkpoint'],map_location='cpu',weights_only=True)
    payload.update(arm='A3',state_dict=model.state_dict(),assignment_conditioned_memory_trained=True,
        memory_settings=settings,memory_actual_updates=steps,write_calibration=dict(scale=scale,intercept=offset))
    torch.save(payload,out/'checkpoint.pt')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),steps=steps,
        settings=settings,counts=dict(counts)),out/'memory_training_state.pt')
    report=dict(status='A3_ASSIGNED_ID_MEMORY_HEAD_AND_TRAIN_ONLY_CALIBRATION_COMPLETE',actual_updates=steps,
        counts=dict(counts),first5=first,trainable_memory_parameters=sum(p.numel() for p in model.write.parameters()),
        all_A2_physical_parameters_bit_exact=True,all2882_RMOT_original_inputs_and_targets_referenced=True,
        RMOT_identity_references_per_epoch=1428,RMOT_assigned_write_references_per_epoch=1454,
        calibration_units=sorted(calibration),calibration_questions=len(y),calibration_positives=int(y.sum()),
        calibration=dict(scale=scale,intercept=offset,before_BCE=float(F.binary_cross_entropy_with_logits(x,y)),
            after_BCE=float(F.binary_cross_entropy_with_logits(x*scale+offset,y))),
        source_actor='Original real Shared Compact causal histories/actual final assignments, off-policy for A2; not own-state A2 rollout or BPTT.',
        same_final_checkpoint_for_all_tasks=str(out/'checkpoint.pt'),new_GPU_processes=0,new_feature_bank_bytes=0,
        tracking_improvement_or_actual_write_flips='pending full trajectory controls',full_goal_complete=False)
    (out/'training_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
