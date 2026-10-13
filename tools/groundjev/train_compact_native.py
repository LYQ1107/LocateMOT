"""GT-dominant native Full distillation and parameter-matched controls."""
import json
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
    torch.set_num_threads(4)
    source=ROOT/'outputs/groundjev/compact/native_pack_v1'
    receipt=json.loads((source/'owner_state.json').read_text())
    assert receipt['status']=='REAL_NATIVE_COMPACT_TRAIN_PACK_COMPLETE'
    examples=torch.load(source/'examples.pt',weights_only=True)
    out=ROOT/'outputs/groundjev/compact/native300_v1';out.mkdir(parents=True,exist_ok=False)
    state={'status':'TRAINING_NATIVE_COMPACT_AND_MATCHED_CONTROLS','steps_per_model':300,
        'seed':0,'examples':len(examples),'MOT_only':True,'completed':{}}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    save(); counts={}
    for name,architecture,alpha in [('compact_distilled','set',.2),('set_gt','set',0.),('mlp_distilled','mlp',.2)]:
        torch.manual_seed(0);rng=random.Random(0)
        model=CompactCore(architecture).cuda();model.train()
        counts[name]=sum(p.numel() for p in model.parameters())
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        logs=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for step in range(1,301):
            e={k:v.cuda() if torch.is_tensor(v) else v for k,v in examples[rng.randrange(len(examples))].items()}
            y=model(e)
            loss=F.cross_entropy(y['match'],e['target'])
            loss+=.3*F.binary_cross_entropy_with_logits(y['update'],e['update_target'])
            loss+=.1*F.cross_entropy(y['reliability'],e['reliability_target'])
            if len(e['history']):loss+=.3*F.cross_entropy(y['life'],e['life_target'])
            teacher=-(e['teacher']*F.log_softmax(y['match'],-1)).sum(-1).mean()
            teacher+=.3*F.binary_cross_entropy_with_logits(y['update'],e['update_teacher'])
            teacher+=.1*-(e['reliability_teacher']*F.log_softmax(y['reliability'],-1)).sum(-1).mean()
            mask=e['life_teacher_mask']
            if mask.any():teacher+=.3*-(e['life_teacher'][mask]*F.log_softmax(y['life'][mask],-1)).sum(-1).mean()
            loss=loss+alpha*teacher
            optimizer.zero_grad();loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            assert torch.isfinite(loss) and gradient>0
            optimizer.step()
            if step==1 or step%50==0:
                row={'model':name,'step':step,'loss':float(loss),'native_teacher_loss':float(teacher),
                     'gradient_norm':gradient,'elapsed_seconds':time.perf_counter()-start};logs.append(row);print(json.dumps(row),flush=True)
        changed=sum(int((initial[k]!=v.detach().cpu()).sum()) for k,v in model.state_dict().items())
        assert changed>0
        model.eval();e={k:v.cuda() if torch.is_tensor(v) else v for k,v in next(x for x in examples if len(x['history'])>1).items()}
        with torch.no_grad():before=model(e)
        checkpoint={'architecture':architecture,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'steps':300,'seed':0,'teacher_coefficient':alpha,'scope':'legal48frameMOTtrainingonly','native_feature_dimension':2560}
        torch.save(checkpoint,out/(name+'.pt'))
        fresh=CompactCore(architecture).cuda();fresh.load_state_dict(torch.load(out/(name+'.pt'),weights_only=True)['state_dict']);fresh.eval()
        with torch.no_grad():after=fresh(e)
        delta=max(float((before[k]-after[k]).abs().max()) for k in before if before[k].numel());assert delta==0
        # Dynamic sets must follow arbitrary candidate and identity ordering.
        cp=torch.randperm(len(e['candidates']),device='cuda');hp=torch.randperm(len(e['history']),device='cuda')
        perm=dict(e)
        for k in ('candidates','candidate_geo'):perm[k]=e[k][cp]
        for k in ('history','history_geo'):perm[k]=e[k][hp]
        perm['pair_geometry']=e['pair_geometry'][cp][:,hp]
        with torch.no_grad():p=fresh(perm)
        expected=before['match'][cp][:,torch.cat([hp,torch.arange(len(hp),len(hp)+3,device='cuda')])]
        permutation_error=max(float((p['match']-expected).abs().max()),float((p['life']-before['life'][hp]).abs().max()))
        assert permutation_error<1e-4
        state['completed'][name]={'parameters':counts[name],'steps':300,'changed_elements':changed,
            'checkpoint_reload_max_difference':delta,'candidate_identity_permutation_max_difference':permutation_error,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint':str(out/(name+'.pt')),
            'checkpoint_bytes':(out/(name+'.pt')).stat().st_size,'training_log':logs,'Full_teacher_coefficient':alpha};save()
        del model,fresh;torch.cuda.empty_cache()
    assert max(counts.values())/min(counts.values())<1.05
    state.update(status='REAL_NATIVE_COMPACT_MATCHED_CONTROL_TRAINING_COMPLETE',
        parameter_counts=counts,matched_parameter_ratio=max(counts.values())/min(counts.values()),
        official_tracking_results='pending actual causal video inference and TrackEval',
        probability_semantics='trained structured heads, not native OneJev decoder; set-only capabilities are controlled');save()


if __name__=='__main__':main()
