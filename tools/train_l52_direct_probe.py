#!/usr/bin/env python3
"""L52 direct query-conditioned region/set probe, fit-only B/C smoke."""
from __future__ import annotations
import argparse, gc, json, math, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT'); DATA=ROOT/'outputs/l49/data'
TEXT=ROOT/'outputs/l48/data/text_cache.pt'; MANIFEST=ROOT/'outputs/l19/protocol/kitti_fast_eval_manifest.json'
L29=ROOT/'outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt'
sys.path.insert(0,str(ROOT))
from locatemot.models.l52_query_region_set_probe import L52QueryRegionSetProbe
from locatemot.rmot.l49_data import sha256_file
from tools.train_l49_kitti_rmot import L29Teacher
from tools.train_l51_streaming_crop_adapter import materialize_units
from tools.l52_streaming_data import CLIP_WEIGHTS,L52StreamingRegionEncoder

EXPECTED='06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'

def fit_units():
    rows=[json.loads(x) for x in (DATA/'train_units.jsonl').read_text().splitlines() if x.strip()]
    if not rows or any(x.get('split')!='fit' for x in rows): raise AssertionError('non-fit training row')
    return rows
def choose(rows,n,seed):
    groups=defaultdict(list)
    for x in rows: groups[(str(x['dataset']),str(x['category']))].append(x)
    rng=random.Random(seed)
    for v in groups.values(): v.sort(key=lambda x:(x['video'],int(x['frame_id']),int(x['query_id']))); rng.shuffle(v)
    order=[]; keys=sorted(groups)
    while any(groups.values()) and len(order)<n:
        for k in keys:
            if groups[k] and len(order)<n: order.append(groups[k].pop())
    if len(order)!=n: raise AssertionError('sampler count drift')
    return order
def bce(s,y):
    a=[]
    if y.any(): a.append(F.binary_cross_entropy_with_logits(s[y],torch.ones_like(s[y])))
    if (~y).any(): a.append(F.binary_cross_entropy_with_logits(s[~y],torch.zeros_like(s[~y])))
    return torch.stack(a).mean() if a else s.new_zeros(())
def loss(out,item):
    s=out['relevance_logit']; y=item['y'].to(s.device).bool(); pos=torch.where(y)[0]; neg=torch.where(~y)[0]
    z=s.new_zeros(())
    hard=neg[torch.argsort(s.detach()[neg],descending=True)[:min(24,len(neg))]] if len(neg) else neg
    pair=F.softplus(.2+s[hard][None,:]-s[pos][:,None]).mean() if len(pos) and len(hard) else z
    ls=torch.logsumexp(s,0)-torch.logsumexp(s[pos],0) if len(pos) else z
    mn=F.softplus(.2+s[hard].max()-s[pos]).mean() if len(pos) and len(hard) else z
    mb=bce(s,y); inactive=bce(s,torch.zeros_like(y)) if not y.any() else z
    total=mb+pair+.5*ls+.5*mn+.25*inactive
    return total,{'total':float(total.detach()),'membership_bce':float(mb.detach()),'hard_pairwise':float(pair.detach()),'multi_positive_listwise':float(ls.detach()),'minimum_positive':float(mn.detach()),'inactive':float(inactive.detach()),'positive_count':int(len(pos)),'negative_count':int(len(neg)),'hard_negative_count':int(len(hard))},pos,hard
def prov(rows,sel,args):
    return {'format':'locatemot-l52-direct-probe-v1','stage':'B/C','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'seed':args.seed,'train_manifest':str((DATA/'train_units.jsonl').resolve()),'train_manifest_sha256':sha256_file(DATA/'train_units.jsonl'),'fit_only':True,'fit_units_total':len(rows),'sampled_fit_units':len(sel),'sampled_domains':dict(Counter(x['dataset'] for x in sel)),'sampled_videos':sorted({x['video'] for x in sel}),'sampled_categories':dict(Counter(x['category'] for x in sel)),'text_cache':str(TEXT.resolve()),'text_cache_sha256':sha256_file(TEXT),'l29_checkpoint':str(L29.resolve()),'l29_checkpoint_sha256':sha256_file(L29),'fixed_manifest_sha256':sha256_file(MANIFEST),'clip_weights':str(CLIP_WEIGHTS.resolve()),'clip_weights_sha256':sha256_file(CLIP_WEIGHTS),'crop_contract':{'box_source':'L19 observation box','padding':.10,'boundary':'clip'},'complete_candidate_set':True,'raw_cache_written':False,'semantic_inputs_excluded':['source_id','pool_id','group_id','state_key','query_id'],'official_test_labels_read':False,'screening_gt_used':False,'calibration_labels_read':False,'validation_labels_read':False,'ordinary_mot_ovmot_touched':False,'token_span_region_alignment':'UNALIGNED','static_motion_language_mask':'UNALIGNED/not claimed'}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--out-root',required=True); p.add_argument('--steps',type=int,default=100); p.add_argument('--units',type=int,default=100); p.add_argument('--seed',type=int,default=20260829); p.add_argument('--device',default='cuda:0'); p.add_argument('--preflight-only',action='store_true'); a=p.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong project root')
    if sha256_file(MANIFEST)!=EXPECTED: raise RuntimeError('manifest SHA mismatch')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows=fit_units(); sel=choose(rows,a.units,a.seed); text=torch.load(TEXT,map_location='cpu',weights_only=False); teacher=L29Teacher(text,torch.device('cpu')); items=materialize_units(sel,text,teacher)
    model=L52QueryRegionSetProbe(); n=min(8,len(items[0]['y'])); patch=torch.randn(n,16,768); ctx=torch.randn(1,16,768); q=text['token_hidden'][items[0]['text_index']].float(); mask=text['attention_mask'][items[0]['text_index']].bool(); o=model(patch,ctx,q,mask,items[0]['frozen_clip'][:n],items[0]['numeric'][:n]); yy=torch.zeros(n,dtype=torch.bool); yy[:2]=True; dummy={'y':yy}; pre,parts,pos,hard=loss(o,dummy); pre.backward(); grads=[x.grad for x in model.parameters() if x.grad is not None];
    if not torch.isfinite(pre) or not any(float(x.abs().sum())>0 for x in grads): raise FloatingPointError('CPU preflight gradient failure')
    (out/'preflight.json').write_text(json.dumps({'status':'pass','model':model.config,'synthetic_patch_shape':list(patch.shape),'context_shape':list(ctx.shape),'text_shape':list(q.shape),'loss':parts,'nonzero_parameter_gradients':sum(float(x.abs().sum())>0 for x in grads),'candidate_set_truncated':False,'fit_only':True},indent=2)+'\n')
    if a.preflight_only: print(json.dumps(parts)); return
    dev=torch.device(a.device)
    if dev.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('authorized GPU required')
    enc=L52StreamingRegionEncoder(dev); model=L52QueryRegionSetProbe().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); (out/'config.json').write_text(json.dumps({'seed':a.seed,'steps':a.steps,'units':a.units,'device':str(dev),'precision':'FP32','model':model.config,'loss':'frame-balanced BCE + hard pairwise + multi-positive listwise + inactive','complete_candidate_set':True,'no_test_or_screening_labels':True},indent=2)+'\n'); (out/'provenance.json').write_text(json.dumps(prov(rows,sel,a),indent=2)+'\n')
    trace=[]; counts=Counter(); start=time.time()
    try:
        for step in range(1,a.steps+1):
            item=items[step-1] if step<=len(items) else items[(step*7919+a.seed)%len(items)]; patch,ctx,meta=enc.encode(item); q=text['token_hidden'][item['text_index']].float().to(dev); mask=text['attention_mask'][item['text_index']].bool().to(dev); o=model(patch.to(dev),ctx.to(dev),q,mask,item['frozen_clip'].to(dev),item['numeric'].to(dev)); total,part,pos,hard=loss(o,item); opt.zero_grad(set_to_none=True); total.backward(); gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.));
            if not torch.isfinite(total) or not math.isfinite(gn) or gn<=0: raise FloatingPointError(f'bad step {step}')
            opt.step(); trace.append({'step':step,**part,'gradient_norm':gn,'candidate_count':len(item['y']),'crop_count':meta['crop_count'],'unit_key':item['unit_key'],'finite':True}); counts.update({'domain_'+item['dataset']:1,'video_'+item['video']:1,'category_'+item['category']:1}); del patch,ctx,o
            if step%25==0: gc.collect(); torch.cuda.empty_cache()
        ck=out/f'checkpoint_l52_step{a.steps}.pt'; torch.save({'format':'locatemot-l52-direct-probe-v1','model':model.state_dict(),'model_config':model.config,'checkpoint_step':a.steps,'seed':a.seed,'provenance':prov(rows,sel,a)},ck); reload=L52QueryRegionSetProbe().to(dev); reload.load_state_dict(torch.load(ck,map_location=dev,weights_only=False)['model']); reload_ok=all(torch.isfinite(x).all().item() for x in reload.state_dict().values() if torch.is_floating_point(x));
        if not reload_ok: raise FloatingPointError('reload nonfinite')
        met={'format':'locatemot-l52-direct-probe-metrics-v1','status':'pass','step':a.steps,'finite_steps':len(trace),'nonzero_gradient_steps':sum(x['gradient_norm']>0 for x in trace),'loss_mean':{k:float(np.mean([x[k] for x in trace])) for k in ('total','membership_bce','hard_pairwise','multi_positive_listwise','minimum_positive','inactive')},'sampling_counts':dict(counts),'candidate_set_truncated':False,'raw_cache_written':False,'checkpoint':str(ck.resolve()),'checkpoint_sha256':sha256_file(ck),'reload_ok':reload_ok,'elapsed_sec':time.time()-start,'peak_memory_bytes':int(torch.cuda.max_memory_allocated(dev)),'train_only':True,'official_test_labels_read':False,'screening_gt_used':False}; (out/'loss_trace.json').write_text(json.dumps(trace,indent=2)+'\n'); (out/'sampling_trace.json').write_text(json.dumps(dict(counts),indent=2)+'\n'); (out/f'metrics_l52_step{a.steps}.json').write_text(json.dumps(met,indent=2)+'\n'); print(json.dumps(met))
    except Exception as e:
        (out/'INCOMPLETE.md').write_text(f'First error after {len(trace)} steps: {type(e).__name__}: {e}\n'); raise
if __name__=='__main__': main()
