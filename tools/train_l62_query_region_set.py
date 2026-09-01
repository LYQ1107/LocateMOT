#!/usr/bin/env python3
"""L62 fit-only smoke for per-level fused-ROI query/set correspondence."""
from __future__ import annotations
import argparse, gc, hashlib, json, random, time, traceback
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve(); DATA=ROOT/'outputs/l49/data'; UNITS=DATA/'train_units.jsonl'; MANIFEST=ROOT/'outputs/l19/protocol/kitti_fast_eval_manifest.json'; PYTHON=Path('/home/lwr/anaconda3/envs/masaenv_debug/bin/python')
EXPECTED_MANIFEST='06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'
import sys; sys.path.insert(0,str(ROOT))
from locatemot.models.l62_query_region_set import L62QueryRegionSet
from tools.l59_fused_common import build_detector, detector_provenance, stream_fused_roi, sha256, WEIGHT

def load_units(): return [json.loads(x) for x in UNITS.read_text().splitlines() if x.strip() and (u:=json.loads(x)).get('split')=='fit' and u.get('dataset') in ('refer_kitti_v1','refer_kitti_v2')]
def ordered(units, seed):
    rng=random.Random(seed); buckets={}
    for u in units: buckets.setdefault((u['dataset'],u.get('category','unknown')),[]).append(u)
    for b in buckets.values(): rng.shuffle(b)
    keys=[(d,c) for d in ('refer_kitti_v1','refer_kitti_v2') for c in ('positive','multi_positive','inactive','present_uncovered')]; out=[]
    while any(buckets.get(k) for k in keys):
        for k in keys:
            if buckets.get(k): out.append(buckets[k].pop(0))
    rest=[u for b in buckets.values() for u in b]; rng.shuffle(rest); return out+rest

def loss_fn(out, target):
    s=out['relevance_logit']; z=s.new_zeros(()); pos=torch.where(target)[0]; neg=torch.where(~target)[0]
    bce_parts=[]
    if len(pos): bce_parts.append(F.binary_cross_entropy_with_logits(s[pos],torch.ones_like(s[pos])))
    if len(neg): bce_parts.append(F.binary_cross_entropy_with_logits(s[neg],torch.zeros_like(s[neg])))
    bce=torch.stack(bce_parts).mean() if bce_parts else z
    hard=neg[torch.argsort(s.detach()[neg],descending=True)[:min(24,len(neg))]] if len(neg) else neg
    pair=F.softplus(.2+s[hard][None,:]-s[pos][:,None]).mean() if len(pos) and len(hard) else z
    listwise=torch.logsumexp(s,0)-torch.logsumexp(s[pos],0) if len(pos) else z
    minimum=F.softplus(.2+s[hard].max()-s[pos]).mean() if len(pos) and len(hard) else z
    inactive=F.binary_cross_entropy_with_logits(s,torch.zeros_like(s)) if not len(pos) else z
    null=F.binary_cross_entropy_with_logits(out['null_logit'],s.new_tensor(float(not len(pos))))
    calibration=((s.sigmoid()-target.float())**2).mean()
    total=bce+.75*pair+.5*listwise+.5*minimum+.15*inactive+.2*null+.1*calibration
    return total, {'total':float(total.detach()),'bce':float(bce.detach()),'hard_pairwise':float(pair.detach()),'multi_positive_listwise':float(listwise.detach()),'minimum_positive':float(minimum.detach()),'inactive':float(inactive.detach()),'null':float(null.detach()),'calibration':float(calibration.detach()),'positive_count':int(len(pos)),'negative_count':int(len(neg)),'hard_negative_count':int(len(hard))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-root',required=True); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--seed',type=int,default=20260829); a=ap.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong cwd')
    if sha256(MANIFEST)!=EXPECTED_MANIFEST: raise RuntimeError('manifest SHA mismatch')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); random.seed(a.seed); torch.manual_seed(a.seed)
    units=ordered(load_units(),a.seed); detector,cfg,loaded=build_detector(); detector.eval()
    if any(p.requires_grad for p in detector.parameters()): raise AssertionError('detector not frozen')
    model=L62QueryRegionSet().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    trace=[]; counts=Counter(); finite=nonzero=0; start=time.time(); peak=0
    progress_path = out / 'progress.jsonl'
    progress_path.write_text('')
    progress_handle = progress_path.open('a')
    try:
      for step in range(1,a.steps+1):
        u=units[(step-1)%len(units)]; bank=torch.load(u['bank_path'],map_location='cpu',weights_only=False); roi,text,tmask,numeric,meta=stream_fused_roi(detector,u,bank)
        n=int(u['end'])-int(u['begin']); assert meta['candidate_count']==n and len(meta['candidate_keys'])==n
        target=torch.zeros(n,dtype=torch.bool,device='cuda:0'); pi=torch.as_tensor(u['positive_indices'],dtype=torch.long,device='cuda:0')
        if len(pi): assert int(pi.min())>=0 and int(pi.max())<n
        target[pi]=True; pred=model(roi,text,tmask,numeric); total,parts=loss_fn(pred,target)
        if not torch.isfinite(total): raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True); total.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        if not torch.isfinite(grad): raise RuntimeError('nonfinite gradient')
        opt.step(); finite+=1; nonzero+=int(float(grad)>0)
        counts.update({'units':1,'candidate_rows':n,'positive':parts['positive_count'],'multi_positive':int(u.get('category')=='multi_positive'),'inactive':int(u.get('category')=='inactive'),'present_uncovered':int(u.get('category')=='present_uncovered'),'domain_'+u['dataset']:1,'video_'+str(u['video']):1})
        parts.update({'step':step,'unit_key':u['unit_key'],'dataset':u['dataset'],'video':u['video'],'category':u.get('category'),'gradient_norm':float(grad),'candidate_key_drift':0,'candidate_truncation':False,'detector_frozen':True,'roi_shape':list(roi.shape),'text_shape':list(text.shape),'null_logit':float(pred['null_logit'].detach()),'representation':'post-fusion per-level ROI tokens'}); trace.append(parts); progress_handle.write(json.dumps(parts)+'\n'); progress_handle.flush(); peak=max(peak,int(torch.cuda.max_memory_allocated())); del bank,roi,text,tmask,numeric,pred,total
    except Exception as exc:
      (out / 'INCOMPLETE.md').write_text('# INCOMPLETE — L62 smoke\n\n' + traceback.format_exc() + '\n')
      raise
    progress_handle.close()
    ck=out/f'checkpoint_l62_step{a.steps}.pt'; torch.save({'format':'locatemot-l62-query-region-set-v1','step':a.steps,'seed':a.seed,'model':model.state_dict(),'config':{'hidden':128,'levels':4,'points_per_level':16,'numeric_dim':24,'representation':'post-fusion per-level tokens; no pre-cross-attention mean','same_class_hard_negative_metadata':'unavailable; all-negative fallback'},'detector_frozen':True,'labels_split':'fit'},ck)
    reload=L62QueryRegionSet().cuda(); reload.load_state_dict(torch.load(ck,map_location='cuda:0',weights_only=False)['model'],strict=True)
    info={'interpreter':str(PYTHON),'torch':torch.__version__,'cuda':torch.version.cuda,'config_sha256':sha256(Path(cfg.filename)) if hasattr(cfg,'filename') and cfg.filename and Path(cfg.filename).exists() else None,'weight_sha256':sha256(WEIGHT),'weight_path':str(WEIGHT),'missing_keys':loaded.get('missing_keys',[]) if isinstance(loaded,dict) else [],'unexpected_keys':loaded.get('unexpected_keys',[]) if isinstance(loaded,dict) else []}
    payload={'format':'locatemot-l62-query-region-set-metrics-v1','status':'complete','stage':'fit-only-smoke','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'seed':a.seed,'steps':a.steps,'finite_steps':finite,'nonzero_gradient_steps':nonzero,'checkpoint':str(ck),'checkpoint_reload':True,'train_split':'fit','fit_units_total':len(units),'sampling_counts':dict(counts),'domains_present':sorted({u['dataset'] for u in units}),'categories_seen':sorted({u.get('category') for u in units[:a.steps]}),'candidate_key_drift':0,'candidate_truncation':False,'persistent_raw_dense_cache_written':False,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'detector_frozen':all(not p.requires_grad for p in detector.parameters()),'adapter_parameter_count':sum(p.numel() for p in model.parameters()),'runtime':{'peak_memory_bytes':peak,'elapsed_sec':time.time()-start,'steps_per_sec':a.steps/max(time.time()-start,1e-9)},'detector':info,'loss_trace':trace}
    metrics_path = out / f'metrics_l62_step{a.steps}.json'
    (metrics_path).write_text(json.dumps(payload,indent=2,default=str)+'\n'); (out/'loss_trace.json').write_text(json.dumps(trace,indent=2)+'\n'); (out/'sampling_trace.json').write_text(json.dumps({'counts':dict(counts),'domains_present':payload['domains_present'],'categories_seen':payload['categories_seen']},indent=2)+'\n'); (out/'config.json').write_text(json.dumps(payload['detector']|{'seed':a.seed,'steps':a.steps,'stage':'fit-only-smoke','complete_candidate_set':True},indent=2)+'\n'); (out/'provenance.json').write_text(json.dumps({'cwd':str(ROOT),'seed':a.seed,'manifest_sha256':EXPECTED_MANIFEST,'train_units_sha256':sha256(UNITS),'fit_only':True,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_raw_dense_cache_written':False},indent=2)+'\n'); print(json.dumps({'status':'complete','metrics':str(metrics_path),'checkpoint':str(ck),'finite_steps':finite,'nonzero_gradient_steps':nonzero}),flush=True)
if __name__=='__main__': main()
