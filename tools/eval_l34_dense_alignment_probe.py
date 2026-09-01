#!/usr/bin/env python3
"""Fixed-calibration evaluation for the isolated L34 region alignment probe."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT'); sys.path.insert(0,str(ROOT))
from locatemot.models.l34_dense_alignment_probe import L34DenseAlignmentProbe
from tools.eval_l25_token_probe import tokens_for
from tools.train_l23_dense_correspondence import fixed_refs
from tools.train_rmot_candidate_scorer import load_bank, load_metadata, make_refs
MANIFEST=ROOT/'outputs/l19/protocol/kitti_fast_eval_manifest.json'; BANK=ROOT/'outputs/l25/candidate_bank_v4'; WEIGHTS=Path('/home/lwr/.cache/clip/ViT-B-16.pt'); CK=ROOT/'outputs/l34/train/alignment_probe_smoke100_corrected2/checkpoint_l34_alignment_step100.pt'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score_refs(model,refs,banks,qcache,device):
 out=[]
 for ref in refs:
  t=banks[ref['video']]['tensors']; sl=slice(ref['begin'],ref['end']); q=qcache[int(ref['query_index'])].to(device); r=t['dense_roi_tokens_v4'][sl].float().to(device); c=t['roi_sample_points_v4'][sl].float().to(device)
  with torch.inference_mode(): s=model(q,r,c)['region_logits'].float().cpu().numpy()
  out.append({'frame':int(ref['frame_id']),'score':s,'label':ref['positive'].astype(bool),'pool':t['pool_id'][sl].numpy()})
 return out
def threshold_cal(refs):
 values=np.concatenate([x['score'] for x in refs]); candidates=np.unique(np.quantile(values,np.linspace(.01,.995,128))); best=None
 for t in candidates:
  tp=fp=fn=0; ff=[]
  for x in refs:
   sel=x['score']>=t; y=x['label']; tp+=int((sel&y).sum()); fp+=int((sel&~y).sum()); fn+=int((~sel&y).sum()); ff.append(int((sel&~y).sum()))
  f=2*tp/max(1,2*tp+fp+fn); item=(f,float(t),tp,fp,fn,float(np.mean(ff)))
  if best is None or item[0]>best[0]: best=item
 return {'threshold':best[1],'frame_f1':best[0],'tp':best[2],'fp':best[3],'fn':best[4],'fp_per_frame':best[5],'source':'calibration_only'}
def metrics(refs,threshold):
 tp=fp=fn=0; top1=[];top5=[];strict=[];best=[];multi=[];null_accept=0;empty=0;fps=[];source={0:[0,0],1:[0,0]}
 for x in refs:
  s,y,pool=x['score'],x['label'],x['pool']; sel=s>=threshold; tp+=int((sel&y).sum());fp+=int((sel&~y).sum());fn+=int((~sel&y).sum());fps.append(int((sel&~y).sum()));empty+=int(not sel.any());null_accept+=int(not y.any() and sel.any()); order=np.argsort(-s,kind='stable')
  if y.any():
   top1.append(float(y[order[:1]].any()));top5.append(float(y[order[:5]].any()));pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
   if len(neg): strict.append(float(s[pos].min()-s[neg].max()));best.append(float(s[pos].max()-s[neg].max()))
   if len(pos)>1: multi.append((float(y[order[:1]].any()),float(y[order[:5]].any())))
  for sid in (0,1):
   rows=np.flatnonzero(pool==sid)
   if len(rows): source[sid][0]+=1;source[sid][1]+=int(y[rows[np.argmax(s[rows])]])
 def st(v): return {'count':len(v),'mean':float(np.mean(v)) if v else None,'median':float(np.median(v)) if v else None}
 return {'frame_units':len(refs),'rows':int(sum(len(x['label']) for x in refs)),'positive_rows':int(sum(x['label'].sum() for x in refs)),'selected':int(tp+fp),'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'fp_per_frame':float(np.mean(fps)),'empty_rate':empty/max(1,len(refs)),'null_false_acceptance':null_accept/max(1,len(refs)),'top1':float(np.mean(top1)) if top1 else None,'top5':float(np.mean(top5)) if top5 else None,'multi_positive_units':len(multi),'multi_positive_top1':float(np.mean([z[0] for z in multi])) if multi else None,'multi_positive_top5':float(np.mean([z[1] for z in multi])) if multi else None,'strict_min_positive_margin':st(strict),'best_positive_margin':st(best),'hard_violation':float(np.mean(np.asarray(strict)<0)) if strict else None,'source_top1_precision':{'main':source[0][1]/max(1,source[0][0]),'reserve':source[1][1]/max(1,source[1][0])},'identity_switches':'N/A at candidate-only probe; no tracker emission'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out-root',required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args();out=Path(a.out_root);out=out if out.is_absolute() else ROOT/out
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True); rows=sorted(json.loads(MANIFEST.read_text())['queries'],key=lambda x:int(x['query_index'])); meta=load_metadata(); vids=sorted({str(q['video']) for q in rows}); banks={v:load_bank(BANK/'kitti'/f'{v}.pt') for v in vids}; refs=make_refs(rows,meta,banks); cal=fixed_refs([r for r in refs if r['split']=='calibration'],6000,17); screen=fixed_refs([r for r in refs if r['split']=='screening'],100,18)
 import clip;device=torch.device(a.device);cm,_=clip.load(str(WEIGHTS),device=device);cm.eval();qcache={int(q['query_index']):tokens_for(cm,str(q['expression']),device).cpu() for q in rows};del cm
 model=L34DenseAlignmentProbe().to(device);model.load_state_dict(torch.load(CK,map_location=device,weights_only=False)['model']);model.eval(); cal_scores=score_refs(model,cal,banks,qcache,device);screen_scores=score_refs(model,screen,banks,qcache,device);thr=threshold_cal(cal_scores)
 payload={'format':'locatemot-l34-alignment-probe-heldout-v1','manifest':str(MANIFEST.resolve()),'manifest_sha256':sha(MANIFEST),'checkpoint':str(CK.resolve()),'weights':str(WEIGHTS),'weights_sha256':sha(WEIGHTS),'query_counts':{'calibration':64,'screening':96},'units':{'calibration':len(cal),'screening':len(screen),'screening_selection':'fixed_refs(seed=18,count=100)'},'calibration_threshold':thr,'screening_metrics':metrics(screen_scores,thr['threshold']),'screening_gt_used_for_threshold':False,'semantic_inputs_excluded':['pool_id','source_id','group_id','state_key'],'alignment_mask_verified':False,'no_tracker_or_trackeval':True}
 (out/'heldout_100.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__':main()
