#!/usr/bin/env python3
"""Train/evaluate L34 on the independent train-only dense bank."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT'); sys.path.insert(0,str(ROOT))
from locatemot.models.l34_dense_alignment_probe import L34DenseAlignmentProbe
from tools.train_l28_track_set_decoder import build_queries
from tools.eval_l25_token_probe import tokens_for

BANK=ROOT/'outputs/l34/train_dense_bank_v1'; WEIGHTS=Path('/home/lwr/.cache/clip/ViT-B-16.pt'); MANIFEST=ROOT/'outputs/l19/protocol/kitti_fast_eval_manifest.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def refs_for(q, bank, frame):
 t=bank['tensors']; fi=int(np.searchsorted(t['frame_ids'].numpy(),frame)); b,e=int(t['frame_ptr'][fi]),int(t['frame_ptr'][fi+1]); labels=bank['candidate_gt'][b:e]; ids={str(x) for x in q['target'].get(int(frame),set())}; y=np.asarray([x is not None and str(x) in ids for x in labels],bool); return b,e,y
def features(bank,b,e):
    t=bank['tensors']; return t['dense_roi_tokens_v4'][b:e].float(), t['roi_sample_points_v4'][b:e].float()
def load_dense_bank(video):
 p=BANK/'kitti'/f'{video}.pt'; raw=torch.load(p,map_location='cpu',weights_only=False); t=raw['tensors']
 keep=('frame_ptr','frame_ids','dense_roi_tokens_v4','roi_sample_points_v4')
 labels=json.loads(p.with_suffix('.labels.json').read_text())['candidate_gt']
 return {'tensors':{k:t[k].contiguous() for k in keep},'candidate_gt':labels}
def balanced(logits,y):
 p=logits[y]; n=logits[~y]; z=logits.new_zeros(()); return ((F.binary_cross_entropy_with_logits(p,torch.ones_like(p)) if len(p) else z)+(F.binary_cross_entropy_with_logits(n,torch.zeros_like(n)) if len(n) else z))/max(1,int(bool(len(p)))+int(bool(len(n))))
def one_loss(model,q,qm,r,c,y,counters):
 out=model(q,r,c); z=out['region_logits']; y=torch.as_tensor(y,device=z.device,dtype=torch.bool); pos=torch.nonzero(y,as_tuple=False).flatten(); neg=torch.nonzero(~y,as_tuple=False).flatten(); hard=neg[torch.argsort(z.detach()[neg],descending=True)[:min(12,len(neg))]] if len(neg) else neg
 bce=balanced(z,y); pair=F.softplus(.5+z[hard][None,:]-z[pos][:,None]).mean() if len(pos) and len(hard) else z.new_zeros(()); setl=torch.logsumexp(z,0)-torch.logsumexp(z[pos],0) if len(pos) else z.new_zeros(()); null=F.binary_cross_entropy_with_logits(out['null_logit'],z.new_tensor(float(not y.any()))); total=bce+pair+.5*setl+.3*null
 counters.update(positive=int(y.sum()),negative=int((~y).sum()),hard_negative=int(len(hard)),multi_positive=int(len(pos)>1),null=int(not y.any())); return total,{'total':float(total.detach()),'bce':float(bce.detach()),'pairwise':float(pair.detach()),'multi_positive_set':float(setl.detach()),'null':float(null.detach())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out-root',required=True); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--seed',type=int,default=20260829); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args(); out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out.mkdir(parents=True,exist_ok=False); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 qs=build_queries(); complete=sorted(p.stem for p in (BANK/'kitti').glob('*.complete')); complete=[x for x in complete if (BANK/'kitti'/f'{x}.pt').exists()]; qs=[q for q in qs if q['video'] in complete]; banks={v:load_dense_bank(v) for v in complete};
 import clip; device=torch.device(a.device); cm,_=clip.load(str(WEIGHTS),device=device); cm.eval(); text_cache={}
 for q in qs:
  if int(q['text_index']) not in text_cache: text_cache[int(q['text_index'])]=tokens_for(cm,q['expression'],device).cpu().half()
 del cm; model=L34DenseAlignmentProbe().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); rng=np.random.default_rng(a.seed); trace=[]; grads=[]; counters=Counter(); start=time.time(); model.train()
 for step in range(a.steps):
  vals=[]
  for _ in range(2):
   q=qs[int(rng.integers(len(qs)))]; bank=banks[q['video']]; frames=bank['tensors']['frame_ids'].numpy(); frame=int(frames[int(rng.integers(len(frames)))]); b,e,y=refs_for(q,bank,frame); r,c=features(bank,b,e); qt=text_cache[int(q['text_index'])].to(device); val,part=one_loss(model,qt,torch.ones(qt.shape[0],device=device),r.to(device),c.to(device),y,counters); vals.append(val)
  loss=torch.stack(vals).mean(); opt.zero_grad(set_to_none=True); loss.backward(); grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5))); opt.step(); trace.append({'step':step+1,'loss':float(loss.detach()),'parts':part})
 ck=out/f'checkpoint_l34_alignment_step{a.steps}.pt'; cfg={'format':'locatemot-l34-alignment-probe-v1','steps':a.steps,'seed':a.seed,'train_only':True,'train_video_count':len(complete),'train_query_count':len(qs),'complete_train_videos':complete,'missing_train_videos':sorted(set(str(q['video']) for q in build_queries())-set(complete)),'bank':str(BANK.resolve()),'weights':str(WEIGHTS),'weights_sha256':sha(WEIGHTS),'manifest':str(MANIFEST.resolve()),'manifest_sha256':sha(MANIFEST),'screening_gt_used_for_fit':False,'input_schema':['word_level_clip_tokens','dense_roi_region_tokens','roi_coordinates'],'semantic_inputs_excluded':['pool_id','source_id','group_id','state_key'],'alignment_mask_verified':False,'counters':dict(counters),'loss_mean':float(np.mean([x['loss'] for x in trace])),'gradient_norm':{'mean':float(np.mean(grads)),'nonzero_steps':int(np.count_nonzero(np.asarray(grads)>0))},'elapsed_sec':time.time()-start}; torch.save({'model':model.state_dict(),'config':cfg},ck); reload=L34DenseAlignmentProbe().to(device); reload.load_state_dict(torch.load(ck,map_location=device,weights_only=False)['model']); cfg.update({'checkpoint':str(ck.resolve()),'checkpoint_reload':True}); (out/f'metrics_l34_alignment_step{a.steps}.json').write_text(json.dumps(cfg,indent=2)+'\n'); (out/'loss_trace.json').write_text(json.dumps(trace,indent=2)+'\n'); print(json.dumps(cfg,indent=2),flush=True)
if __name__=='__main__': main()
