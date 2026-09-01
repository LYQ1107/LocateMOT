"""Train/evaluate one L25 frame-level token correspondence stage."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT');sys.path.insert(0,str(ROOT))
from locatemot.models.rmot_l25_token_correspondence import L25TokenCorrespondence
from tools.eval_l25_token_probe import tokens_for
from tools.train_l23_dense_correspondence import fixed_refs
from tools.train_rmot_candidate_scorer import load_bank,load_metadata,make_refs,auc,average_precision,scalar_stats
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vals(ref,bank,device):
 t=bank['tensors'];sl=slice(ref['begin'],ref['end']);n=ref['end']-ref['begin'];q=None
 return {'roi':t['dense_roi_tokens_v4'][sl].float().to(device),'coords':t['roi_sample_points_v4'][sl].float().to(device),'context':torch.cat((t['dense_context_1p5_tokens_v4'][sl].float(),t['dense_context_3_tokens_v4'][sl].float()),1).to(device),'prev':t['dense_prev_roi_tokens_v4'][sl].float().to(device),'motion':t['motion_v2'][sl].float().to(device),'objectness':t['objectness'][sl].float().reshape(n).to(device)}
def score(model,q,a):
 n=a['roi'].shape[0];q_exp=q[None].expand(n,-1,-1);teacher=None
 if model.stage=='F2':
  teacher=torch.einsum('nld,nkd->nlk',F.normalize(q_exp,dim=-1),F.normalize(a['roi'],dim=-1)).amax((1,2))
 return model(q_exp,a['roi'],a['coords'],a['context'],a['prev'],a['motion'],teacher)
def choose(y,obj,s):
 neg=np.flatnonzero(~y);pre=neg[np.argsort(-obj[neg],kind='stable')[:min(96,len(neg))]];online=pre[np.argsort(-s[pre],kind='stable')[:min(24,len(pre))]] if len(pre) else pre;full=neg[np.argsort(-s[neg],kind='stable')[:min(24,len(neg))]] if len(neg) else neg;easy=np.setdiff1d(neg,online,assume_unique=False)[:16];return np.flatnonzero(y),online,easy,pre,full
def frame_loss(model,ref,bank,device,qcache):
 a=vals(ref,bank,device)
 q=qcache[ref['query_index']].to(device)
 with torch.inference_mode():
  preliminary=score(model,q,a).cpu().numpy()
 y=ref['positive'].astype(bool);pos,hard,easy,pre,full_hard=choose(y,a['objectness'].cpu().numpy(),preliminary)
 if model.stage=='F5': hard=full_hard
 s=score(model,q,a);z=s.new_zeros(());p=s[torch.as_tensor(pos,device=device)];h=s[torch.as_tensor(hard,device=device)];e=s[torch.as_tensor(easy,device=device)]
 pb=F.binary_cross_entropy_with_logits(p,torch.ones_like(p)) if len(p) else z;hb=F.binary_cross_entropy_with_logits(h,torch.zeros_like(h)) if len(h) else z;eb=F.binary_cross_entropy_with_logits(e,torch.zeros_like(e)) if len(e) else z;pair=F.softplus(1+h[None,:]-p[:,None]).mean() if len(p) and len(h) else z;violation=F.softplus(h[None,:]-p[:,None]).mean() if len(p) and len(h) else z;listwise=torch.logsumexp(s,0)-torch.logsumexp(p,0) if len(p) else z;total=pb+hb+.1*eb+pair+.5*listwise+.2*violation
 return total,{'total':float(total.detach()),'positive_bce':float(pb.detach()),'hard_bce':float(hb.detach()),'easy_bce':float(eb.detach()),'pairwise':float(pair.detach()),'listwise':float(listwise.detach()),'violation':float(violation.detach()),'positive_count':len(pos),'hard_count':len(hard),'easy_count':len(easy),'prefilter_count':len(pre)}
def train(model,refs,banks,device,qcache,steps,seed):
 train_refs=[r for r in refs if r['positive'].any()];rng=random.Random(seed);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);rows=[];grads=[];start=time.time();model.train()
 for _ in range(steps):
  chosen=[train_refs[rng.randrange(len(train_refs))] for _ in range(4)];ls=[];rs=[]
  for ref in chosen:
   l,r=frame_loss(model,ref,banks[ref['video']],device,qcache);ls.append(l);rs.append(r)
  l=torch.stack(ls).mean();opt.zero_grad(set_to_none=True);l.backward();grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5)));opt.step();rows.append({k:float(np.mean([r[k] for r in rs])) for k in rs[0]})
 return {'steps':steps,'positive_frame_units':len(train_refs),'elapsed_sec':time.time()-start,'loss':{k:scalar_stats([r[k] for r in rows]) for k in ('total','positive_bce','hard_bce','easy_bce','pairwise','listwise','violation')},'bucket_counts':{k:int(sum(r[k] for r in rows)) for k in ('positive_count','hard_count','easy_count','prefilter_count')},'gradient_norm':scalar_stats(grads)}
def evaluate(model,refs,banks,device,qcache):
 model.eval();ss=[];yy=[];marg=[];fullm=[];viol=[];miss=[];top1=top5=pf=mpf=mp1=mp5=zero=zp=0;null=[];source={0:[0,0,0],1:[0,0,0]}
 for ref in refs:
  a=vals(ref,banks[ref['video']],device);q=qcache[ref['query_index']].to(device)
  with torch.inference_mode():s=score(model,q,a).cpu().numpy()
  y=ref['positive'].astype(bool);ss.append(s);yy.append(y);zero+=int((s>=0).sum());zp+=int(((s>=0)&y).sum());pos,hard,easy,pre,full=choose(y,a['objectness'].cpu().numpy(),s)
  if not len(pos):null.append(float(s.max()) if len(s) else 0.);continue
  pf+=1;o=np.argsort(-s);top1+=int(y[o[:1]].any());top5+=int(y[o[:5]].any());
  if len(pos)>1:mpf+=1;mp1+=int(y[o[:1]].any());mp5+=int(y[o[:5]].any())
  if len(hard):marg.append(float(s[pos].min()-s[hard].max()));viol.append(marg[-1]<0);fullm.append(float(s[pos].min()-s[full].max()) if len(full) else marg[-1]);miss.append(int(len(full) and full[0] not in set(pre.tolist())))
  for sid in (0,1):
   pool=banks[ref['video']]['tensors']['pool_id'][ref['begin']:ref['end']].numpy()==sid;pr=np.flatnonzero(pool)
   if len(pr):so=pr[np.argsort(-s[pr])];source[sid][0]+=1;source[sid][1]+=int(y[so[:1]].any());source[sid][2]+=int(y[so[:5]].sum())
 s=np.concatenate(ss);y=np.concatenate(yy);d=lambda x:scalar_stats(x) if x else {'count':0,'mean':None}
 return {'candidate_count':int(len(y)),'positive_count':int(y.sum()),'positive_frame_count':pf,'null_frame_count':len(null),'roc_auc':auc(s,y),'pr_auc':average_precision(s,y),'top1_frame_recall':top1/max(1,pf),'top5_frame_recall':top5/max(1,pf),'multi_positive_frame_count':mpf,'multi_positive_top1_recall':mp1/max(1,mpf),'multi_positive_top5_recall':mp5/max(1,mpf),'positive_min_model_hard_margin':d(marg),'full_frame_model_hard_margin':d(fullm),'hard_violation_rate':float(np.mean(viol)) if viol else None,'objectness_prefilter_missed_full_hard_top1_rate':float(np.mean(miss)) if miss else None,'source_internal_precision':{'main':{'top1':source[0][1]/max(1,source[0][0]),'top5':source[0][2]/max(1,source[0][0]*5),'frames':source[0][0]},'reserve':{'top1':source[1][1]/max(1,source[1][0]),'top5':source[1][2]/max(1,source[1][0]*5),'frames':source[1][0]}},'null_highest_candidate_score':d(null),'zero_threshold':{'predictions':zero,'positive':zp,'predictions_per_positive':zero/max(1,int(y.sum()))}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='outputs/l19/protocol/kitti_fast_eval_manifest.json');ap.add_argument('--v4-root',default='outputs/l25/candidate_bank_v4');ap.add_argument('--out-root',required=True);ap.add_argument('--stage',default='D0',choices=['D0','D1','D2','D3','D4','F1','F2','F3','F4','F5','F6']);ap.add_argument('--steps',type=int,default=50);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--device',default='cuda:0');ap.add_argument('--frames-per-split',type=int,default=6000);args=ap.parse_args();P=lambda x:(Path(x) if Path(x).is_absolute() else ROOT/Path(x));manifest,root,out=map(P,(args.manifest,args.v4_root,args.out_root));
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);qs=sorted(json.loads(manifest.read_text())['queries'],key=lambda x:int(x['query_index']));meta=load_metadata();vids=sorted({str(q['video']) for q in qs});banks={v:load_bank(root/'kitti'/f'{v}.pt') for v in vids};refs=make_refs(qs,meta,banks);cal=fixed_refs([r for r in refs if r['split']=='calibration'],args.frames_per_split,args.seed);screen=fixed_refs([r for r in refs if r['split']=='screening'],args.frames_per_split,args.seed+1);device=torch.device(args.device);import clip;cm,_=clip.load('/home/lwr/.cache/clip/ViT-B-16.pt',device=device);cm.eval();qcache={int(q['query_index']):tokens_for(cm,q['expression'],device).cpu() for q in qs};model=L25TokenCorrespondence(args.stage).to(device);t0=time.time();report={'format':'locatemot-l25-token-correspondence-v1','stage':args.stage,'manifest':str(manifest),'manifest_sha256':sha(manifest),'v4_root':str(root),'device':str(device),'seed':args.seed,'steps':args.steps,'calibration_query_count':64,'screening_query_count':96,'calibration_frame_units':len(cal),'screening_frame_units':len(screen),'screening_gt_used_for_selection':False,'hard_rule':{'objectness_prefilter':96,'current_model_topk':24,'training_validation_same':True},'excluded':['pool_id','source','group','state','tracker','old_checkpoint'],'motion_language_decomposition':'not claimed; word tokens are unsplit because no verified motion token mask'};report['train']=train(model,cal,banks,device,qcache,args.steps,args.seed) if args.steps else {'steps':0};report['calibration_metrics']=evaluate(model,cal,banks,device,qcache);report['screening_metrics']=evaluate(model,screen,banks,device,qcache);ck=out/f'checkpoint_{args.stage.lower()}_step{args.steps}.pt';torch.save({'model':model.state_dict(),'stage':args.stage,'manifest_sha256':report['manifest_sha256']},ck);report['checkpoint']=str(ck);reload=L25TokenCorrespondence(args.stage).to(device);reload.load_state_dict(torch.load(ck,map_location=device,weights_only=False)['model']);reload.eval();report['checkpoint_reload']=True;report['elapsed_sec']=time.time()-t0;(out/f'metrics_{args.stage.lower()}_step{args.steps}.json').write_text(json.dumps(report,indent=2)+'\n');(out/'README.md').write_text(f'# L25 {args.stage} token correspondence\n\nCalibration-only frame-level candidate-set training; screening is reporting-only.\n');print(json.dumps({'output':str(out),'stage':args.stage,'screening':{k:report['screening_metrics'].get(k) for k in ('roc_auc','pr_auc','top1_frame_recall','positive_min_model_hard_margin','hard_violation_rate','objectness_prefilter_missed_full_hard_top1_rate')}},indent=2))
if __name__=='__main__':main()
