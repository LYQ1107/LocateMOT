"""C0-C5 L25 token-to-region upper-limit probes on the fixed fast manifest."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT');sys.path.insert(0,str(ROOT))
from tools.train_l23_dense_correspondence import fixed_refs
from tools.train_rmot_candidate_scorer import load_bank,load_metadata,make_refs,auc,average_precision,scalar_stats
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tokens_for(model,text,device):
    ids=torch.cat([torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long)],1) if False else __import__('clip').tokenize([text]).to(device)
    with torch.inference_mode():
        x=model.token_embedding(ids).type(model.dtype);x=x+model.positional_embedding.type(model.dtype);x=x.permute(1,0,2);x=model.transformer(x);x=x.permute(1,0,2);x=model.ln_final(x).type(model.dtype);x=x@model.text_projection
    ids=ids[0].tolist();valid=[i for i,t in enumerate(ids) if t not in (0,49407) and i>0];valid=valid or [1]
    return F.normalize(x[0,valid].float(),dim=-1)
class TokenProbe(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;self.q=nn.Linear(512,64,bias=False);self.k=nn.Linear(512,64,bias=False);self.v=nn.Linear(512,64,bias=False);self.attn=nn.MultiheadAttention(64,4,batch_first=True);self.coord=nn.Linear(2,64,bias=False);self.head=nn.Sequential(nn.LayerNorm(64+(16 if variant=='C5' else 0)),nn.Linear(64+(16 if variant=='C5' else 0),1));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,q,p,coords=None,context=None,prev=None,motion=None):
        key_parts=[self.k(p)]; value_parts=[self.v(p)]
        if coords is not None:key_parts[0]=key_parts[0]+self.coord(coords)
        if context is not None:key_parts.extend([self.k(context)]);value_parts.extend([self.v(context)])
        if prev is not None:key_parts.extend([self.k(prev)]);value_parts.extend([self.v(prev)])
        p=torch.cat(key_parts,1);v=torch.cat(value_parts,1)
        q=self.q(q);o,_=self.attn(q,p,v,need_weights=False);z=o.mean(1)
        if self.variant=='C5':z=torch.cat((z,motion),1)
        return self.head(z).squeeze(-1)
def arrays(ref,bank,device):
    t=bank['tensors'];sl=slice(ref['begin'],ref['end']);
    return {'patch':t['dense_roi_tokens_v4'][sl].float().to(device),'coords':t['roi_sample_points_v4'][sl].float().to(device),'context':torch.cat((t['dense_context_1p5_tokens_v4'][sl].float(),t['dense_context_3_tokens_v4'][sl].float()),1).to(device),'prev':t['dense_prev_roi_tokens_v4'][sl].float().to(device),'motion':t['motion_v2'][sl].float().to(device),'geometry':t['geometry_v2'][sl].float().to(device),'objectness':t['objectness'][sl].float().reshape(-1).to(device)}
def fixed_score(variant,q,p):
    q=F.normalize(q,dim=-1);p=F.normalize(p,dim=-1);sim=torch.einsum('ld,nkd->nlk',q,p)
    if variant=='C1':return sim.mean((1,2))
    return sim.amax((1,2))
def score(variant,model,q,a):
    if variant in ('C1','C2'):return fixed_score(variant,q,a['patch'])
    return model(q[None].expand(a['patch'].shape[0],-1,-1),a['patch'],a['coords'] if variant in ('C3','C4','C5') else None,a['context'] if variant in ('C4','C5') else None,a['prev'] if variant in ('C4','C5') else None,a['motion'] if variant=='C5' else None)
def metric(refs,banks,device,variant,model=None,text_cache=None):
    ss=[];yy=[];marg=[];fullm=[];viol=[];top1=top5=pf=mpf=mp1=mp5=zero=zp=0;source={0:[0,0,0],1:[0,0,0]};null=[];breaks=[]
    if model:model.eval()
    for ref in refs:
        a=arrays(ref,banks[ref['video']],device);q=text_cache[ref['query_index']].to(device);with_ctx=torch.inference_mode() if model else torch.inference_mode()
        with with_ctx:
            s=score(variant,model,q,a).detach().cpu().numpy() if model else score(variant,None,q,a).detach().cpu().numpy()
        y=ref['positive'].astype(bool);ss.append(s);yy.append(y);zero+=int((s>=0).sum());zp+=int(((s>=0)&y).sum());pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
        if not len(pos):null.append(float(s.max()) if len(s) else 0.);continue
        pf+=1;o=np.argsort(-s);top1+=int(y[o[:1]].any());top5+=int(y[o[:5]].any());
        if len(pos)>1:mpf+=1;mp1+=int(y[o[:1]].any());mp5+=int(y[o[:5]].any())
        obj=a['objectness'].cpu().numpy();pre=neg[np.argsort(-obj[neg],kind='stable')[:min(96,len(neg))]];hard=pre[np.argsort(-s[pre],kind='stable')[:min(24,len(pre))]] if len(pre) else pre;gh=neg[np.argsort(-s[neg],kind='stable')[:min(24,len(neg))]] if len(neg) else neg
        if len(hard):marg.append(float(s[pos].min()-s[hard].max()));viol.append(marg[-1]<0);fullm.append(float(s[pos].min()-s[gh].max()) if len(gh) else marg[-1]);breaks.append(float(np.mean(np.sign((s[pos][:,None]-s[hard][None,:]))<0)))
        pool=banks[ref['video']]['tensors']['pool_id'][ref['begin']:ref['end']].numpy()
        for sid in (0,1):
            rows=np.flatnonzero(pool==sid)
            if len(rows):so=rows[np.argsort(-s[rows])];source[sid][0]+=1;source[sid][1]+=int(y[so[:1]].any());source[sid][2]+=int(y[so[:5]].sum())
    s=np.concatenate(ss);y=np.concatenate(yy);d=lambda x:scalar_stats(x) if x else {'count':0,'mean':None}
    return {'candidate_count':int(len(y)),'positive_count':int(y.sum()),'roc_auc':auc(s,y),'pr_auc':average_precision(s,y),'positive_frame_count':pf,'top1_frame_recall':top1/max(1,pf),'top5_frame_recall':top5/max(1,pf),'multi_positive_frame_count':mpf,'multi_positive_top1_recall':mp1/max(1,mpf),'multi_positive_top5_recall':mp5/max(1,mpf),'positive_min_model_hard_margin':d(marg),'full_frame_model_hard_margin':d(fullm),'hard_violation_rate':float(np.mean(viol)) if viol else None,'selected_hard_negative_rate':float(np.mean(breaks)) if breaks else None,'source_internal_precision':{'main':{'top1':source[0][1]/max(1,source[0][0]),'top5':source[0][2]/max(1,source[0][0]*5),'frames':source[0][0]},'reserve':{'top1':source[1][1]/max(1,source[1][0]),'top5':source[1][2]/max(1,source[1][0]*5),'frames':source[1][0]}},'null_highest_candidate_score':d(null),'zero_threshold':{'predictions':zero,'positive':zp,'predictions_per_positive':zero/max(1,int(y.sum()))}}
def train_probe(model,refs,banks,text_cache,device,steps,seed):
    rng=random.Random(seed);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);loss=[];grad=[];model.train()
    for _ in range(steps):
        chosen=[refs[rng.randrange(len(refs))] for _ in range(4)];terms=[]
        for ref in chosen:
            a=arrays(ref,banks[ref['video']],device);q=text_cache[ref['query_index']].to(device);s=score(model.variant,model,q,a);y=torch.as_tensor(ref['positive'],device=device).float();pos=s[y>.5];neg=s[y<=.5]
            if len(pos):terms.append(F.binary_cross_entropy_with_logits(pos,torch.ones_like(pos)))
            if len(pos) and len(neg):terms.append(F.softplus(1+neg.topk(min(24,len(neg))).values[None,:]-pos[:,None]).mean())
        l=torch.stack(terms).mean();opt.zero_grad(set_to_none=True);l.backward();grad.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5)));opt.step();loss.append(float(l.detach()))
    return {'steps':steps,'loss':scalar_stats(loss),'gradient_norm':scalar_stats(grad)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='outputs/l19/protocol/kitti_fast_eval_manifest.json');ap.add_argument('--v4-root',default='outputs/l25/candidate_bank_v4');ap.add_argument('--out-root',default='outputs/l25/eval/token_probe');ap.add_argument('--device',default='cuda:0');ap.add_argument('--frames-per-split',type=int,default=6000);ap.add_argument('--steps',type=int,default=100);ap.add_argument('--seed',type=int,default=17);args=ap.parse_args();P=lambda x:(Path(x) if Path(x).is_absolute() else ROOT/Path(x));manifest,root,out=map(P,(args.manifest,args.v4_root,args.out_root));
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);qs=sorted(json.loads(manifest.read_text())['queries'],key=lambda x:int(x['query_index']));meta=load_metadata();vids=sorted({str(q['video']) for q in qs});banks={v:load_bank(root/'kitti'/f'{v}.pt') for v in vids};refs=make_refs(qs,meta,banks);cal=fixed_refs([r for r in refs if r['split']=='calibration'],args.frames_per_split,args.seed);screen=fixed_refs([r for r in refs if r['split']=='screening'],args.frames_per_split,args.seed+1);device=torch.device(args.device);import clip;clip_model,_=clip.load('/home/lwr/.cache/clip/ViT-B-16.pt',device=device);clip_model.eval();text_cache={int(q['query_index']):tokens_for(clip_model,q['expression'],device).cpu() for q in qs}
 # Avoid a second model load for text extraction in subsequent work: the cache is
 # stored as tensors and all probes use only these frozen token representations.
 result={'format':'locatemot-l25-token-probe-v1','manifest':str(manifest),'manifest_sha256':sha(manifest),'v4_root':str(root),'weights':'/home/lwr/.cache/clip/ViT-B-16.pt','weights_sha256':sha('/home/lwr/.cache/clip/ViT-B-16.pt'),'calibration_queries':64,'screening_queries':96,'calibration_frame_units':len(cal),'screening_frame_units':len(screen),'screening_gt_used_for_selection':False,'variants':{}}
 # C0 is the previously validated L24 teacher-only result, retained as the
 # frozen baseline; C1/C2 are deterministic token compatibility upper bounds.
 c0=json.loads((ROOT/'outputs/l24/eval/R0_teacher_only_retry/metrics_f1_step0.json').read_text())['screening_metrics'];result['variants']['C0_L24_frozen_teacher']={'source':'outputs/l24/eval/R0_teacher_only_retry/metrics_f1_step0.json','metrics':c0}
 for name in ('C1','C2'):
  result['variants'][name]={'train':{'steps':0,'frozen':True},'calibration_metrics':metric(cal,banks,device,name,None,text_cache),'screening_metrics':metric(screen,banks,device,name,None,text_cache)}
 for name in ('C3','C4','C5'):
  model=TokenProbe(name).to(device);start=time.time();tr=train_probe(model,[r for r in cal if r['positive'].any()],banks,text_cache,device,args.steps,args.seed);result['variants'][name]={'train':tr,'calibration_metrics':metric(cal,banks,device,name,model,text_cache),'screening_metrics':metric(screen,banks,device,name,model,text_cache),'elapsed_sec':time.time()-start};torch.save({'model':model.state_dict(),'variant':name,'manifest_sha256':result['manifest_sha256']},out/f'checkpoint_{name.lower()}_step{args.steps}.pt')
 (out/'token_probe.json').write_text(json.dumps(result,indent=2)+'\n');(out/'README.md').write_text('# L25 token-level probe\n\nC0 is the prior L24 frozen teacher; C1/C2 are frozen token compatibility controls; C3-C5 are calibration-only small cross-attention probes.\n');print(json.dumps({'output':str(out/'token_probe.json'),'variants':{k:{m:v.get('screening_metrics',{}).get(m) for m in ('roc_auc','pr_auc','top1_frame_recall','positive_min_model_hard_margin')} for k,v in result['variants'].items()}},indent=2))
if __name__=='__main__':main()
