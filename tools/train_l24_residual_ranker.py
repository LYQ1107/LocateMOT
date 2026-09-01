"""Train one isolated L24 teacher-preserving residual ranker.

Calibration refs are the only optimization data.  Screening refs are used for
reporting only.  The hard-negative contract is identical in train/eval:
objectness top-96, current-score top-24, plus teacher/objectness hard samples.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT'); sys.path.insert(0,str(ROOT))
from locatemot.models.rmot_l24_residual_ranker import L24ResidualDenseRanker
from tools.fit_l24_dense_teacher import feat as teacher_feat
from tools.train_l23_dense_correspondence import fixed_refs, arrays_for
from tools.train_rmot_candidate_scorer import load_bank, load_metadata, make_refs, auc, average_precision, scalar_stats

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def path(x):
    x=Path(x); return x if x.is_absolute() else ROOT/x
def values(ref, bank, rows, device):
    rows=np.asarray(rows,np.int64); a=arrays_for(ref,bank,rows); absolute=ref['begin']+rows; t=bank['tensors']
    a['neighbor']=t['neighbor_v2'][absolute].float().numpy(); a['lifecycle']=t['lifecycle_v2'][absolute].float().numpy()
    return {k:torch.as_tensor(v,device=device) for k,v in a.items()}
def score(model,v):
    return model(query=v['query'],dense_points=v['dense_points'],dense_roi=v['dense_roi'],geometry=v['geometry'],objectness=v['objectness'],dense_context_1p5=v['dense_context_1p5'],dense_context_3=v['dense_context_3'],dense_prev_roi=v['dense_prev_roi'],motion=v['motion'],neighbor=v['neighbor'],lifecycle=v['lifecycle'])
def mine(ref,bank,model,device,full_scores=None):
    n=ref['end']-ref['begin']; rows=np.arange(n); y=ref['positive'].astype(bool); neg=np.flatnonzero(~y)
    obj=bank['tensors']['objectness'][ref['begin']:ref['end']].float().numpy().reshape(-1)
    pre=neg[np.argsort(-obj[neg],kind='stable')[:min(96,len(neg))]]
    if full_scores is None:
        with torch.inference_mode(): full_scores=score(model,values(ref,bank,rows,device))['final_score'].cpu().numpy()
    current_pre=pre[np.argsort(-full_scores[pre],kind='stable')[:min(24,len(pre))]] if len(pre) else np.zeros(0,dtype=np.int64)
    current_full=neg[np.argsort(-full_scores[neg],kind='stable')[:min(24,len(neg))]]
    with torch.inference_mode(): teacher=score(model,values(ref,bank,neg,device))['teacher_score'].cpu().numpy() if len(neg) else np.zeros(0)
    teacher_hard=neg[np.argsort(-teacher,kind='stable')[:min(12,len(neg))]]
    object_hard=neg[np.argsort(-obj[neg],kind='stable')[:min(12,len(neg))]]
    mixed=np.unique(np.concatenate((current_pre,teacher_hard,object_hard))).astype(np.int64)
    if len(mixed)>24: mixed=mixed[np.argsort(-full_scores[mixed],kind='stable')[:24]]
    easy=np.setdiff1d(neg,mixed,assume_unique=False)[:16]
    return np.flatnonzero(y),mixed,easy,pre,current_pre,current_full
def frame_loss(model,ref,bank,device):
    n=ref['end']-ref['begin']; rows=np.arange(n); v=values(ref,bank,rows,device); out=score(model,v); full=out['final_score']; teacher=out['teacher_score'].detach(); y=ref['positive'].astype(bool)
    pos,hard,easy,pre,current_pre,current_full=mine(ref,bank,model,device,full.detach().cpu().numpy()); z=full.new_zeros(())
    p=full[torch.as_tensor(pos,device=device)]; h=full[torch.as_tensor(hard,device=device)]; e=full[torch.as_tensor(easy,device=device)]
    pb=F.binary_cross_entropy_with_logits(p,torch.ones_like(p)) if len(p) else z
    hb=F.binary_cross_entropy_with_logits(h,torch.zeros_like(h)) if len(h) else z
    eb=F.binary_cross_entropy_with_logits(e,torch.zeros_like(e)) if len(e) else z
    pair=F.softplus(1.0+h[None,:]-p[:,None]).mean() if len(p) and len(h) else z
    violation=F.softplus(h[None,:]-p[:,None]).mean() if len(p) and len(h) else z
    listwise=(torch.logsumexp(full,0)-torch.logsumexp(p,0)) if len(p) else z
    tp=teacher[torch.as_tensor(pos,device=device)]; th=teacher[torch.as_tensor(hard,device=device)]
    fd=full[torch.as_tensor(pos,device=device)][:,None]-full[torch.as_tensor(hard,device=device)][None,:] if len(p) and len(h) else z
    td=tp[:,None]-th[None,:] if len(p) and len(h) else z
    preserve=F.relu(.1-torch.sign(td.detach())*fd).mean() if len(p) and len(h) else z
    total=pb+hb+.1*eb+pair+.5*listwise+.2*violation+.25*preserve
    return total,{'total':float(total.detach()),'positive_bce':float(pb.detach()),'hard_bce':float(hb.detach()),'easy_bce':float(eb.detach()),'pairwise':float(pair.detach()),'listwise':float(listwise.detach()),'violation':float(violation.detach()),'teacher_preservation':float(preserve.detach()),'positive_count':len(pos),'hard_count':len(hard),'easy_count':len(easy),'prefilter_count':len(pre)}
def train(model,refs,banks,device,steps,seed,batch_frames):
    rng=random.Random(seed); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-4); history=[]; grads=[]; start=time.time(); model.train()
    for _ in range(steps):
        chosen=[refs[rng.randrange(len(refs))] for _ in range(min(batch_frames,len(refs)))]; terms=[]; reports=[]
        for ref in chosen:
            l,r=frame_loss(model,ref,banks[ref['video']],device);terms.append(l);reports.append(r)
        loss=torch.stack(terms).mean();opt.zero_grad(set_to_none=True);loss.backward();g=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step();grads.append(g)
        row={k:float(np.mean([r[k] for r in reports])) for k in reports[0]};history.append(row)
    return {'steps':steps,'elapsed_sec':time.time()-start,'loss':{k:scalar_stats([r[k] for r in history]) for k in ('total','positive_bce','hard_bce','easy_bce','pairwise','listwise','violation','teacher_preservation')},'bucket_counts':{k:int(sum(r[k] for r in history)) for k in ('positive_count','hard_count','easy_count','prefilter_count')},'gradient_norm':scalar_stats(grads)}
def evaluate(model,refs,banks,device):
    model.eval(); all_s=[];all_y=[]; posvals=[];hardvals=[];easyvals=[];marg=[];tmarg=[];rmarg=[];viol=[];rankbreak=[];fullmarg=[];null=[];top1=top5=pf=zero_pred=zero_pos=0; source={0:[0,0,0],1:[0,0,0]}; norms={'query':[],'roi':[],'points':[]}
    for ref in refs:
        bank=banks[ref['video']];n=ref['end']-ref['begin'];rows=np.arange(n);v=values(ref,bank,rows,device)
        with torch.inference_mode(): out=score(model,v); s=out['final_score'].cpu().numpy();t=out['teacher_score'].cpu().numpy();r=out['residual_score'].cpu().numpy()
        y=ref['positive'].astype(bool);all_s.append(s);all_y.append(y);zero_pred+=int((s>=0).sum());zero_pos+=int(((s>=0)&y).sum());pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
        norms['query'].append(float(np.linalg.norm(ref['spec']))); norms['roi'].extend(np.linalg.norm(bank['tensors']['dense_roi'][ref['begin']:ref['end']].float().numpy(),axis=1).tolist()); norms['points'].extend(np.linalg.norm(bank['tensors']['dense_points'][ref['begin']:ref['end']].float().numpy(),axis=2).reshape(-1).tolist())
        if not len(pos): null.append(float(s.max()) if len(s) else 0.);continue
        pf+=1;o=np.argsort(-s);top1+=int(y[o[:1]].any());top5+=int(y[o[:5]].any());positive_h,hard,easy,pre,online,global_h=mine(ref,bank,model,device,s); 
        if len(hard):
            posvals.extend(s[pos].tolist());hardvals.extend(s[hard].tolist());easyvals.extend(s[easy].tolist());marg.append(float(s[pos].min()-s[hard].max()));tmarg.append(float(t[pos].min()-t[hard].max()));rmarg.append(float(r[pos].min()-r[hard].max()));viol.append(marg[-1]<0);fd=s[pos][:,None]-s[hard][None,:];td=t[pos][:,None]-t[hard][None,:];valid=np.abs(td)>1e-8;rankbreak.append(float(np.mean((np.sign(fd[valid])!=np.sign(td[valid]))) if valid.any() else 0.))
        if len(global_h): fullmarg.append(float(s[pos].min()-s[global_h].max()))
        for sid in (0,1):
            pool=bank['tensors']['pool_id'][ref['begin']:ref['end']].numpy()==sid;pr=np.flatnonzero(pool)
            if len(pr): so=pr[np.argsort(-s[pr])];source[sid][0]+=1;source[sid][1]+=int(y[so[:1]].any());source[sid][2]+=int(y[so[:5]].sum())
    s=np.concatenate(all_s);y=np.concatenate(all_y)
    def dist(x): return scalar_stats(x) if x else {'count':0,'mean':None}
    return {'candidate_count':int(len(y)),'positive_count':int(y.sum()),'positive_frame_count':pf,'null_frame_count':len(null),'roc_auc':auc(s,y),'pr_auc':average_precision(s,y),'top1_frame_recall':top1/max(1,pf),'top5_frame_recall':top5/max(1,pf),'positive_score':dist(posvals),'online_hard_score':dist(hardvals),'easy_negative_score':dist(easyvals),'positive_min_vs_mixed_hard_margin':dist(marg),'teacher_margin':dist(tmarg),'residual_margin':dist(rmarg),'full_frame_model_hard_margin':dist(fullmarg),'hard_negative_violation_rate':float(np.mean(viol)) if viol else None,'teacher_ranking_break_rate':float(np.mean(rankbreak)) if rankbreak else None,'source_internal_precision':{'main':{'top1':source[0][1]/max(1,source[0][0]),'top5':source[0][2]/max(1,source[0][0]*5),'frames':source[0][0]},'reserve':{'top1':source[1][1]/max(1,source[1][0]),'top5':source[1][2]/max(1,source[1][0]*5),'frames':source[1][0]}},'null_highest_candidate_score':dist(null),'zero_threshold':{'predictions':zero_pred,'positive':zero_pos,'predictions_per_positive':zero_pred/max(1,int(y.sum()))},'feature_norms':{k:scalar_stats(v) for k,v in norms.items()}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='outputs/l19/protocol/kitti_fast_eval_manifest.json');ap.add_argument('--v3-root',default='outputs/l23/candidate_bank_v3');ap.add_argument('--teacher',default='outputs/l24/teacher_c5/teacher_c5_linear.pt');ap.add_argument('--out-root',required=True);ap.add_argument('--stage',default='R1',choices=['R1','R2','R3','R4','F1','F2','F3','F4','F5','F6']);ap.add_argument('--steps',type=int,default=50);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--device',default='cuda:0');ap.add_argument('--frames-per-split',type=int,default=6000);ap.add_argument('--batch-frames',type=int,default=4);ap.add_argument('--alpha',type=float,default=.1);args=ap.parse_args(); manifest,root,teacher_path,out=map(path,(args.manifest,args.v3_root,args.teacher,args.out_root));
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True); data=json.loads(manifest.read_text());qs=sorted(data['queries'],key=lambda x:int(x['query_index'])); assert len(qs)==160 and sum(x['split']=='calibration' for x in qs)==64 and sum(x['split']=='screening' for x in qs)==96;meta=load_metadata();vids=sorted({str(q['video']) for q in qs});banks={v:load_bank(root/'kitti'/f'{v}.pt') for v in vids};refs=make_refs(qs,meta,banks);cal=fixed_refs([r for r in refs if r['split']=='calibration'],args.frames_per_split,args.seed);screen=fixed_refs([r for r in refs if r['split']=='screening'],args.frames_per_split,args.seed+1);device=torch.device(args.device); pack=torch.load(teacher_path,map_location='cpu',weights_only=False); model=L24ResidualDenseRanker(pack['weight'],pack['bias'],stage=args.stage,alpha=args.alpha).to(device); torch.manual_seed(args.seed);np.random.seed(args.seed)
    report={'format':'locatemot-l24-residual-ranker-v1','stage':args.stage,'manifest':str(manifest),'manifest_sha256':sha(manifest),'v3_root':str(root),'teacher':str(teacher_path),'teacher_sha256':sha(teacher_path),'seed':args.seed,'steps':args.steps,'device':str(device),'calibration_query_count':64,'screening_query_count':96,'calibration_frame_units':len(cal),'screening_frame_units':len(screen),'alpha':args.alpha,'hard_rule':{'objectness_prefilter':96,'current_prefilter_topk':24,'teacher_hard':12,'objectness_hard':12,'mixed_max':24,'full_frame_gate_reported':True,'train_validation_same_contract':True},'loss_weights':{'positive_bce':1.,'hard_bce':1.,'easy_bce':.1,'pairwise':1.,'listwise':.5,'violation':.2,'teacher_preservation':.25},'excluded':['pool_id','source_score','observation_group_id','grouping','membership','source_acceptance','null_scalar_subtraction','temporal_gru','tracker'],'motion_language_decomposition':'not claimed; motion_v2 is visual causal motion and query is unchanged'}
    # F1 is deliberately an offline teacher-preservation control: there is no
    # free residual head and therefore no differentiable parameter to train.
    if args.stage == 'F1':
        report['train']={'steps':0,'requested_steps':args.steps,'teacher_only':True,'reason':'no_free_head_by_design'}
    else:
        report['train']=train(model,cal,banks,device,args.steps,args.seed,args.batch_frames) if args.steps else {'steps':0,'teacher_only':True}
    report['calibration_metrics']=evaluate(model,cal,banks,device); report['screening_metrics']=evaluate(model,screen,banks,device); ck=out/f'checkpoint_{args.stage.lower()}_step{args.steps}.pt'; torch.save({'model':model.state_dict(),'stage':args.stage,'alpha':args.alpha,'teacher_sha256':report['teacher_sha256'],'manifest_sha256':report['manifest_sha256'],'step':args.steps},ck); report['checkpoint']=str(ck); report['checkpoint_reload']=False; reload=L24ResidualDenseRanker(pack['weight'],pack['bias'],stage=args.stage,alpha=args.alpha).to(device);reload.load_state_dict(torch.load(ck,map_location=device,weights_only=False)['model']);reload.eval();report['checkpoint_reload']=True
    gate={'auc_gt_065':report['screening_metrics']['roc_auc']>.65,'top1_gt_060':report['screening_metrics']['top1_frame_recall']>.60,'hard_margin_ge_teacher':report['screening_metrics']['positive_min_vs_mixed_hard_margin']['mean']>=report['screening_metrics']['teacher_margin']['mean'],'zero_threshold_lt3':report['screening_metrics']['zero_threshold']['predictions_per_positive']<3,'all_candidates_accepted':False};gate['candidate_gate']=all(gate.values())
    report['gate']=gate;(out/f'metrics_{args.stage.lower()}_step{args.steps}.json').write_text(json.dumps(report,indent=2)+'\n');(out/'README.md').write_text(f'# L24 {args.stage}\n\nCalibration-only residual ranker; screening is reporting-only.\n');print(json.dumps({'out_root':str(out),'stage':args.stage,'checkpoint':str(ck),'screening':{k:report['screening_metrics'].get(k) for k in ('roc_auc','pr_auc','top1_frame_recall','positive_min_vs_mixed_hard_margin','teacher_margin','hard_negative_violation_rate','teacher_ranking_break_rate')},'gate':gate},indent=2),flush=True)
if __name__=='__main__':main()
