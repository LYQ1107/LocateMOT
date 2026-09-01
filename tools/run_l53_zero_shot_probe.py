#!/usr/bin/env python3
"""L53 zero-shot GroundingDINO -> complete L19 candidate probe.

The child process owns the TTAOD-F model.  It writes prediction boxes only;
the parent maps them to L19 rows using a pre-registered IoU rule without
opening labels until metric calculation.
"""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT'); DATA=ROOT/'outputs/l49/data'; TTAOD=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main'); PYTHON=Path('/home/lwr/anaconda3/envs/ttaod_f/bin/python'); CONFIG=TTAOD/'configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'; WEIGHT=TTAOD/'download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'; BERT=Path('/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'); MANIFEST=ROOT/'outputs/l19/protocol/kitti_fast_eval_manifest.json'; L29=ROOT/'outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt'; L52_RECORDS=ROOT/'outputs/l52/eval/b0_semantic_probe_retry1/score_records.jsonl'
sys.path.insert(0,str(ROOT))
from locatemot.rmot.l49_data import load_bank, sha256_file
from tools.eval_l49_validation import l29_score
from tools.train_l49_kitti_rmot import build_teacher_cache
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
EXPECTED='06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'

def load_units(path, split):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if any(x.get('split')!=split for x in rows): raise AssertionError(f'{path} split mismatch')
    return rows
def subset(rows,n,seed):
    by=defaultdict(lambda:defaultdict(list))
    for x in rows: by[x['dataset']][x['category']].append(x)
    rng=random.Random(seed); answer=[]
    for ds in sorted(by):
        for vals in by[ds].values(): vals.sort(key=lambda x:(x['video'],int(x['frame_id']),int(x['query_id']))); rng.shuffle(vals)
        cats=sorted(by[ds]); c={x:0 for x in cats}
        while sum(x['dataset']==ds for x in answer)<n:
            ok=False
            for cat in cats:
                if sum(x['dataset']==ds for x in answer)>=n: break
                if c[cat]<len(by[ds][cat]): answer.append(by[ds][cat][c[cat]]); c[cat]+=1; ok=True
            if not ok: raise AssertionError(f'not enough {ds}')
    return answer
def iou(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); lt=np.maximum(a[:,None,:2],b[None,:, :2]); rb=np.minimum(a[:,None,2:],b[None,:,2:]); inter=np.prod(np.maximum(0,rb-lt),axis=2); aa=np.prod(np.maximum(0,a[:,2:]-a[:,:2]),axis=1)[:,None]; bb=np.prod(np.maximum(0,b[:,2:]-b[:,:2]),axis=1)[None,:]; return inter/np.maximum(aa+bb-inter,1e-9)
def metric(recs,t):
    tp=fp=pn=0; top1=[]; top5=[]; viol=[]; strict=[]; best=[]; avg=[]; mrec=[]; multi_n=multi_tp=0; empty=0; nullfp=0; cov=[]
    for r in recs:
        s=np.asarray(r['score']); y=np.asarray(r['label'],bool); order=np.argsort(-s,kind='stable'); chosen=s>=t; tp+=int((chosen&y).sum()); fp+=int((chosen&~y).sum()); pn+=int(y.sum()); top1.append(float(y[order[0]]) if len(order) else 0); top5.append(float(y[order[:5]].any()) if len(order) else 0); empty+=int(not chosen.any()); nullfp+=int(r['category']=='inactive' and chosen.any()); cov.append(float(np.mean(s>-19.9)) if len(s) else 0)
        pos=np.where(y)[0]; neg=np.where(~y)[0]
        if len(pos) and len(neg): strict.append(float(s[pos].min()-s[neg].max())); best.append(float(s[pos].max()-s[neg].max())); avg.append(float(s[pos].mean()-s[neg].max())); viol.append(float(s[pos].min()<=s[neg].max()))
        if r['category']=='multi_positive': multi_n+=int(y.sum()); multi_tp+=int((chosen&y).sum())
    return {'units':len(recs),'top1':float(np.mean(top1)) if top1 else None,'top5':float(np.mean(top5)) if top5 else None,'candidate_precision':float(tp/max(1,tp+fp)),'candidate_recall':float(tp/max(1,pn)),'fp_per_frame':float(fp/max(1,len(recs))),'pred_per_positive':float((tp+fp)/max(1,pn)),'hard_violation':float(np.mean(viol)) if viol else None,'strict_min_positive_margin':float(np.mean(strict)) if strict else None,'best_positive_margin':float(np.mean(best)) if best else None,'average_positive_margin':float(np.mean(avg)) if avg else None,'multi_positive_recall':float(multi_tp/max(1,multi_n)) if multi_n else None,'empty_rate':float(empty/max(1,len(recs))),'null_false_acceptance':float(nullfp/max(1,sum(r['category']=='inactive' for r in recs))) if any(r['category']=='inactive' for r in recs) else None,'proposal_coverage':float(np.mean(cov)) if cov else None}
def threshold(recs):
    vals=np.concatenate([np.asarray(r['score']) for r in recs]); candidates=np.linspace(float(vals.min()-1),float(vals.max()+1),101); best=None
    for t in candidates:
        m=metric(recs,float(t)); f1=2*m['candidate_precision']*m['candidate_recall']/max(1e-12,m['candidate_precision']+m['candidate_recall']); key=(f1,-m['fp_per_frame'],-float(t))
        if best is None or key>best[0]: best=(key,float(t),m)
    return {'threshold':best[1],'rule':'calibration-only 101-point score range +/-1, max frame F1 then lower FP/frame then lower threshold','metrics':best[2]}
def child_code():
    return r'''
import json,os,sys,time,traceback,torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
import mmdet.models
import mmdet.datasets
register_all_modules(init_default_scope=True)
from mmdet.apis import inference_detector
jobs=json.load(open(os.environ['L53_JOBS'])); cfg=Config.fromfile(os.environ['L53_CONFIG']); cfg.model.backbone.init_cfg=None; cfg.model.language_model.name=os.environ['L53_BERT']; model=MODELS.build(cfg.model); load_checkpoint(model,os.environ['L53_WEIGHT'],map_location='cpu',strict=False); model.to('cuda:0').eval(); model.cfg=cfg; out=[]
try:
 for job in jobs:
  t=time.time(); result=inference_detector(model,job['image_path'],text_prompt=job['expression'],custom_entities=True); p=result.pred_instances; boxes=p.bboxes.detach().cpu().float().tolist(); scores=p.scores.detach().cpu().float().tolist(); out.append({'job_id':job['job_id'],'unit_key':job['unit_key'],'image_path':job['image_path'],'expression':job['expression'],'pred_boxes':boxes,'pred_scores':scores,'elapsed_sec':time.time()-t,'finite':all(torch.isfinite(x.float()).all().item() for x in p.values() if hasattr(x,'dtype') and torch.is_floating_point(x))}); del result,p; torch.cuda.empty_cache()
except Exception as e:
 json.dump({'completed':out,'failed_job':jobs[len(out)] if len(out)<len(jobs) else None,'error_type':type(e).__name__,'error':str(e)},open(os.environ['L53_PRED']+'.partial','w'),indent=2); raise
json.dump(out,open(os.environ['L53_PRED'],'w'),indent=2)
'''
def main():
    p=argparse.ArgumentParser(); p.add_argument('--out-root',required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=20260830); a=p.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong LocateMOT cwd')
    if sha256_file(MANIFEST)!=EXPECTED: raise RuntimeError('manifest SHA mismatch')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); cal=subset(load_units(DATA/'calibration_units.jsonl','calibration'),8,a.seed); val=subset(load_units(DATA/'validation_units.jsonl','validation'),12,a.seed+1); units=cal+val; jobs=[]
    for i,u in enumerate(units):
        bank=load_bank(u['dataset'],u['video']); t=bank['tensors']; frame=int(u['frame_id']); image=ROOT/'data/kitti_tracking_training/image_02'/str(u['video'])/f'{frame:06d}.png'; b,e=int(u['begin']),int(u['end']); boxes=t['box'][b:e].float().tolist(); jobs.append({'job_id':i,'unit_key':u['unit_key'],'dataset':u['dataset'],'video':u['video'],'query_id':int(u['query_id']),'frame_id':frame,'expression':str(u['sentence']),'image_path':str(image.resolve()),'candidate_boxes':boxes})
    jobs_path=out/'jobs_no_labels.json'; jobs_path.write_text(json.dumps(jobs,indent=2)); pred_path=out/'predictions.json'; env=os.environ.copy(); env.update({'PYTHONPATH':str(TTAOD),'L53_JOBS':str(jobs_path),'L53_PRED':str(pred_path),'L53_CONFIG':str(CONFIG),'L53_WEIGHT':str(WEIGHT),'L53_BERT':str(BERT),'TRANSFORMERS_OFFLINE':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}); start=time.time(); run=subprocess.run([str(PYTHON),'-c',child_code()],cwd=str(TTAOD),env=env,text=True,capture_output=True,timeout=1800)
    if run.returncode!=0:
        (out/'INCOMPLETE.md').write_text('Isolated zero-shot child failed:\n'+run.stderr[-8000:]); raise RuntimeError(run.stderr[-2000:])
    predictions=json.loads(pred_path.read_text()); pmap={x['unit_key']:x for x in predictions}; records=[]; l29_model=L29FrameMembershipSetDecoder().to(a.device); l29_model.load_state_dict(torch.load(L29,map_location=a.device,weights_only=False)['model'],strict=True); l29_model.eval(); tc={}; text=torch.load(ROOT/'outputs/l48/data/text_cache.pt',map_location='cpu',weights_only=False)
    for u in units:
        key=(u['dataset'],u['video']); bank=load_bank(*key)
        if key not in tc: tc[key]=build_teacher_cache(bank)
        ts=l29_score(l29_model,tc[key],bank,u,text,torch.device(a.device)); b,e=int(u['begin']),int(u['end']); cand=np.asarray(bank['tensors']['box'][b:e].float()); q=pmap[u['unit_key']]; pb=np.asarray(q['pred_boxes']); ps=np.asarray(q['pred_scores']); ov=iou(cand,pb) if len(pb) else np.zeros((len(cand),0)); score=np.max(np.where(ov>=.30,ps[None,:],-20.),axis=1) if len(pb) else np.full(len(cand),-20.); y=np.zeros(e-b,dtype=bool); y[np.asarray(u['positive_indices'],dtype=int)]=True; records.append({'unit_key':u['unit_key'],'dataset':u['dataset'],'video':u['video'],'frame_id':int(u['frame_id']),'category':u['category'],'label':y.tolist(),'score':score.tolist(),'teacher':np.asarray(ts).tolist(),'proposal_count':len(pb),'proposal_coverage':float(np.mean(score>-19.9)) if len(score) else 0.0,'native_pred_finite':q['finite']})
    grouped={'l29_teacher':{'calibration':[r for r in records if r['unit_key'] in {x['unit_key'] for x in cal}],'validation':[r for r in records if r['unit_key'] in {x['unit_key'] for x in val}]},'groundingdino':{'calibration':[r for r in records if r['unit_key'] in {x['unit_key'] for x in cal}],'validation':[r for r in records if r['unit_key'] in {x['unit_key'] for x in val}]}}; grouped['groundingdino']['calibration']=[dict(r,score=r['score']) for r in grouped['groundingdino']['calibration']]; grouped['l29_teacher']['calibration']=[dict(r,score=r['teacher']) for r in grouped['l29_teacher']['calibration']]; grouped['l29_teacher']['validation']=[dict(r,score=r['teacher']) for r in grouped['l29_teacher']['validation']]
    decisions={n:threshold(x['calibration']) for n,x in grouped.items()}; results={n:{'calibration':decisions[n]['metrics'],'validation':metric(x['validation'],decisions[n]['threshold'])} for n,x in grouped.items()}; l52_note={'available':False,'reason':'L52 direct score cache retained as historical reference; not used for L53 selection'}
    payload={'format':'locatemot-l53-zero-shot-grounding-probe-v1','status':'pass','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'seed':a.seed,'compatibility_audit':'outputs/l53/audit/compatibility_attempt4/compatibility.json','isolated_interpreter':str(PYTHON),'groundingdino_config':str(CONFIG),'weight':str(WEIGHT),'weight_sha256':sha256_file(WEIGHT),'bert_snapshot':str(BERT),'calibration_units':len(cal),'validation_units':len(val),'calibration_videos':sorted({x['video'] for x in cal}),'validation_videos':sorted({x['video'] for x in val}),'mapping_rule':'candidate score=max native proposal score for IoU(candidate,proposal)>=0.30 else -20; inference-only geometry; no GT','full_candidate_set':True,'thresholds':decisions,'results':results,'l52_reference':l52_note,'proposal_counts':{'mean':float(np.mean([r['proposal_count'] for r in records])),'native_finite':all(r['native_pred_finite'] for r in records)},'records':records,'official_test_labels_read':False,'screening_gt_used':False,'ordinary_mot_ovmot_touched':False,'elapsed_sec':time.time()-start,'native_child_stderr_tail':run.stderr[-2000:]}
    (out/'zero_shot.json').write_text(json.dumps(payload,indent=2)+'\n'); (out/'provenance.json').write_text(json.dumps({'manifest_sha256':sha256_file(MANIFEST),'weight_sha256':sha256_file(WEIGHT),'config_sha256':sha256_file(CONFIG),'train_fit_only_for_selection':True,'calibration_labels_only_for_threshold':True,'validation_used_after_freeze':True,'screening_gt_used':False,'official_test_labels_read':False,'mapping_uses_gt':False},indent=2)+'\n'); print(json.dumps({'status':'pass','output':str(out/'zero_shot.json'),'results':results}))
if __name__=='__main__': main()
