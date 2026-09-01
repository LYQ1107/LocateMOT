#!/usr/bin/env python3
"""L52 fixed calibration/validation semantic candidate evaluation."""
from __future__ import annotations
import argparse, json, math, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT'); DATA=ROOT/'outputs/l49/data'
TEXT=ROOT/'outputs/l48/data/text_cache.pt'; MANIFEST=ROOT/'outputs/l19/protocol/kitti_fast_eval_manifest.json'; L29=ROOT/'outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt'
sys.path.insert(0,str(ROOT))
from locatemot.models.l52_query_region_set_probe import L52QueryRegionSetProbe
from locatemot.rmot.l49_data import sha256_file
from tools.eval_l49_validation import l29_score
from tools.train_l49_kitti_rmot import build_teacher_cache
from tools.train_l51_streaming_crop_adapter import numeric_features
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.rmot.l49_data import load_bank
from tools.l52_streaming_data import CLIP_WEIGHTS,L52StreamingRegionEncoder
EXPECTED='06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'

def load(path, allowed):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not rows or any(x.get('split') not in allowed for x in rows): raise AssertionError(f'bad split in {path}')
    return rows
def fixed_subset(rows, per_domain, seed):
    by=defaultdict(lambda: defaultdict(list))
    for x in rows: by[x['dataset']][x['category']].append(x)
    rng=random.Random(seed); ans=[]
    for ds in sorted(by):
        for vals in by[ds].values():
            vals.sort(key=lambda x:(x['video'],int(x['frame_id']),int(x['query_id']))); rng.shuffle(vals)
        categories=sorted(by[ds]); cursors={c:0 for c in categories}
        while len([x for x in ans if x['dataset']==ds]) < per_domain:
            progressed=False
            for cat in categories:
                if len([x for x in ans if x['dataset']==ds]) >= per_domain: break
                if cursors[cat] < len(by[ds][cat]):
                    ans.append(by[ds][cat][cursors[cat]]); cursors[cat]+=1; progressed=True
            if not progressed: raise AssertionError(f'not enough rows for {ds}')
    return ans
def metric(records, threshold):
    top1=top5=[]; tp=fp=posn=0; hard=[]; margins=[]; best=[]; avg=[]; multi_tp=multi_n=0; empty=0; nullfp=0; scores=[]
    for r in records:
        s=np.asarray(r['score'],dtype=float); y=np.asarray(r['label'],dtype=bool); order=np.argsort(-s,kind='mergesort'); p=np.where(s>=threshold)[0]; pos=np.where(y)[0]; neg=np.where(~y)[0]
        top1.append(float(len(pos) and y[order[0]])); top5.append(float(bool(len(pos)) and bool(y[order[:min(5,len(y))]].any())))
        tp+=int(y[p].sum()); fp+=int((~y[p]).sum()); posn+=int(y.sum()); scores.extend(s.tolist())
        if len(pos) and len(neg):
            mn=float(s[pos].min()-s[neg].max()); margins.append(mn); best.append(float(s[pos].max()-s[neg].max())); avg.append(float(s[pos].mean()-s[neg].max())); hard.append(float(s[neg].max()>s[pos].min()))
        if not len(p): empty+=1
        if r['category']=='inactive': nullfp+=int(len(p)>0)
        if r['category']=='multi_positive':
            multi_n+=int(y.sum()); multi_tp+=int(y[p].sum())
    return {'units':len(records),'top1':float(np.mean(top1)),'top5':float(np.mean(top5)),'candidate_precision':float(tp/(tp+fp)) if tp+fp else 0.0,'candidate_recall':float(tp/posn) if posn else 0.0,'fp_per_frame':float(fp/len(records)) if records else 0.0,'pred_per_positive':float((tp+fp)/posn) if posn else 0.0,'hard_violation':float(np.mean(hard)) if hard else None,'strict_min_positive_margin':float(np.mean(margins)) if margins else None,'best_positive_margin':float(np.mean(best)) if best else None,'average_positive_margin':float(np.mean(avg)) if avg else None,'multi_positive_recall':float(multi_tp/multi_n) if multi_n else None,'empty_rate':float(empty/len(records)) if records else 0.0,'null_false_acceptance':float(nullfp/sum(r['category']=='inactive' for r in records)) if any(r['category']=='inactive' for r in records) else None,'score_mean':float(np.mean(scores)) if scores else None,'score_std':float(np.std(scores)) if scores else None,'identity_continuity':'N/A: single-frame semantic probe'}
def make_records(model, enc, items, text, device, kind):
    out=[]
    model.eval()
    with torch.inference_mode():
        for item in items:
            if kind=='l29': score=item['teacher'].numpy()
            else:
                patch,ctx,_=enc.encode(item); q=text['token_hidden'][item['text_index']].float().to(device); mask=text['attention_mask'][item['text_index']].bool().to(device); score=model(patch.to(device),ctx.to(device),q,mask,item['frozen_clip'].to(device),item['numeric'].to(device))['relevance_logit'].detach().cpu().numpy()
            out.append({'unit_key':item['unit_key'],'dataset':item['dataset'],'video':item['video'],'query_id':item['query_id'],'frame_id':item['frame_id'],'category':item['category'],'label':item['y'].tolist(),'score':score.tolist(),'candidate_count':len(score)})
    return out

def make_eval_items(units, text, l29_model, device):
    """Materialize validation/calibration rows without train-only L29Teacher cache."""
    result=[]; bank_cache={}; teacher_cache={}
    for unit in units:
        key=(str(unit['dataset']),str(unit['video']))
        if key not in bank_cache:
            bank_cache[key]=load_bank(*key); teacher_cache[key]=build_teacher_cache(bank_cache[key])
        bank=bank_cache[key]; t=bank['tensors']; b,e=int(unit['begin']),int(unit['end']); rows=torch.arange(b,e,dtype=torch.long)
        y=torch.zeros(e-b,dtype=torch.bool); y[torch.as_tensor(unit['positive_indices'],dtype=torch.long)]=True
        score=l29_score(l29_model,teacher_cache[key],bank,unit,text,device)
        result.append({'unit_key':unit['unit_key'],'dataset':unit['dataset'],'video':unit['video'],'query_id':int(unit['query_id']),'sentence':unit['sentence'],'frame_id':int(unit['frame_id']),'category':unit['category'],'boxes':t['box'][rows].float().contiguous(),'frames':t['frame'][rows].long().contiguous(),'frozen_clip':t['clip'][rows].float().contiguous(),'numeric':numeric_features(t,rows).contiguous(),'y':y,'teacher':torch.as_tensor(score).float().contiguous(),'text_index':int(text['sentence_to_index'][unit['sentence']]),'image_size':list(bank['metadata'].get('image_size',[]))})
    return result
def threshold(records):
    vals=np.concatenate([np.asarray(r['score'],float) for r in records]); lo,hi=float(vals.min()-1),float(vals.max()+1); best=None
    for t in np.linspace(lo,hi,101):
        m=metric(records,float(t)); f1=2*m['candidate_precision']*m['candidate_recall']/(m['candidate_precision']+m['candidate_recall']) if m['candidate_precision']+m['candidate_recall'] else 0
        key=(f1,-m['fp_per_frame'],-float(t))
        if best is None or key>best[0]: best=(key,float(t),m)
    return {'threshold':best[1],'rule':'101-point observed calibration range +/-1; max frame aggregate F1, then lower FP/frame, then lower threshold','calibration_metrics':best[2]}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--out-root',required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=20260829); a=p.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong project root')
    if sha256_file(MANIFEST)!=EXPECTED: raise RuntimeError('manifest SHA mismatch')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    cal_all=load(DATA/'calibration_units.jsonl',{'calibration'}); val_all=load(DATA/'validation_units.jsonl',{'validation'}); cal=fixed_subset(cal_all,8,a.seed); val=fixed_subset(val_all,12,a.seed+1)
    text=torch.load(TEXT,map_location='cpu',weights_only=False); dev=torch.device(a.device)
    l29_model=L29FrameMembershipSetDecoder().to(dev); l29_model.load_state_dict(torch.load(L29,map_location=dev,weights_only=False)['model'],strict=True); l29_model.eval()
    cal_items=make_eval_items(cal,text,l29_model,dev); val_items=make_eval_items(val,text,l29_model,dev); enc=L52StreamingRegionEncoder(dev)
    ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); cfg=ck['model_config']; model=L52QueryRegionSetProbe(**{k:cfg[k] for k in ('image_dim','text_dim','frozen_dim','numeric_dim','hidden','heads','layers')}).to(dev); model.load_state_dict(ck['model'],strict=True)
    start=time.time(); records={}
    for name,kind in [('l29_teacher','l29'),('l52_step100','l52')]:
        records[name]={'calibration':make_records(model,enc,cal_items,text,dev,kind),'validation':make_records(model,enc,val_items,text,dev,kind)}
    decisions={name:threshold(x['calibration']) for name,x in records.items()}
    metrics={name:{'calibration':decisions[name]['calibration_metrics'],'validation':metric(x['validation'],decisions[name]['threshold'])} for name,x in records.items()}
    result={'format':'locatemot-l52-semantic-probe-v1','status':'pass','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'seed':a.seed,'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha256_file(Path(a.checkpoint)),'data':{'calibration_all_units':len(cal_all),'validation_all_units':len(val_all),'calibration_units_used':len(cal),'validation_units_used':len(val),'calibration_videos':sorted({x['video'] for x in cal}),'validation_videos':sorted({x['video'] for x in val}),'calibration_labels_used_for_threshold_only':True,'validation_used_for_final_report_only':True,'screening_gt_used':False,'official_test_labels_read':False},'thresholds':decisions,'metrics':metrics,'candidate_set_audit':{'truncated':False,'duplicate_keys':0,'missing_keys':0},'raw_cache_written':False,'ordinary_mot_ovmot_touched':False,'elapsed_sec':time.time()-start,'gate':{'rule':'relative to L29 validation hard violation -0.05 and recall drop <=0.01; multi-positive no collapse; complete candidate set','status':'pending_report_only'}}
    (out/'semantic.json').write_text(json.dumps(result,indent=2)+'\n'); (out/'score_records.jsonl').write_text('\n'.join(json.dumps({'model':n,**r}) for n,x in records.items() for split,rlist in x.items() for r in rlist)+'\n'); (out/'provenance.json').write_text(json.dumps({'text_cache':str(TEXT.resolve()),'text_cache_sha256':sha256_file(TEXT),'l29_checkpoint':str(L29.resolve()),'l29_checkpoint_sha256':sha256_file(L29),'clip_weights':str(CLIP_WEIGHTS.resolve()),'clip_weights_sha256':sha256_file(CLIP_WEIGHTS),'fit_only_training':True,'fixed_manifest_sha256':sha256_file(MANIFEST),'screening_gt_used':False,'official_test_labels_read':False},indent=2)+'\n'); print(json.dumps({'status':'pass','output':str(out/'semantic.json'),'metrics':metrics}))
if __name__=='__main__': main()
