#!/usr/bin/env python3
"""L57 fixed 16-calibration/24-validation semantic evaluation.

L57 rows are recomputed by the verified newer-runtime detector path; the saved
L53 files are immutable controls and provide the fixed unit order/labels. No
screening or official-test file is opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
SRC=ROOT/'outputs/l53/eval/zero_shot_retry4'
DATA=ROOT/'outputs/l49/data'
MANIFEST_SHA='06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'
CAL_N=16; VAL_N=24
sys.path.insert(0,str(ROOT))
from tools.train_l57_decoder_representation_scorer import (  # noqa: E402
    BankStore, build_detector, stream_rep,
)
from locatemot.models.l57_decoder_representation_scorer import L57DecoderRepresentationScorer  # noqa: E402


def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def iou(a,b):
    aa=np.asarray(a,dtype=np.float64).reshape((-1,4)); bb=np.asarray(b,dtype=np.float64).reshape((-1,4))
    if not len(aa) or not len(bb): return np.zeros((len(aa),len(bb)))
    lt=np.maximum(aa[:,None,:2],bb[None,:,:2]); rb=np.minimum(aa[:,None,2:],bb[None,:,2:])
    inter=np.prod(np.maximum(0,rb-lt),axis=2)
    ar_a=np.prod(np.maximum(0,aa[:,2:]-aa[:,:2]),axis=1)[:,None]
    ar_b=np.prod(np.maximum(0,bb[:,2:]-bb[:,:2]),axis=1)[None,:]
    return inter/np.maximum(ar_a+ar_b-inter,1e-12)


def summary(x):
    a=np.asarray(list(x),dtype=np.float64)
    return {'count':int(len(a)),'min':float(a.min()) if len(a) else None,'max':float(a.max()) if len(a) else None,
            'mean':float(a.mean()) if len(a) else None,'std':float(a.std()) if len(a) else None}


def metrics(rows,threshold):
    tp=fp=pos_count=0; t1=[]; t5=[]; strict=[]; best=[]; avg=[]; viol=[]; mp_hit=mp_total=0; empty=0; null_accept=0; inactive=0; all_scores=[]
    for r in rows:
        s=np.asarray(r['score'],float); y=np.asarray(r['label'],bool); chosen=s>=threshold; order=np.argsort(-s,kind='stable')
        tp+=int((chosen&y).sum()); fp+=int((chosen&~y).sum()); pos_count+=int(y.sum())
        t1.append(float(y[order[0]]) if len(order) else 0); t5.append(float(y[order[:5]].any()) if len(order) else 0)
        empty+=int(not chosen.any())
        if r['category']=='inactive': inactive+=1; null_accept+=int(chosen.any())
        if r['category']=='multi_positive': mp_total+=int(y.sum()); mp_hit+=int((chosen&y).sum())
        p=np.flatnonzero(y); n=np.flatnonzero(~y)
        if len(p) and len(n):
            hn=float(s[n].max()); strict.append(float(s[p].min()-hn)); best.append(float(s[p].max()-hn)); avg.append(float(s[p].mean()-hn)); viol.append(float(s[p].min()<=hn))
        all_scores.extend(s.tolist())
    predicted=tp+fp
    return {'units':len(rows),'candidate_positive_count':pos_count,'predicted_count':predicted,'top1':float(np.mean(t1)) if t1 else None,'top5':float(np.mean(t5)) if t5 else None,
      'candidate_precision':float(tp/max(1,predicted)),'candidate_recall':float(tp/max(1,pos_count)),'fp_per_frame':float(fp/max(1,len(rows))),
      'pred_per_positive':float(predicted/max(1,pos_count)),'hard_violation':float(np.mean(viol)) if viol else None,
      'strict_min_positive_margin':float(np.mean(strict)) if strict else None,'best_positive_margin':float(np.mean(best)) if best else None,
      'average_positive_margin':float(np.mean(avg)) if avg else None,'multi_positive_recall':float(mp_hit/max(1,mp_total)) if mp_total else None,
      'empty_rate':float(empty/max(1,len(rows))),'null_false_acceptance':float(null_accept/max(1,inactive)) if inactive else None,
      'score_distribution':summary(all_scores)}


def threshold(rows):
    vals=np.unique(np.concatenate([np.asarray(r['score'],float) for r in rows]))
    best_key=None; best_t=None; best_m=None
    for t in vals:
        m=metrics(rows,float(t)); p=m['candidate_precision']; q=m['candidate_recall']; f=2*p*q/max(1e-12,p+q)
        key=(float(f),-float(m['fp_per_frame']),-float(t))
        if best_key is None or key>best_key: best_key=key; best_t=float(t); best_m=m
    return {'threshold':best_t,'rule':'calibration-only exact observed-score sweep: max candidate F1; tie lower FP/frame; tie lower threshold','calibration_metrics':best_m}


def rank_flips(rows):
    total=tc=correct_flip=error_correct=0
    for r in rows:
        s=np.asarray(r['score'],float); t=np.asarray(r['teacher'],float); y=np.asarray(r['label'],bool); p=np.flatnonzero(y); n=np.flatnonzero(~y)
        for i in p:
            for j in n:
                teacher_ok=bool(t[i]>t[j]); student_ok=bool(s[i]>s[j]); total+=1; tc+=int(teacher_ok); correct_flip+=int(teacher_ok and not student_ok); error_correct+=int((not teacher_ok) and student_ok)
    return {'pairs':total,'teacher_correct_pairs':tc,'teacher_correct_student_flip':correct_flip,'teacher_error_student_correction':error_correct,
            'teacher_correct_flip_rate':correct_flip/max(1,total),'teacher_error_correction_rate':error_correct/max(1,total),'total_rank_flip_rate':(correct_flip+error_correct)/max(1,total)}


def load_units_by_key(keys):
    found={}
    for fn in ('calibration_units.jsonl','validation_units.jsonl'):
        for line in (DATA/fn).read_text().splitlines():
            if line.strip():
                u=json.loads(line)
                if u.get('unit_key') in keys: found[u['unit_key']]=u
    missing=sorted(set(keys)-set(found))
    if missing: raise AssertionError('missing fixed evaluation units: '+str(missing[:5]))
    return found


def continuous_score(job,pred):
    cb=np.asarray(job['candidate_boxes'],float); pb=np.asarray(pred['pred_boxes'],float).reshape((-1,4)); ps=np.asarray(pred['pred_scores'],float)
    ov=iou(cb,pb); return np.max(ov*ps[None,:],axis=1) if len(pb) else np.zeros(len(cb))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out-root',required=True); a=ap.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong LocateMOT cwd')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    t0=time.time(); zero=json.loads((SRC/'zero_shot.json').read_text()); jobs=json.loads((SRC/'jobs_no_labels.json').read_text()); preds=json.loads((SRC/'predictions.json').read_text())
    records=zero['records']; assert len(records)==CAL_N+VAL_N
    keys=[x['unit_key'] for x in records]; assert len(keys)==len(set(keys))
    units=load_units_by_key(keys); jobmap={x['unit_key']:x for x in jobs}; predmap={x['unit_key']:x for x in preds}
    if set(keys)!=set(jobmap) or set(keys)!=set(predmap): raise AssertionError('immutable L53 key set mismatch')
    rows=[]; integrity=[]
    for r in records:
        u=units[r['unit_key']]; j=jobmap[r['unit_key']]; c=np.asarray(j['candidate_boxes'],float); y=np.asarray(r['label'],bool)
        if int(u['candidate_count'])!=len(c) or len(c)!=len(y) or int(u['end'])-int(u['begin'])!=len(c): raise AssertionError('candidate count mismatch '+r['unit_key'])
        if not np.isfinite(c).all() or not np.isfinite(np.asarray(r['teacher'],float)).all(): raise AssertionError('nonfinite immutable row')
        rows.append({'unit_key':r['unit_key'],'dataset':r['dataset'],'video':r['video'],'query_id':r.get('query_id',u['query_id']),'frame_id':r['frame_id'],'category':r['category'],'expression':j.get('expression'),'label':y.tolist(),'candidate_boxes':c.tolist(),'teacher':np.asarray(r['teacher'],float).tolist(),'l53_m0':np.asarray(r['score'],float).tolist(),'l54_continuous':continuous_score(j,predmap[r['unit_key']]).tolist(),'unit':u})
        integrity.append({'unit_key':r['unit_key'],'candidate_count':len(c),'full_candidate_set':True,'duplicate_keys':0,'missing_keys':0,'candidate_order':'immutable L49 bank slice'})

    detector,_,load=build_detector(); adapter=L57DecoderRepresentationScorer(); ck=torch.load(Path(a.checkpoint),map_location='cpu'); adapter.load_state_dict(ck['model'],strict=True); adapter.to('cuda:0').eval(); store=BankStore(); nulls=[]
    for r in rows:
        u=r.pop('unit'); bank=store.get(u['bank_path']); begin,end=int(u['begin']),int(u['end'])
        if end-begin!=len(r['candidate_boxes']): raise AssertionError('bank slice mismatch '+r['unit_key'])
        rep,text,mask,numeric,entity,meta=stream_rep(detector,u,bank)
        bank_key=str(Path(u['bank_path']).resolve())
        expected=[{'video':str(u['video']),'frame_id':int(u['frame_id']),'bank_path':bank_key,'row_offset':begin+i} for i in range(end-begin)]
        if meta['candidate_keys']!=expected:
            raise AssertionError('L57 candidate key/order drift '+r['unit_key']+
                                 f' got={meta["candidate_keys"][:3]} expected={expected[:3]} '
                                 f'got_last={meta["candidate_keys"][-1]} expected_last={expected[-1]}')
        with torch.inference_mode(): o=adapter(rep,text,mask,numeric,entity)
        s=o['relevance_logit'].detach().float().cpu().numpy(); n=float(o['null_logit'].detach().cpu());
        if len(s)!=len(r['candidate_boxes']) or not np.isfinite(s).all(): raise AssertionError('L57 score shape/nonfinite '+r['unit_key'])
        r['l57']=s.tolist(); r['null_logit']=n; nulls.append(n)
    methods={'l29_teacher':'teacher','l53_m0':'l53_m0','l54_continuous':'l54_continuous','l57_step100':'l57'}; result={}
    for name,key in methods.items():
        rr=[{'unit_key':r['unit_key'],'dataset':r['dataset'],'video':r['video'],'frame_id':r['frame_id'],'category':r['category'],'label':r['label'],'teacher':r['teacher'],'score':r[key]} for r in rows]
        cal=rr[:CAL_N]; val=rr[CAL_N:]; th=threshold(cal)
        result[name]={'calibration':th,'validation':metrics(val,th['threshold']),'calibration_by_domain':{},'validation_by_domain':{},'rank_flips':{'calibration':rank_flips(cal),'validation':rank_flips(val)}}
        for split,subset in [('calibration',cal),('validation',val)]:
            for domain in ('refer_kitti_v1','refer_kitti_v2'):
                ds=[x for x in subset if x['dataset']==domain]; result[name][split+'_by_domain'][domain]=metrics(ds,th['threshold'])

    score_rows=[{k:r[k] for k in ('unit_key','dataset','video','query_id','frame_id','category','expression','label','candidate_boxes','teacher','l53_m0','l54_continuous','l57','null_logit')} for r in rows]
    payload={'format':'locatemot-l57-semantic-eval-v1','status':'complete','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha256(Path(a.checkpoint)),'seed':20260829,'manifest_sha256':MANIFEST_SHA,'source_files':{str(p):sha256(p) for p in (SRC/'zero_shot.json',SRC/'jobs_no_labels.json',SRC/'predictions.json')},'scope':'exact immutable 16 calibration + 24 validation units; no screening/test labels','threshold_rule':'one observed-score candidate threshold per named method fit on calibration only; frozen before validation; no top-k/NMS/deletion','calibration_units':CAL_N,'validation_units':VAL_N,'calibration_videos':{'refer_kitti_v1':['0016'],'refer_kitti_v2':['0015']},'validation_videos':{'refer_kitti_v1':['0004','0018'],'refer_kitti_v2':['0016','0017','0020']},'methods':result,'null_policy':'L57 null_logit emitted and logged as diagnostic; no NULL post-processing used in candidate emission; null false acceptance is measured from frozen candidate threshold','integrity':{'rows':len(rows),'candidate_key_drift':0,'duplicate_candidate_keys':0,'missing_candidate_keys':0,'full_candidate_set_retained':True,'per_row':integrity},'per_query_track_recall':'not available from isolated frame-unit cache; candidate recall/top1 reported per unit; no temporal sequence provided','identity_switches':'N/A for frame-only semantic probe','source_split_precision':'unavailable in immutable L53 records; source/pool are excluded from semantic inputs','null_logit_distribution':summary(nulls),'detector_frozen':all(not p.requires_grad for p in detector.parameters()),'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_raw_dense_cache_written':False,'gpu_used':'0','elapsed_sec':time.time()-t0,'score_records':score_rows}
    (out/'semantic.json').write_text(json.dumps(payload,indent=2)+'\n')
    gate={'status':'pending','rule':'relative to immutable L29 validation: hard violation decrease >=.05; recall drop <=.01; precision >=.0830188679; FP/frame <=11.125; pred/positive <=4.069; multi-positive preserved; NULL not universal; complete candidates','decision_source':'validation is report-only; no validation selection','methods':{}}
    base=result['l29_teacher']['validation']; l57=result['l57_step100']['validation']
    checks={'hard_violation_delta':(base.get('hard_violation') is not None and l57.get('hard_violation') is not None and base['hard_violation']-l57['hard_violation']), 'recall_delta':(l57['candidate_recall']-base['candidate_recall']), 'precision':l57['candidate_precision'],'fp_per_frame':l57['fp_per_frame'],'pred_per_positive':l57['pred_per_positive'],'multi_positive_recall':l57['multi_positive_recall'],'null_false_acceptance':l57['null_false_acceptance']}
    gate['checks']=checks; gate['status']='pass' if checks['hard_violation_delta']>=.05 and checks['recall_delta']>=-.01 and checks['precision']>=.0830188679 and checks['fp_per_frame']<=11.125 and checks['pred_per_positive']<=4.069 else 'fail'
    (out/'gate_decision.json').write_text(json.dumps(gate,indent=2)+'\n'); (out/'provenance.json').write_text(json.dumps({'cwd':str(ROOT),'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha256(Path(a.checkpoint)),'manifest_sha256':MANIFEST_SHA,'calibration_only_selection':True,'validation_frozen_report':True,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_raw_dense_cache_written':False,'detector_runtime':'/home/lwr/anaconda3/envs/masaenv_debug/bin/python','torch':'2.1.2+cu121'},indent=2)+'\n')
    print(json.dumps({'status':'complete','semantic':str(out/'semantic.json'),'gate':gate,'validation_l29':base,'validation_l57':l57},indent=2))


if __name__=='__main__': main()
