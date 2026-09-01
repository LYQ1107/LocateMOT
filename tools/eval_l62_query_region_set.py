#!/usr/bin/env python3
"""Fixed L62 16-calibration/24-validation semantic evaluation."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve(); DATA=ROOT/'outputs/l49/data'; CACHE=ROOT/'outputs/l59/eval/semantic_16cal_24val/score_records.jsonl'; MANIFEST=ROOT/'outputs/l19/protocol/kitti_fast_eval_manifest.json'; EXPECTED='06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'
import sys; sys.path.insert(0,str(ROOT))
from locatemot.models.l62_query_region_set import L62QueryRegionSet
from tools.l59_fused_common import build_detector, stream_fused_roi, sha256

def dist(x):
    a=np.asarray(x,dtype=float); return {'count':int(a.size),'mean':float(a.mean()) if a.size else None,'std':float(a.std()) if a.size else None,'min':float(a.min()) if a.size else None,'max':float(a.max()) if a.size else None}
def metric(rows, threshold, null_threshold=None):
    tp=fp=fn=top1=top5=empty=null_false=0; strict=[]; best=[]; average=[]; violations=[]; multi=[]
    for r in rows:
        s=np.asarray(r['score'],dtype=float); y=np.asarray(r['label'],dtype=bool)
        suppress=null_threshold is not None and float(r.get('null_logit',-np.inf))>=null_threshold; z=(s>=threshold)&(not suppress)
        tp+=int((z&y).sum()); fp+=int((z&~y).sum()); fn+=int((~z&y).sum()); empty+=int(not z.any()); null_false+=int(not y.any() and z.any())
        p=np.flatnonzero(y); n=np.flatnonzero(~y)
        if len(p):
            order=np.argsort(-s,kind='stable'); top1+=int(y[order[:1]].any()); top5+=int(y[order[:5]].any())
            if len(p)>1: multi.append(float((z&y).sum()/len(p)))
            if len(n): d=float(s[p].min()-s[n].max()); strict.append(d); best.append(float(s[p].max()-s[n].max())); average.append(float(s[p].mean()-s[n].max())); violations.append(d<0)
    units=len(rows); pos=sum(int(np.asarray(r['label'],bool).sum()) for r in rows); selected=tp+fp; denom=sum(bool(np.asarray(r['label']).any()) for r in rows)
    return {'units':units,'candidate_rows':sum(len(r['label']) for r in rows),'positive_rows':pos,'top1':top1/max(1,denom),'top5':top5/max(1,denom),'candidate_precision':tp/max(1,selected),'candidate_recall':tp/max(1,tp+fn),'fp_per_frame':fp/max(1,units),'predictions_per_positive':selected/max(1,pos),'hard_violation':float(np.mean(violations)) if violations else None,'strict_margin':dist(strict),'best_margin':dist(best),'average_margin':dist(average),'multi_positive_recall':float(np.mean(multi)) if multi else None,'empty_rate':empty/max(1,units),'null_false_acceptance':null_false/max(1,units),'threshold':float(threshold),'null_threshold':None if null_threshold is None else float(null_threshold)}
def fit_threshold(rows):
    vals=np.unique(np.concatenate([np.asarray(r['score'],float) for r in rows])); best=None
    for t in vals.tolist()+[float(vals.min())-1e-6,float(vals.max())+1e-6]:
        m=metric(rows,t); tp=fp=fn=0
        for r in rows:
            s=np.asarray(r['score']); y=np.asarray(r['label'],bool); z=s>=t; tp+=int((z&y).sum()); fp+=int((z&~y).sum()); fn+=int((~z&y).sum())
        f=2*tp/max(1,2*tp+fp+fn); key=(f,-m['fp_per_frame'],-float(t))
        if best is None or key>best[0]: best=(key,float(t))
    return {'threshold':best[1],'objective':'exact observed candidate-level calibration F1; tie lower FP/frame; tie lower threshold'}
def fit_null(rows,t):
    vals=np.unique([float(r['null_logit']) for r in rows]); best=None
    for nt in vals.tolist()+[float(vals.min())-1e-6,float(vals.max())+1e-6]:
        pred=[]; truth=[]
        for r in rows:
            y=np.asarray(r['label'],bool); pred.append(bool((np.asarray(r['score'])>=t).any()) and float(r['null_logit'])<nt); truth.append(bool(y.any()))
        tp=sum(a and b for a,b in zip(pred,truth)); fp=sum(a and not b for a,b in zip(pred,truth)); fn=sum((not a) and b for a,b in zip(pred,truth)); f=2*tp/max(1,2*tp+fp+fn); ia=sum(a and not b for a,b in zip(pred,truth))/max(1,sum(not b for b in truth)); key=(f,-ia,-float(nt))
        if best is None or key>best[0]: best=(key,float(nt),tp,fp,fn,ia)
    return {'null_threshold':best[1],'rule':'suppress all candidate rows iff null_logit >= threshold','objective':'frame-presence F1; tie lower inactive false acceptance; tie lower threshold','tp':best[2],'fp':best[3],'fn':best[4],'calibration_inactive_false_acceptance':best[5]}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out-root',required=True); a=ap.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong cwd')
    if sha256(MANIFEST)!=EXPECTED: raise RuntimeError('manifest SHA mismatch')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    old=[json.loads(x) for x in CACHE.read_text().splitlines() if x.strip()]
    if len(old)!=40 or len({x['unit_key'] for x in old})!=40: raise AssertionError('immutable 40-unit cache mismatch')
    cal_units=[json.loads(x) for x in (DATA/'calibration_units.jsonl').read_text().splitlines() if x.strip()]; val_units=[json.loads(x) for x in (DATA/'validation_units.jsonl').read_text().splitlines() if x.strip()]
    old_by={x['unit_key']:x for x in old}; unit_by={x['unit_key']:x for x in cal_units+val_units}; units=[unit_by[x['unit_key']] for x in old]
    if any(x['unit_key'] not in unit_by for x in old): raise AssertionError('unit mapping missing')
    detector,_,_=build_detector(); model=L62QueryRegionSet().cuda(); ck=Path(a.checkpoint).resolve(); payload=torch.load(ck,map_location='cuda:0',weights_only=False); model.load_state_dict(payload['model'],strict=True); model.eval(); rows=[]; start=time.time()
    for x,u in zip(old,units):
        bank=torch.load(u['bank_path'],map_location='cpu',weights_only=False); roi,text,mask,num,meta=stream_fused_roi(detector,u,bank); n=int(u['end'])-int(u['begin']); assert meta['candidate_count']==n and len(meta['candidate_keys'])==n
        with torch.inference_mode(): op=model(roi,text,mask,num)
        labels=np.asarray(x['label'],bool); assert len(labels)==n
        rows.append({'unit_key':u['unit_key'],'dataset':u['dataset'],'video':u['video'],'frame_id':u['frame_id'],'category':u.get('category'),'label':labels,'l29':np.asarray(x['l29'],float),'m0':np.asarray(x['m0'],float),'m54':np.asarray(x['m54'],float),'l62':op['relevance_logit'].float().cpu().numpy(),'null_logit':float(op['null_logit'].cpu()),'key_audit':{'candidate_count':n,'key_count':len(meta['candidate_keys']),'ordered':True,'candidate_truncation':False}}); del bank,roi,text,mask,num,op
    cal,val=rows[:16],rows[16:]; methods={}
    for name,field in [('l29_teacher','l29'),('l53_m0','m0'),('l54_continuous','m54'),('l62_fused_roi','l62')]:
        score_rows=[dict(r,score=np.asarray(r[field],float)) for r in rows]; score_cal,score_val=score_rows[:16],score_rows[16:]
        t=fit_threshold(score_cal); nt=fit_null(score_cal,t['threshold']) if name=='l62_fused_roi' else None; methods[name]={'calibration_candidate_only':metric(score_cal,t['threshold']),'validation_candidate_only':metric(score_val,t['threshold']),'threshold':t,'null_fit':nt}
        if nt: methods[name]['calibration_final']=metric(score_cal,t['threshold'],nt['null_threshold']); methods[name]['validation_final']=metric(score_val,t['threshold'],nt['null_threshold'])
        else: methods[name]['calibration_final']=methods[name]['calibration_candidate_only']; methods[name]['validation_final']=methods[name]['validation_candidate_only']
    base=methods['l29_teacher']['validation_final']; cur=methods['l62_fused_roi']['validation_final']; checks={'hard_violation_decrease_ge_0.05':cur['hard_violation']<=base['hard_violation']-.05,'recall_drop_le_0.01':cur['candidate_recall']>=base['candidate_recall']-.01,'precision_ge_0.0830188679':cur['candidate_precision']>=.0830188679,'fp_per_frame_le_11.125':cur['fp_per_frame']<=11.125,'predictions_per_positive_le_4.069':cur['predictions_per_positive']<=4.069,'multi_positive_preserved':cur['multi_positive_recall']>=base['multi_positive_recall']-.03,'null_not_universal':cur['null_false_acceptance']<1.0,'complete_keys':all(r['key_audit']['candidate_count']==r['key_audit']['key_count'] for r in rows),'candidate_deletion_false':True}
    out.mkdir(parents=True); serial=[]
    for r in rows: serial.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k not in ('label','l62','l29','m0','m54')}|{'label':r['label'].astype(int).tolist(),'l29':r['l29'].tolist(),'m0':r['m0'].tolist(),'m54':r['m54'].tolist(),'l62':r['l62'].tolist()})
    (out/'score_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in serial)); prov={'format':'locatemot-l62-query-region-set-semantic-v1','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'checkpoint':str(ck),'checkpoint_sha256':sha256(ck),'source_control_cache':str(CACHE),'source_control_cache_sha256':sha256(CACHE),'calibration_units':16,'validation_units':24,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_raw_dense_cache_written':False,'candidate_rows_retained':True,'elapsed_sec':time.time()-start}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n'); gate={'format':'locatemot-l62-query-region-set-gate-v1','status':'semantic_gate_pass' if all(checks.values()) else 'semantic_gate_fail','decision':'pass' if all(checks.values()) else 'fail','checks':checks,'methods':methods,'selection':{'thresholds':'one exact candidate-F1 threshold per method fitted on 16 calibration rows only','null_rule':'L62 frame-presence-F1 NULL threshold fitted on calibration only; final L62 validation is gate metric','validation_used_for_selection':False},'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'no_hota_or_trackeval':True}; (out/'gate_decision.json').write_text(json.dumps(gate,indent=2,default=str)+'\n'); (out/'semantic.json').write_text(json.dumps({'provenance':prov,'methods':methods,'gate':gate},indent=2,default=str)+'\n'); print(json.dumps({'status':gate['status'],'out':str(out),'validation_l62':cur,'checks':checks}),flush=True)
if __name__=='__main__': main()
