#!/usr/bin/env python3
"""CPU-only L54 audit using immutable L53 predictions and records."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT')
L53=ROOT/'outputs/l53/eval/zero_shot_retry4'

def iou(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)))
    lt=np.maximum(a[:,None,:2],b[None,:, :2]); rb=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.prod(np.maximum(0,rb-lt),axis=2)
    aa=np.prod(np.maximum(0,a[:,2:]-a[:,:2]),axis=1)[:,None]
    bb=np.prod(np.maximum(0,b[:,2:]-b[:,:2]),axis=1)[None,:]
    return inter/np.maximum(aa+bb-inter,1e-9)

def metric(rows,t):
    tp=fp=pn=0; top1=[]; top5=[]; viol=[]; strict=[]; best=[]; avg=[]; mp_tp=mp_n=0; empty=0; nullfp=0
    for r in rows:
        s=np.asarray(r['score'],float); y=np.asarray(r['label'],bool); order=np.argsort(-s,kind='stable'); chosen=s>=t
        tp+=int((chosen&y).sum()); fp+=int((chosen&~y).sum()); pn+=int(y.sum())
        top1.append(float(y[order[0]]) if len(order) else 0.0); top5.append(float(y[order[:5]].any()) if len(order) else 0.0)
        empty+=int(not chosen.any()); nullfp+=int(r['category']=='inactive' and chosen.any())
        pos=np.where(y)[0]; neg=np.where(~y)[0]
        if len(pos) and len(neg):
            strict.append(float(s[pos].min()-s[neg].max())); best.append(float(s[pos].max()-s[neg].max())); avg.append(float(s[pos].mean()-s[neg].max())); viol.append(float(s[pos].min()<=s[neg].max()))
        if r['category']=='multi_positive': mp_n+=int(y.sum()); mp_tp+=int((chosen&y).sum())
    return {'units':len(rows),'top1':float(np.mean(top1)) if top1 else None,'top5':float(np.mean(top5)) if top5 else None,
            'candidate_precision':float(tp/max(1,tp+fp)),'candidate_recall':float(tp/max(1,pn)),
            'fp_per_frame':float(fp/max(1,len(rows))),'pred_per_positive':float((tp+fp)/max(1,pn)),
            'hard_violation':float(np.mean(viol)) if viol else None,'strict_min_positive_margin':float(np.mean(strict)) if strict else None,
            'best_positive_margin':float(np.mean(best)) if best else None,'average_positive_margin':float(np.mean(avg)) if avg else None,
            'multi_positive_recall':float(mp_tp/max(1,mp_n)) if mp_n else None,'empty_rate':float(empty/max(1,len(rows)),),
            'null_false_acceptance':float(nullfp/max(1,sum(r['category']=='inactive' for r in rows))) if any(r['category']=='inactive' for r in rows) else None}

def fit_threshold(rows):
    vals=np.concatenate([np.asarray(r['score'],float) for r in rows])
    candidates=np.unique(np.concatenate(([float(vals.min()-1)],vals,[float(vals.max()+1)])))
    best=None
    for t in candidates:
        m=metric(rows,float(t)); p=m['candidate_precision']; q=m['candidate_recall']; f1=2*p*q/max(1e-12,p+q)
        key=(f1,-m['fp_per_frame'],-float(t))
        if best is None or key>best[0]: best=(key,float(t),m)
    return {'threshold':best[1],'rule':'calibration-only exact observed-score sweep; max frame F1, then lower FP/frame, then lower threshold','metrics':best[2]}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out-root',required=True); a=p.parse_args()
    if Path.cwd().resolve()!=ROOT: raise RuntimeError('wrong LocateMOT cwd')
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out=out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    source=json.loads((L53/'zero_shot.json').read_text()); preds={x['unit_key']:x for x in json.loads((L53/'predictions.json').read_text())}
    if len(preds)!=len(source['records']): raise AssertionError('prediction/record key mismatch')
    jobs_by_key={x['unit_key']:x for x in json.loads((L53/'jobs_no_labels.json').read_text())}
    rows=[]
    for r in source['records']:
        p=preds[r['unit_key']]; cand=np.asarray(json.loads((L53/'jobs_no_labels.json').read_text())[0]['candidate_boxes']) if False else None
        cand=np.asarray(jobs_by_key[r['unit_key']]['candidate_boxes'],float); pb=np.asarray(p['pred_boxes'],float); ps=np.asarray(p['pred_scores'],float)
        ov=iou(cand,pb)
        continuous=np.max(ps[None,:]*ov,axis=1) if len(pb) else np.zeros(len(cand))
        geom=np.max(ov,axis=1) if len(pb) else np.zeros(len(cand))
        rows.append({'unit_key':r['unit_key'],'dataset':r['dataset'],'video':r['video'],'frame_id':r['frame_id'],'category':r['category'],'label':r['label'],'l53_m0':r['score'],'continuous_score':continuous.tolist(),'geometry_score':geom.tolist(),'proposal_count':len(pb),'proposal_coverage':r['proposal_coverage']})
    cal_keys={r['unit_key'] for r in rows[:16]}; val_keys={r['unit_key'] for r in rows[16:]}
    methods={}
    for name in ['l53_m0','continuous_score','geometry_score']:
        rr=[dict(r,score=r[name]) for r in rows]; c=[r for r in rr if r['unit_key'] in cal_keys]; v=[r for r in rr if r['unit_key'] in val_keys]
        th=fit_threshold(c); methods[name]={'threshold':th,'calibration':th['metrics'],'validation':metric(v,th['threshold'])}
    key_audit={'records':len(rows),'prediction_keys':len(preds),'duplicate_keys':len(preds)!=len(rows),'missing_keys':sorted(set(x['unit_key'] for x in rows)-set(preds)),'full_candidate_rows':all(len(r['continuous_score'])==len(r['label']) for r in rows),'finite':bool(all(np.isfinite(r['continuous_score']).all() and np.isfinite(r['geometry_score']).all() for r in rows))}
    payload={'format':'locatemot-l54-score-contract-audit-v1','status':'complete','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'source_l53':str(L53/'zero_shot.json'),'calibration_records':16,'validation_records':24,'calibration_selection_only':True,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'mapping_contract':{'name':'continuous_final_proposal_score_times_iou','formula':'max_j(final_native_proposal_score_j * IoU(candidate_i, proposal_j))','proposal_source':'immutable L53 top-300 final scores only; raw 900-query logits unavailable in cache','unmatched_score':0.0,'iou_lower_bound':0.0,'candidate_rows_retained':True,'gt_used_for_score':False},'methods':methods,'key_audit':key_audit,'rows':rows}
    (out/'score_contract_audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'status':'complete','output':str(out/'score_contract_audit.json'),'methods':{k:v['validation'] for k,v in methods.items()}}))
if __name__=='__main__': main()
