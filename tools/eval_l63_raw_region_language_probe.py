#!/usr/bin/env python3
"""Fixed 16 calibration/24 validation evaluation for L63-C."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT').resolve()
DATA = ROOT / 'outputs/l49/data'
SOURCE = ROOT / 'outputs/l59/eval/semantic_16cal_24val/score_records.jsonl'
MANIFEST = ROOT / 'outputs/l19/protocol/kitti_fast_eval_manifest.json'
EXPECTED = '06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa'
FROZEN_L29_THRESHOLD = -1.030576229095459
import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l63_raw_region_language_probe import L63RawRegionLanguageProbe
from locatemot.rmot.l49_data import load_bank, unit_features


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def units40():
    scores=[json.loads(x) for x in SOURCE.read_text().splitlines() if x.strip()]
    if len(scores)!=40: raise AssertionError(len(scores))
    by_key={}
    for fn in ('calibration_units.jsonl','validation_units.jsonl'):
        for x in (DATA/fn).read_text().splitlines():
            if x.strip():
                u=json.loads(x); by_key[u['unit_key']]=u
    out=[]
    for i,s in enumerate(scores):
        u=dict(by_key[s['unit_key']]); u['eval_split']='calibration' if i<16 else 'validation'
        u['source_scores']={'l29':s['l29'],'l53_m0':s['m0'],'l54':s['m54'],'l59':s['l59']}
        out.append(u)
    return out


def fit_threshold(rows, field):
    vals=sorted(set(float(v) for r in rows for v in r[field]))
    best=None
    for t in vals:
        tp=fp=fn=0
        for r in rows:
            s=np.asarray(r[field]); y=np.asarray(r['label'],bool); z=s>=t
            tp+=int((z&y).sum()); fp+=int((z&~y).sum()); fn+=int((~z&y).sum())
        f1=2*tp/max(1,2*tp+fp+fn); key=(f1,-fp,-t)
        if best is None or key>best[0]: best=(key,float(t))
    return {'threshold':best[1],'objective':'candidate-level F1 on 16 calibration units; ties lower FP then lower threshold'}


def fit_null(rows, threshold):
    vals=sorted(set(float(r['null_logit']) for r in rows))
    best=None
    for t in vals:
        tp=fp=fn=inactive_fp=0
        for r in rows:
            y=np.asarray(r['label'],bool); pred=bool((np.asarray(r['score'])>=threshold).any()) and float(r['null_logit'])<t
            truth=bool(y.any()); tp+=int(pred and truth); fp+=int(pred and not truth); fn+=int((not pred) and truth); inactive_fp+=int(pred and not truth)
        f1=2*tp/max(1,2*tp+fp+fn); key=(f1,-inactive_fp,-t)
        if best is None or key>best[0]: best=(key,float(t))
    return {'null_threshold':best[1],'objective':'frame-presence F1 on calibration; ties lower inactive false acceptance then lower threshold'}


def metric(rows, threshold, null_threshold):
    tp=fp=fn=top1=top5=pos=selected=empty=inactive_false=0; hard=[]; strict=[]; best=[]; avg=[]; multi=[]
    for r in rows:
        s=np.asarray(r['score'],float); y=np.asarray(r['label'],bool); pred=s>=threshold
        if null_threshold is not None and float(r['null_logit'])>=null_threshold: pred=np.zeros_like(pred)
        tp+=int((pred&y).sum()); fp+=int((pred&~y).sum()); fn+=int((~pred&y).sum()); pos+=int(y.sum()); selected+=int(pred.sum()); empty+=int(not pred.any())
        if y.any():
            order=np.argsort(-s,kind='stable'); top1+=int(y[order[0]]); top5+=int(y[order[:5]].any())
            neg=s[~y]
            if len(neg):
                strict.append(float(s[y].min()-neg.max())); best.append(float(s[y].max()-neg.max())); avg.append(float(s[y].mean()-neg.max())); hard.append(float(s[y].min()<=neg.max()))
            if r['category']=='multi_positive':
                k=int(y.sum()); multi.append(float(y[np.argsort(-s,kind='stable')[:k]].mean()))
        if r['category']=='inactive': inactive_false+=int(pred.any())
    present=sum(bool(np.asarray(r['label'],bool).any()) for r in rows)
    return {'units':len(rows),'candidate_rows':sum(len(r['label']) for r in rows),'positive_rows':pos,'top1':top1/max(1,present),'top5':top5/max(1,present),'candidate_precision':tp/max(1,tp+fp),'candidate_recall':tp/max(1,tp+fn),'fp_per_frame':fp/max(1,len(rows)),'predictions_per_positive':selected/max(1,pos),'hard_violation':float(np.mean(hard)) if hard else None,'strict_margin_mean':float(np.mean(strict)) if strict else None,'best_margin_mean':float(np.mean(best)) if best else None,'average_margin_mean':float(np.mean(avg)) if avg else None,'multi_positive_recall':float(np.mean(multi)) if multi else None,'empty_rate':empty/max(1,len(rows)),'null_false_acceptance':inactive_false/max(1,sum(r['category']=='inactive' for r in rows)),'threshold':threshold,'null_threshold':null_threshold}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--out',type=Path,default=ROOT/'outputs/l63/eval/raw_region_probe_16cal24val'); a=p.parse_args(); out=a.out.resolve()
    if out.exists() and any(out.iterdir()): raise RuntimeError(f'refusing overwrite {out}')
    out.mkdir(parents=True); start=time.time()
    if sha(MANIFEST)!=EXPECTED: raise AssertionError('manifest mismatch')
    text=torch.load(ROOT/'outputs/l48/data/text_cache.pt',map_location='cpu',weights_only=False)
    model=L63RawRegionLanguageProbe(hidden=128).cuda(); ck=torch.load(Path(a.checkpoint),map_location='cuda:0',weights_only=False); model.load_state_dict(ck['model'],strict=True); model.eval()
    rows=[]
    for u in units40():
        bank=load_bank(u['dataset'], u['video']); v=unit_features(u,bank,text,history=8)
        with torch.inference_mode(): o=model(v['clip'].cuda(),v['text'].cuda(),v['text_mask'].cuda())
        score=o['relevance_logit'].float().cpu().numpy(); n=len(score); labels=np.asarray(u['positive_indices'],dtype=int); y=np.zeros(n,bool); y[labels]=True
        rows.append({'unit_key':u['unit_key'],'dataset':u['dataset'],'video':u['video'],'frame_id':u['frame_id'],'category':u['category'],'eval_split':u['eval_split'],'label':y.astype(int).tolist(),'score':score.tolist(),'null_logit':float(o['null_logit'].cpu()),'key_audit':{'candidate_count':n,'candidate_truncation':False,'ordered':True}})
        del bank,v,o
    cal=rows[:16]; val=rows[16:]
    methods={}
    for name,field in [('l29','l29'),('l63_raw_region','score')]:
        if name=='l29':
            for r,u in zip(rows,units40()): r[field]=u['source_scores']['l29']
        t={'threshold': FROZEN_L29_THRESHOLD, 'objective': 'immutable L29 threshold from accepted L62 calibration control'} if name=='l29' else fit_threshold(cal,field)
        nt=fit_null(cal,t['threshold']) if name=='l63_raw_region' else None
        methods[name]={'calibration':metric(cal,t['threshold'],nt['null_threshold'] if nt else None),'validation':metric(val,t['threshold'],nt['null_threshold'] if nt else None),'threshold':t,'null_fit':nt}
    base=methods['l29']['validation']; cur=methods['l63_raw_region']['validation']
    checks={'hard_violation_decrease_ge_0.05':cur['hard_violation'] is not None and base['hard_violation'] is not None and cur['hard_violation']<=base['hard_violation']-.05,'recall_drop_le_0.01':cur['candidate_recall']>=base['candidate_recall']-.01,'precision_ge_l29':cur['candidate_precision']>=base['candidate_precision'],'fp_per_frame_le_l29_plus_1':cur['fp_per_frame']<=base['fp_per_frame']+1.0,'predictions_per_positive_le_4.069':cur['predictions_per_positive']<=4.069,'multi_positive_preserved':cur['multi_positive_recall'] is not None and base['multi_positive_recall'] is not None and cur['multi_positive_recall']>=base['multi_positive_recall']-.03,'null_not_universal':cur['null_false_acceptance']<1.0,'complete_keys':all(r['key_audit']['candidate_count']==len(r['label']) for r in rows),'candidate_deletion_false':True}
    payload={'format':'locatemot-l63-raw-region-language-semantic-v1','status':'complete','project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha(Path(a.checkpoint)),'source_units':str(SOURCE),'source_units_sha256':sha(SOURCE),'calibration_units':16,'validation_units':24,'methods':methods,'gate':{'status':'semantic_gate_pass' if all(checks.values()) else 'semantic_gate_fail','checks':checks,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'no_hota_or_trackeval':True},'elapsed_sec':time.time()-start}
    serial=[]
    for r in rows: serial.append(r)
    (out/'semantic.json').write_text(json.dumps(payload,indent=2)+'\n'); (out/'gate_decision.json').write_text(json.dumps(payload['gate'],indent=2)+'\n'); (out/'score_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in serial)); (out/'provenance.json').write_text(json.dumps({'project_root':str(ROOT),'cwd':str(Path.cwd().resolve()),'manifest_sha256':sha(MANIFEST),'checkpoint_sha256':sha(Path(a.checkpoint)),'calibration_units':16,'validation_units':24,'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'candidate_rows_retained':True},indent=2)+'\n')
    print(json.dumps({'status':payload['gate']['status'],'out':str(out),'validation':cur,'checks':checks},indent=2))


if __name__=='__main__': main()
