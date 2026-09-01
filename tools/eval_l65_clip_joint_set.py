#!/usr/bin/env python3
"""L65 fixed 16-calibration/24-validation semantic evaluation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA = ROOT / "outputs/l49/data"
IMMUTABLE = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
L64_JSON = ROOT / "outputs/l64/eval/raw_patch_16cal24val/semantic.json"
L64_RECORDS = ROOT / "outputs/l64/eval/raw_patch_16cal24val/score_records.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L29_THRESHOLD = -1.030576229095459

import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l65_clip_joint_set import L65ClipJointSet
from tools.l65_clip_joint_common import StreamingClipJoint, sha256


def fixed_units():
    lookup = {}
    for fn in ("calibration_units.jsonl", "validation_units.jsonl"):
        for line in (DATA / fn).read_text().splitlines():
            if line.strip():
                u = json.loads(line); lookup[u["unit_key"]] = u
    records = [json.loads(x) for x in IMMUTABLE.read_text().splitlines() if x.strip()]
    if len(records) != 40 or len({r["unit_key"] for r in records}) != 40: raise AssertionError("40-unit contract")
    out = []
    for i, r in enumerate(records):
        u = dict(lookup[r["unit_key"]]); u["eval_split"] = "calibration" if i < 16 else "validation"; u["record"] = r; out.append(u)
    return out


def numeric(t, b, e):
    return torch.cat((t["geometry"][b:e].float(), t["motion"][b:e].float(), t["lifecycle"][b:e].float(), t["context"][b:e].float(), t["objectness"][b:e].float().reshape(-1, 1)), 1)


def dist(x):
    a = np.asarray(x, float); return {"count": int(a.size), "mean": float(a.mean()) if a.size else None, "std": float(a.std()) if a.size else None, "min": float(a.min()) if a.size else None, "max": float(a.max()) if a.size else None}


def metric(rows, field, threshold, null_threshold=None):
    tp=fp=fn=sel=pos=top1=top5=empty=null_false=0; strict=[]; best=[]; avg=[]; viol=[]; multi=[]; vals=[]
    for r in rows:
        s=np.asarray(r[field],float); y=np.asarray(r["label"],bool); assert len(s)==len(y) and np.isfinite(s).all()
        suppress=null_threshold is not None and float(r.get("null_logit",-np.inf))>=float(null_threshold); z=(s>=threshold)&(not suppress); tp+=int((z&y).sum());fp+=int((z&~y).sum());fn+=int((~z&y).sum());sel+=int(z.sum());pos+=int(y.sum());empty+=int(not z.any());null_false+=int(not y.any() and z.any());vals.extend(s.tolist())
        p=np.flatnonzero(y); n=np.flatnonzero(~y)
        if len(p):
            order=np.argsort(-s,kind="stable");top1+=int(y[order[:1]].any());top5+=int(y[order[:5]].any())
            if len(n):
                d=float(s[p].min()-s[n].max());strict.append(d);best.append(float(s[p].max()-s[n].max()));avg.append(float(s[p].mean()-s[n].max()));viol.append(d<0)
            if len(p)>1: multi.append(float((z&y).sum()/len(p)))
    present=sum(bool(np.asarray(r["label"],bool).any()) for r in rows)
    return {"units":len(rows),"candidate_rows":int(sum(len(r["label"]) for r in rows)),"positive_rows":pos,"top1":top1/max(1,present),"top5":top5/max(1,present),"candidate_precision":tp/max(1,sel),"candidate_recall":tp/max(1,tp+fn),"fp_per_frame":fp/max(1,len(rows)),"predictions_per_positive":sel/max(1,pos),"hard_violation":float(np.mean(viol)) if viol else None,"strict_margin":dist(strict),"best_margin":dist(best),"average_margin":dist(avg),"multi_positive_recall":float(np.mean(multi)) if multi else None,"empty_rate":empty/max(1,len(rows)),"null_false_acceptance":null_false/max(1,len(rows)),"score_distribution":dist(vals),"threshold":float(threshold),"null_threshold":None if null_threshold is None else float(null_threshold)}


def fit_threshold(rows, field):
    values=np.unique(np.concatenate([np.asarray(r[field],float) for r in rows])); best=None
    for t in values.tolist()+[float(values.min())-1e-6,float(values.max())+1e-6]:
        tp=fp=fn=0
        for r in rows:
            s=np.asarray(r[field]);y=np.asarray(r["label"],bool);z=s>=t;tp+=int((z&y).sum());fp+=int((z&~y).sum());fn+=int((~z&y).sum())
        f1=2*tp/max(1,2*tp+fp+fn); key=(f1,-fp,-float(t))
        if best is None or key>best[0]:best=(key,float(t))
    return best[1]


def fit_null(rows, threshold):
    values=np.unique([float(r["null_logit"]) for r in rows]); best=None
    for nt in values.tolist()+[float(values.min())-1e-6,float(values.max())+1e-6]:
        pred=[]; truth=[]
        for r in rows:
            pred.append(bool((np.asarray(r["score"])>=threshold).any()) and float(r["null_logit"])<nt); truth.append(bool(np.asarray(r["label"],bool).any()))
        tp=sum(a and b for a,b in zip(pred,truth));fp=sum(a and not b for a,b in zip(pred,truth));fn=sum((not a) and b for a,b in zip(pred,truth));f1=2*tp/max(1,2*tp+fp+fn);key=(f1,-fp,-float(nt))
        if best is None or key>best[0]:best=(key,float(nt))
    return best[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
    if Path.cwd().resolve()!=ROOT:raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256(MANIFEST)!=EXPECTED:raise AssertionError("manifest mismatch")
    out=Path(args.out);out=out if out.is_absolute() else ROOT/out;out=out.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    units=fixed_units(); model=L65ClipJointSet(hidden=128).cuda();ck=Path(args.checkpoint).resolve();model.load_state_dict(torch.load(ck,map_location="cuda:0",weights_only=False)["model"],strict=True);model.eval();encoder=StreamingClipJoint("cuda:0",batch_size=32);rows=[];start=time.time()
    for u in units:
        bank=torch.load(Path(u["bank_path"]),map_location="cpu",weights_only=False);t=bank["tensors"];b,e=int(u["begin"]),int(u["end"]);n=e-b;patches,image=encoder.encode_unit(u["video"],int(u["frame_id"]),t["box"][b:e].float().tolist());words,valid,text_global,_=encoder.text_joint_tokens(u["sentence"]);nums=numeric(t,b,e)
        if patches.shape!=(n,17,512) or not torch.isfinite(patches).all() or not torch.isfinite(words).all():raise AssertionError(f"feature contract {u['unit_key']}")
        with torch.inference_mode():o=model(patches.cuda(),words.cuda(),valid.cuda(),nums.cuda())
        patch=patches[:,1:]; token_cos=patch@words[valid].T
        labels=list(map(int,u["record"]["label"])); score=o["relevance_logit"].float().cpu().tolist();
        rows.append({"unit_key":u["unit_key"],"dataset":u["dataset"],"video":u["video"],"frame_id":int(u["frame_id"]),"category":u["category"],"eval_split":u["eval_split"],"label":labels,"joint_zero_shot":(patches[:,0]@text_global).tolist(),"joint_point_max":token_cos.max(1).values.max(1).values.tolist(),"score":score,"null_logit":float(o["null_logit"].cpu()),"image":str(image),"key_audit":{"candidate_count":n,"candidate_rows_retained":n,"candidate_truncation":False,"ordered":True}})
        del bank,patches,words,valid,text_global,nums,o,patch,token_cos
    del encoder,model
    cal,val=rows[:16],rows[16:]; methods={}
    for name in ("joint_zero_shot","joint_point_max"):
        th=fit_threshold(cal,name);methods[name]={"calibration":metric(cal,name,th),"validation":metric(val,name,th),"threshold":{"threshold":th,"fit":"16 calibration units only"},"null_rule":"N/A; zero-shot control has no NULL head"}
    th=fit_threshold(cal,"score");nt=fit_null(cal,th);methods["l65_learned_head"]={"candidate_only_calibration":metric(cal,"score",th),"candidate_only_validation":metric(val,"score",th),"final_calibration":metric(cal,"score",th,nt),"final_validation":metric(val,"score",th,nt),"threshold":{"threshold":th,"fit":"16 calibration units only"},"null_rule":{"null_threshold":nt,"fit":"16 calibration units only","rule":"suppress all candidates when null_logit >= threshold"}}
    # Historical L64 is included as a read-only comparison, never selected.
    if L64_JSON.exists():
        l64=json.load(open(L64_JSON)); l64_methods=l64.get("methods",{}); methods["l64_historical"]={"source":str(L64_JSON),"validation":(l64_methods.get("l64_raw_patch") or l64_methods.get("l63_raw_patch") or l64_methods.get("l63_raw_region")),"selection":"historical only"}
    # L29 is reconstructed directly from the accepted immutable rows.
    l29_rows=[{"label":r["record"]["label"],"l29":r["record"]["l29"]} for r in units];methods["l29_teacher"]={"validation":metric(l29_rows[16:],"l29",L29_THRESHOLD),"calibration":metric(l29_rows[:16],"l29",L29_THRESHOLD),"threshold":{"threshold":L29_THRESHOLD,"source":"accepted immutable L62 control"}}
    base=methods["l29_teacher"]["validation"]; candidates={"joint_zero_shot":methods["joint_zero_shot"]["validation"],"joint_point_max":methods["joint_point_max"]["validation"],"l65_learned_head":methods["l65_learned_head"]["final_validation"]}; gates={}
    for name,cur in candidates.items():
        gates[name]={"hard_violation_decrease_ge_0.05":cur["hard_violation"] is not None and cur["hard_violation"]<=base["hard_violation"]-.05,"recall_floor":cur["candidate_recall"]>=.7233333,"precision_floor":cur["candidate_precision"]>=.0830188679,"fp_frame_floor":cur["fp_per_frame"]<=11.125,"predictions_per_positive_floor":cur["predictions_per_positive"]<=4.069,"multi_positive_floor":cur["multi_positive_recall"] is not None and cur["multi_positive_recall"]>=.7894444,"complete_keys":all(r["key_audit"]["candidate_count"]==len(r["score"])==len(r["label"]) for r in rows),"candidate_deletion_false":True,"null_not_universal":cur["null_false_acceptance"]<1.0}
    usable=[n for n,g in gates.items() if all(g.values())]; gate={"format":"locatemot-l65-clip-joint-semantic-gate-v1","status":"semantic_gate_pass" if usable else "semantic_gate_fail","usable_methods":usable,"checks_by_method":gates,"calibration_units":16,"validation_units":24,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"no_hota_or_trackeval":True}
    out.mkdir(parents=True);(out/"semantic.json").write_text(json.dumps({"format":"locatemot-l65-clip-joint-semantic-v1","status":"complete","project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"checkpoint":str(ck),"checkpoint_sha256":sha256(ck),"methods":methods,"gate":gate,"elapsed_sec":time.time()-start},indent=2,default=str)+"\n");(out/"gate_decision.json").write_text(json.dumps(gate,indent=2)+"\n");(out/"score_records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows));(out/"provenance.json").write_text(json.dumps({"project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"manifest_sha256":sha256(MANIFEST),"immutable_l29_source":str(IMMUTABLE),"immutable_l29_source_sha256":sha256(IMMUTABLE),"checkpoint_sha256":sha256(ck),"clip_weights":"/home/lwr/.cache/clip/ViT-B-16.pt","clip_weights_sha256":sha256(Path("/home/lwr/.cache/clip/ViT-B-16.pt")),"calibration_units":16,"validation_units":24,"threshold_fit_calibration_only":True,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"persistent_raw_dense_cache_written":False,"candidate_rows_retained":True,"token_span_alignment":"UNALIGNED","static_motion_mask":"UNALIGNED"},indent=2)+"\n");print(json.dumps({"status":gate["status"],"usable_methods":usable,"validation":candidates["l65_learned_head"],"output":str(out)},indent=2),flush=True)


if __name__=="__main__":main()
