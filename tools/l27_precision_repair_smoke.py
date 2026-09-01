#!/usr/bin/env python3
"""One minimal post-score precision repair smoke for L27.

The repair is deliberately output-side: calibration precision-first threshold
plus an at-most-two-candidates-per-frame cap. It does not modify a model,
backbone, tracker or GT and is evaluated on the first 100 screening frame
units only as a targeted smoke.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.eval_l27_fast_rmot import base_metrics, calibrate, frame_groups, pool_frame_units, selection
from tools.summarize_l27_fast_rmot import cache_name, load_caches


def smoke(data, threshold, steps=100):
    y=data["label"].astype(bool); s=data["score"]; chosen=np.zeros(len(s),bool); rows=[]; used=0
    for frame, idx in frame_groups(data):
        if used >= steps: break
        order=idx[np.argsort(-s[idx],kind="stable")]; keep=order[:2]; keep=keep[s[keep]>=threshold]; chosen[keep]=True
        rows.append((idx,chosen[idx],y[idx])); used += 1
    if not rows: return {"smoke_frame_units":0}
    selected=np.concatenate([x[1] for x in rows]); labels=np.concatenate([x[2] for x in rows]); fp_per=[]; null_accept=[]; empty=[]
    for idx,keep,lab in rows:
        pred=bool(keep.any()); pos=bool(lab.any()); empty.append(not pred); null_accept.append((not pos) and pred); fp_per.append(int((keep & ~lab).sum()))
    tp=int((selected&labels).sum()); fp=int((selected&~labels).sum())
    return {"smoke_frame_units":used,"selected":int(selected.sum()),"positive_rows":int(labels.sum()),"tp":tp,"fp":fp,"precision":float(tp/max(1,tp+fp)),"recall":float(tp/max(1,int(labels.sum()))),"false_positive_candidates_per_frame":float(np.mean(fp_per)),"empty_output_rate":float(np.mean(empty)),"null_frame_false_acceptance":float(np.mean(null_accept))}


def main():
    out=ROOT/"outputs/l27/repair/precision_top2_smoke100_final"; out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/"outputs/l19/protocol/kitti_fast_eval_manifest.json").read_text()); entries=[]
    from tools.eval_l27_fast_rmot import make_entries
    entries=make_entries(); all_results={}; score_root=ROOT/"outputs/l27/fast_rmot_validation_retry"
    # Reuse the completed immutable calibration sweep. Re-running the full
    # threshold grid is unnecessary for a 100-frame smoke and is much more
    # expensive than the repair itself.
    formal=json.loads((ROOT/"outputs/l27/fast_rmot_validation_formal/summary.json").read_text())
    for model in ("A_C1_S2000","B_F4_bounded_residual"):
        qs=load_caches(score_root,entries,(model,))[model]
        c=formal["candidate_metrics"][model]["calibration"]
        null_threshold=c.get("null_max_threshold")
        gap_threshold=c.get("gap_threshold")
        threshold_value=c["precision_first"].get("threshold")
        if threshold_value is None:
            raise ValueError(f"{model}: calibration precision-first threshold is None")
        threshold=float(threshold_value)
        screen=pool_frame_units([qs[(e["video"],e["expression"])] for e in entries if e["split"]=="screening"])
        all_results[model]={"repair":"calibration precision-first threshold + max-two-candidates-per-frame","threshold":threshold,"calibration_provenance":{"null_max_samples":c["null_max_samples"],"null_max_threshold":null_threshold,"null_max_fallback":c.get("null_max_fallback") if null_threshold is not None else "explicit_no_null","gap_samples":c["gap_samples"],"gap_threshold":gap_threshold,"gap_fallback":c.get("gap_fallback") if gap_threshold is not None else "explicit_no_gap","precision_threshold_selected":c["precision_first"].get("calibration_metrics",{}).get("selected")},"screening_gt_used_for_repair_selection":False,"smoke":smoke(screen,threshold,100)}
        del qs, screen
    report={"format":"locatemot-l27-precision-repair-smoke-v1","manifest":str(ROOT/"outputs/l19/protocol/kitti_fast_eval_manifest.json"),"repair_scope":"one output-side precision repair; no model/backbone/tracker change","models":all_results}
    (out/"smoke.json").write_text(json.dumps(report,indent=2)+"\n"); (out/"README.md").write_text("# L27 precision repair smoke\n\nCalibration precision-first threshold plus an at-most-two-candidates-per-frame cap; 100 screening frame units per model.\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
