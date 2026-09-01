#!/usr/bin/env python3
"""Summarize completed L27 A/B score caches and run frozen fast TrackEval."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l27_fast_rmot import (MANIFEST, V5, base_metrics, calibrate,
                                      make_entries, selection,
                                      pool_frame_units,
                                      sha, threshold_metrics,
                                      trackeval_outputs)

def cache_name(entry):
    return f"{entry['video']}_{hashlib.sha1(entry['expression'].encode()).hexdigest()[:12]}.npz"


def load_caches(score_root, entries, model_names=("A_C1_S2000", "B_F4_bounded_residual")):
    arrays = {model: {} for model in model_names}
    for model in model_names:
        directory = score_root / "scores" / model
        for entry in entries:
            key = (entry["video"], entry["expression"]); path = directory / cache_name(entry)
            if not path.exists(): raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as z:
                arrays[model][key] = {x: np.asarray(z[x]) for x in z.files}
    return arrays


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--score-root", required=True); ap.add_argument("--out-root", required=True)
    args = ap.parse_args(); score_root = (ROOT / args.score_root).resolve(); out = (ROOT / args.out_root).resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); entries = make_entries(); summaries={}; trackeval={}
    # Keep one model's cache in memory at a time; the NPZs are immutable score
    # provenance and this avoids retaining both 160-query arrays during
    # TrackEval materialization.
    for model_name in ("A_C1_S2000", "B_F4_bounded_residual"):
        loaded = load_caches(score_root, entries, (model_name,)); qs = loaded[model_name]
        cal_items=[qs[(x["video"],x["expression"])] for x in entries if x["split"]=="calibration"]
        cal=pool_frame_units(cal_items); calibration=calibrate(cal)
        print(json.dumps({"model":model_name,"null_max_samples":calibration.get("null_max_samples"),"null_max_threshold":calibration.get("null_max_threshold"),"null_max_fallback":calibration.get("null_max_fallback"),"gap_samples":calibration.get("gap_samples"),"gap_threshold":calibration.get("gap_threshold"),"gap_fallback":calibration.get("gap_fallback")}), flush=True)
        summaries[model_name]={"calibration":calibration,"calibration_ranking":base_metrics(cal)}
        for strat_name in ("precision_first","recall_first","balanced"):
            threshold=calibration[strat_name]["threshold"]
            for strategy in ("threshold","null_max","gap_top1","threshold_top1"):
                nt=calibration["null_max_threshold"] if strategy=="null_max" else None
                gt=calibration["gap_threshold"] if strategy=="gap_top1" else None
                items=[(x,qs[(x["video"],x["expression"])]) for x in entries if x["split"]=="screening"]
                screen=pool_frame_units([d for _e,d in items])
                chosen=selection(screen,threshold,strategy,nt,gt); key=f"{strat_name}__{strategy}"
                query_rows=[]
                for entry,data in items:
                    qsel=selection(data,threshold,strategy,nt,gt)
                    qm=threshold_metrics(data,qsel); qm.update({"video":entry["video"],"expression":entry["expression"],"query_index":int(entry["query_index"])})
                    query_rows.append(qm)
                ranking=base_metrics(screen); tm=threshold_metrics(screen,chosen)
                summaries[model_name][key]={"threshold":threshold,"strategy":strategy,"null_threshold":nt,"gap_threshold":gt,"ranking":ranking,"threshold_metrics":tm,"query_metrics":query_rows,"worst_query_frame_recall":sorted(query_rows,key=lambda x:(x["frame_recall"],x["frame_f1"]))[:5],"screening_gt_used_for_threshold":False}
                trackeval.setdefault(model_name,{})[key]=trackeval_outputs(out,model_name,key,threshold,{model_name: qs},entries,"screening",nt,gt,selection_strategy=strategy)
        del loaded
    payload={"format":"locatemot-l27-fast-rmot-summary-v1","manifest":str(MANIFEST),"manifest_sha256":sha(MANIFEST),"score_root":str(score_root),"checkpoint_provenance":{"A_C1_S2000":str((ROOT/'outputs/l26/train/C1_crossmodal_adapter_S2000/checkpoint_c1_step2000.pt').resolve()),"B_F4_bounded_residual":str((ROOT/'outputs/l26/fallback/F4_bounded_residual_S500_retry/checkpoint_bounded_residual_step500.pt').resolve())},"v5_root":str(V5),"query_counts":{"all":len(entries),"calibration":64,"screening":96},"calibration_labels_used_for_threshold":True,"screening_gt_used_for_threshold":False,"candidate_metrics":summaries,"trackeval":trackeval}
    (out/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"out":str(out),"models":list(trackeval),"strategy_count":sum(len(x) for x in trackeval.values())},indent=2),flush=True)


if __name__=="__main__": main()
