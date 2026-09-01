#!/usr/bin/env python
"""Stage L0-D: build two-frame sequences and run official TrackEval."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Official TrackEval (commit 12c8791b) uses numpy aliases removed in modern
# NumPy. Provide compatibility aliases without modifying the reference repo.
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

from locatemot.evaluation.two_frame_trackeval import TwoFrameLocateMOT  # noqa: E402
from trackeval.eval import Evaluator  # noqa: E402
from trackeval.metrics.hota import HOTA  # noqa: E402
from trackeval.metrics.clear import CLEAR  # noqa: E402
from trackeval.metrics.identity import Identity  # noqa: E402
from tools.l0d_analyze import hard_flag, _bucket_targets  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs/l0_d")

MODEL_KEYS = ["B0_IoU", "B2_PBDCos", "B2_PBDBoxEnd", "B3_PairwiseMLP", "B4_TrackDecoder"]


def _seq_ids_by_key(records):
    ids = {}
    for i, r in enumerate(records):
        ids[(r["current_token_id"], r["temporal_gap"], tuple(t["track_id"] for t in r["reference_targets"]),
             r["protocol"])] = i
    return ids


def build_arrays(pairs, assignments, subset_idx=None):
    """Build per-sequence GT/tracker arrays for the two-frame diagnostic."""
    idxs = range(len(pairs)) if subset_idx is None else subset_idx
    S = len(idxs)
    gt0 = [None] * S
    gt1 = [None] * S
    tr0 = [None] * S
    tr1 = [None] * S
    seq_ids = np.zeros(S, dtype=np.int64)
    for k, i in enumerate(idxs):
        p = pairs[i]
        M = len(p["ref_boxes"])
        seq_ids[k] = i
        ref_id_map = {t["track_id"]: r for r, t in enumerate(p["reference_targets"])}
        # frame0 GT = references
        b0 = np.asarray(p["ref_boxes"], dtype=np.float64)
        i0 = np.arange(1, M + 1, dtype=np.int64)
        gt0[k] = (b0, i0)
        tr0[k] = (b0.copy(), i0.copy())
        # frame1 GT = refs present (assigned or candidate_missing)
        b1, i1 = [], []
        for t in p["assignment_targets"]:
            gb = p["cur_gt"].get(str(t["track_id"]))
            if gb is None:
                continue
            b1.append(gb)
            i1.append(ref_id_map[t["track_id"]] + 1)
        for tid in p["candidate_missing_targets"]:
            gb = p["cur_gt"].get(str(tid))
            if gb is None:
                continue
            b1.append(gb)
            i1.append(ref_id_map[tid] + 1)
        gt1[k] = (np.asarray(b1, dtype=np.float64) if b1 else np.empty((0, 4)),
                  np.asarray(i1, dtype=np.int64))
        # frame1 tracker
        tb, ti = [], []
        used = set()
        assign = assignments[i]
        by_idx = {ti_: tag for ti_, tag in assign}
        for r, t in enumerate(p["reference_targets"]):
            tag = by_idx.get(r)
            if tag is not None and tag.startswith("candidate:"):
                j = int(tag.split(":")[1])
                if j < len(p["cur_boxes"]):
                    tb.append(p["cur_boxes"][j])
                    ti.append(r + 1)
                    used.add(j)
        new_id = max(M + 1, 1000)
        for j in range(p["n_model"]):
            if j not in used:
                tb.append(p["cur_boxes"][j])
                ti.append(new_id)
                new_id += 1
        tr1[k] = (np.asarray(tb, dtype=np.float64) if tb else np.empty((0, 4)),
                  np.asarray(ti, dtype=np.int64))
    # stack to fixed max dims per side
    def stack(rows, side0_max, side1_max):
        g0 = np.zeros((len(rows), side0_max, 4), dtype=np.float64)
        g1 = np.zeros((len(rows), side1_max, 4), dtype=np.float64)
        i0 = np.zeros((len(rows), side0_max), dtype=np.int64)
        i1 = np.zeros((len(rows), side1_max), dtype=np.int64)
        for k, (a, ia, b, ib) in enumerate(rows):
            g0[k, :len(ia)] = a
            i0[k, :len(ia)] = ia
            g1[k, :len(ib)] = b
            i1[k, :len(ib)] = ib
        return g0, i0, g1, i1
    rows = [(gt0[k][0], gt0[k][1], gt1[k][0], gt1[k][1]) for k in range(S)]
    max0 = max((r[1].shape[0] for r in rows), default=1)
    max1 = max((r[3].shape[0] for r in rows), default=1)
    g0, i0, g1, i1 = stack(rows, max0, max1)
    rows_t = [(tr0[k][0], tr0[k][1], tr1[k][0], tr1[k][1]) for k in range(S)]
    max0t = max((r[1].shape[0] for r in rows_t), default=1)
    max1t = max((r[3].shape[0] for r in rows_t), default=1)
    t0, ti0, t1, ti1 = stack(rows_t, max0t, max1t)
    return {
        "seq_ids": seq_ids,
        "gt_boxes0": g0, "gt_ids0": i0, "gt_boxes1": g1, "gt_ids1": i1,
        "tr_boxes0": t0, "tr_ids0": ti0, "tr_boxes1": t1, "tr_ids1": ti1,
    }


def run_eval(npz_path, seq_ids=None, out_prefix=None):
    dataset = TwoFrameLocateMOT({
        "DATA_FILE": npz_path,
        "SEQ_IDS": None if seq_ids is None else seq_ids.tolist(),
        "PRINT_CONFIG": False,
        "DO_PREPROC": False,
        "TRACKERS_TO_EVAL": ["model"],
        "TRACKER_DISPLAY_NAMES": ["model"],
    })
    evaluator = Evaluator({
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    })
    metrics = [HOTA(), CLEAR(), Identity()]
    res, _ = evaluator.evaluate([dataset], metrics, show_progressbar=False)
    dataset_name = list(res.keys())[0]
    combined = res[dataset_name]["model"]["COMBINED_SEQ"]["pedestrian"]
    out = {}
    for metric_name, vals in combined.items():
        for k, v in vals.items():
            out[f"{metric_name}_{k}"] = float(v) if np.isscalar(v) else \
                (v.tolist() if hasattr(v, "tolist") else v)
    # scalar alpha=0.05 fields
    for prefix, field in (("HOTA", "HOTA"), ("DetA", "DetA"), ("AssA", "AssA"), ("LocA", "LocA")):
        arr = out.get(f"HOTA_{field}")
        if isinstance(arr, list) and arr:
            out[f"HOTA_{field}(0)"] = float(arr[0])
    if out_prefix:
        with open(out_prefix + ".json", "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign-file", default=os.path.join(OUT, "baseline_assignments_clean.pt"))
    ap.add_argument("--models", default=",".join(MODEL_KEYS))
    ap.add_argument("--out", default=os.path.join(OUT, "trackeval"))
    ap.add_argument("--stratified", action="store_true")
    args = ap.parse_args()
    pairs = torch.load(os.path.join(OUT, "pairs_heldout.pt"), map_location="cpu", weights_only=False)
    assigns = torch.load(args.assign_file, map_location="cpu", weights_only=False)
    os.makedirs(args.out, exist_ok=True)
    models = [m for m in args.models.split(",") if m in assigns]
    summary = {}
    for model in models:
        arr = build_arrays(pairs, assigns[model])
        npz = os.path.join(args.out, f"seqs_{model}.npz")
        np.savez(npz, **arr)
        res = run_eval(npz, out_prefix=os.path.join(args.out, model))
        summary[model] = res
        if args.stratified:
            groups = {}
            for pi, p in enumerate(pairs):
                rec = p["rec"]
                for dim, val in {
                    "target_count": _bucket_targets(rec["reference_target_count"]),
                    "dataset": rec["dataset"].replace("_train", ""),
                    "protocol": rec["protocol"],
                    "hard_competition": "hard" if hard_flag(p) else "easy",
                }.items():
                    groups.setdefault((dim, val), []).append(pi)
            for (dim, val), idxs in groups.items():
                sub = build_arrays(pairs, assigns[model], subset_idx=idxs)
                sub_npz = os.path.join(args.out, f"seqs_{model}_{dim}_{val}.npz")
                np.savez(sub_npz, **sub)
                sres = run_eval(sub_npz, seq_ids=sub["seq_ids"],
                                out_prefix=os.path.join(args.out, f"{model}_{dim}_{val}"))
                summary[f"{model}|{dim}|{val}"] = sres
        print(model, {k: round(v, 4) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in res.items() if k in
                      ("HOTA_HOTA(0)", "HOTA_DetA(0)", "HOTA_AssA(0)", "HOTA_LocA(0)",
                       "CLEAR_MOTA", "CLEAR_MOTP", "Identity_IDF1", "CLEAR_FP", "CLEAR_FN",
                       "CLEAR_IDSW", "CLEAR_Frag", "CLEAR_MT", "CLEAR_PT", "CLEAR_ML")})
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
