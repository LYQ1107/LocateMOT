#!/usr/bin/env python
"""Calibrate a global no-match threshold shift on calibration only.

The shift is added to the per-track NO_MATCH dummy cost (equivalently to the
no-match logit), mirroring MOTIP id_thresh / GTR overlap_thresh calibration.
Selection: maximize NO_MATCH F1 subject to calibration conditional accuracy
being at least B0's calibration conditional accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.evaluation.assignment import assign_tracks_to_candidates  # noqa: E402
from locatemot.evaluation.pair_metrics import evaluate_assignments  # noqa: E402
from tools.evaluate_l0d import make_model  # noqa: E402
from tools.l0d_analyze import (  # noqa: E402
    _load_pairs,
    _records_from_pairs,
    _cur_gt,
    _batch_tensors,
    OUT,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def raw_logits(model, split):
    pairs = _load_pairs(split)
    raw = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(pairs), 32):
            b = _batch_tensors(pairs, s, min(s + 32, len(pairs)))
            pred = model(b)
            for bi in range(len(b["ref_boxes"])):
                p = pairs[s + bi]
                M = len(p["ref_boxes"])
                N = p["n_model"]
                if N == 0:
                    raw.append((None, None))
                else:
                    raw.append((
                        pred["match_logits"][bi, :M, :N].cpu().numpy(),
                        pred["no_match_logits"][bi, :M].cpu().numpy(),
                    ))
    return pairs, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--cond-floor", type=float, default=0.75)
    args = ap.parse_args()
    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    model = make_model(args.model)
    ck = torch.load(os.path.join(ROOT, f"outputs/l0_d/checkpoints/{args.model}/best.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device)
    cal_pairs, cal_raw = raw_logits(model, "calibration")
    cal_recs = _records_from_pairs(cal_pairs)
    cal_gt = _cur_gt(cal_pairs)
    best = None
    for th in np.arange(-4.0, 4.0, 0.25):
        preds = [[] if mm is None else assign_tracks_to_candidates(mm, nn - th)
                 for mm, nn in cal_raw]
        m = evaluate_assignments(preds, cal_recs, cal_gt)
        if m["conditional_accuracy"] < args.cond_floor:
            continue
        key = (m["no_match_f1"], m["conditional_accuracy"] + m["no_match_f1"])
        if best is None or key > best[0]:
            best = (key, th, m["conditional_accuracy"], m["no_match_f1"], m["e2e_accuracy"])
    if best is None:
        best = (None, 0.0, 0.0, 0.0, 0.0)
    theta = float(best[1])
    out_dir = os.path.join(OUT, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.model}.json"), "w") as f:
        json.dump({"model": args.model, "theta": theta,
                   "calib_cond": best[2], "calib_nm_f1": best[3],
                   "calib_e2e": best[4], "cond_floor": args.cond_floor,
                   "seed": 20260806}, f, ensure_ascii=False, indent=2)
    # write calibrated held-out assignments
    ho_pairs, ho_raw = raw_logits(model, "heldout")
    ho_recs = _records_from_pairs(ho_pairs)
    preds = [[] if mm is None else assign_tracks_to_candidates(mm, nn - theta)
             for mm, nn in ho_raw]
    os.makedirs(os.path.join(OUT, "assignments_clean"), exist_ok=True)
    torch.save(preds, os.path.join(OUT, f"assignments_clean/{args.model}_calibrated.pt"))
    m = evaluate_assignments(preds, ho_recs, _cur_gt(ho_pairs))
    print(json.dumps({"theta": theta, "heldout_cond": round(m["conditional_accuracy"], 4),
                      "heldout_nm_f1": round(m["no_match_f1"], 4),
                      "heldout_e2e": round(m["e2e_accuracy"], 4),
                      "heldout_id_f1": round(m["id_f1"], 4)}, indent=2))


if __name__ == "__main__":
    main()
