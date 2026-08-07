#!/usr/bin/env python
"""Stage L0-D: clean-style held-out evaluation for B5/B6 + merged tables."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.evaluation.pair_metrics import evaluate_assignments  # noqa: E402
from locatemot.models.track_decoder.relation_pairwise import RelationPairwiseModel  # noqa: E402
from locatemot.models.track_decoder.relation_track_decoder import RelationTrackDecoderModel  # noqa: E402
from tools.l0d_analyze import (  # noqa: E402
    _load_pairs,
    _records_from_pairs,
    _cur_gt,
    _model_assignments,
    hard_flag,
    _bucket_gap,
    _bucket_targets,
    _bucket_density,
    OUT,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_model(name):
    if name in ("b5a",):
        return RelationPairwiseModel(use_pbd_base=False, use_region_geom=False, residual=True)
    if name == "b5b":
        return RelationPairwiseModel(use_pbd_base=True, use_region_geom=False, residual=True)
    if name == "b5c":
        return RelationPairwiseModel(use_pbd_base=True, use_region_geom=True, residual=True)
    if name == "b6":
        return RelationTrackDecoderModel(use_pbd_base=True, use_region_geom=True, residual=True)
    if name == "b6_nores":
        return RelationTrackDecoderModel(use_pbd_base=True, use_region_geom=True, residual=False)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="b5a,b5b,b5c,b6,b6_nores")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    pairs = _load_pairs("heldout")
    recs = _records_from_pairs(pairs)
    cur_gt = _cur_gt(pairs)
    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    merged = torch.load(os.path.join(OUT, "baseline_assignments_clean.pt"),
                        map_location="cpu", weights_only=False)
    for name in args.models.split(","):
        ck_path = os.path.join(ROOT, f"outputs/l0_d/checkpoints/{name}/best.pt")
        if not os.path.exists(ck_path):
            print(f"[eval] missing {ck_path}", flush=True)
            continue
        model = make_model(name)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        preds = _model_assignments(model, pairs, device, batch_size=32, official_style=False,
                                   nm_sigmoid=False)
        merged[f"B5" if name.startswith("b5") else "B6"] = preds
        merged[name.upper()] = preds
        os.makedirs(os.path.join(OUT, "assignments_clean"), exist_ok=True)
        torch.save(preds, os.path.join(OUT, f"assignments_clean/{name}.pt"))
        m = evaluate_assignments(preds, recs, cur_gt)
        print(f"[eval] {name}: cond={m['conditional_accuracy']:.4f} e2e={m['e2e_accuracy']:.4f} "
              f"nm_f1={m['no_match_f1']:.4f} id_f1={m['id_f1']:.4f}", flush=True)
    torch.save(merged, os.path.join(OUT, "assignments_clean_all.pt"))
    # merged tables
    rows = []
    groups = defaultdict(list)
    for pi, p in enumerate(pairs):
        rec = recs[pi]
        for k, v in {
            "dataset": rec["dataset"].replace("_train", ""),
            "protocol": rec["protocol"],
            "gap": _bucket_gap(rec["temporal_gap"]),
            "target_count": _bucket_targets(rec["reference_target_count"]),
            "candidate_density": _bucket_density(rec["current_candidate_count"]),
            "hard_competition": "hard" if hard_flag(p) else "easy",
        }.items():
            groups[(k, v)].append(pi)
    for model, preds in merged.items():
        m = evaluate_assignments(preds, recs, cur_gt)
        rows.append({"model": model, "group": "all", "value": "all", "samples": len(pairs),
                     "e2e": round(m["e2e_accuracy"], 4), "cond": round(m["conditional_accuracy"], 4),
                     "nm_f1": round(m["no_match_f1"], 4), "id_f1": round(m["id_f1"], 4)})
        for (dim, val), idxs in groups.items():
            mm = evaluate_assignments([preds[i] for i in idxs], [recs[i] for i in idxs], cur_gt)
            rows.append({"model": model, "group": dim, "value": val, "samples": len(idxs),
                         "e2e": round(mm["e2e_accuracy"], 4), "cond": round(mm["conditional_accuracy"], 4),
                         "nm_f1": round(mm["no_match_f1"], 4), "id_f1": round(mm["id_f1"], 4)})
    with open(os.path.join(OUT, "diagnosis/clean_stratified_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[eval] tables written")


if __name__ == "__main__":
    main()
