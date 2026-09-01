"""Stratified metrics grouped by dataset/protocol/gap/target count etc."""
from __future__ import annotations

import csv
from collections import defaultdict

from .pair_metrics import evaluate_assignments


def _bucket_gap(gap):
    if gap <= 4:
        return "1-4"
    if gap <= 16:
        return "5-16"
    if gap <= 64:
        return "17-64"
    return ">64"


def _bucket_targets(m):
    if m == 1:
        return "1"
    if m <= 4:
        return "2-4"
    return "5-8"


def _bucket_density(n):
    if n <= 5:
        return "0-5"
    if n <= 15:
        return "6-15"
    return ">15"


def stratify(pred_assignments, records, cur_gt_boxes_by_key, out_dir):
    groups = defaultdict(list)
    for rec, preds in zip(records, pred_assignments):
        keys = {
            "dataset": rec["dataset"].replace("_train", ""),
            "protocol": rec["protocol"],
            "gap": _bucket_gap(rec["temporal_gap"]),
            "target_count": _bucket_targets(rec["reference_target_count"]),
            "candidate_density": _bucket_density(rec.get("current_candidate_count", 0)),
            "candidate_presence": "present" if rec["visible_positives"] > 0 else "missing",
        }
        for k, v in keys.items():
            groups[(k, v)].append((rec, preds))
    rows = []
    for (dim, val), items in groups.items():
        recs = [x[0] for x in items]
        preds = [x[1] for x in items]
        m = evaluate_assignments(preds, recs, cur_gt_boxes_by_key)
        rows.append({
            "dimension": dim, "value": val, "samples": len(recs),
            "e2e_accuracy": round(m["e2e_accuracy"], 4),
            "conditional_accuracy": round(m["conditional_accuracy"], 4),
            "no_match_f1": round(m["no_match_f1"], 4),
        })
    path = f"{out_dir}/stratified_metrics.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dimension"])
        w.writeheader()
        w.writerows(rows)
    return rows
