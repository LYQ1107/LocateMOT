#!/usr/bin/env python3
"""Read-only audit of the L63/L62/L59 score-record contract.

This deliberately does not load a model or read any new labels.  It compares
the immutable L59 records with the accepted L62 records and audits the L63
record serialization that was used by its historical probe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
L62 = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
L59 = ROOT / "outputs/l59/eval/semantic_16cal_24val/score_records.jsonl"
L63 = ROOT / "outputs/l63/eval/raw_region_probe_16cal24val_retry2/score_records.jsonl"
OUT = ROOT / "outputs/l64/audit/control_contract"
THRESHOLD = -1.030576229095459
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fixed_metrics(rows, score_field="l29"):
    tp = fp = fn = selected = positives = 0
    top1 = top5 = empty = null_false = 0
    strict = []
    violations = []
    multi = []
    present_units = 0
    for row in rows:
        y = np.asarray(row["label"], dtype=bool)
        s = np.asarray(row[score_field], dtype=float)
        if len(y) != len(s) or not np.isfinite(s).all():
            raise ValueError(f"nonfinite/length mismatch: {row['unit_key']}")
        z = s >= THRESHOLD
        tp += int((z & y).sum())
        fp += int((z & ~y).sum())
        fn += int((~z & y).sum())
        selected += int(z.sum())
        positives += int(y.sum())
        if y.any():
            present_units += 1
            order = np.argsort(-s, kind="stable")
            top1 += int(y[order[:1]].any())
            top5 += int(y[order[:5]].any())
            neg = s[~y]
            if len(neg):
                d = float(s[y].min() - neg.max())
                strict.append(d)
                violations.append(d < 0)
            if y.sum() > 1:
                multi.append(float((z & y).sum() / y.sum()))
        else:
            null_false += int(z.any())
        empty += int(not z.any())
    return {
        "units": len(rows),
        "candidate_rows": int(sum(len(r["label"]) for r in rows)),
        "positive_rows": positives,
        "top1": top1 / max(1, present_units),
        "top5": top5 / max(1, present_units),
        "candidate_precision": tp / max(1, selected),
        "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(rows)),
        "predictions_per_positive": selected / max(1, positives),
        "hard_violation": float(np.mean(violations)) if violations else None,
        "strict_margin_mean": float(np.mean(strict)) if strict else None,
        "multi_positive_recall": float(np.mean(multi)) if multi else None,
        "empty_rate": empty / max(1, len(rows)),
        "null_false_acceptance": null_false / max(1, len(rows)),
        "threshold": THRESHOLD,
    }


def main() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("manifest SHA mismatch")
    old = read_rows(L59)
    accepted = read_rows(L62)
    historical = read_rows(L63)
    issues = []
    if len(old) != 40 or len(accepted) != 40:
        issues.append({"type": "record_count", "l59": len(old), "l62": len(accepted)})
    old_keys = [r.get("unit_key") for r in old]
    accepted_keys = [r.get("unit_key") for r in accepted]
    if old_keys != accepted_keys:
        issues.append({"type": "unit_order_or_key_mismatch", "l59_only": [k for k in old_keys if k not in accepted_keys], "l62_only": [k for k in accepted_keys if k not in old_keys]})
    by_old = {r["unit_key"]: r for r in old}
    by_acc = {r["unit_key"]: r for r in accepted}
    pair_checks = []
    for key in old_keys:
        a, b = by_old.get(key), by_acc.get(key)
        if a is None or b is None:
            continue
        check = {"unit_key": key, "labels_equal": a.get("label") == b.get("label"),
                 "candidate_length_l59": len(a.get("label", [])),
                 "candidate_length_l62": len(b.get("label", [])),
                 "l29_length_equal": len(a.get("l29", [])) == len(b.get("l29", [])),
                 "l29_max_abs_diff": None,
                 "l29_exact_equal": a.get("l29") == b.get("l29"),
                 "key_audit_l59": a.get("key_audit"),
                 "key_audit_l62": b.get("key_audit")}
        if len(a.get("l29", [])) == len(b.get("l29", [])):
            check["l29_max_abs_diff"] = float(np.max(np.abs(np.asarray(a["l29"], float) - np.asarray(b["l29"], float)))) if a["l29"] else 0.0
        pair_checks.append(check)
        if not check["labels_equal"] or check["candidate_length_l59"] != check["candidate_length_l62"] or not check["l29_length_equal"] or (check["l29_max_abs_diff"] is not None and check["l29_max_abs_diff"] > 0):
            issues.append({"type": "immutable_l29_or_label_mismatch", "unit_key": key, "check": check})

    l63_checks = []
    l63_by = {r.get("unit_key"): r for r in historical}
    for key in old_keys:
        row = l63_by.get(key)
        ref = by_old.get(key)
        if row is None or ref is None:
            issues.append({"type": "l63_missing_key", "unit_key": key})
            continue
        lengths = {field: len(row.get(field, [])) for field in ("label", "l29", "score")}
        check = {"unit_key": key, "lengths": lengths,
                 "labels_equal_l59": row.get("label") == ref.get("label"),
                 "l29_equal_l59": row.get("l29") == ref.get("l29"),
                 "score_field_present": "score" in row,
                 "score_equals_l29": row.get("score") == row.get("l29"),
                 "key_audit": row.get("key_audit")}
        l63_checks.append(check)
        if not check["labels_equal_l59"] or not check["l29_equal_l59"]:
            issues.append({"type": "l63_label_or_l29_order_mismatch", "unit_key": key})
        if len(set(lengths.values())) != 1:
            issues.append({"type": "l63_score_field_length_mismatch", "unit_key": key, "lengths": lengths})

    # Core metric is recalculated solely from accepted L62 rows.  Multi-positive
    # uses the historical all-positive threshold recall definition.
    recalculated = {"calibration": fixed_metrics(accepted[:16]), "validation": fixed_metrics(accepted[16:])}
    expected = {"validation": {"candidate_recall": 0.7333333333333333,
                                "candidate_precision": 0.0830188679245283,
                                "fp_per_frame": 10.125,
                                "predictions_per_positive": 8.833333333333334,
                                "hard_violation": 0.9166666666666666,
                                "multi_positive_recall": 0.8194444444444443}}
    metric_diffs = {k: recalculated["validation"][k] - v for k, v in expected["validation"].items()}
    payload = {
        "format": "locatemot-l64-control-contract-audit-v1",
        "status": "complete",
        "project_root": str(ROOT),
        "cwd": str(Path.cwd().resolve()),
        "sources": {"l59": str(L59), "l59_sha256": sha256(L59), "l62": str(L62), "l62_sha256": sha256(L62), "l63": str(L63), "l63_sha256": sha256(L63), "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST)},
        "alignment": {"l59_count": len(old), "l62_count": len(accepted), "l63_count": len(historical), "l59_l62_order_equal": old_keys == accepted_keys, "pair_checks": pair_checks},
        "l63_audit": {"checks": l63_checks, "score_field_overwrite_present": any(c["score_equals_l29"] for c in l63_checks)},
        "fixed_l29_threshold": THRESHOLD,
        "recalculated_l29_from_accepted_l62_records": recalculated,
        "accepted_l62_validation_reference": expected["validation"],
        "validation_metric_diffs_from_reference": metric_diffs,
        "issues": issues,
        "control_status": "VALID" if not issues and all(abs(v) < 1e-12 for v in metric_diffs.values()) else "INVALID_CONTROL",
        "selection": {"labels_used": "accepted records only for audit metrics", "screening_gt_used": False, "official_test_labels_read": False, "training": False},
        "ordinary_mot_ovmot_touched": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "control_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({"project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "source_sha256": payload["sources"], "read_only": True, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False}, indent=2) + "\n")
    print(json.dumps({"status": payload["control_status"], "issues": len(issues), "validation": recalculated["validation"], "output": str(OUT / "control_contract.json")}, indent=2))


if __name__ == "__main__":
    main()
