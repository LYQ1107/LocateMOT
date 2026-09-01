#!/usr/bin/env python3
"""L55-B: immutable, CPU-only native-score slice audit.

The script consumes only the saved L53 no-label predictions and the saved
L53 score records.  It never calls the detector and never creates a cache.
The first 16 records are the pre-registered calibration slice and the next
24 are the pre-registered validation slice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
SRC = ROOT / "outputs/l53/eval/zero_shot_retry4"
MANIFEST_SHA256 = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
CAL_N = 16
VAL_N = 24


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iou_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64).reshape((-1, 4))
    bb = np.asarray(b, dtype=np.float64).reshape((-1, 4))
    if not len(aa) or not len(bb):
        return np.zeros((len(aa), len(bb)), dtype=np.float64)
    lt = np.maximum(aa[:, None, :2], bb[None, :, :2])
    rb = np.minimum(aa[:, None, 2:], bb[None, :, 2:])
    inter = np.prod(np.maximum(0.0, rb - lt), axis=2)
    area_a = np.prod(np.maximum(0.0, aa[:, 2:] - aa[:, :2]), axis=1)[:, None]
    area_b = np.prod(np.maximum(0.0, bb[:, 2:] - bb[:, :2]), axis=1)[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def finite_list(x: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(x, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def safe_mean(vals: List[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def score_summary(vals: Iterable[float]) -> Dict[str, Any]:
    v = np.asarray(list(vals), dtype=np.float64)
    if not len(v):
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {"count": int(len(v)), "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "std": float(v.std())}


def frame_metrics(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    tp = fp = positives = 0
    top1: List[float] = []
    top5: List[float] = []
    strict: List[float] = []
    best: List[float] = []
    average: List[float] = []
    violations: List[float] = []
    mp_hit = mp_total = 0
    empty = 0
    null_accept = 0
    inactive_units = 0
    all_scores: List[float] = []
    all_ious: List[float] = []
    for row in rows:
        scores = np.asarray(row["score"], dtype=np.float64)
        labels = np.asarray(row["label"], dtype=bool)
        chosen = scores >= threshold
        order = np.argsort(-scores, kind="stable")
        tp += int(np.logical_and(chosen, labels).sum())
        fp += int(np.logical_and(chosen, ~labels).sum())
        positives += int(labels.sum())
        top1.append(float(labels[order[0]]) if len(order) else 0.0)
        top5.append(float(labels[order[:5]].any()) if len(order) else 0.0)
        empty += int(not chosen.any())
        if row["category"] == "inactive":
            inactive_units += 1
            null_accept += int(chosen.any())
        pos = np.flatnonzero(labels)
        neg = np.flatnonzero(~labels)
        if len(pos) and len(neg):
            neg_max = float(scores[neg].max())
            strict.append(float(scores[pos].min() - neg_max))
            best.append(float(scores[pos].max() - neg_max))
            average.append(float(scores[pos].mean() - neg_max))
            violations.append(float(scores[pos].min() <= neg_max))
        if row["category"] == "multi_positive":
            mp_total += int(labels.sum())
            mp_hit += int(np.logical_and(chosen, labels).sum())
        all_scores.extend(scores.tolist())
        all_ious.extend(row["candidate_iou_max"].tolist())
    denom = max(1, tp + fp)
    return {
        "units": len(rows),
        "candidate_positive_count": positives,
        "predicted_count": tp + fp,
        "top1": safe_mean(top1),
        "top5": safe_mean(top5),
        "candidate_precision": float(tp / denom),
        "candidate_recall": float(tp / max(1, positives)),
        "fp_per_frame": float(fp / max(1, len(rows))),
        "pred_per_positive": float((tp + fp) / max(1, positives)),
        "hard_violation": safe_mean(violations),
        "strict_min_positive_margin": safe_mean(strict),
        "best_positive_margin": safe_mean(best),
        "average_positive_margin": safe_mean(average),
        "multi_positive_recall": float(mp_hit / max(1, mp_total)) if mp_total else None,
        "empty_rate": float(empty / max(1, len(rows))),
        "null_false_acceptance": float(null_accept / inactive_units) if inactive_units else None,
        "score_distribution": score_summary(all_scores),
        "candidate_iou_distribution": score_summary(all_ious),
    }


def fit_threshold(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = np.concatenate([np.asarray(r["score"], dtype=np.float64) for r in rows])
    candidates = np.unique(np.concatenate(([float(values.min() - 1.0)], values,
                                            [float(values.max() + 1.0)])))
    best: Tuple[Tuple[float, float, float], float, Dict[str, Any]] | None = None
    for threshold in candidates:
        metrics = frame_metrics(rows, float(threshold))
        p = metrics["candidate_precision"]
        r = metrics["candidate_recall"]
        f1 = 2.0 * p * r / max(1e-12, p + r)
        # Pre-registered tie breaking: max F1, lower FP/frame, lower threshold.
        key = (float(f1), -float(metrics["fp_per_frame"]), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return {
        "threshold": best[1],
        "rule": "calibration-only exact observed-score sweep: max frame F1; tie lower FP/frame; tie lower threshold",
        "calibration_metrics": best[2],
    }


def unavailable(reason: str) -> Dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def add_bucket_reports(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    categories = {
        "positive": lambda r: r["category"] == "positive",
        "multi_positive": lambda r: r["category"] == "multi_positive",
        "inactive_NULL": lambda r: r["category"] == "inactive",
        "present_uncovered": lambda r: r["category"] == "present_uncovered",
    }
    result: Dict[str, Any] = {}
    for name, predicate in categories.items():
        selected = [r for r in rows if predicate(r)]
        result[name] = {"status": "available", "metrics": frame_metrics(selected, threshold)}
    # The saved L53 no-label metadata contains no class/attribute annotations.
    # Do not relabel all negative rows as same-class hard negatives.
    result["same_class_hard_negative"] = unavailable(
        "no verified class/identity annotation in immutable L53 no-label cache; negative proxy is not promoted")
    result["relation_position_expression"] = unavailable(
        "no verified relation/position metadata field in immutable L53 records")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()} != {ROOT}")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    files = {name: SRC / name for name in ("predictions.json", "zero_shot.json", "jobs_no_labels.json")}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    zero = json.loads(files["zero_shot.json"].read_text())
    predictions = json.loads(files["predictions.json"].read_text())
    jobs = json.loads(files["jobs_no_labels.json"].read_text())
    records = zero["records"]
    pred_map = {x["unit_key"]: x for x in predictions}
    job_map = {x["unit_key"]: x for x in jobs}
    record_keys = [x["unit_key"] for x in records]
    duplicate_record_keys = len(record_keys) != len(set(record_keys))
    if duplicate_record_keys:
        raise AssertionError("duplicate score-record keys")
    rows: List[Dict[str, Any]] = []
    integrity_errors: List[str] = []
    for record in records:
        key = record["unit_key"]
        if key not in pred_map or key not in job_map:
            integrity_errors.append(f"missing:{key}")
            continue
        pred = pred_map[key]
        job = job_map[key]
        candidate_boxes = np.asarray(job["candidate_boxes"], dtype=np.float64)
        proposal_boxes = np.asarray(pred["pred_boxes"], dtype=np.float64).reshape((-1, 4))
        proposal_scores = np.asarray(pred["pred_scores"], dtype=np.float64)
        teacher = np.asarray(record["teacher"], dtype=np.float64)
        m0 = np.asarray(record["score"], dtype=np.float64)
        labels = np.asarray(record["label"], dtype=bool)
        if len(candidate_boxes) != len(labels) or len(candidate_boxes) != len(teacher) or len(candidate_boxes) != len(m0):
            integrity_errors.append(f"length:{key}")
            continue
        if len(proposal_boxes) != len(proposal_scores):
            integrity_errors.append(f"proposal_length:{key}")
            continue
        ov = iou_matrix(candidate_boxes, proposal_boxes)
        continuous = np.max(ov * proposal_scores[None, :], axis=1) if len(proposal_boxes) else np.zeros(len(candidate_boxes))
        iou_max = ov.max(axis=1) if len(proposal_boxes) else np.zeros(len(candidate_boxes))
        finite = all(finite_list(x) for x in (candidate_boxes, proposal_boxes, proposal_scores, teacher, m0, labels, continuous, iou_max))
        if not finite:
            integrity_errors.append(f"nonfinite:{key}")
            continue
        rows.append({
            "unit_key": key, "dataset": record["dataset"], "video": record["video"],
            "frame_id": record["frame_id"], "category": record["category"],
            "expression": job.get("expression"), "label": labels.tolist(),
            "candidate_boxes": candidate_boxes.tolist(), "teacher": teacher.tolist(),
            "l53_m0": m0.tolist(), "continuous_score": continuous.tolist(),
            "candidate_iou_max": iou_max, "proposal_count": int(len(proposal_boxes)),
            "proposal_coverage": record.get("proposal_coverage"),
        })
    if integrity_errors:
        raise AssertionError(";".join(integrity_errors[:10]))
    if len(rows) != CAL_N + VAL_N:
        raise AssertionError(f"expected {CAL_N + VAL_N} rows, got {len(rows)}")
    if set(pred_map) != set(record_keys) or set(job_map) != set(record_keys):
        raise AssertionError("prediction/job key set mismatch")

    source_hashes = {k: sha256(v) for k, v in files.items()}
    method_payload: Dict[str, Any] = {}
    for method in ("teacher", "l53_m0", "continuous_score"):
        transformed = []
        for row in rows:
            score_key = "teacher" if method == "teacher" else method
            transformed.append(dict(row, score=row[score_key]))
        cal = transformed[:CAL_N]
        val = transformed[CAL_N:]
        fit = fit_threshold(cal)
        method_payload[method] = {
            "calibration": fit,
            "validation": frame_metrics(val, fit["threshold"]),
            "calibration_buckets": add_bucket_reports(cal, fit["threshold"]),
            "validation_buckets": add_bucket_reports(val, fit["threshold"]),
        }

    payload = {
        "format": "locatemot-l55-native-score-slice-audit-v1",
        "status": "complete",
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "source_dir": str(SRC), "source_sha256": source_hashes,
        "manifest_sha256": MANIFEST_SHA256,
        "calibration_records": CAL_N, "validation_records": VAL_N,
        "slice_contract": "immutable L53 record order: first 16 calibration, next 24 validation; full candidate rows retained",
        "selection_contract": "one formula and one calibration threshold per method; frozen before validation; no top-k/NMS/deletion",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "gpu_used": False, "persistent_raw_cache_written": False,
        "raw_query_attempt": str(ROOT / "outputs/l55/audit/raw_query_attempt1/raw_query.json"),
        "raw_query_40_unit_scores_used": False,
        "integrity": {"rows": len(rows), "duplicate_record_keys": duplicate_record_keys,
                      "missing_prediction_keys": sorted(set(record_keys) - set(pred_map)),
                      "missing_job_keys": sorted(set(record_keys) - set(job_map)),
                      "candidate_key_drift": 0, "duplicate_candidate_keys": 0,
                      "nonfinite_rows": 0, "full_candidate_set_retained": True},
        "methods": method_payload,
        "metadata_limits": {"same_class_hard_negative": "unavailable",
                            "relation_position_expression": "unavailable",
                            "token_span_alignment": "UNALIGNED",
                            "note": "L53 saved score records have category labels but no verified class/attribute annotation."},
    }
    output = out / "native_score_slices.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "source_sha256": source_hashes, "manifest_sha256": MANIFEST_SHA256,
        "formulae": {"teacher": "immutable L29 record teacher", "l53_m0": "immutable L53 mapped score",
                      "continuous_score": "max_j(final_native_score_j * IoU(candidate_i, proposal_j)); zero for no overlap/no proposal"},
        "calibration_only": True, "validation_frozen": True, "labels_used_for_score": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "gpu_used": False,
    }, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(output),
                      "validation": {k: v["validation"] for k, v in method_payload.items()}}, indent=2))


if __name__ == "__main__":
    main()
