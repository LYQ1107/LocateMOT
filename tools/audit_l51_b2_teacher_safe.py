#!/usr/bin/env python3
"""Calibration-only audit of teacher-safe L51 residual contracts.

The reader stops at the first non-calibration record so validation/test labels
are not consumed. No image or dense feature is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
CHECKPOINT_SHA = "38685e38a63e2ccaef4b9036bec449b8ae9fb2bc2d14d3f31e52c68e030e52bd"
THRESHOLDS = {"refer_kitti_v1": -0.4991305879518098,
              "refer_kitti_v2": -1.1109599880143708}
BOUND = 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dist(values):
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None,
                "q10": None, "q90": None}
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "median": float(np.median(x)), "q10": float(np.quantile(x, .1)),
            "q90": float(np.quantile(x, .9))}


def load_calibration(path: Path):
    rows = []
    stopped = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "calibration":
                stopped = True
                break
            if row.get("dataset") not in THRESHOLDS:
                raise RuntimeError(f"unexpected calibration dataset: {row.get('dataset')}")
            if str(row.get("video")) not in {"0016", "0015"}:
                raise RuntimeError(f"unexpected calibration video: {row.get('video')}")
            rows.append(row)
    if not rows:
        raise RuntimeError("no calibration rows found")
    return rows, stopped


def relation_order(score):
    return np.argsort(-score, kind="stable")


def metrics(rows, variant, bound=None):
    tp = fp = fn = empty = null_false = top1 = top5 = 0
    positive_units = multi_units = multi_recall = 0
    strict, best, average, saturation = [], [], [], []
    source_counts = {}
    teacher_error = teacher_correct = corrected = correct_flips = all_flips = 0
    rank_changed = rank_relation_changed = 0
    candidate_count = residual_count = 0
    score_values = []
    for row in rows:
        score = np.asarray(variant(row), dtype=np.float64)
        teacher = np.asarray(row["teacher_score"], dtype=np.float64)
        raw_residual = np.asarray(row["residual"], dtype=np.float64)
        label = np.asarray(row["label"], dtype=bool)
        if score.size != label.size or teacher.size != label.size:
            raise RuntimeError(f"candidate shape mismatch at {row.get('unit_key')}")
        if not np.isfinite(score).all():
            raise RuntimeError(f"nonfinite score at {row.get('unit_key')}")
        candidate_count += int(label.size); residual_count += int(raw_residual.size)
        score_values.extend(score.tolist())
        threshold = THRESHOLDS[row["dataset"]]
        order = relation_order(score)
        chosen = score >= threshold
        tp += int((chosen & label).sum()); fp += int((chosen & ~label).sum())
        fn += int((~chosen & label).sum()); empty += int(not chosen.any())
        null_false += int(not label.any() and chosen.any())
        for name, mask in row.get("sources", {}).items():
            mask = np.asarray(mask, dtype=bool)
            entry = source_counts.setdefault(name, [0, 0])
            entry[0] += int((chosen & mask).sum()); entry[1] += int((chosen & mask & label).sum())
        pos = np.flatnonzero(label); neg = np.flatnonzero(~label)
        if len(pos):
            positive_units += 1
            top1 += int(label[order[:1]].any()); top5 += int(label[order[:5]].any())
            if len(pos) > 1:
                multi_units += 1; multi_recall += float((chosen & label).sum() / len(pos))
            if len(neg):
                strict.append(float(score[pos].min() - score[neg].max()))
                best.append(float(score[pos].max() - score[neg].max()))
                average.append(float(score[pos].mean() - score[neg].max()))
                for i in pos:
                    for j in neg:
                        t_ok = teacher[i] > teacher[j]
                        s_ok = score[i] > score[j]
                        teacher_correct += int(t_ok); teacher_error += int(not t_ok)
                        corrected += int((not t_ok) and s_ok)
                        correct_flips += int(t_ok and not s_ok)
                        all_flips += int(t_ok != s_ok)
        raw = np.asarray(row["score"], dtype=np.float64)
        raw_order = relation_order(raw)
        rank_changed += int(not np.array_equal(raw_order, order))
        rank_relation_changed += int(any((raw[i] > raw[j]) != (score[i] > score[j])
                                         for i in range(len(raw)) for j in range(i + 1, len(raw))))
        if bound is not None:
            vals = np.asarray(variant._residual(row), dtype=np.float64)
            saturation.extend(np.abs(vals).tolist())
    selected = tp + fp
    result = {
        "frame_units": len(rows), "candidate_rows": candidate_count,
        "positive_rows": int(sum(np.asarray(r["label"], dtype=bool).sum() for r in rows)),
        "precision": tp / max(1, selected), "recall": tp / max(1, tp + fn),
        "top1_frame_recall": top1 / max(1, positive_units),
        "top5_frame_recall": top5 / max(1, positive_units),
        "false_positive_candidates_per_frame": fp / max(1, len(rows)),
        "predictions_per_positive": selected / max(1, int(sum(np.asarray(r["label"], dtype=bool).sum() for r in rows))),
        "multi_positive_frame_count": multi_units,
        "multi_positive_recall": multi_recall / max(1, multi_units),
        "empty_output_rate": empty / max(1, len(rows)),
        "null_frame_false_acceptance": null_false / max(1, len(rows)),
        "strict_min_positive_margin": dist(strict), "best_positive_margin": dist(best),
        "average_positive_margin": dist(average),
        "hard_violation_rate": float(np.mean(np.asarray(strict) < 0)) if strict else None,
        "source_precision": {k: {"accepted": v[0], "true_positive": v[1],
                                  "precision": v[1] / max(1, v[0])}
                             for k, v in sorted(source_counts.items())},
        "score_distribution": dist(score_values),
        "teacher_error_correction_rate": corrected / max(1, teacher_error),
        "teacher_correct_flip_rate": correct_flips / max(1, teacher_correct),
        "pair_flip_rate_vs_teacher": all_flips / max(1, teacher_error + teacher_correct),
        "teacher_error_pairs": teacher_error, "teacher_correct_pairs": teacher_correct,
        "teacher_error_corrected_pairs": corrected,
        "teacher_correct_flipped_pairs": correct_flips,
        "raw_rank_changed_frame_count": rank_changed,
        "raw_pair_relation_changed_frame_count": rank_relation_changed,
    }
    if bound is not None:
        result["residual_distribution"] = dist(saturation)
        result["residual_bound"] = bound
        result["residual_bound_satisfied"] = bool(saturation and max(abs(x) for x in saturation) <= bound + 1e-6)
        result["near_bound_fraction"] = float(np.mean(np.abs(saturation) >= .95 * bound)) if saturation else None
    else:
        result["residual_bound"] = None
        result["residual_bound_satisfied"] = None
        result["near_bound_fraction"] = None
    return result


def make_variant(name):
    if name == "teacher":
        fn = lambda r: np.asarray(r["teacher_score"], dtype=np.float64)
        fn._residual = lambda r: np.zeros(len(r["teacher_score"]))
        return fn
    if name == "raw_residual":
        fn = lambda r: np.asarray(r["score"], dtype=np.float64)
        fn._residual = lambda r: np.asarray(r["residual"], dtype=np.float64)
        return fn
    if name == "centering_only_rank_control":
        fn = lambda r: np.asarray(r["score"], dtype=np.float64) - np.asarray(r["residual"], dtype=np.float64).mean()
        fn._residual = lambda r: np.asarray(r["residual"], dtype=np.float64) - np.asarray(r["residual"], dtype=np.float64).mean()
        return fn
    if name == "bounded_centered":
        fn = lambda r: np.asarray(r["teacher_score"], dtype=np.float64) + .5 * np.tanh((np.asarray(r["residual"], dtype=np.float64) - np.asarray(r["residual"], dtype=np.float64).mean()) / .5)
        fn._residual = lambda r: .5 * np.tanh((np.asarray(r["residual"], dtype=np.float64) - np.asarray(r["residual"], dtype=np.float64).mean()) / .5)
        return fn
    if name == "teacher_safe_gate":
        def fn(r):
            m = np.asarray(r["teacher_score"], dtype=np.float64)
            rc = fn._residual(r)
            q = float(np.median(m))
            gate = 1.0 / (1.0 + np.exp(-np.clip((.25 - (m - q)) / .10, -60, 60)))
            return m + gate * rc
        def residual(r):
            raw = np.asarray(r["residual"], dtype=np.float64)
            return .5 * np.tanh((raw - raw.mean()) / .5)
        fn._residual = residual
        return fn
    if name == "label_gated_oracle":
        fn = lambda r: np.asarray(r["teacher_score"], dtype=np.float64) + .5 * (2 * np.asarray(r["label"], dtype=np.float64) - 1)
        fn._residual = lambda r: .5 * (2 * np.asarray(r["label"], dtype=np.float64) - 1)
        return fn
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    source = Path(args.input).resolve()
    rows, stopped = load_calibration(source)
    names = ["teacher", "raw_residual", "centering_only_rank_control",
             "bounded_centered", "teacher_safe_gate", "label_gated_oracle"]
    variants = {}
    for name in names:
        fn = make_variant(name)
        variants[name] = metrics(rows, fn, BOUND if name in {"raw_residual", "bounded_centered", "teacher_safe_gate", "label_gated_oracle"} else None)
    center_rank = variants["centering_only_rank_control"]
    raw_rank = variants["raw_residual"]
    result = {
        "format": "locatemot-l51-b2-teacher-safe-audit-v1",
        "status": "diagnostic_complete",
        "decision_status": "pending_fixed_calibration_rule",
        "root": str(ROOT), "source": str(source), "source_sha256": sha256(source),
        "manifest_sha256": MANIFEST_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
        "calibration_only": True, "calibration_rows": len(rows),
        "calibration_videos": sorted({str(r["video"]) for r in rows}),
        "stopped_at_first_non_calibration_record": stopped,
        "validation_labels_read": False, "screening_gt_used": False,
        "official_test_labels_read": False, "raw_cache_written": False,
        "fixed_thresholds": THRESHOLDS,
        "pre_registered_formula": {
            "bound": BOUND,
            "bounded_centered": "r_c=.5*tanh((r-mean_frame(r))/.5); s=m+r_c",
            "teacher_safe_gate": "q=median_frame(m); g=sigmoid((.25-(m-q))/.10); s=m+g*r_c",
            "oracle": "GT_PRIVILEGED_ORACLE: s=m+.5 for positive, m-.5 for negative",
        },
        "rank_invariance_evidence": {
            "raw_residual_frame_units": raw_rank["frame_units"],
            "centering_only_raw_order_changed_frames": center_rank["raw_rank_changed_frame_count"],
            "centering_only_raw_pair_relation_changed_frames": center_rank["raw_pair_relation_changed_frame_count"],
            "statement": "m+r-mean(r) is a constant shift of m+r within each frame; centering alone cannot repair frame-internal correspondence rank.",
        },
        "deployable_variants": {k: variants[k] for k in names if k != "label_gated_oracle"},
        "oracle_variant": variants["label_gated_oracle"],
        "fixed_continue_rule": {
            "hard_violation_reduction_min": 0.05,
            "recall_drop_max": 0.01,
            "teacher_correct_flip_rate_max": 0.01,
            "requires_deployable_variant": True,
            "oracle_eligible": False,
        },
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "calibration_rows": len(rows),
                      "output": str(out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
