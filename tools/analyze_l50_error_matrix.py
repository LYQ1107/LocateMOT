#!/usr/bin/env python3
"""CPU-only detailed fit/validation decomposition for L50-B score caches.

This reads only L50 train/calibration/validation score records and the
immutable L29 validation controls.  It intentionally has no test-path
constant and cannot select a checkpoint from official labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.eval_l49_validation import summarize  # noqa: E402


def load_rows(path: Path, split: str | None = None, checkpoint: int | None = None):
    rows = []
    with path.open() as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if split is not None and row.get("split") != split:
                continue
            if checkpoint is not None and int(row.get("checkpoint_step", -1)) != int(checkpoint):
                continue
            row["score"] = np.asarray(row["score"], dtype=np.float32)
            row["label"] = np.asarray(row["label"], dtype=bool)
            row["sources"] = {key: np.asarray(value, dtype=bool)
                               for key, value in row.get("sources", {}).items()}
            rows.append(row)
    return rows


def baseline_key(row):
    return (str(row["dataset"]), str(row["video"]), int(row["query_id"]), int(row["frame_id"]))


def expression_type(sentence: str) -> str:
    text = str(sentence).lower()
    if any(word in text for word in ("left", "right", "front", "ahead", "before", "behind")):
        return "space"
    if any(word in text for word in
           ("black", "white", "silver", "red", "light", "green", "color", "colour")):
        return "appearance"
    if any(word in text for word in
           ("moving", "walk", "walking", "park", "parked", "standing", "direction",
            "transit", "opposite", "same", "heading", "abandoned")):
        return "action"
    if any(word in text for word in
           ("car", "cars", "vehicle", "vehicles", "auto", "autos", "automobile",
            "automobiles", "pedestrian", "pedestrians", "person", "persons", "people",
            "men", "males", "women", "females", "individual")):
        return "category"
    return "other"


def candidate_bucket(count: int) -> str:
    if count <= 20:
        return "1-20"
    if count <= 40:
        return "21-40"
    if count <= 60:
        return "41-60"
    return "61+"


def length_bucket(sentence: str) -> str:
    words = len(str(sentence).split())
    return "1-5" if words <= 5 else "6-10" if words <= 10 else "11+"


def margin(row):
    label = np.asarray(row["label"], dtype=bool)
    pos = np.flatnonzero(label); neg = np.flatnonzero(~label)
    if not len(pos) or not len(neg):
        return None
    return float(np.asarray(row["score"])[pos].min() - np.asarray(row["score"])[neg].max())


def rank_flip(rows, baseline):
    pair_count = correct_flip = correction = total_flip = 0
    for row in rows:
        base = baseline.get(baseline_key(row))
        if base is None:
            continue
        label = np.asarray(row["label"], dtype=bool)
        pos = np.flatnonzero(label); neg = np.flatnonzero(~label)
        if not len(pos) or not len(neg):
            continue
        teacher_order = (np.asarray(base["score"])[pos, None] > np.asarray(base["score"])[None, neg]).reshape(-1)
        student_order = (np.asarray(row["score"])[pos, None] > np.asarray(row["score"])[None, neg]).reshape(-1)
        pair_count += len(teacher_order)
        correct_flip += int((teacher_order & ~student_order).sum())
        correction += int((~teacher_order & student_order).sum())
        total_flip += int((teacher_order != student_order).sum())
    denom = max(1, pair_count)
    return {"pair_count": pair_count, "teacher_correct_flip": correct_flip,
            "teacher_error_correction": correction, "total_flip": total_flip,
            "teacher_correct_flip_rate": correct_flip / denom,
            "teacher_error_correction_rate": correction / denom,
            "total_flip_rate": total_flip / denom}


def group_summary(rows, thresholds):
    groups = defaultdict(list)
    for row in rows:
        domain = str(row["dataset"])
        group = (expression_type(row.get("sentence", "")),
                 candidate_bucket(int(row.get("candidate_count", len(row["score"])))))
        groups[(domain, group)].append(row)
    result = {}
    for (domain, group), values in sorted(groups.items()):
        result.setdefault(domain, {})["%s|%s" % group] = {
            "units": len(values),
            "summary": summarize(values, thresholds[domain]),
            "margin_mean": float(np.mean([x for x in (margin(row) for row in values) if x is not None]))
            if any(margin(row) is not None for row in values) else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="outputs/l50/eval/long5000")
    parser.add_argument("--out", default="outputs/l50/eval/long5000/error_matrix_detailed.json")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    started = time.time()
    eval_root = Path(args.eval_root)
    if not eval_root.is_absolute():
        eval_root = ROOT / eval_root
    matrix = json.loads((eval_root / "error_matrix.json").read_text())
    if matrix.get("official_test_labels_read") or matrix.get("test_paths_opened"):
        raise RuntimeError("L50 matrix is not test-free")
    scores_path = eval_root / "scores.jsonl"
    baseline_path = ROOT / "outputs/l49/val/validation_baseline_scores.jsonl"
    baseline = load_rows(baseline_path, split="validation")
    baseline_by_key = {baseline_key(row): row for row in baseline}
    details = {}
    for step in sorted(matrix["checkpoint_results"], key=int):
        rows = load_rows(scores_path, split="validation", checkpoint=int(step))
        fit = load_rows(scores_path, split="fit", checkpoint=int(step))
        thresholds = {domain: float(matrix["checkpoint_results"][step]["calibration_thresholds"][domain]["threshold"])
                      for domain in matrix["checkpoint_results"][step]["per_domain"]}
        val_group = group_summary(rows, thresholds)
        fit_group = group_summary(fit, thresholds)
        details[step] = {
            "validation_units": len(rows), "fit_units": len(fit),
            "validation_by_expression_candidate_bucket": val_group,
            "fit_by_expression_candidate_bucket": fit_group,
            "validation_rank_flips_vs_l29": rank_flip(rows, baseline_by_key),
            "fit_rank_flips_vs_l29": rank_flip(fit, {baseline_key(x): x for x in
                                                       load_rows(ROOT / "outputs/l49/val/fit_baseline_scores_selected.jsonl", split="fit")}),
        }
    output = {
        "format": "locatemot-l50-detailed-error-matrix-v1", "stage": "L50-C-decision-input",
        "project_root": str(ROOT), "source_matrix": str((eval_root / "error_matrix.json").resolve()),
        "source_scores": str(scores_path.resolve()), "official_test_labels_read": False,
        "test_paths_opened": False, "screening_gt_used": False,
        "calibration_thresholds_source": "embedded L50 calibration-only thresholds",
        "checkpoint_details": details, "completed_at_unix": time.time(),
        "elapsed_sec": time.time() - started,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "pass", "checkpoints": sorted(details, key=int),
                      "official_test_labels_read": False, "elapsed_sec": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
