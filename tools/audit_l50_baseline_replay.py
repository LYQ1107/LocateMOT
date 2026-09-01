#!/usr/bin/env python3
"""Read-only L50-A replay of the frozen L49 train/validation evidence.

This audit deliberately never opens ``outputs/l49/test``.  It re-aggregates
the immutable fit/validation score caches, checks the L49 contract and hashes,
and records the calibration-only thresholds and teacher rank flips needed
before the L50-B experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l49_fit_error_matrix import (  # noqa: E402
    load_score_records,
    rank_flip_decomposition,
    record_key,
)
from tools.eval_l49_validation import summarize  # noqa: E402


MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
CONTRACT = ROOT / "outputs/l49/audit/kitti_data_contract.json"
CALIBRATION = ROOT / "outputs/l49/val/calibration.json"
SELECTION = ROOT / "outputs/l49/val/selected_checkpoint.json"
VALIDATION_METRICS = ROOT / "outputs/l49/val/validation_metrics.json"
PRETEST_MATRIX = ROOT / "outputs/l49/val/error_matrix_pretest.json"
TRAIN_SUMMARY = ROOT / "outputs/l49/train/joint_long5000/metrics_l49_training_summary.json"
FIT = ROOT / "outputs/l49/val/fit_scores_selected.jsonl"
FIT_BASELINE = ROOT / "outputs/l49/val/fit_baseline_scores_selected.jsonl"
VAL = ROOT / "outputs/l49/val/validation_scores.jsonl"
VAL_BASELINE = ROOT / "outputs/l49/val/validation_baseline_scores.jsonl"

EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L49_STEPS = (100, 250, 500, 1000, 2500, 5000)
SUMMARY_FIELDS = (
    "frame_units", "candidate_rows", "positive_rows", "positive_frame_units",
    "precision", "recall", "top1_frame_recall", "top5_frame_recall",
    "false_positive_candidates_per_frame", "empty_output_rate",
    "null_frame_false_acceptance", "predictions_per_positive",
    "multi_positive_recall", "hard_violation_rate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def check_disjoint(contract: dict) -> dict:
    result = {}
    for domain, data in contract["domains"].items():
        buckets = {
            "fit": set(data["fit_videos"]),
            "calibration": set(data["calibration_videos"]),
            "validation": set(data["validation_videos"]),
            "official_eval_metadata_only": set(data["official_eval_videos_metadata_only"]),
        }
        intersections = {}
        names = list(buckets)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                overlap = sorted(buckets[left] & buckets[right])
                intersections[f"{left}∩{right}"] = overlap
        result[domain] = {
            "counts": {key: len(value) for key, value in buckets.items()},
            "intersections": intersections,
            "within_domain_disjoint": not any(intersections.values()),
        }
    return result


def simple_summary(records, threshold):
    summary = summarize(records, threshold)
    return {key: summary.get(key) for key in SUMMARY_FIELDS}


def assert_close(actual: dict, expected: dict, fields, tolerance=2e-6):
    checks = {}
    for field in fields:
        left = actual.get(field)
        right = expected.get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            checks[field] = {
                "actual": left, "expected": right,
                "abs_error": abs(float(left) - float(right)),
                "match": abs(float(left) - float(right)) <= tolerance,
            }
        else:
            checks[field] = {"actual": left, "expected": right, "match": left == right}
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "outputs/l50/audit/baseline_replay.json"))
    args = parser.parse_args()
    started = time.time()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")

    contract = json.loads(CONTRACT.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    selection = json.loads(SELECTION.read_text())
    validation_metrics = json.loads(VALIDATION_METRICS.read_text())
    pretest = json.loads(PRETEST_MATRIX.read_text())
    train_summary = json.loads(TRAIN_SUMMARY.read_text())

    manifest_sha = sha256(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError(f"fixed manifest SHA mismatch: {manifest_sha}")
    if selection["official_test_labels_read"] or validation_metrics["official_test_labels_read"]:
        raise RuntimeError("L49 train/validation provenance unexpectedly says test labels were read")
    if calibration["validation_or_test_labels_used"]:
        raise RuntimeError("calibration artifact says validation/test labels were used")

    checkpoint_hashes = {}
    checkpoint_hash_failures = []
    for step in L49_STEPS:
        path = Path(train_summary["checkpoints"][str(step)])
        actual = sha256(path)
        checkpoint_hashes[str(step)] = {"path": str(path.resolve()), "sha256": actual,
                                        "expected": train_summary["checkpoint_sha256"][str(step)],
                                        "match": actual == train_summary["checkpoint_sha256"][str(step)]}
        if actual != train_summary["checkpoint_sha256"][str(step)]:
            checkpoint_hash_failures.append(str(step))
    if checkpoint_hash_failures:
        raise RuntimeError(f"checkpoint SHA mismatch at steps {checkpoint_hash_failures}")

    selected_step = int(selection["selected_step"])
    thresholds = {domain: float(value["threshold"])
                  for domain, value in selection["selected_thresholds"].items()}
    calibration_selected = {
        domain: calibration["selected_thresholds"][domain]["threshold"]
        for domain in thresholds
    }
    if thresholds != {key: float(value) for key, value in calibration_selected.items()}:
        raise RuntimeError("selected thresholds do not match calibration artifact")

    fit = load_score_records(FIT, "fit", checkpoint_step=selected_step)
    fit_baseline = load_score_records(FIT_BASELINE, "fit")
    val_all = load_score_records(VAL, "validation", checkpoint_step=selected_step)
    val_baseline = load_score_records(VAL_BASELINE, "validation")
    if len(fit) != 5314 or len(fit_baseline) != 5314 or len(val_all) != 1218 or len(val_baseline) != 1218:
        raise RuntimeError(f"unexpected replay record counts: fit={len(fit)}, fit_base={len(fit_baseline)}, "
                           f"val={len(val_all)}, val_base={len(val_baseline)}")

    def by_domain(records):
        return {domain: [row for row in records if row["dataset"] == domain]
                for domain in ("refer_kitti_v1", "refer_kitti_v2")}

    fit_by_domain = by_domain(fit)
    fit_base_by_domain = by_domain(fit_baseline)
    val_by_domain = by_domain(val_all)
    val_base_by_domain = by_domain(val_baseline)
    fit_summaries = {domain: {
        "l49": simple_summary(rows, thresholds[domain]),
        "l29": simple_summary(fit_base_by_domain[domain], thresholds[domain]),
    } for domain, rows in fit_by_domain.items()}
    val_summaries = {domain: {
        "l49": simple_summary(rows, thresholds[domain]),
        "l29": simple_summary(val_base_by_domain[domain], thresholds[domain]),
    } for domain, rows in val_by_domain.items()}

    fit_base_map = {record_key(row): row for row in fit_baseline}
    val_base_map = {record_key(row): row for row in val_baseline}
    fit_rank = rank_flip_decomposition(fit, fit_base_map)["semantic_only"]
    val_rank = rank_flip_decomposition(val_all, val_base_map)["semantic_only"]

    expected_fit = pretest["selected_detailed_matrix"]["fit"]["per_domain"]
    expected_val = pretest["selected_detailed_matrix"]["validation"]["per_domain"]
    replay_checks = {"fit": {}, "validation": {}}
    for domain in ("refer_kitti_v1", "refer_kitti_v2"):
        replay_checks["fit"][domain] = assert_close(
            fit_summaries[domain]["l49"], expected_fit[domain], SUMMARY_FIELDS)
        replay_checks["validation"][domain] = assert_close(
            val_summaries[domain]["l49"], expected_val[domain], SUMMARY_FIELDS)

    duplicate_keys = {
        "fit": len(fit) - len({record_key(row) for row in fit}),
        "fit_baseline": len(fit_baseline) - len({record_key(row) for row in fit_baseline}),
        "validation": len(val_all) - len({record_key(row) for row in val_all}),
        "validation_baseline": len(val_baseline) - len({record_key(row) for row in val_baseline}),
    }

    all_checks = []
    for group in replay_checks.values():
        for domain in group.values():
            all_checks.extend(check["match"] for check in domain.values())
    coverage = {
        domain: {
            "candidate_coverage": contract["domains"][domain]["positive_frame_recall_coverage"],
            "fit_videos": contract["domains"][domain]["fit_videos"],
            "calibration_videos": contract["domains"][domain]["calibration_videos"],
            "validation_videos": contract["domains"][domain]["validation_videos"],
            "official_eval_videos_metadata_only": contract["domains"][domain]["official_eval_videos_metadata_only"],
            "candidate_size": contract["domains"][domain]["candidate_sizes"],
            "multi_positive_rate": contract["domains"][domain]["multi_positive_rate"],
            "inactive_rate": contract["domains"][domain]["inactive_rate"],
        } for domain in ("refer_kitti_v1", "refer_kitti_v2")
    }

    output = {
        "format": "locatemot-l50-baseline-replay-v1",
        "stage": "L50-A",
        "project_root": str(ROOT),
        "started_at_unix": started,
        "completed_at_unix": time.time(),
        "status": "pass" if all(all_checks) and not any(
            item["within_domain_disjoint"] is False
            for item in check_disjoint(contract).values()
        ) and not any(duplicate_keys.values()) else "fail",
        "official_test_labels_read": False,
        "screening_labels_used_for_selection": False,
        "test_paths_opened": False,
        "fixed_manifest": {"path": str(MANIFEST.resolve()), "sha256": manifest_sha,
                           "expected_sha256": EXPECTED_MANIFEST_SHA, "match": manifest_sha == EXPECTED_MANIFEST_SHA},
        "video_overlap": check_disjoint(contract),
        "candidate_coverage": coverage,
        "checkpoint_hashes": checkpoint_hashes,
        "selected_checkpoint": {"step": selected_step,
                                 "path": selection["selected_checkpoint"],
                                 "sha256": selection["selected_checkpoint_sha256"],
                                 "selection_rule": selection["selection_rule"]},
        "calibration": {"selected_thresholds": thresholds,
                         "calibration_artifact_thresholds": calibration_selected,
                         "labels_source": calibration["labels_source"],
                         "validation_or_test_labels_used": False},
        "record_counts": {"fit": len(fit), "fit_baseline": len(fit_baseline),
                           "validation": len(val_all), "validation_baseline": len(val_baseline)},
        "duplicate_key_counts": duplicate_keys,
        "fit": fit_summaries,
        "validation": val_summaries,
        "rank_flip": {"fit": fit_rank, "validation": val_rank},
        "replay_against_l49_pretest_matrix": replay_checks,
        "all_replay_metric_checks_pass": bool(all(all_checks)),
        "all_l49_checkpoint_validation_metrics_present": all(
            str(step) in validation_metrics["checkpoint_results"] for step in L49_STEPS),
        "stored_fit_curve_artifact": str(PRETEST_MATRIX.resolve()),
        "stored_fit_curve_artifact_status": pretest.get("matrix_status"),
        "ordinary_mot_ovmot_touched": False,
        "elapsed_sec": time.time() - started,
    }
    if not finite(output):
        raise FloatingPointError("nonfinite value in L50-A audit output")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, out)
    print(json.dumps({"output": str(out.resolve()), "status": output["status"],
                      "official_test_labels_read": False,
                      "all_replay_metric_checks_pass": output["all_replay_metric_checks_pass"],
                      "elapsed_sec": output["elapsed_sec"]}, indent=2), flush=True)
    if output["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
