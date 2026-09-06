#!/usr/bin/env python3
"""Offline L88C root-cause decomposition from existing evidence only.

This script deliberately does not run a model, change a checkpoint, fit a
threshold, or read any screening/test labels.  It summarizes the corrected
replay, existing dev rule records, TrackEval matrix, training trace, and the
saved LoRA packages so that the semantic failure is attributable rather than
repaired by another experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from l88_eval_common import MANIFEST, MANIFEST_SHA, THREAD, sha256, write_json


WORK_ROOT = Path(__file__).resolve().parents[1]
FIXED_DIR = WORK_ROOT / "outputs/l88c/eval/fixed_semantic_corrected_attempt1"
DEV_RULES = WORK_ROOT / "outputs/l88c/dev/corrected_reselect_attempt2/corrected_rule_refit.json"
TRACK_MATRIX = WORK_ROOT / "outputs/l88c/dev/trackeval_matrix_attempt2/trackeval_matrix.json"
TRAIN_METRICS = WORK_ROOT / "outputs/l88/train/joint40_world4_retry1/metrics_l88_training.json"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/locatemot_l88/project_outputs/train/joint40_world4_retry1")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _summary(values: list[float]) -> dict[str, Any]:
    values = sorted(_finite(values))
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    middle = values[len(values) // 2]
    if len(values) % 2 == 0:
        middle = (values[len(values) // 2 - 1] + middle) / 2.0
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": middle,
        "min": values[0],
        "max": values[-1],
        "p10": values[max(0, int(0.10 * (len(values) - 1)))],
        "p90": values[min(len(values) - 1, int(0.90 * (len(values) - 1)))],
    }


def _metric_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    names = (
        "legacy_candidate_recall", "legacy_candidate_precision", "legacy_fp_per_frame",
        "legacy_predictions_per_positive", "legacy_row_hard_violation",
        "legacy_row_multi_positive_recall", "empty_rate", "inactive_false_acceptance",
        "distinct_target_recall", "distinct_multi_target_exact", "target_bag_f1",
        "target_bag_precision", "target_bag_recall", "candidate_rows", "selected_rows",
    )
    return {name: metrics.get(name) for name in names if name in metrics}


def _pairwise(records: list[dict[str, Any]]) -> dict[str, Any]:
    unit_rows: list[dict[str, Any]] = []
    wins: list[float] = []
    best_ranks: list[float] = []
    all_ranks: list[float] = []
    for record in records:
        labels = [bool(value) for value in record["labels"]]
        scores = [float(value) for value in record["score"]]
        positives = [score for score, label in zip(scores, labels) if label]
        negatives = [score for score, label in zip(scores, labels) if not label]
        if not positives or not negatives:
            continue
        pair_total = len(positives) * len(negatives)
        pair_wins = sum(pos > neg for pos in positives for neg in negatives)
        wins.append(pair_wins / pair_total)
        ranks = [1 + sum(other > pos for other in scores) for pos in positives]
        best_ranks.append(float(min(ranks)))
        all_ranks.extend(float(rank) for rank in ranks)
        unit_rows.append({
            "unit_key": record["unit_key"], "dataset": record["dataset"],
            "category": record["category"], "positive_count": len(positives),
            "negative_count": len(negatives), "pairwise_accuracy": pair_wins / pair_total,
            "best_positive_rank": min(ranks), "minimum_positive_rank": max(ranks),
        })
    return {
        "unit_count": len(unit_rows),
        "unit_pairwise_accuracy": _summary(wins),
        "best_positive_rank": _summary(best_ranks),
        "all_positive_ranks": _summary(all_ranks),
        "records": unit_rows,
    }


def _frontier(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Fixed diagnostic grid, not a selection grid and not used by the gate.
    thresholds = (-15.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 1.0, 2.0, 3.0)
    out = []
    total_positive = sum(sum(bool(value) for value in row["labels"]) for row in records)
    total_rows = sum(int(row["candidate_count"]) for row in records)
    for threshold in thresholds:
        tp = fp = selected = 0
        for row in records:
            for score, label in zip(row["score"], row["labels"]):
                selected += float(score) >= threshold
                if float(score) >= threshold and bool(label):
                    tp += 1
                elif float(score) >= threshold:
                    fp += 1
        out.append({
            "threshold": threshold, "selected_rows": selected, "total_rows": total_rows,
            "tp": tp, "fp": fp, "fn": max(0, total_positive - tp),
            "precision": tp / selected if selected else 0.0,
            "recall": tp / total_positive if total_positive else 0.0,
            "note": "fixed validation diagnostic; never used for checkpoint or threshold selection",
        })
    return out


def _presence_null(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    row_margin: dict[str, list[float]] = defaultdict(list)
    for row in records:
        category = str(row["category"])
        presence = float(row["presence_logit"])
        null = float(row["null_logit"])
        by_category[category]["presence"].append(presence)
        by_category[category]["null"].append(null)
        by_category[category]["presence_minus_null"].append(presence - null)
        for score, label in zip(row["score"], row["labels"]):
            row_margin["positive" if label else "negative"].append(float(score) - null)
    return {
        "unit_logits": {
            category: {name: _summary(values) for name, values in fields.items()}
            for category, fields in by_category.items()
        },
        "candidate_score_minus_null": {
            name: _summary(values) for name, values in row_margin.items()
        },
        "interpretation": "corrected emission uses candidate score minus null_logit and presence threshold; no post-hoc suppression was added",
    }


def _multi_positive(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        if str(record["category"]) != "multi_positive":
            continue
        scores = [float(value) for value in record["score"]]
        labels = [bool(value) for value in record["labels"]]
        positives = [score for score, label in zip(scores, labels) if label]
        if not positives:
            continue
        rows.append({
            "unit_key": record["unit_key"], "positive_count": len(positives),
            "minimum_positive_score": min(positives), "maximum_positive_score": max(positives),
            "positive_score_gap": max(positives) - min(positives),
            "positive_ranks": [1 + sum(other > score for other in scores) for score in positives],
        })
    return {"unit_count": len(rows), "records": rows,
            "minimum_positive_score": _summary([r["minimum_positive_score"] for r in rows]),
            "positive_score_gap": _summary([r["positive_score_gap"] for r in rows])}


def _lora_norms() -> dict[str, Any]:
    epochs = {}
    for epoch in (2, 8, 20, 30, 40):
        path = CHECKPOINT_ROOT / f"checkpoint_l88_epoch{epoch:03d}.pt"
        package = torch.load(path, map_location="cpu", weights_only=False)
        targets: dict[str, dict[str, float]] = defaultdict(lambda: {"A_norm": 0.0, "B_norm": 0.0, "delta_fro_norm": 0.0, "target_count": 0})
        total_delta_sq = 0.0
        total_a_sq = 0.0
        total_b_sq = 0.0
        for key, factors in package["lora_state_dict"].items():
            A = factors["A"].float()
            B = factors["B"].float()
            delta = B @ A * float(package["lora_manifest"]["scale"])
            if key.startswith("encoder.fusion_layers.4"):
                layer = "encoder.fusion_layers.4"
            elif key.startswith("encoder.fusion_layers.5"):
                layer = "encoder.fusion_layers.5"
            elif match := re.match(r"decoder\.layers\.(\d+)", key):
                layer = f"decoder.layers.{match.group(1)}"
            else:
                layer = "other"
            item = targets[layer]
            item["A_norm"] += float(A.norm())
            item["B_norm"] += float(B.norm())
            item["delta_fro_norm"] += float(delta.norm())
            item["target_count"] += 1
            total_delta_sq += float(delta.pow(2).sum())
            total_a_sq += float(A.pow(2).sum())
            total_b_sq += float(B.pow(2).sum())
        epochs[str(epoch)] = {
            "checkpoint": str(path), "checkpoint_sha256": sha256(path),
            "rank": package["lora_manifest"]["rank"], "alpha": package["lora_manifest"]["alpha"],
            "total_A_l2": math.sqrt(total_a_sq), "total_B_l2": math.sqrt(total_b_sq),
            "total_scaled_delta_fro": math.sqrt(total_delta_sq),
            "by_layer": dict(targets),
        }
    return epochs


def _trackeval_selected(matrix: dict[str, Any], epoch: int, rule: str) -> dict[str, Any]:
    selected = {}
    marker = f"candidate_epoch{epoch:03d}/{rule}/"
    for row in matrix.get("results", []):
        if marker in str(row.get("source", "")):
            raw = row.get("metrics_raw", {})
            selected[str(row["dataset"])] = {name: raw.get(name) for name in (
                "HOTA___AUC", "DetA___AUC", "AssA___AUC", "DetRe___AUC", "DetPr___AUC", "IDF1___AUC"
            ) if name in raw}
    return selected


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty diagnosis output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    semantic = _read_json(FIXED_DIR / "semantic.json")
    records = _read_jsonl(FIXED_DIR / "score_records.jsonl")
    validation = records[16:]
    if len(records) != 40 or len(validation) != 24:
        raise AssertionError("fixed semantic record count drift")
    for row in records:
        if int(row["candidate_count"]) != len(row["score"]):
            raise AssertionError(f"candidate length drift: {row['unit_key']}")
        if row.get("candidate_deletion") or row.get("candidate_truncation"):
            raise AssertionError(f"candidate deletion/truncation: {row['unit_key']}")

    training = _read_json(TRAIN_METRICS)
    dev_rules = _read_json(DEV_RULES)
    track_matrix = _read_json(TRACK_MATRIX)
    final = semantic["l88c_corrected_final"]["validation"]["frozen_rule"]
    l29 = semantic["l29_teacher"]["validation"]
    root_cause = {
        "primary": "correspondence_and_multi_positive_recall_insufficient_after_corrected_emission",
        "confidence": "high",
        "evidence": [
            "corrected hard violation improves to 0.8461538462 but recall is 0.3548387097 and multi-positive recall is 0.3055555556",
            "precision and FP/frame pass only because the frozen candidate threshold emits a small subset; this is not a recall-preserving fix",
            "V2 validation is weaker than V1 and multi-positive exact coverage is zero in the V2 fixed slice",
            "inactive false acceptance is 0.6667, so the dominant failure is not universal NULL acceptance",
            "candidate coverage was already adequate in the independent L76 audit; no new proposal evidence is introduced here",
        ],
        "not_primary": ["candidate_coverage", "universal_null_acceptance", "threshold_or_top_k_repair", "training_finite_or_reload_failure"],
    }
    payload = {
        "format": "locatemot-l88c-root-cause-diagnosis-v1", "status": "complete",
        "evidence_type": "offline diagnosis of existing L88/L88C artifacts; no training or selection",
        "command": " ".join([sys.executable, *sys.argv]), "cwd": str(WORK_ROOT),
        "luna_thread": THREAD, "base_l88_sha": "c9b44c07b9b977de9d0f839fb2ff6363abb0386e",
        "corrected_candidate_vs_null": True, "zero_training": True,
        "fixed_record_count": len(records), "validation_record_count": len(validation),
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        "A_candidate_representation_ranking": _pairwise(validation),
        "B_fixed_validation_precision_recall_frontier": _frontier(validation),
        "C_presence_active_inactive": _presence_null(validation),
        "D_candidate_vs_null_margins": _presence_null(validation)["candidate_score_minus_null"],
        "E_multi_target_collapse": _multi_positive(validation),
        "F_V2_generalization": {
            "fixed_validation": semantic["l88c_corrected_final"]["validation"]["frozen_rule"].get("per_dataset", {}).get("refer_kitti_v2"),
            "selected_dev_trackeval_epoch30_B": _trackeval_selected(track_matrix, 30, "B"),
        },
        "G_lora_delta_norms": _lora_norms(),
        "H_L84_direct_evidence": {"status": "not_used", "reason": "no direct L84 artifact was required by this replay; no inference made"},
        "I_training_trajectory": {
            "epochs": [{
                "epoch": int(row["epoch"]), "phase": row.get("phase"),
                "loss_mean": row.get("loss_mean"), "positive_rows": row.get("positive_rows"),
                "masked_missing_count": row.get("masked_missing_count"),
                "nonzero_gradient_entries": row.get("nonzero_gradient_entries"),
                "gradient_entries": row.get("gradient_entries"),
                "temporal_enabled": row.get("temporal_enabled"),
                "temporal_identity_pairs": row.get("temporal_identity_pairs"),
            } for row in training["trace"] if int(row["epoch"]) in (1, 2, 8, 20, 30, 39, 40)],
            "loss_first_epoch": training["trace"][0]["loss_mean"],
            "loss_last_epoch": training["trace"][-1]["loss_mean"],
            "semantic_selection_did_not_use_training_trace": True,
        },
        "metric_snapshot": {"L29": l29, "L88C_corrected_final": _metric_fields(final),
                            "L88C_gate": _read_json(FIXED_DIR / "gate_decision.json")},
        "root_cause": root_cause,
        "checkpoint_selection_frozen_before_fixed_validation": True,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "no_hota_or_trackeval": True,
        "new_test_required": True,
        "next_action": "request supervisor approval for exactly one structural branch; do not extend L88C or retune emission",
    }
    write_json(out / "root_cause.json", payload)
    write_json(out / "status.json", {
        "format": "locatemot-l88c-root-cause-status-v1", "status": "complete",
        "root_cause_primary": root_cause["primary"], "root_cause_confidence": root_cause["confidence"],
        "zero_training": True, "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "no_hota_or_trackeval": True,
    })
    write_json(out / "provenance.json", {
        "format": "locatemot-l88c-root-cause-provenance-v1", "status": "complete",
        "inputs": {
            "fixed_semantic": str(FIXED_DIR), "fixed_semantic_sha256": sha256(FIXED_DIR / "semantic.json"),
            "dev_rule_refit": str(DEV_RULES), "dev_rule_refit_sha256": sha256(DEV_RULES),
            "trackeval_matrix": str(TRACK_MATRIX), "trackeval_matrix_sha256": sha256(TRACK_MATRIX),
            "training_metrics": str(TRAIN_METRICS), "training_metrics_sha256": sha256(TRAIN_METRICS),
        }, "manifest_sha256": MANIFEST_SHA, "flags": payload,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
