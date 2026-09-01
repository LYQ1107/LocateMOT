#!/usr/bin/env python3
"""Calibration/validation evaluation for the L77 correspondence probe.

All 40 L62 units are scored label-free first.  Labels for the first 16 are
then attached to fit the registered calibration threshold and checkpoint
tuple.  Only after that tuple is frozen are the final 24 validation labels
attached.  No top-k/NMS/NULL filtering is applied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l77_common import (  # noqa: E402
    FIXED_FORBIDDEN_LABEL_FIELDS, L77Bank, L62_RECORDS, MANIFEST, MANIFEST_SHA256,
    attach_labels, load_fixed_label_free, load_fixed_label_units, load_text_cache, safe_torch_load,
    sha256_file, unit_tensors, write_json,
)
from locatemot.models.l77_region_cross_attention import L77RegionCrossAttention  # noqa: E402

CONTROL_CONTRACT = ROOT / "outputs/l64/audit/control_contract/control_contract.json"
L29_THRESHOLD_FALLBACK = -1.030576229095459
SEED = 20260829
GATE = {
    "recall_floor": 0.7233333,
    "precision_floor": 0.0830188679,
    "fp_per_frame_ceiling": 11.125,
    "predictions_per_positive_ceiling": 4.069,
    "hard_improvement": 0.05,
    "multi_positive_floor": 0.7894444,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None,
                "p50": None, "p95": None}
    return {"count": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max()),
            "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95))}


def fit_threshold(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = np.unique(np.concatenate([np.asarray(row[field], dtype=np.float64) for row in rows]))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best: tuple[tuple[float, int, float], float] | None = None
    for threshold in candidates:
        tp = fp = fn = 0
        for row in rows:
            score = np.asarray(row[field], dtype=np.float64)
            label = np.asarray(row["labels"], dtype=bool)
            selected = score >= threshold
            tp += int((selected & label).sum())
            fp += int((selected & ~label).sum())
            fn += int((~selected & label).sum())
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        key = (f1, -fp, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    if best is None:
        raise AssertionError("cannot fit threshold on empty calibration")
    return {
        "threshold": best[1], "objective": "candidate-level observed-score F1 on 16 calibration units",
        "tie_rule": "higher F1, fewer false positives, then higher threshold",
        "validation_used": False,
    }


def metric(rows: list[dict[str, Any]], field: str, threshold: float) -> dict[str, Any]:
    tp = fp = fn = selected = positives = top1 = top5 = empty = 0
    target_present_units = candidate_present_units = present_uncovered_units = 0
    inactive_accept = inactive_fp_rows = 0
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    violations: list[bool] = []
    multi: list[float] = []
    minimum_coverage: list[float] = []
    unit_recall: list[float] = []
    values: list[float] = []
    key_complete = True
    for row in rows:
        score = np.asarray(row[field], dtype=np.float64)
        label = np.asarray(row["labels"], dtype=bool)
        n = int(row["candidate_count"])
        key_values = row.get("row_keys", row.get("candidate_keys", []))
        if score.size != n or label.size != n or len(key_values) != n or not np.isfinite(score).all():
            raise AssertionError(f"score/label/key failure for {row['unit_key']}")
        values.extend(score.tolist())
        target_present = bool(row["target_present"])
        if target_present:
            target_present_units += 1
        if bool(row["candidate_present"]):
            candidate_present_units += 1
        if str(row["category"]) == "present_uncovered":
            present_uncovered_units += 1
        selected_mask = score >= float(threshold)
        row_tp = int((selected_mask & label).sum())
        row_fp = int((selected_mask & ~label).sum())
        row_fn = int((~selected_mask & label).sum())
        tp += row_tp; fp += row_fp; fn += row_fn
        selected += int(selected_mask.sum()); positives += int(label.sum())
        empty += int(not selected_mask.any())
        if str(row["category"]) == "inactive":
            inactive_accept += int(selected_mask.any())
            inactive_fp_rows += row_fp
        if target_present and label.any():
            order = np.argsort(-score, kind="stable")
            top1 += int(bool(label[order[:1]].any()))
            top5 += int(bool(label[order[:5]].any()))
            unit_recall.append(row_tp / float(label.sum()))
        if label.any():
            positive = np.flatnonzero(label)
            negative = np.flatnonzero(~label)
            if negative.size:
                negative_max = float(score[negative].max())
                strict_value = float(score[positive].min() - negative_max)
                strict.append(strict_value)
                best.append(float(score[positive].max() - negative_max))
                average.append(float(score[positive].mean() - negative_max))
                violations.append(strict_value < 0.0)
                minimum_coverage.append(float(selected_mask[positive].all()))
            if positive.size > 1:
                multi.append(float((selected_mask & label).sum() / positive.size))
    units = len(rows)
    inactive_units = sum(str(row["category"]) == "inactive" for row in rows)
    return {
        "units": units, "candidate_rows": int(sum(int(row["candidate_count"]) for row in rows)),
        "positive_rows": positives, "target_present_units": target_present_units,
        "candidate_present_units": candidate_present_units, "present_uncovered_units": present_uncovered_units,
        "top1": top1 / max(1, target_present_units), "top5": top5 / max(1, target_present_units),
        "candidate_recall": tp / max(1, tp + fn), "candidate_precision": tp / max(1, selected),
        "fp_per_frame": fp / max(1, units), "predictions_per_positive": selected / max(1, positives),
        "hard_violation": float(np.mean(violations)) if violations else None,
        "strict_margin": stats(strict), "best_margin": stats(best), "average_margin": stats(average),
        "multi_positive_recall": float(np.mean(multi)) if multi else None,
        "minimum_positive_coverage": float(np.mean(minimum_coverage)) if minimum_coverage else None,
        "query_track_recall": stats(unit_recall), "empty_rate": empty / max(1, units),
        "inactive_false_acceptance": inactive_accept / max(1, inactive_units),
        "inactive_false_positive_rows": inactive_fp_rows, "false_positive_rows": fp,
        "score_mean": float(np.mean(values)) if values else None,
        "score_std": float(np.std(values)) if values else None, "threshold": float(threshold),
        "complete_finite": bool(key_complete),
    }


def load_l29_rows() -> tuple[list[dict[str, Any]], float]:
    old = read_jsonl(L62_RECORDS)
    if len(old) != 40:
        raise AssertionError("immutable L62 records must contain 40 rows")
    threshold = L29_THRESHOLD_FALLBACK
    if CONTROL_CONTRACT.exists():
        payload = json.loads(CONTROL_CONTRACT.read_text())
        if "fixed_l29_threshold" in payload:
            threshold = float(payload["fixed_l29_threshold"])
    result: list[dict[str, Any]] = []
    for row in old:
        labels = [bool(value) for value in row["label"]]
        result.append({
            "unit_key": str(row["unit_key"]), "dataset": str(row["dataset"]),
            "video": str(row["video"]), "frame_id": int(row["frame_id"]),
            "category": str(row.get("category", "unknown")), "candidate_count": len(labels),
            "candidate_keys": [None for _ in labels], "labels": labels,
            "target_present": bool(any(labels) or str(row.get("category", "")) == "present_uncovered"),
            "candidate_present": bool(any(labels)), "l29": [float(value) for value in row["l29"]],
        })
    return result, threshold


def attach_orders(records: list[dict[str, Any]], orders: Iterable[int]) -> None:
    """Attach only explicitly authorized labels, reusing native bank pointers."""
    selected = {int(order) for order in orders}
    units = load_fixed_label_units(selected)
    by_video: dict[str, list[int]] = defaultdict(list)
    for order in sorted(selected):
        by_video[str(records[order]["video"])].append(order)
    for video in sorted(by_video):
        bank = L77Bank(video)
        try:
            for order in by_video[video]:
                record = attach_labels(records[order], units[order], bank)
                record["declared_category"] = str(units[order].get("category", "unknown"))
                records[order].update({key: record[key] for key in (
                    "target_ids", "sidecar_candidate_gt", "labels", "positive_indices", "positive_count",
                    "target_present", "candidate_present", "coverage_mask", "null_target", "category", "label_source",
                    "declared_category")})
        finally:
            bank.close()


def score_records(records: list[dict[str, Any]], checkpoints: dict[str, Path], device: torch.device) -> dict[str, Any]:
    models: dict[str, L77RegionCrossAttention] = {}
    norms: dict[str, float] = {}
    for name, path in checkpoints.items():
        package = safe_torch_load(path)
        config = dict(package["config"])
        model = L77RegionCrossAttention(region_dim=int(config["region_dim"]), text_dim=int(config["text_dim"]),
                                         hidden=int(config["hidden"]), heads=int(config["heads"]),
                                         dropout=float(config["dropout"])).to(device).eval()
        model.load_state_dict(package["model"], strict=True)
        models[name] = model
        norms[name] = math.sqrt(sum(float(value.float().square().sum()) for value in model.state_dict().values()))
    by_video: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_video[str(record["video"])].append(index)
    text_cache = load_text_cache()
    for video in sorted(by_video):
        bank = L77Bank(video)
        try:
            for index in by_video[video]:
                record = records[index]
                data_cpu = record["unit_tensors"]
                data = {key: value.to(device) for key, value in data_cpu.items()
                        if key in ("region", "text", "text_mask")}
                with torch.inference_mode():
                    for name, model in models.items():
                        output = model(data)
                        score = output["match_logits"].float().cpu()
                        if tuple(score.shape) != (int(record["candidate_count"]),) or not bool(torch.isfinite(score).all()):
                            raise AssertionError(f"score shape/finite failure {record['unit_key']} {name}")
                        record[name] = score.tolist()
                del data, data_cpu
        finally:
            bank.close()
    for model in models.values():
        del model
    return norms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint100", type=Path, required=True)
    parser.add_argument("--checkpoint500", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    running: dict[str, Any] = {
        "format": "locatemot-l77-semantic-evaluation-v1", "status": "running",
        "project_root": str(ROOT), "cwd": os.getcwd(), "command": " ".join(sys.argv),
        "seed": SEED, "calibration_units": 16, "validation_units": 24,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        "training_run": False, "raw_dense_feature_cache_written": False,
        "validation_labels_used_for_selection": False, "candidate_deletion": False,
        "candidate_truncation": False,
    }
    write_json(out / "status.json", running)
    started = time.perf_counter()
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA256:
            raise AssertionError("fixed manifest SHA mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("L77 evaluator requires GPU0")
        device = torch.device(args.device)
        if device.type != "cuda" or device.index not in (None, 0):
            raise RuntimeError(f"L77 evaluator requires GPU0, got {device}")
        for path in (args.checkpoint100, args.checkpoint500):
            if not path.exists():
                raise FileNotFoundError(path)
        # Row/feature construction is label-free.  The helper intentionally
        # does not retain target_ids in these records.
        records = load_fixed_label_free(load_text_cache())
        if len(records) != 40 or [int(row["fixed_eval_order"]) for row in records] != list(range(40)):
            raise AssertionError("fixed L62 order drift")
        norms = score_records(records, {"step100": args.checkpoint100, "step500": args.checkpoint500}, device)
        for row in records:
            n = int(row["candidate_count"])
            for name in ("step100", "step500"):
                if len(row[name]) != n or len(row["row_keys"]) != n:
                    raise AssertionError(f"candidate row/key drift {row['unit_key']}")

        forbidden_preselection = tuple(FIXED_FORBIDDEN_LABEL_FIELDS)
        preselection_forbidden_absent = all(
            field not in record and field not in record["unit_metadata"]
            for record in records for field in forbidden_preselection
        )
        preselection_audit = {
            "format": "locatemot-l77-preselection-label-isolation-v1",
            "status": "complete" if preselection_forbidden_absent else "INVALID",
            "phase": "after_label_free_scores_before_calibration_attach",
            "records": 40, "calibration_records": 16, "validation_records": 24,
            "fixed_eval_order": [int(record["fixed_eval_order"]) for record in records],
            "record_keys": sorted({key for record in records for key in record if key != "unit_tensors"}),
            "unit_metadata_keys": sorted({key for record in records for key in record["unit_metadata"]}),
            "forbidden_label_fields": list(forbidden_preselection),
            "forbidden_fields_absent_from_records_and_metadata": bool(preselection_forbidden_absent),
            "key_only_l62_loader": True,
            "key_only_l49_metadata_loader": True,
            "raw_label_payloads_retained": False,
            "label_fields_accessed": False,
            "calibration_labels_attached": False,
            "validation_labels_attached": False,
            "candidate_rows_and_scores_complete": True,
            "candidate_deletion": False, "candidate_truncation": False,
            "validation_labels_read": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "assertion": "pre-selection fixed records expose only dataset/video/query/frame/sentence/expression/unit_key plus native row/score data",
        }
        if not preselection_forbidden_absent:
            raise AssertionError("forbidden label field present before calibration attach")
        write_json(out / "preselection_label_isolation.json", preselection_audit)

        # Calibration labels only: this is the first authorized label access.
        calibration_attach_event = {
            "phase": "calibration_label_attach",
            "orders": list(range(16)), "preselection_audit_written": True,
            "validation_label_loader_called": False,
        }
        attach_orders(records, range(16))
        for row in records[:16]:
            if "labels" not in row:
                raise AssertionError("calibration label missing")
        thresholds = {name: fit_threshold(records[:16], name) for name in ("step100", "step500")}
        cal_methods = {name: metric(records[:16], name, thresholds[name]["threshold"]) for name in thresholds}

        # Checkpoint tuple is fixed exclusively from calibration.
        selection_rows: list[dict[str, Any]] = []
        for name in ("step100", "step500"):
            cm = cal_methods[name]
            key = (
                float(cm["hard_violation"] if cm["hard_violation"] is not None else 1e9),
                -float(cm["minimum_positive_coverage"] if cm["minimum_positive_coverage"] is not None else -1.0),
                float(cm["inactive_false_acceptance"]), float(cm["false_positive_rows"]),
                100 if name == "step100" else 500, float(norms[name]),
            )
            selection_rows.append({"step": 100 if name == "step100" else 500, "method": name,
                                   "threshold": thresholds[name], "calibration_metrics": cm,
                                   "lexicographic_key": list(key)})
        selection_rows.sort(key=lambda row: tuple(row["lexicographic_key"]))
        selected = selection_rows[0]
        selected_name = str(selected["method"])
        selected_threshold = float(thresholds[selected_name]["threshold"])
        selection_frozen_event = {
            "phase": "calibration_selection_frozen",
            "selected_method": selected_name, "selected_step": int(selected["step"]),
            "threshold": selected_threshold, "validation_labels_attached": False,
            "validation_used_for_selection": False,
        }

        # Only now attach the 24 validation labels.
        validation_attach_event = {
            "phase": "validation_label_attach",
            "orders": list(range(16, 40)), "selection_frozen": True,
        }
        attach_orders(records, range(16, 40))
        if any("labels" not in row for row in records[16:]):
            raise AssertionError("validation label missing after strategy freeze")
        methods: dict[str, Any] = {
            "l29_teacher_immutable": {},
            "l77_step100": {"calibration": cal_methods["step100"], "validation": metric(records[16:], "step100", thresholds["step100"]["threshold"])},
            "l77_step500": {"calibration": cal_methods["step500"], "validation": metric(records[16:], "step500", thresholds["step500"]["threshold"])},
        }
        methods["l77_step100"]["threshold"] = thresholds["step100"]
        methods["l77_step500"]["threshold"] = thresholds["step500"]
        methods["l77_selected"] = {
            "method": selected_name, "step": int(selected["step"]),
            "threshold": thresholds[selected_name], "calibration": cal_methods[selected_name],
            "validation": methods[f"l77_{selected_name}"]["validation"],
            "selection_source": "calibration-only registered tuple",
        }
        l29_rows, l29_threshold = load_l29_rows()
        l29_cal = metric(l29_rows[:16], "l29", l29_threshold)
        l29_val = metric(l29_rows[16:], "l29", l29_threshold)
        methods["l29_teacher_immutable"] = {
            "calibration": l29_cal, "validation": l29_val,
            "threshold": {"threshold": l29_threshold, "source": "L64 immutable control-contract audit"},
        }
        base_val = l29_val
        final_val = methods["l77_selected"]["validation"]
        both_domains = {}
        for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
            l29_d = [row for row in l29_rows[16:] if row["dataset"] == dataset]
            l77_d = [row for row in records[16:] if row["dataset"] == dataset]
            both_domains[dataset] = {
                "l29": metric(l29_d, "l29", l29_threshold),
                "l77_selected": metric(l77_d, selected_name, selected_threshold),
            }
        checks = {
            "hard_negative_decrease_ge_0.05": final_val["hard_violation"] is not None and base_val["hard_violation"] is not None and final_val["hard_violation"] <= base_val["hard_violation"] - GATE["hard_improvement"],
            "recall_floor": final_val["candidate_recall"] >= GATE["recall_floor"],
            "precision_floor": final_val["candidate_precision"] >= GATE["precision_floor"],
            "fp_per_frame_ceiling": final_val["fp_per_frame"] <= GATE["fp_per_frame_ceiling"],
            "predictions_per_positive_ceiling": final_val["predictions_per_positive"] <= GATE["predictions_per_positive_ceiling"],
            "multi_positive_floor": final_val["multi_positive_recall"] is not None and final_val["multi_positive_recall"] >= GATE["multi_positive_floor"],
            "inactive_false_acceptance_lt_1": final_val["inactive_false_acceptance"] < 1.0,
            "complete_finite_keys": all(len(row["row_keys"]) == int(row["candidate_count"]) == len(row[selected_name]) and np.isfinite(np.asarray(row[selected_name], dtype=np.float64)).all() for row in records),
            "candidate_deletion_false": all(not bool(row["candidate_deletion"]) for row in records),
            "candidate_truncation_false": all(not bool(row["candidate_truncation"]) for row in records),
            "both_domains_reported": all(dataset in both_domains for dataset in ("refer_kitti_v1", "refer_kitti_v2")),
        }
        decision = "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail"
        gate = {
            "format": "locatemot-l77-region-cross-attention-gate-v1", "status": decision,
            "decision": "pass" if decision == "semantic_gate_pass" else "fail",
            "formal_method": selected_name, "formal_step": int(selected["step"]),
            "threshold": selected_threshold, "selection": selection_rows,
            "checks": checks, "fixed_thresholds": thresholds,
            "calibration_units": 16, "validation_units": 24,
            "validation_used_for_selection": False, "candidate_set": "all L69 rows; no top-k/NMS/deletion",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }
        serial: list[dict[str, Any]] = []
        for row in records:
            selected_field = row[selected_name]
            serial.append({
                "format": "locatemot-l77-score-record-v1", "unit_key": row["unit_key"],
                "dataset": row["dataset"], "video": row["video"], "query_id": row["query_id"],
                "frame_id": row["frame_id"], "category": row.get("category", "unattached"),
                "declared_category": row["declared_category"], "fixed_eval_order": row["fixed_eval_order"],
                "candidate_count": row["candidate_count"], "candidate_keys": row["row_keys"],
                "candidate_index_provenance": row["candidate_index"],
                "labels": [int(value) for value in row.get("labels", [])],
                "l77_step100": row["step100"], "l77_step500": row["step500"],
                "selected_method": selected_name, "selected_score": selected_field,
                "target_present": row.get("target_present"), "candidate_present": row.get("candidate_present"),
                "present_uncovered": row.get("category") == "present_uncovered",
                "row_contract": {"candidate_count": row["candidate_count"], "key_count": len(row["row_keys"]),
                                 "ordered": True, "candidate_deletion": False, "candidate_truncation": False,
                                 "old_l49_begin_end_used": False, "old_l49_positive_indices_used": False},
            })
        (out / "score_records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in serial))
        previous_dir = ROOT / "outputs/l77/eval/semantic_16cal24val_retry4"
        previous_score_path = previous_dir / "score_records.jsonl"
        previous_gate_path = previous_dir / "gate_decision.json"
        previous_semantic_path = previous_dir / "semantic.json"
        previous_selection_path = previous_dir / "checkpoint_selection.json"
        comparison = {
            "format": "locatemot-l77-retry-comparison-v1", "status": "complete",
            "current": str(out), "previous": str(previous_dir),
            "score_records_sha256": sha256_file(out / "score_records.jsonl"),
            "previous_score_records_sha256": None,
            "score_records_byte_identical": False,
            "semantic_metrics_identical": False,
            "gate_decision_identical": False,
            "checkpoint_selection_identical": False,
            "all_scores_and_decisions_identical": False,
            "difference_policy": "stop if any registered score or decision differs; never select the nicer result",
        }
        if previous_score_path.exists() and previous_gate_path.exists() and previous_semantic_path.exists() and previous_selection_path.exists():
            comparison["previous_score_records_sha256"] = sha256_file(previous_score_path)
            comparison["score_records_byte_identical"] = comparison["score_records_sha256"] == comparison["previous_score_records_sha256"]
            previous_gate = json.loads(previous_gate_path.read_text())
            previous_semantic = json.loads(previous_semantic_path.read_text())
            previous_selection = json.loads(previous_selection_path.read_text())
            comparison["semantic_metrics_identical"] = (
                previous_semantic.get("methods") == methods and
                previous_semantic.get("domain_slices") == both_domains
            )
            comparison["gate_decision_identical"] = previous_gate == gate
            comparison["checkpoint_selection_identical"] = (
                previous_selection.get("selected") == selected and
                previous_selection.get("candidates") == selection_rows
            )
            comparison["all_scores_and_decisions_identical"] = all(bool(comparison[key]) for key in (
                "score_records_byte_identical", "semantic_metrics_identical",
                "gate_decision_identical", "checkpoint_selection_identical"))
        else:
            comparison["status"] = "INCOMPLETE"
            comparison["failure_root_cause"] = "retry4 comparison inputs missing"
        write_json(out / "retry4_comparison.json", comparison)
        if not comparison["all_scores_and_decisions_identical"]:
            raise AssertionError(f"retry4 comparison differs: {comparison}")
        label_isolation_audit = {
            "format": "locatemot-l77-label-isolation-audit-v1", "status": "complete",
            "preselection": preselection_audit,
            "events": [calibration_attach_event, selection_frozen_event, validation_attach_event],
            "assertions": {
                "preselection_forbidden_fields_absent": True,
                "calibration_attached_only_after_label_free_scores": True,
                "validation_attached_only_after_selection_frozen": True,
                "validation_not_used_for_selection": True,
                "all_40_orders_and_scores_unchanged_vs_retry4": True,
            },
            "calibration_labels_attached_after_preselection_audit": True,
            "validation_labels_attached_after_calibration_selection": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }
        write_json(out / "label_isolation_audit.json", label_isolation_audit)
        checkpoint_info = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in {"step100": args.checkpoint100, "step500": args.checkpoint500}.items()}
        provenance = {
            **running, "status": "complete", "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
            "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
            "l69_bank_sha256": {video: sha256_file(ROOT / "outputs/l69/attempt9/budget40_features/kitti" / f"{video}.pt") for video in sorted({str(row["video"]) for row in records})},
            "l48_text_cache": str(ROOT / "outputs/l48/data/text_cache.pt"),
            "l48_text_cache_sha256": sha256_file(ROOT / "outputs/l48/data/text_cache.pt"),
            "l49_calibration_units": str(ROOT / "outputs/l49/data/calibration_units.jsonl"),
            "l49_validation_units": str(ROOT / "outputs/l49/data/validation_units.jsonl"),
            "l62_records": str(L62_RECORDS), "l62_records_sha256": sha256_file(L62_RECORDS),
            "l64_control_contract": str(CONTROL_CONTRACT),
            "checkpoints": checkpoint_info, "selected_checkpoint": checkpoint_info[selected_name],
            "candidate_rows_total": int(sum(int(row["candidate_count"]) for row in records)),
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "calibration_only_selection": True, "validation_labels_read_after_selection": True,
            "label_isolation_audit": str(out / "label_isolation_audit.json"),
            "preselection_label_isolation": str(out / "preselection_label_isolation.json"),
            "preselection_forbidden_fields_absent": True,
            "calibration_attach_before_selection_only": True,
            "validation_attach_after_selection_only": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "token_region_alignment": "UNALIGNED", "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(out / "semantic.json", {"format": running["format"], "status": "complete", "methods": methods,
                                            "domain_slices": both_domains, "gate": gate, "provenance": provenance})
        write_json(out / "gate_decision.json", gate)
        write_json(out / "checkpoint_selection.json", {"format": "locatemot-l77-calibration-selection-v1", "status": "complete", "selected": selected, "candidates": selection_rows, "validation_used": False})
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {**running, "status": "complete", "failure_root_cause": None,
                                          "next_action": "stop L77 branch; supervisor approval required for one raw-image end-to-end correspondence/proposal redesign",
                                          "retry4_comparison": str(out / "retry4_comparison.json"),
                                          "label_isolation_audit": str(out / "label_isolation_audit.json")})
        print(json.dumps({"status": "complete", "decision": decision, "selected": selected_name,
                          "validation": final_val, "out": str(out)}), flush=True)
        return 0
    except Exception as exc:
        failure = {**running, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "elapsed_seconds": time.perf_counter() - started,
                   "next_action": "fix only first L77 evaluator root cause and rerun in a new output directory"}
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text("# L77 evaluator INCOMPLETE\n\n" +
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n```text\n" + traceback.format_exc() + "```\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
