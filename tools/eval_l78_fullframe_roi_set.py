#!/usr/bin/env python3
"""Strict-label-isolated L78 fixed 16-calibration/24-validation evaluator."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l78_fullframe_roi_set import L78FullFrameROISet
from tools.l78_common import (
    CLIP_WEIGHTS, EXPECTED_CLIP_SHA, EXPECTED_MANIFEST_SHA, FORBIDDEN_LABEL_FIELDS,
    L29_VALIDATION_CONTROL, MANIFEST, L78Bank, StreamingOpenAIClipFullFrame,
    authorized_fixed_labels, boxes_to_normalized, fixed_key_metadata, image_path,
    sha256_file, write_json,
)


def _ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)


def summary(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p50": None}
    return {"count": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), "p50": float(np.quantile(a, 0.5))}


def attach_labels(record: dict[str, Any], label_unit: dict[str, Any], bank: L78Bank) -> dict[str, Any]:
    return bank.attach_labels(record, label_unit)


def metric(records: list[dict[str, Any]], field: str, threshold: float) -> dict[str, Any]:
    tp = fp = fn = selected_count = positive_rows = 0
    top1 = top5 = empty = 0
    hard_violations: list[bool] = []
    strict: list[float] = []; best: list[float] = []; average: list[float] = []
    multi_values: list[float] = []; minimum_values: list[float] = []
    inactive_units = inactive_accept = inactive_fp_rows = 0
    target_present_units = candidate_present_units = present_uncovered_units = 0
    score_values: list[float] = []
    by_category: dict[str, list[dict[str, Any]]] = {}
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        scores = np.asarray(record[field], dtype=float)
        labels = np.asarray(record["labels"], dtype=bool)
        if len(scores) != len(labels) or not np.isfinite(scores).all():
            raise AssertionError(f"score/label contract failed {record['unit_key']} {field}")
        selected = scores >= float(threshold)
        tp += int((selected & labels).sum()); fp += int((selected & ~labels).sum())
        fn += int((~selected & labels).sum()); selected_count += int(selected.sum())
        positive_rows += int(labels.sum()); score_values.extend(scores.tolist())
        category = str(record["category"])
        by_category.setdefault(category, []).append(record)
        by_dataset.setdefault(str(record["dataset"]), []).append(record)
        if category != "inactive":
            target_present_units += 1
        if bool(labels.any()):
            candidate_present_units += 1
            order = np.argsort(-scores, kind="stable")
            top1 += int(bool(labels[order[:1]].any())); top5 += int(bool(labels[order[:5]].any()))
        if category == "present_uncovered":
            present_uncovered_units += 1
        if category == "inactive":
            inactive_units += 1; inactive_accept += int(bool(selected.any())); inactive_fp_rows += int((selected & ~labels).sum())
        empty += int(not bool(selected.any()))
        pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
        if len(pos) and len(neg):
            strict_margin = float(scores[pos].min() - scores[neg].max())
            strict.append(strict_margin); best.append(float(scores[pos].max() - scores[neg].max())); average.append(float(scores[pos].mean() - scores[neg].max()))
            hard_violations.append(strict_margin < 0)
        if len(pos) > 1:
            multi_values.append(float((selected & labels).sum() / len(pos)))
            minimum_values.append(float(bool(np.all(selected[pos]))))
    candidate_rows = sum(len(record["labels"]) for record in records)
    present_for_top = candidate_present_units
    result = {
        "units": len(records), "target_present_units": target_present_units,
        "candidate_present_units": candidate_present_units, "present_uncovered_units": present_uncovered_units,
        "candidate_rows": candidate_rows, "positive_rows": positive_rows,
        "selected_rows": selected_count, "false_positive_rows": fp,
        "top1": top1 / max(1, present_for_top), "top5": top5 / max(1, present_for_top),
        "candidate_precision": tp / max(1, selected_count), "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(records)), "predictions_per_positive": selected_count / max(1, positive_rows),
        "hard_violation": float(np.mean(hard_violations)) if hard_violations else None,
        "strict_margin": summary(strict), "best_margin": summary(best), "average_margin": summary(average),
        "multi_positive_recall": float(np.mean(multi_values)) if multi_values else None,
        "minimum_positive_coverage": float(np.mean(minimum_values)) if minimum_values else None,
        "empty_rate": empty / max(1, len(records)),
        "inactive_false_acceptance": inactive_accept / max(1, inactive_units),
        "inactive_false_positive_rows": inactive_fp_rows, "inactive_units": inactive_units,
        "score_distribution": summary(score_values), "threshold": float(threshold),
        "per_category": {}, "per_dataset": {},
    }
    for key, values in by_category.items():
        result["per_category"][key] = metric(values, field, threshold) if len(values) != len(records) else None
    for key, values in by_dataset.items():
        result["per_dataset"][key] = metric(values, field, threshold) if len(values) != len(records) else None
    # The recursive fields above are deliberately computed with the same fixed
    # threshold and score field; remove their nested stratification to keep
    # the JSON compact and unambiguous.
    for branch in (result["per_category"], result["per_dataset"]):
        for value in branch.values():
            if value is not None:
                value.pop("per_category", None); value.pop("per_dataset", None)
    return result


def fit_threshold(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = np.unique(np.concatenate([np.asarray(record[field], dtype=float) for record in records]))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best_key = None; best_threshold = None; best_counts = None
    for threshold in candidates:
        tp = fp = fn = 0
        for record in records:
            score = np.asarray(record[field], dtype=float); label = np.asarray(record["labels"], dtype=bool); selected = score >= float(threshold)
            tp += int((selected & label).sum()); fp += int((selected & ~label).sum()); fn += int((~selected & label).sum())
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        # Registered tie: higher F1, fewer FP, then higher threshold.
        key = (f1, -fp, float(threshold))
        if best_key is None or key > best_key:
            best_key = key; best_threshold = float(threshold); best_counts = {"tp": tp, "fp": fp, "fn": fn, "f1": f1}
    return {"threshold": best_threshold, "objective": "candidate-level observed-score F1 on 16 calibration units", "tie_rule": "higher F1, fewer false positives, higher threshold", "validation_used": False, "counts": best_counts}


def raw_record(meta: dict[str, Any], bank: L78Bank, encoder: StreamingOpenAIClipFullFrame, device: torch.device, models: dict[str, L78FullFrameROISet]) -> dict[str, Any]:
    record = bank.label_free_record(meta)
    path = image_path(record["video"], record["frame_id"])
    spatial, global_token, geometry = encoder.image_map(path)
    boxes, box_details = boxes_to_normalized(record["boxes"], geometry, padding=0.10)
    text, text_mask, token_ids = encoder.text_tokens(record["sentence"])
    # Explicitly clone frozen/inference tensors before model use.  Eval itself
    # has no backward graph, but the boundary is retained for auditability.
    features = {"spatial_map": spatial.detach().clone().to(device), "global_token": global_token.detach().clone().to(device), "text": text.detach().clone().to(device), "text_mask": text_mask.to(device), "boxes": boxes.detach().clone().to(device)}
    scores: dict[str, list[float]] = {}
    absent: dict[str, float] = {}
    for name, model in models.items():
        with torch.inference_mode():
            output = model(features)
        scores[name] = output["match_logits"].float().cpu().tolist()
        absent[name] = float(output["absent_logit"].float().cpu())
        if len(scores[name]) != record["candidate_count"] or not np.isfinite(np.asarray(scores[name])).all():
            raise AssertionError(f"candidate score contract {record['unit_key']} {name}")
    output_record = {
        "format": "locatemot-l78-score-record-v1", "unit_key": record["unit_key"],
        "dataset": record["dataset"], "video": record["video"], "query_id": record["query_id"], "frame_id": record["frame_id"],
        "fixed_eval_order": meta["fixed_eval_order"], "fixed_eval_split": meta["fixed_eval_split"],
        "sentence": record["sentence"], "bank_path": record["bank_path"],
        "row_offsets": record["row_offsets"], "row_keys": record["row_keys"],
        "candidate_index_provenance": record["candidate_index_provenance"], "pool_id_provenance": record["pool_id_provenance"],
        "track_id_provenance": record["track_id_provenance"], "raw_rank_provenance": record["raw_rank_provenance"],
        "candidate_count": record["candidate_count"], "normalized_boxes": boxes.tolist(),
        "image_path": str(path), "image_geometry": geometry, "text_valid_tokens": int(text_mask.sum()),
        "text_token_count": int(token_ids.numel()), "score_fields": scores, "null_logits": absent,
        "candidate_rows_retained": record["candidate_count"], "candidate_deletion": False, "candidate_truncation": False,
        "sidecar_labels_loaded": False, "finite_scores": True,
    }
    del bank, record, spatial, global_token, geometry, boxes, box_details, text, text_mask, token_ids, features
    for model in models.values():
        model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_record


def attach_record_labels(record: dict[str, Any], label_unit: dict[str, Any], bank: L78Bank) -> dict[str, Any]:
    # L78Bank expects native row offsets and only reads its sidecar here.
    base = {"unit_key": record["unit_key"], "row_offsets": record["row_offsets"], "candidate_count": record["candidate_count"]}
    labeled = bank.attach_labels(base, label_unit)
    result = dict(record)
    result.update({"labels": labeled["labels"], "positive_indices": labeled["positive_indices"], "positive_count": labeled["positive_count"], "target_ids": labeled["target_ids"], "target_present": labeled["target_present"], "candidate_present": labeled["candidate_present"], "coverage_mask": labeled["coverage_mask"], "null_target": labeled["null_target"], "category": labeled["category"], "sidecar_candidate_gt": labeled["sidecar_candidate_gt"], "label_source": labeled["label_source"], "sidecar_labels_loaded": True})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint100", required=True)
    parser.add_argument("--checkpoint500", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd {Path.cwd()}")
    out = Path(args.out); out = out if out.is_absolute() else ROOT / out; out = out.resolve(); _ensure_empty(out)
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA or sha256_file(CLIP_WEIGHTS) != EXPECTED_CLIP_SHA:
        raise AssertionError("immutable manifest/CLIP SHA mismatch")
    device = torch.device(args.device)
    checkpoint_paths = {"step100": Path(args.checkpoint100).resolve(), "step500": Path(args.checkpoint500).resolve()}
    models: dict[str, L78FullFrameROISet] = {}
    checkpoint_hashes = {}
    for name, path in checkpoint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        package = torch.load(path, map_location=device, weights_only=False)
        config = package.get("model_config", {})
        model = L78FullFrameROISet(**{key: int(config[key]) for key in ("visual_dim", "text_dim", "hidden", "heads", "roi_grid") if key in config}).to(device)
        model.load_state_dict(package["model"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model; checkpoint_hashes[name] = sha256_file(path)
    metadata = fixed_key_metadata()
    if len(metadata) != 40 or [row["unit_key"] for row in metadata] != fixed_key_order_local():
        raise AssertionError("fixed metadata order drift")
    encoder = StreamingOpenAIClipFullFrame(str(device))
    start_time = time.time()
    preselection = []
    for meta in metadata:
        bank = L78Bank(str(meta["video"]))
        preselection.append(raw_record(meta, bank, encoder, device, models))
        bank.close()
    forbidden_present = sorted({field for row in preselection for field in FORBIDDEN_LABEL_FIELDS if field in row})
    if forbidden_present:
        raise AssertionError(f"preselection label fields present: {forbidden_present}")
    key_counts = [len(row["row_keys"]) for row in preselection]
    preselection_audit = {
        "format": "locatemot-l78-preselection-label-isolation-v1", "status": "complete",
        "records": len(preselection), "fixed_order": [row["fixed_eval_order"] for row in preselection],
        "preselection_schema": sorted(preselection[0].keys()), "forbidden_label_fields": list(FORBIDDEN_LABEL_FIELDS),
        "forbidden_fields_absent": not forbidden_present, "forbidden_fields_found": forbidden_present,
        "candidate_rows_and_scores_complete": all(row["candidate_count"] == len(row["row_keys"]) == len(row["score_fields"]["step100"]) == len(row["score_fields"]["step500"]) for row in preselection),
        "candidate_deletion": False, "candidate_truncation": False, "sidecar_labels_loaded": False,
        "calibration_labels_attached": False, "validation_labels_attached": False,
        "validation_labels_read": False, "selection_frozen": False,
        "event": "all raw features and scores constructed before authorized label loader",
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
    }
    write_json(out / "preselection_label_isolation.json", preselection_audit)
    cal_labels = authorized_fixed_labels(range(16))
    cal_rows = []
    for order in range(16):
        meta = metadata[order]; bank = L78Bank(str(meta["video"]))
        cal_rows.append(attach_record_labels(preselection[order], cal_labels[order], bank)); bank.close()
    threshold_info: dict[str, Any] = {}
    calibration_metrics: dict[str, Any] = {}
    selection_candidates = []
    for method in ("step100", "step500"):
        threshold_info[method] = fit_threshold(cal_rows, f"score_fields.{method}") if False else None
        # Keep score fields flat for metric/selection, without changing score records.
        flat = [dict(row, **{method: row["score_fields"][method]}) for row in cal_rows]
        threshold_info[method] = fit_threshold(flat, method)
        calibration_metrics[method] = metric(flat, method, threshold_info[method]["threshold"])
        cm = calibration_metrics[method]
        selection_candidates.append({
            "method": method, "step": int(method.replace("step", "")),
            "threshold": threshold_info[method], "calibration_metrics": cm,
            "lexicographic_key": [
                float(cm["hard_violation"] if cm["hard_violation"] is not None else 1.0),
                -float(cm["minimum_positive_coverage"] if cm["minimum_positive_coverage"] is not None else 0.0),
                float(cm["inactive_false_acceptance"]), float(cm["false_positive_rows"]), int(method.replace("step", "")),
            ],
        })
    selection_candidates.sort(key=lambda item: tuple(item["lexicographic_key"]))
    selected = selection_candidates[0]
    preselection_audit.update({"calibration_labels_attached": True, "validation_labels_attached": False, "selection_frozen": True})
    write_json(out / "preselection_label_isolation.json", preselection_audit)
    write_json(out / "checkpoint_selection.json", {"format": "locatemot-l78-checkpoint-selection-v1", "status": "complete", "selection_source": "calibration-only", "tuple": "lower hard violation, higher minimum-positive coverage, lower inactive acceptance, lower false-positive rows, earlier step", "candidates": selection_candidates, "selected": selected, "validation_used": False, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
    val_labels = authorized_fixed_labels(range(16, 40))
    val_rows = []
    for order in range(16, 40):
        meta = metadata[order]; bank = L78Bank(str(meta["video"]))
        val_rows.append(attach_record_labels(preselection[order], val_labels[order], bank)); bank.close()
    def flat(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
        return [dict(row, **{method: row["score_fields"][method]}) for row in rows]
    methods = {}
    for method in ("step100", "step500"):
        methods[method] = {"checkpoint": {"path": str(checkpoint_paths[method]), "sha256": checkpoint_hashes[method]}, "threshold": threshold_info[method], "calibration": calibration_metrics[method], "validation": metric(flat(val_rows, method), method, threshold_info[method]["threshold"])}
    selected_method = str(selected["method"])
    selected_validation = methods[selected_method]["validation"]
    checks = {
        "hard_negative_improvement_ge_0.05": selected_validation["hard_violation"] is not None and selected_validation["hard_violation"] <= L29_VALIDATION_CONTROL["hard_violation"] - 0.05,
        "recall_floor": selected_validation["candidate_recall"] >= 0.7233333,
        "precision_floor": selected_validation["candidate_precision"] >= 0.0830188679,
        "fp_per_frame_ceiling": selected_validation["fp_per_frame"] <= 11.125,
        "predictions_per_positive_ceiling": selected_validation["predictions_per_positive"] <= 4.069,
        "multi_positive_floor": selected_validation["multi_positive_recall"] is not None and selected_validation["multi_positive_recall"] >= 0.7894444,
        "inactive_false_acceptance_lt_1": selected_validation["inactive_false_acceptance"] < 1.0,
        "complete_finite_keys": all(row["candidate_count"] == len(row["row_keys"]) == len(row["score_fields"]["step100"]) == len(row["score_fields"]["step500"]) == len(row["labels"]) and row["finite_scores"] for row in val_rows),
        "candidate_deletion_false": all(row["candidate_rows_retained"] == row["candidate_count"] and not row["candidate_deletion"] and not row["candidate_truncation"] for row in preselection),
    }
    decision = "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail"
    gate = {"format": "locatemot-l78-semantic-gate-v1", "status": decision, "decision": decision, "selected_method": selected_method, "selected_step": int(selected["step"]), "checks": checks, "l29_validation_control": L29_VALIDATION_CONTROL, "calibration_units": 16, "validation_units": 24, "selection_and_threshold_calibration_only": True, "candidate_set": "complete L69 rows; no sampling/top-k/NMS/deletion", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True}
    semantic = {
        "format": "locatemot-l78-semantic-evaluation-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "methods": {"l29_immutable": {"validation": L29_VALIDATION_CONTROL, "source": "accepted L62/L64 control contract; L78 candidate rows differ"}, "l78_step100": methods["step100"], "l78_step500": methods["step500"]}, "checkpoint_selection": selected, "gate": gate, "candidate_rows": {"calibration": int(sum(row["candidate_count"] for row in cal_rows)), "validation": int(sum(row["candidate_count"] for row in val_rows))}, "label_events": {"preselection_scores_complete": True, "calibration_labels_attached_before_selection": True, "selection_frozen_before_validation_attach": True, "validation_labels_attached_after_selection": True}, "elapsed_sec": time.time() - start_time,
    }
    write_json(out / "semantic.json", semantic)
    write_json(out / "gate_decision.json", gate)
    with (out / "score_records.jsonl").open("w") as handle:
        for order, row in enumerate(preselection):
            labeled = cal_rows[order] if order < 16 else val_rows[order - 16]
            payload = dict(row)
            payload.update({key: labeled[key] for key in ("labels", "positive_indices", "positive_count", "target_ids", "target_present", "candidate_present", "coverage_mask", "null_target", "category", "sidecar_candidate_gt", "label_source")})
            payload["sidecar_labels_loaded"] = True
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    write_json(out / "provenance.json", {"format": "locatemot-l78-evaluation-provenance-v1", "status": "complete", "command": " ".join([str(Path.cwd() / "tools/eval_l78_fullframe_roi_set.py")] + list(__import__("sys").argv[1:])), "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "inputs": {"l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"), "l49_calibration_units": str(ROOT / "outputs/l49/data/calibration_units.jsonl"), "l49_validation_units": str(ROOT / "outputs/l49/data/validation_units.jsonl"), "l62_fixed_records": str(ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"), "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST), "clip_weights": str(CLIP_WEIGHTS), "clip_weights_sha256": sha256_file(CLIP_WEIGHTS)}, "checkpoints": {name: {"path": str(path), "sha256": checkpoint_hashes[name]} for name, path in checkpoint_paths.items()}, "fixed_order": [row["unit_key"] for row in metadata], "preselection_label_isolation": str(out / "preselection_label_isolation.json"), "calibration_labels_attached_only_after_scores": True, "validation_labels_attached_only_after_selection": True, "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False, "token_region_alignment": "UNALIGNED", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
    write_json(out / "status.json", {"format": "locatemot-l78-status-v1", "status": decision, "stage": "fixed-calibration-validation-semantic-gate", "command": " ".join(__import__("sys").argv), "inputs": [str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"), str(MANIFEST)], "outputs": [str(out / "semantic.json"), str(out / "gate_decision.json"), str(out / "score_records.jsonl")], "failure_root_cause": None if decision == "semantic_gate_pass" else "to_be_decomposed_from_selected_validation_metrics", "next_action": "stop L78 and write failure decomposition" if decision != "semantic_gate_pass" else "freeze method and request supervisor authorization", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
    write_json(out / "config.json", {"format": "locatemot-l78-eval-config-v1", "fixed_order_units": 40, "calibration_units": 16, "validation_units": 24, "threshold_rule": "calibration-only observed-score candidate F1; higher F1, fewer FP, higher threshold", "checkpoint_rule": "calibration-only lower hard violation, higher minimum-positive coverage, lower inactive acceptance, lower false-positive rows, earlier step", "candidate_rows": "all L69 rows; no deletion/truncation/top-k/NMS", "null_logit": "diagnostic only; no semantic NULL filter", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
    if decision == "semantic_gate_fail":
        # This is a completed semantic evaluation, not an incomplete attempt.
        pass
    print(json.dumps({"status": decision, "selected_method": selected_method, "selected_validation": selected_validation, "checks": checks, "output": str(out)}, indent=2), flush=True)


def fixed_key_order_local() -> list[str]:
    rows = []
    for line in (ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(str(json.loads(line)["unit_key"]))
    return rows


if __name__ == "__main__":
    main()
