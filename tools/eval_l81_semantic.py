#!/usr/bin/env python3
"""L81 fixed calibration/validation semantic gate.

The metric, threshold, NULL, and immutable-control functions are imported
directly from ``tools.eval_l80_v12``.  Only the L81 raw-input/model scoring
adapter is new.  All three L81 checkpoints are scored before any L81 labels
are attached; calibration labels are attached before selection and validation
labels only after the selection tuple is frozen.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l81_hierarchical_early_fusion import L81Config, L81HierarchicalEarlyFusion  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    EXPECTED_MANIFEST_SHA,
    FORBIDDEN_LABEL_FIELDS,
    L80BankStore,
    L49_DATA,
    L62_ROWS,
    MANIFEST,
    load_full_unit_for_labels,
    sha256_file,
)
from locatemot.rmot.l81_runtime import (  # noqa: E402
    CLIP_SHA256,
    CLIP_WEIGHT,
    FrameFeatureCache,
    load_clip,
    raw_inputs_for_l81,
)
from tools.eval_l80_v12 import (  # noqa: E402
    FORBIDDEN_PRESELECTION_FIELDS,
    L29_VALIDATION_CONTROL,
    attach_record_labels,
    checkpoint_norm,
    checkpoint_step,
    fit_candidate_threshold,
    fit_null_threshold,
    fixed_metadata,
    immutable_control_thresholds,
    make_control_records,
    metric,
    safe_load,
    sha256_file as canonical_sha256_file,
    summary,
    write_json,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829


def score_record(meta: dict[str, Any], store: L80BankStore, clip_model: Any,
                 cache: FrameFeatureCache, model: L81HierarchicalEarlyFusion,
                 device: torch.device, method: str) -> dict[str, Any]:
    """Score all native L69 candidate rows without opening labels."""
    batch = store.build_unit(meta)
    raw = raw_inputs_for_l81(clip_model, batch, device, cache)
    history = batch.history_observations.to(device=device).clone()
    history_mask = batch.history_mask.to(device=device).clone()
    history_frames = batch.history_frame_ids.to(device=device).clone()
    with torch.inference_mode():
        output = model(
            raw["visual_pyramid"], raw["local_tokens"], raw["text_tokens"], raw["text_mask"],
            history, history_mask, history_frames, int(batch.frame_id), raw["boxes_norm"],
        )
    arrays = {
        "score": output["candidate_logits"].float().cpu().tolist(),
        "track": output["track_logits"].float().cpu().tolist(),
        "continuation": output["continuation_logits"].float().cpu().tolist(),
        "quality": output["quality_logits"].float().cpu().tolist(),
    }
    null_value = float(output["null_logit"].float().cpu())
    cardinality_value = float(output["cardinality_logit"].float().cpu())
    if any(not np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in arrays.values()):
        raise FloatingPointError(f"nonfinite L81 score: {batch.unit_key}/{method}")
    if not np.isfinite([null_value, cardinality_value]).all():
        raise FloatingPointError(f"nonfinite L81 frame score: {batch.unit_key}/{method}")
    if len(arrays["score"]) != batch.candidate_count:
        raise AssertionError(f"candidate score length drift: {batch.unit_key}/{method}")
    if batch.row_offsets != list(range(batch.row_offsets[0], batch.row_offsets[0] + batch.candidate_count)):
        raise AssertionError(f"native L69 row offset drift: {batch.unit_key}")
    record = {
        "format": "locatemot-l81-score-record-v1",
        "unit_key": batch.unit_key, "dataset": batch.dataset, "video": batch.video,
        "query_id": int(batch.query_id), "frame_id": int(batch.frame_id),
        "fixed_eval_order": int(meta["fixed_eval_order"]),
        "fixed_eval_split": str(meta["fixed_eval_split"]), "sentence": batch.sentence,
        "bank_path": batch.bank_path, "row_offsets": [int(x) for x in batch.row_offsets],
        "row_keys": [list(key) for key in batch.row_keys],
        "candidate_index_provenance": [int(x) for x in batch.candidate_indices],
        "track_id_provenance": [int(x) for x in batch.track_ids],
        "pool_id_provenance": [int(x) for x in batch.pool_ids],
        "candidate_count": int(batch.candidate_count), "boxes": batch.boxes.tolist(),
        "image_path": batch.image_path, "image_size": list(batch.image_size),
        "score_fields": {method: arrays["score"]}, "track_score_fields": {method: arrays["track"]},
        "continuation_score_fields": {method: arrays["continuation"]},
        "quality_score_fields": {method: arrays["quality"]},
        "null_logits": {method: null_value}, "cardinality_logits": {method: cardinality_value},
        "history_future_rows": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
        "text_valid_tokens": int(batch.text_mask.sum()),
        "candidate_rows_retained": int(batch.candidate_count),
        "candidate_deletion": False, "candidate_truncation": False,
        "sidecar_labels_loaded": False, "finite_scores": True,
        "source_pool_ids_provenance_only": True,
        "raw_visual_pyramid_shape": list(raw["visual_pyramid"].shape),
        "raw_local_token_shape": list(raw["local_tokens"].shape),
        "raw_cache_persistent": False,
    }
    del output, raw, history, history_mask, history_frames, batch
    return record


def score_checkpoint(metadata: list[dict[str, Any]], path: Path, method: str,
                     clip_model: Any, cache: FrameFeatureCache, store: L80BankStore,
                     device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    package = safe_load(path, map_location="cpu")
    config = L81Config(**package["model_config"])
    model = L81HierarchicalEarlyFusion(config).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"L81 strict checkpoint load failed: {result}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records = []
    for meta in metadata:
        current = score_record(meta, store, clip_model, cache, model, device, method)
        if current["history_future_rows"] != 0:
            raise AssertionError(f"future history in L81 evaluation: {current['unit_key']}")
        records.append(current)
    info = {
        "path": str(path.resolve()), "sha256": sha256_file(path), "method": method,
        "step": checkpoint_step(package, path), "parameter_norm": checkpoint_norm(package),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "model_config": config.__dict__, "strict_reload": True,
    }
    del model, package
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, info


def merge_records(base: dict[str, Any], current: dict[str, Any], method: str) -> None:
    for field in ("unit_key", "fixed_eval_order", "row_offsets", "row_keys", "candidate_count",
                  "candidate_index_provenance", "pool_id_provenance", "bank_path"):
        if base[field] != current[field]:
            raise AssertionError(f"L81 candidate row drift at {base['unit_key']} field={field} method={method}")
    for field in ("score_fields", "track_score_fields", "continuation_score_fields",
                  "quality_score_fields", "null_logits", "cardinality_logits"):
        base[field].update(current[field])


def label_payload(record: dict[str, Any]) -> dict[str, Any]:
    source = load_full_unit_for_labels(record["unit_key"])
    return attach_record_labels(record, source)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L81 evaluation output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA changed")
    if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
        raise AssertionError("CLIP SHA changed")
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("L81 evaluation requires GPU0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    metadata = fixed_metadata()
    if len(metadata) != 40 or [int(x["fixed_eval_order"]) for x in metadata] != list(range(40)):
        raise AssertionError("immutable fixed L62 order drift")
    checkpoint_specs = []
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError("--checkpoint must be NAME=PATH")
        name, value_path = value.split("=", 1)
        path = Path(value_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_specs.append((str(name), path))
    if not checkpoint_specs or len({x[0] for x in checkpoint_specs}) != len(checkpoint_specs):
        raise AssertionError("checkpoint names are missing or duplicated")
    names = [name for name, _ in checkpoint_specs]
    clip_model = load_clip(device)
    cache = FrameFeatureCache(max_items=max(64, len(metadata)))
    store = L80BankStore(max_history=8)
    preselection: list[dict[str, Any]] = []
    checkpoint_info: dict[str, Any] = {}
    try:
        for method, path in checkpoint_specs:
            current, info = score_checkpoint(metadata, path, method, clip_model, cache, store, device)
            checkpoint_info[method] = info
            if not preselection:
                preselection = current
            else:
                for base, value in zip(preselection, current):
                    merge_records(base, value, method)
        forbidden = sorted({field for row in preselection
                            for field in FORBIDDEN_PRESELECTION_FIELDS if field in row})
        if forbidden:
            raise AssertionError(f"preselection label fields leaked: {forbidden}")
        if len(preselection) != 40 or [x["fixed_eval_order"] for x in preselection] != list(range(40)):
            raise AssertionError("L81 fixed preselection order drift")
        for row in preselection:
            if row["candidate_count"] != len(row["row_keys"]) or row["candidate_count"] != len(row["row_offsets"]):
                raise AssertionError(f"candidate row completeness drift: {row['unit_key']}")
            for method in names:
                if len(row["score_fields"][method]) != row["candidate_count"]:
                    raise AssertionError(f"candidate score length drift: {row['unit_key']}/{method}")
                if not np.isfinite(np.asarray(row["score_fields"][method], dtype=np.float64)).all():
                    raise AssertionError(f"nonfinite candidate score: {row['unit_key']}/{method}")
        preselection_audit = {
            "format": "locatemot-l81-preselection-label-isolation-v1", "status": "complete",
            "fixed_records": 40, "calibration_records": 16, "validation_records": 24,
            "preselection_schema": sorted(preselection[0].keys()),
            "forbidden_label_fields": sorted(FORBIDDEN_PRESELECTION_FIELDS),
            "forbidden_fields_absent": not forbidden, "forbidden_fields_found": forbidden,
            "candidate_rows_and_scores_complete": all(
                row["candidate_count"] == len(row["row_keys"]) == len(row["row_offsets"]) and
                all(len(row["score_fields"][method]) == row["candidate_count"] for method in names)
                for row in preselection
            ),
            "native_fixed_order": [row["unit_key"] for row in preselection],
            "candidate_deletion": False, "candidate_truncation": False,
            "sidecar_labels_loaded": False, "calibration_labels_attached": False,
            "validation_labels_attached": False, "selection_frozen": False,
            "event": "all L81 checkpoint scores and complete native L69 rows constructed before L81 label attachment",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        write_json(out / "preselection_label_isolation.json", preselection_audit)

        calibration = [label_payload(preselection[index]) for index in range(16)]
        thresholds: dict[str, Any] = {}
        null_rules: dict[str, Any] = {}
        selection_candidates = []
        calibration_results: dict[str, Any] = {}
        for method in names:
            thresholds[method] = fit_candidate_threshold(calibration, method)
            null_rules[method] = fit_null_threshold(calibration, method)
            calibration_results[method] = {
                "candidate_only": metric(calibration, method, thresholds[method]["threshold"]),
                "candidate_plus_null": metric(
                    calibration, method, thresholds[method]["threshold"], True,
                    null_rules[method]["threshold"]),
            }
            selected_metric = calibration_results[method]["candidate_plus_null"]
            selection_candidates.append({
                "method": method, "step": checkpoint_info[method]["step"],
                "threshold": thresholds[method], "null_rule": null_rules[method],
                "calibration_metrics_for_selection": selected_metric,
                "lexicographic_key": [
                    float(selected_metric["hard_violation"] if selected_metric["hard_violation"] is not None else 1.0),
                    -float(selected_metric["minimum_positive_coverage"] if selected_metric["minimum_positive_coverage"] is not None else 0.0),
                    float(selected_metric["inactive_false_acceptance"]),
                    float(selected_metric["false_positive_rows"]),
                    int(checkpoint_info[method]["step"]),
                    float(checkpoint_info[method]["parameter_norm"]),
                ],
            })
        selection_candidates.sort(key=lambda item: tuple(item["lexicographic_key"]))
        selected = selection_candidates[0]
        preselection_audit.update({
            "calibration_labels_attached": True, "validation_labels_attached": False,
            "selection_frozen": True, "selection_method": selected["method"],
            "selection_step": selected["step"],
        })
        write_json(out / "preselection_label_isolation.json", preselection_audit)
        write_json(out / "checkpoint_selection.json", {
            "format": "locatemot-l81-checkpoint-selection-v1", "status": "complete",
            "selection_source": "fit/internal calibration only",
            "tuple": "lower calibration hard violation, higher calibration minimum-positive coverage, lower inactive false acceptance, lower calibration false-positive rows, earlier step, smaller parameter norm",
            "candidate_threshold_rule": "calibration-only exact observed candidate F1; higher F1, fewer FP rows, higher threshold",
            "null_rule": "calibration-only inactive-vs-present F1; present-uncovered excluded; null_logit threshold and cardinality_logit < 0",
            "candidates": selection_candidates, "selected": selected, "validation_used": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })

        validation = [label_payload(preselection[index]) for index in range(16, 40)]
        preselection_audit.update({"validation_labels_attached": True, "validation_labels_read": True})
        write_json(out / "preselection_label_isolation.json", preselection_audit)
        results = {}
        for method in names:
            results[method] = {
                "checkpoint": checkpoint_info[method], "threshold": thresholds[method],
                "null_rule": null_rules[method],
                "calibration_candidate_only": calibration_results[method]["candidate_only"],
                "calibration_candidate_plus_null": calibration_results[method]["candidate_plus_null"],
                "validation_candidate_only": metric(validation, method, thresholds[method]["threshold"]),
                "validation_candidate_plus_null": metric(
                    validation, method, thresholds[method]["threshold"], True,
                    null_rules[method]["threshold"]),
            }
        selected_method = str(selected["method"])
        selected_validation = results[selected_method]["validation_candidate_plus_null"]
        gate_checks = {
            "hard_negative_improvement_ge_0.05": selected_validation["hard_violation"] is not None and selected_validation["hard_violation"] <= L29_VALIDATION_CONTROL["hard_violation"] - 0.05,
            "recall_floor": selected_validation["candidate_recall"] >= 0.7233333,
            "precision_floor": selected_validation["candidate_precision"] >= 0.0830188679,
            "fp_per_frame_ceiling": selected_validation["fp_per_frame"] <= 11.125,
            "predictions_per_positive_ceiling": selected_validation["predictions_per_positive"] <= 4.069,
            "multi_positive_floor": selected_validation["multi_positive_recall"] is not None and selected_validation["multi_positive_recall"] >= 0.7894444,
            "inactive_false_acceptance_lt_1": selected_validation["inactive_false_acceptance"] < 1.0,
            "candidate_keys_complete": all(
                row["candidate_count"] == len(row["row_keys"]) == len(row["row_offsets"]) == len(row["labels"])
                for row in validation
            ),
            "candidate_deletion_false": all(
                row["candidate_rows_retained"] == row["candidate_count"] and
                not row["candidate_deletion"] and not row["candidate_truncation"]
                for row in preselection
            ),
            "finite_scores": all(row["finite_scores"] for row in preselection),
            "both_domains_reported": {row["dataset"] for row in validation} == {"refer_kitti_v1", "refer_kitti_v2"},
        }
        decision = "semantic_gate_pass_pending_supervisor" if all(gate_checks.values()) else "semantic_gate_fail"
        controls = make_control_records()
        control_thresholds = immutable_control_thresholds()
        control_results = {}
        for name, field in (("l29_teacher", "l29"), ("l53_m0", "m0"), ("l54_continuous", "m54")):
            control_results[name] = {
                "source": "immutable L62 score_records.jsonl; L81 L69 rows are not paired row-by-row",
                "calibration": metric(controls[:16], field, control_thresholds[name]),
                "validation": metric(controls[16:], field, control_thresholds[name]),
                "fixed_threshold": control_thresholds[name],
            }
        semantic = {
            "format": "locatemot-l81-semantic-evaluation-v1", "status": "complete",
            "decision": decision, "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "luna_thread": THREAD, "seed": SEED, "methods": results,
            "immutable_controls": control_results, "checkpoint_selection": selected,
            "gate": {"final_output": "selected checkpoint candidate_plus_null",
                      "checks": gate_checks, "l29_validation_control": L29_VALIDATION_CONTROL,
                      "thresholds": {"hard_violation_max": 0.8666667, "recall_min": 0.7233333,
                                     "precision_min": 0.0830188679, "fp_per_frame_max": 11.125,
                                     "predictions_per_positive_max": 4.069,
                                     "multi_positive_min": 0.7894444}},
            "candidate_rows": {"calibration": int(sum(x["candidate_count"] for x in calibration)),
                               "validation": int(sum(x["candidate_count"] for x in validation)),
                               "all_rows_retained": True, "candidate_deletion": False,
                               "candidate_truncation": False},
            "label_events": {"preselection_scores_complete": True,
                              "calibration_labels_attached_before_selection": True,
                              "selection_and_thresholds_frozen_before_validation_attach": True,
                              "validation_labels_attached_after_selection": True},
            "evidence_type": "fixed 16-calibration/24-validation semantic probe; not screening, official test, HOTA or TrackEval",
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "elapsed_sec": time.perf_counter() - started,
        }
        gate = {
            "format": "locatemot-l81-semantic-gate-v1", "status": decision, "decision": decision,
            "selected_method": selected_method, "selected_step": int(selected["step"]),
            "selected_output": "candidate_plus_null", "checks": gate_checks,
            "l29_validation_control": L29_VALIDATION_CONTROL,
            "calibration_units": 16, "validation_units": 24,
            "selection_and_threshold_calibration_only": True,
            "candidate_set": "complete L69 rows; no sampling/top-k/NMS/deletion",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }
        write_json(out / "semantic.json", semantic)
        write_json(out / "gate_decision.json", gate)
        with (out / "score_records.jsonl").open("w") as handle:
            for order, row in enumerate(preselection):
                labeled = calibration[order] if order < 16 else validation[order - 16]
                payload = dict(row)
                payload.update({key: labeled[key] for key in (
                    "labels", "positive_indices", "positive_count", "target_ids", "target_present",
                    "candidate_present", "coverage_mask", "null_target", "category", "declared_category",
                    "sidecar_candidate_gt", "label_source", "sidecar_labels_loaded",
                )})
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        source_eval = ROOT / "tools/eval_l80_v12.py"
        source_model = ROOT / "locatemot/models/l81_hierarchical_early_fusion.py"
        source_runtime = ROOT / "locatemot/rmot/l81_runtime.py"
        write_json(out / "protocol_diff.json", {
            "format": "locatemot-l81-canonical-protocol-diff-v1", "status": "complete",
            "canonical_metric_source": str(source_eval), "canonical_metric_sha256": sha256_file(source_eval),
            "canonical_functions_imported": ["metric", "fit_candidate_threshold", "fit_null_threshold",
                                              "make_control_records", "immutable_control_thresholds",
                                              "fixed_metadata", "attach_record_labels", "checkpoint_norm", "checkpoint_step"],
            "new_l81_score_adapter": [str(source_model), str(source_runtime)],
            "differences": ["L81 uses complete L69 budget-40 rows and L81 canonical heads; immutable controls remain L62 records"],
            "fixed_order_equal_immutable_l62": True, "row_order_or_score_overwrite": False,
            "candidate_deletion": False, "candidate_truncation": False,
        })
        write_json(out / "provenance.json", {
            "format": "locatemot-l81-semantic-provenance-v1", "status": "complete", "command": command,
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD,
            "inputs": {"manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)},
                       "clip_weight": {"path": str(CLIP_WEIGHT), "sha256": sha256_file(CLIP_WEIGHT)},
                       "l69_features": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                       "l49_units": str(L49_DATA), "l62_rows": {"path": str(L62_ROWS), "sha256": sha256_file(L62_ROWS)},
                       "canonical_eval": {"path": str(source_eval), "sha256": sha256_file(source_eval)},
                       "l81_model": {"path": str(source_model), "sha256": sha256_file(source_model)},
                       "l81_runtime": {"path": str(source_runtime), "sha256": sha256_file(source_runtime)}},
            "checkpoints": checkpoint_info, "fixed_order": [row["unit_key"] for row in metadata],
            "preselection_label_isolation": str(out / "preselection_label_isolation.json"),
            "calibration_labels_attached_only_after_scores": True,
            "validation_labels_attached_only_after_selection": True,
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "same_class_hard_negative_metadata": "unavailable; all-negative diagnostics",
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "raw_dense_cache_written": False, "process_local_frame_cache_serialized": False,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
        write_json(out / "config.json", {
            "format": "locatemot-l81-semantic-config-v1", "fixed_units": 40,
            "calibration_units": 16, "validation_units": 24, "seed": SEED,
            "checkpoint_specs": [[name, str(path)] for name, path in checkpoint_specs],
            "selection_tuple": ["lower calibration hard violation", "higher calibration minimum-positive coverage",
                                "lower inactive false acceptance", "lower calibration false-positive rows",
                                "earlier step", "smaller parameter norm"],
            "canonical_metric_source": str(source_eval),
            "threshold_rule": "calibration-only exact observed candidate F1; higher F1, fewer FP rows, higher threshold",
            "null_rule": "calibration-only inactive-vs-present F1; present-uncovered excluded; null_logit >= threshold and cardinality_logit < 0",
            "candidate_set": "all native L69 rows; no sampling/top-k/NMS/deletion",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        })
        write_json(out / "status.json", {
            "format": "locatemot-l81-semantic-status-v1", "status": decision,
            "stage": "fixed-calibration-validation-semantic-gate", "command": command,
            "inputs": [str(MANIFEST), str(L62_ROWS)],
            "outputs": [str(out / name) for name in ("semantic.json", "gate_decision.json", "score_records.jsonl")],
            "failure_root_cause": None if decision != "semantic_gate_fail" else "see semantic.json gate checks and failure decomposition",
            "next_action": "await supervisor review; no automatic expansion" if decision != "semantic_gate_fail" else "stop L81 branch and write one evidence-based structural next action",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        })
        return {"status": decision, "selected_method": selected_method, "selected_step": selected["step"],
                "selected_validation": selected_validation, "checks": gate_checks, "output": str(out)}
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# L81 semantic evaluation — INCOMPLETE\n\n" + traceback.format_exc() +
            "\nNo screening/official-test labels, TrackEval/HOTA, ordinary MOT or OVMOT action was run.\n")
        raise
    finally:
        cache.clear(); store._store._bank = None; store._store._text_cache = None
        del clip_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="NAME=PATH; repeat for step100/250/500")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
