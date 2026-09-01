#!/usr/bin/env python3
"""Fixed L80-R1 calibration/validation evaluation.

This is a new evaluator for the R1 region-interface variant.  It reuses only
the immutable metric/control helpers from ``eval_l80_v12``; all R1 scores are
generated with the R1 model and the 8x8/4x4 runtime interface.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))
from locatemot.models.l80_r1_region import L80R1Config, L80R1RegionCorrespondence  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    EXPECTED_MANIFEST_SHA, FORBIDDEN_LABEL_FIELDS, L80BankStore, MANIFEST,
    load_full_unit_for_labels, sha256_file,
)
from locatemot.rmot.l80_r1_runtime import (  # noqa: E402
    CLIP_SHA256, CLIP_WEIGHT, FrameFeatureCache, load_clip, raw_inputs_for_unit_r1,
)
from tools import eval_l80_v12 as base  # noqa: E402

SEED = 20260829


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def build_score_record(meta: dict[str, Any], store: L80BankStore, clip_model: Any,
                       cache: FrameFeatureCache, model: L80R1RegionCorrespondence,
                       device: torch.device, method: str) -> dict[str, Any]:
    batch = store.build_unit(meta)
    if not Path(batch.image_path).is_file():
        raise FileNotFoundError(batch.image_path)
    raw = raw_inputs_for_unit_r1(clip_model, batch, device, cache)
    history = batch.history_observations.to(device=device).clone()
    history_mask = batch.history_mask.to(device=device).clone()
    history_frames = batch.history_frame_ids.to(device=device).clone()
    with torch.inference_mode():
        output = model(raw["visual_tokens"], raw["text_tokens"], raw["text_mask"], history,
                       history_mask, history_frames, int(batch.frame_id))
    score = output["candidate_logits"].float().cpu().tolist()
    track = output["track_logits"].float().cpu().tolist()
    continuation = output["continuation_logits"].float().cpu().tolist()
    quality = output["quality_logits"].float().cpu().tolist()
    null_logit = float(output["null_logit"].float().cpu())
    cardinality = float(output["cardinality_logit"].float().cpu())
    arrays = [score, track, continuation, quality, [null_logit], [cardinality]]
    if any(not np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in arrays):
        raise FloatingPointError(f"nonfinite R1 score: {batch.unit_key}/{method}")
    if len(score) != batch.candidate_count:
        raise AssertionError(f"R1 candidate score length drift: {batch.unit_key}/{method}")
    record = {
        "format": "locatemot-l80-r1-score-record-v1",
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
        "score_fields": {method: score}, "track_score_fields": {method: track},
        "continuation_score_fields": {method: continuation},
        "quality_score_fields": {method: quality}, "null_logits": {method: null_logit},
        "cardinality_logits": {method: cardinality},
        "history_future_rows": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
        "text_valid_tokens": int(batch.text_mask.sum()),
        "candidate_rows_retained": int(batch.candidate_count),
        "candidate_deletion": False, "candidate_truncation": False,
        "sidecar_labels_loaded": False, "finite_scores": True,
        "source_pool_ids_provenance_only": True,
        "region_interface": {"roi_grid": 8, "context_grid": 4, "region_tokens": 243},
    }
    if record["history_future_rows"] != 0:
        raise AssertionError(f"future history in R1 evaluation: {batch.unit_key}")
    del output, raw, history, history_mask, history_frames, batch
    return record


def score_checkpoint(metadata: list[dict[str, Any]], checkpoint: Path, method: str,
                     clip_model: Any, cache: FrameFeatureCache, store: L80BankStore,
                     device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    package = base.safe_load(checkpoint, map_location="cpu")
    config = L80R1Config(**package["model_config"])
    if config.tokens_per_scale != 81:
        raise AssertionError(f"R1 checkpoint has wrong region contract: {config.tokens_per_scale}")
    model = L80R1RegionCorrespondence(config).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"R1 strict checkpoint load failed: {loaded}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records = [build_score_record(item, store, clip_model, cache, model, device, method)
               for item in metadata]
    info = {
        "path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint), "method": method,
        "step": base.checkpoint_step(package, checkpoint), "epoch": int(package.get("epoch", 0)),
        "parameter_norm": base.checkpoint_norm(package),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "model_config": config.__dict__, "strict_reload": True,
    }
    del model, package
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, info


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 evaluation output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA changed")
    if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
        raise AssertionError("CLIP SHA changed")
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    metadata = base.fixed_metadata()
    specs = []
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError("--checkpoint must be NAME=PATH")
        name, value_path = value.split("=", 1)
        path = Path(value_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append((str(name), path))
    if not specs or len({name for name, _ in specs}) != len(specs):
        raise ValueError("unique R1 checkpoint specifications are required")
    names = [name for name, _ in specs]
    clip_model = load_clip(device)
    cache = FrameFeatureCache(max_items=64)
    store = L80BankStore(max_history=8)
    preselection: list[dict[str, Any]] = []
    checkpoint_info: dict[str, Any] = {}
    try:
        for method, checkpoint in specs:
            current, info = score_checkpoint(metadata, checkpoint, method, clip_model, cache, store, device)
            checkpoint_info[method] = info
            if not preselection:
                preselection = current
            else:
                for base_row, current_row in zip(preselection, current):
                    base.merge_score_record(base_row, current_row, method)
        if len(preselection) != 40 or [x["fixed_eval_order"] for x in preselection] != list(range(40)):
            raise AssertionError("R1 fixed 40-unit order drift")
        forbidden = sorted({field for row in preselection for field in FORBIDDEN_LABEL_FIELDS if field in row})
        if forbidden:
            raise AssertionError(f"R1 labels leaked before selection: {forbidden}")
        for row in preselection:
            if row["candidate_count"] != len(row["row_keys"]) or row["candidate_count"] != len(row["row_offsets"]):
                raise AssertionError(f"R1 row count drift: {row['unit_key']}")
            for method in names:
                values = np.asarray(row["score_fields"][method], dtype=np.float64)
                if len(values) != row["candidate_count"] or not np.isfinite(values).all():
                    raise AssertionError(f"R1 score length/finite drift: {row['unit_key']}/{method}")
        preselection_audit = {
            "format": "locatemot-l80-r1-preselection-label-isolation-v1", "status": "complete",
            "fixed_records": 40, "calibration_records": 16, "validation_records": 24,
            "preselection_schema": sorted(preselection[0].keys()),
            "forbidden_label_fields": sorted(FORBIDDEN_LABEL_FIELDS), "forbidden_fields_absent": not forbidden,
            "candidate_rows_and_scores_complete": True, "candidate_deletion": False,
            "candidate_truncation": False, "sidecar_labels_loaded": False,
            "calibration_labels_attached": False, "validation_labels_attached": False,
            "selection_frozen": False,
            "event": "all R1 checkpoints and complete native L69 candidate rows scored before label attachment",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        write_json(out / "preselection_label_isolation.json", preselection_audit)
        cal_rows = [base.attach_record_labels(preselection[index], load_full_unit_for_labels(metadata[index]["unit_key"]))
                    for index in range(16)]
        thresholds = {}; null_rules = {}; calibration_metrics = {}; selection_candidates = []
        for method in names:
            thresholds[method] = base.fit_candidate_threshold(cal_rows, method)
            null_rules[method] = base.fit_null_threshold(cal_rows, method)
            calibration_metrics[method] = {
                "candidate_only": base.metric(cal_rows, method, thresholds[method]["threshold"]),
                "candidate_plus_null": base.metric(cal_rows, method, thresholds[method]["threshold"], True,
                                                    null_rules[method]["threshold"]),
            }
            selected_metric = calibration_metrics[method]["candidate_plus_null"]
            selection_candidates.append({
                "method": method, "step": checkpoint_info[method]["step"],
                "threshold": thresholds[method], "null_rule": null_rules[method],
                "calibration_metrics_for_selection": selected_metric,
                "lexicographic_key": [
                    float(selected_metric["hard_violation"] if selected_metric["hard_violation"] is not None else 1.0),
                    -float(selected_metric["minimum_positive_coverage"] if selected_metric["minimum_positive_coverage"] is not None else 0.0),
                    float(selected_metric["inactive_false_acceptance"]),
                    float(selected_metric["false_positive_rows"]), int(checkpoint_info[method]["step"]),
                    float(checkpoint_info[method]["parameter_norm"]),
                ],
            })
        selection_candidates.sort(key=lambda value: tuple(value["lexicographic_key"]))
        selected = selection_candidates[0]
        preselection_audit.update({"calibration_labels_attached": True, "selection_frozen": True})
        write_json(out / "preselection_label_isolation.json", preselection_audit)
        write_json(out / "checkpoint_selection.json", {
            "format": "locatemot-l80-r1-checkpoint-selection-v1", "status": "complete",
            "selection_source": "calibration-only; no validation selection",
            "tuple": "lower calibration hard violation, higher minimum-positive coverage, lower inactive false acceptance, lower false-positive rows, earlier step, smaller parameter norm",
            "candidate_threshold_rule": "calibration-only observed candidate F1; higher F1, fewer FP rows, higher threshold",
            "null_rule": "calibration-only inactive-vs-present F1; present-uncovered excluded; null_logit threshold and fixed cardinality_logit < 0",
            "candidates": selection_candidates, "selected": selected, "validation_used": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        val_rows = [base.attach_record_labels(preselection[index], load_full_unit_for_labels(metadata[index]["unit_key"]))
                    for index in range(16, 40)]
        preselection_audit.update({"validation_labels_attached": True, "validation_labels_read": True})
        write_json(out / "preselection_label_isolation.json", preselection_audit)
        methods = {}
        for method in names:
            methods[method] = {
                "checkpoint": checkpoint_info[method], "threshold": thresholds[method], "null_rule": null_rules[method],
                "calibration_candidate_only": calibration_metrics[method]["candidate_only"],
                "calibration_candidate_plus_null": calibration_metrics[method]["candidate_plus_null"],
                "validation_candidate_only": base.metric(val_rows, method, thresholds[method]["threshold"]),
                "validation_candidate_plus_null": base.metric(val_rows, method, thresholds[method]["threshold"], True,
                                                               null_rules[method]["threshold"]),
            }
        selected_method = str(selected["method"])
        selected_validation = methods[selected_method]["validation_candidate_plus_null"]
        checks = {
            "hard_negative_improvement_ge_0.05": selected_validation["hard_violation"] is not None and selected_validation["hard_violation"] <= base.L29_VALIDATION_CONTROL["hard_violation"] - 0.05,
            "recall_floor": selected_validation["candidate_recall"] >= 0.7233333,
            "precision_floor": selected_validation["candidate_precision"] >= 0.0830188679,
            "fp_per_frame_ceiling": selected_validation["fp_per_frame"] <= 11.125,
            "predictions_per_positive_ceiling": selected_validation["predictions_per_positive"] <= 4.069,
            "multi_positive_floor": selected_validation["multi_positive_recall"] is not None and selected_validation["multi_positive_recall"] >= 0.7894444,
            "inactive_false_acceptance_lt_1": selected_validation["inactive_false_acceptance"] < 1.0,
            "candidate_keys_complete": all(row["candidate_count"] == len(row["row_keys"]) == len(row["row_offsets"]) == len(row["labels"]) for row in val_rows),
            "candidate_deletion_false": all(row["candidate_rows_retained"] == row["candidate_count"] and not row["candidate_deletion"] and not row["candidate_truncation"] for row in preselection),
            "finite_scores": all(row["finite_scores"] for row in preselection),
            "both_domains_reported": {row["dataset"] for row in val_rows} == {"refer_kitti_v1", "refer_kitti_v2"},
        }
        decision = "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail"
        controls = base.make_control_records(); fixed_thresholds = base.immutable_control_thresholds()
        control_results = {}
        for name, field in (("l29_teacher", "l29"), ("l53_m0", "m0"), ("l54_continuous", "m54")):
            control_results[name] = {
                "source": "immutable L62 score_records.jsonl; not overwritten by R1 rows",
                "calibration": base.metric(controls[:16], field, fixed_thresholds[name]),
                "validation": base.metric(controls[16:], field, fixed_thresholds[name]),
                "fixed_threshold": fixed_thresholds[name],
            }
        semantic = {
            "format": "locatemot-l80-r1-semantic-evaluation-v1", "status": "complete", "decision": decision,
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "seed": SEED,
            "methods": methods, "immutable_controls": control_results, "checkpoint_selection": selected,
            "gate": {"final_output": "selected checkpoint candidate_plus_null", "checks": checks,
                      "l29_validation_control": base.L29_VALIDATION_CONTROL,
                      "thresholds": {"recall": 0.7233333, "precision": 0.0830188679, "fp_per_frame": 11.125,
                                     "predictions_per_positive": 4.069, "hard_violation_delta": 0.05,
                                     "multi_positive": 0.7894444}},
            "candidate_rows": {"calibration": int(sum(row["candidate_count"] for row in cal_rows)),
                               "validation": int(sum(row["candidate_count"] for row in val_rows)),
                               "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False},
            "label_events": {"preselection_scores_complete": True, "calibration_labels_attached_before_selection": True,
                              "selection_and_thresholds_frozen_before_validation_attach": True,
                              "validation_labels_attached_after_selection": True},
            "evidence_type": "fixed 16-calibration/24-validation R1 semantic probe; not screening, official test, HOTA or TrackEval",
            "elapsed_sec": time.perf_counter() - started,
        }
        gate = {"format": "locatemot-l80-r1-semantic-gate-v1", "status": decision, "decision": decision,
                "selected_method": selected_method, "selected_step": int(selected["step"]),
                "selected_output": "candidate_plus_null", "checks": checks,
                "l29_validation_control": base.L29_VALIDATION_CONTROL, "calibration_units": 16,
                "validation_units": 24, "selection_and_threshold_calibration_only": True,
                "candidate_set": "complete L69 rows; no sampling/top-k/NMS/deletion",
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True}
        write_json(out / "semantic.json", semantic); write_json(out / "gate_decision.json", gate)
        with (out / "score_records.jsonl").open("w") as handle:
            for order, row in enumerate(preselection):
                labeled = cal_rows[order] if order < 16 else val_rows[order - 16]
                payload = dict(row)
                payload.update({key: labeled[key] for key in ("labels", "positive_indices", "positive_count", "target_ids",
                    "target_present", "candidate_present", "coverage_mask", "null_target", "category", "declared_category",
                    "sidecar_candidate_gt", "label_source", "sidecar_labels_loaded")})
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        write_json(out / "provenance.json", {
            "format": "locatemot-l80-r1-evaluation-provenance-v1", "status": "complete", "command": command,
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
            "inputs": {"manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)},
                       "clip_weight": {"path": str(CLIP_WEIGHT), "sha256": sha256_file(CLIP_WEIGHT)},
                       "l69_features": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                       "l62_fixed_rows": {"path": str(base.L62_ROWS), "sha256": sha256_file(base.L62_ROWS)}},
            "checkpoints": checkpoint_info, "fixed_order": [row["unit_key"] for row in metadata],
            "preselection_label_isolation": str(out / "preselection_label_isolation.json"),
            "region_interface": {"roi_grid": 8, "context_grid": 4, "tokens_per_scale": 81, "region_tokens": 243,
                                 "only_structural_change": "R0 4x4/2x2 to R1 8x8/4x4"},
            "calibration_labels_attached_only_after_all_scores": True,
            "validation_labels_attached_only_after_selection": True, "candidate_rows_retained": True,
            "candidate_deletion": False, "candidate_truncation": False, "same_class_hard_negative_metadata": "unavailable",
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "raw_dense_cache_written": False, "process_local_frame_cache_serialized": False,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        })
        write_json(out / "config.json", {"format": "locatemot-l80-r1-eval-config-v1", "fixed_order_units": 40,
            "calibration_units": 16, "validation_units": 24, "seed": SEED,
            "checkpoint_specs": [[name, str(path)] for name, path in specs],
            "threshold_rule": "calibration-only observed candidate F1; higher F1, fewer FP rows, higher threshold",
            "checkpoint_rule": "calibration-only lower hard violation, higher minimum-positive coverage, lower inactive acceptance, lower FP rows, earlier step, smaller parameter norm",
            "null_rule": "calibration-only inactive-vs-present F1 with fixed cardinality_logit < 0; present-uncovered excluded",
            "candidate_set": "all native L69 rows; no top-k/NMS/sampling/deletion", "region_tokens": 243,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
        write_json(out / "status.json", {"format": "locatemot-l80-r1-status-v1", "status": decision,
            "stage": "fixed-calibration-validation-semantic-gate", "command": command,
            "outputs": [str(out / name) for name in ("semantic.json", "gate_decision.json", "score_records.jsonl")],
            "failure_root_cause": None if decision == "semantic_gate_pass" else "R1 fixed semantic gate checks failed",
            "next_action": "request supervisor review only if gate passes" if decision == "semantic_gate_pass" else "stop R1 and decompose the held-out failure",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
        return {"status": decision, "selected_method": selected_method, "selected_step": int(selected["step"]),
                "selected_validation": selected_validation, "checks": checks, "output": str(out)}
    except Exception:
        (out / "INCOMPLETE.md").write_text("# L80-R1 evaluation — INCOMPLETE\n\n" +
            __import__("traceback").format_exc() +
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
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
