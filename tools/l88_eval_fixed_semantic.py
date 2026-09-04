#!/usr/bin/env python3
"""Evaluate the frozen L88 dev selection on the fixed internal 16+24 slice.

All forty fixed units are first scored from key/text metadata with no label
fields present.  Only after the preselection audit is written are calibration
labels attached and evaluated, followed by the validation labels.  No
checkpoint, rule, threshold, or candidate set is selected here.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from l88_eval_common import (
    ASSET_ROOT, L88_CACHE, L88ClipStore, MANIFEST, MANIFEST_SHA, SEED, THREAD,
    EncoderCacheReader, load_checkpoint_into, make_runtime, sha256, write_json,
)
from l88_eval_metrics import metric


WORK_ROOT = Path(__file__).resolve().parents[1]
L69_FEATURE_ROOT = ASSET_ROOT / "outputs/l69/attempt9/budget40_features/kitti"
SELECTION_DEFAULT = WORK_ROOT / "outputs/l88/dev/final_selection_attempt1/checkpoint_selection.json"
FORBIDDEN_LABEL_FIELDS = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "target_present", "candidate_gt", "coverage_mask", "declared_category",
}
L29_VALIDATION = {
    "legacy_candidate_recall": 0.7333333333333333,
    "legacy_candidate_precision": 0.0830188679245283,
    "legacy_fp_per_frame": 10.125,
    "legacy_predictions_per_positive": 8.833333333333334,
    "legacy_row_hard_violation": 0.9166666666666666,
    "legacy_row_multi_positive_recall": 0.8194444444444444,
}


def key_only(row: dict[str, Any]) -> dict[str, Any]:
    allowed = ("unit_key", "dataset", "video", "query_id", "frame_id", "sentence", "expression",
               "evaluation_partition", "fixed_eval_order")
    result = {key: row[key] for key in allowed if key in row}
    sentence = str(result.get("sentence") or result.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty fixed semantic sentence: {row.get('unit_key')}")
    result["sentence"] = sentence; result["expression"] = sentence
    leaked = FORBIDDEN_LABEL_FIELDS.intersection(result)
    if leaked:
        raise AssertionError(f"fixed key-only record leaked labels: {sorted(leaked)}")
    return result


def load_fixed_key_units() -> list[dict[str, Any]]:
    import locatemot.rmot.l80_data as data
    rows = [key_only(row) for row in data.load_fixed_key_units()]
    if len(rows) != 40:
        raise AssertionError(f"fixed semantic unit count drift: {len(rows)}")
    for index, row in enumerate(rows):
        expected = "calibration" if index < 16 else "validation"
        if row.get("evaluation_partition") != expected:
            raise AssertionError(f"fixed semantic partition drift at {index}")
        if int(row.get("fixed_eval_order", index)) != index and "fixed_eval_order" in row:
            raise AssertionError(f"fixed semantic order drift at {index}")
        row["fixed_eval_order"] = index
    return rows


def valid_mean(value: torch.Tensor, mask: torch.Tensor | None, name: str) -> torch.Tensor:
    if value.ndim != 3:
        raise AssertionError(f"{name} rank drift: {tuple(value.shape)}")
    valid = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device) if mask is None else mask.bool()
    if tuple(valid.shape) != tuple(value.shape[:2]):
        raise AssertionError(f"{name} mask drift: {tuple(value.shape)} / {tuple(valid.shape)}")
    weights = valid.to(value.dtype).unsqueeze(-1)
    result = (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if not bool(torch.isfinite(result.float()).all()):
        raise FloatingPointError(f"nonfinite {name}")
    return result


def make_label_free_record(batch: Any, output: dict[str, torch.Tensor], row: dict[str, Any],
                           checkpoint_info: dict[str, Any]) -> dict[str, Any]:
    score = output["candidate_energy"].float().detach().cpu().numpy()
    r_total = output["r_total"].float().detach().cpu().numpy()
    prior = output["candidate_prior"].float().detach().cpu().numpy()
    presence = float(output["presence_logit"].float().detach().cpu().item())
    null = float(output["null_logit"].float().detach().cpu().item())
    if score.shape != (int(batch.candidate_count),):
        raise AssertionError(f"fixed semantic score/candidate drift: {batch.unit_key}")
    if not all(np.isfinite(value).all() for value in (score, r_total, prior, np.asarray([presence, null]))):
        raise FloatingPointError(f"nonfinite fixed semantic score: {batch.unit_key}")
    row_keys = [list(value) for value in batch.row_keys]
    if len(row_keys) != batch.candidate_count or [int(value[-1]) for value in row_keys] != batch.row_offsets:
        raise AssertionError(f"fixed semantic row key drift: {batch.unit_key}")
    return {
        "format": "locatemot-l88-fixed-semantic-score-v1",
        "unit_key": str(batch.unit_key), "dataset": str(batch.dataset), "video": str(batch.video),
        "query_id": int(batch.query_id), "frame_id": int(batch.frame_id),
        "fixed_eval_order": int(row["fixed_eval_order"]),
        "evaluation_partition": str(row["evaluation_partition"]),
        "candidate_count": int(batch.candidate_count), "row_offsets": [int(x) for x in batch.row_offsets],
        "row_keys": row_keys, "candidate_indices": [int(x) for x in batch.candidate_indices],
        "track_ids": [int(x) for x in batch.track_ids], "pool_ids": [int(x) for x in batch.pool_ids],
        "score": score.astype(np.float64).tolist(), "r_total": r_total.astype(np.float64).tolist(),
        "candidate_prior": prior.astype(np.float64).tolist(), "presence_logit": presence,
        "null_logit": null, "future_history_count": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
        "candidate_rows_retained": True, "candidate_deletion": False,
        "candidate_truncation": False, "finite_scores": True,
        "labels_attached": False, "label_fields_present": [],
        "checkpoint": checkpoint_info,
    }


def attach_labels(batch: Any, record: dict[str, Any]) -> None:
    from locatemot.rmot.l80_data import load_full_unit_for_labels
    full = load_full_unit_for_labels(str(batch.unit_key))
    labels = batch_store.attach_labels(batch, full)
    candidate_gt = labels.get("candidate_gt", labels.get("sidecar_candidate_gt"))
    if candidate_gt is None:
        raise AssertionError(f"fixed semantic sidecar field missing: {batch.unit_key}")
    record.update({
        "labels": [bool(x) for x in labels["labels"].tolist()],
        "target_ids": [str(x) for x in labels["target_ids"]],
        "candidate_gt": [None if x is None else str(x) for x in candidate_gt],
        "positive_indices": [int(x) for x in labels["positive_indices"]],
        "positive_count": int(labels["positive_count"]),
        "target_present": bool(labels["target_present"]),
        "candidate_present": bool(labels["candidate_present"]),
        "coverage_mask": bool(labels["coverage_mask"]),
        "category": str(labels["category"]),
        "declared_category": str(labels.get("declared_category", "unknown")),
        "label_source": str(labels["label_source"]), "labels_attached": True,
        "label_fields_present": sorted(FORBIDDEN_LABEL_FIELDS.intersection(record) | {"labels"}),
    })


def run(args: argparse.Namespace) -> int:
    global batch_store
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 fixed semantic output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    runtime = reader = sidecar = None
    batch_store = None
    try:
        selection_path = args.selection.resolve()
        selection = json.loads(selection_path.read_text())
        if selection.get("status") != "complete" or not selection.get("selection_frozen_before_fixed_validation"):
            raise AssertionError("L88 selection is not frozen")
        final = selection["final_selection"]
        checkpoint_path = Path(str(final["checkpoint_info"]["path"])).resolve()
        rule = str(final["rule"])
        rule_fit = final["rule_fit"]
        thresholds = {name: float(rule_fit[name]) for name in ("candidate_threshold", "presence_threshold", "null_margin")}
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        rows = load_fixed_key_units()
        if [str(row["evaluation_partition"]) for row in rows[:16]] != ["calibration"] * 16:
            raise AssertionError("calibration prefix drift")
        if [str(row["evaluation_partition"]) for row in rows[16:]] != ["validation"] * 24:
            raise AssertionError("validation suffix drift")
        reader = EncoderCacheReader(L88_CACHE)
        batch_store = __import__("locatemot.rmot.l80_data", fromlist=["L80BankStore"]).L80BankStore(max_history=8)
        runtime, injector, base_digest = make_runtime(device)
        sidecar, checkpoint_info = load_checkpoint_into(runtime, injector, checkpoint_path, device)
        records: list[dict[str, Any]] = []
        batches: list[Any] = []
        with torch.inference_mode():
            for row in rows:
                batch = batch_store.build_unit(row)
                if batch.candidate_count <= 0:
                    raise AssertionError(f"empty fixed semantic candidate set: {batch.unit_key}")
                if int((batch.history_frame_ids > int(batch.frame_id)).sum()) != 0:
                    raise AssertionError(f"future fixed semantic history: {batch.unit_key}")
                item = reader.read(f"{batch.dataset}|{batch.video}|{int(batch.frame_id):06d}", device)
                from locatemot.rmot.l88_grounding_runtime import forward_l88_z1
                replay = forward_l88_z1(runtime.model, item, batch.boxes, [str(row["sentence"])], device,
                                        query_tile=1, autocast_bf16=False)
                text_global = valid_mean(replay["memory_text"].float(), replay.get("text_token_mask"), "fixed_text_global")
                mmask = replay.get("memory_mask")
                frame_global = valid_mean(replay["memory"].float(), None if mmask is None else ~mmask.bool(), "fixed_frame_global")
                output = sidecar(
                    replay["z1"].float(), text_global, frame_global,
                    batch.observations.float().to(device), batch.history_observations.float().to(device),
                    batch.history_mask.to(device), batch.history_frame_ids.to(device),
                    batch.frame_id, temporal_enabled=True,
                )
                record = make_label_free_record(batch, output, row, checkpoint_info)
                records.append(record); batches.append(batch)
                del replay, output, item, text_global, frame_global
                batch_store._store.load_video(str(batch.video))
                gc.collect()
        if len(records) != 40:
            raise AssertionError(f"fixed semantic score record count drift: {len(records)}")
        # This file is written before any calibration or validation labels are
        # attached and is the machine-readable label-isolation boundary.
        preselection = {
            "format": "locatemot-l88-fixed-semantic-preselection-audit-v1", "status": "complete",
            "record_count": len(records), "order": [str(row["unit_key"]) for row in records],
            "forbidden_fields": sorted(FORBIDDEN_LABEL_FIELDS),
            "records_without_label_fields": [not bool(FORBIDDEN_LABEL_FIELDS.intersection(row)) for row in records],
            "all_forbidden_fields_absent": all(not FORBIDDEN_LABEL_FIELDS.intersection(row) for row in records),
            "thresholds_and_checkpoint_frozen_before_labels": True,
            "checkpoint_selection_path": str(selection_path), "rule": rule, "thresholds": thresholds,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "candidate_deletion": False,
            "candidate_truncation": False, "no_hota_or_trackeval": True,
        }
        write_json(out / "preselection_label_isolation.json", preselection)
        if not preselection["all_forbidden_fields_absent"]:
            raise AssertionError("fixed semantic preselection label leak")
        # Calibration labels are attached and evaluated before validation is
        # read.  The threshold/rule itself was already frozen from fit/dev.
        for index in range(16):
            attach_labels(batches[index], records[index])
        calibration = records[:16]
        calibration_candidate_only = metric(calibration, thresholds["candidate_threshold"], -1e30, -1e30)
        calibration_frozen = metric(calibration, thresholds["candidate_threshold"], thresholds["presence_threshold"], thresholds["null_margin"])
        for index in range(16, 40):
            attach_labels(batches[index], records[index])
        validation = records[16:]
        validation_candidate_only = metric(validation, thresholds["candidate_threshold"], -1e30, -1e30)
        validation_frozen = metric(validation, thresholds["candidate_threshold"], thresholds["presence_threshold"], thresholds["null_margin"])
        gate = {
            "hard_violation_target": float(L29_VALIDATION["legacy_row_hard_violation"] - 0.05),
            "recall_floor": 0.7233333, "precision_floor": 0.0830188679,
            "fp_per_frame_ceiling": 11.125, "predictions_per_positive_ceiling": 4.069,
            "multi_positive_floor": 0.7894444,
            "checks": {
                "hard_violation": float(validation_frozen["legacy_row_hard_violation"]) <= 0.8666666666666667,
                "recall": float(validation_frozen["legacy_candidate_recall"]) >= 0.7233333,
                "precision": float(validation_frozen["legacy_candidate_precision"]) >= 0.0830188679,
                "fp_per_frame": float(validation_frozen["legacy_fp_per_frame"]) <= 11.125,
                "predictions_per_positive": float(validation_frozen["legacy_predictions_per_positive"]) <= 4.069,
                "multi_positive": float(validation_frozen["legacy_row_multi_positive_recall"]) >= 0.7894444,
                "finite_keys": bool(validation_frozen["finite_scores"] and validation_frozen["candidate_rows_retained"]),
                "no_deletion": not bool(validation_frozen["candidate_deletion"] or validation_frozen["candidate_truncation"]),
            },
        }
        gate["decision"] = "semantic_gate_pass" if all(gate["checks"].values()) else "semantic_gate_fail"
        semantic = {
            "format": "locatemot-l88-fixed-semantic-v1", "status": "complete",
            "evidence_type": "fixed 16 calibration / 24 validation semantic report after dev selection",
            "checkpoint": checkpoint_info, "rule": rule, "thresholds": thresholds,
            "selection_source": str(selection_path), "selection_frozen_before_fixed_validation": True,
            "calibration_candidate_only": calibration_candidate_only, "calibration_frozen_rule": calibration_frozen,
            "validation_candidate_only": validation_candidate_only, "validation_frozen_rule": validation_frozen,
            "l29_validation_control": L29_VALIDATION, "gate": gate,
            "record_count": len(records), "calibration_count": 16, "validation_count": 24,
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "base_detector_digest": base_digest, "adapter_target_manifest": injector.manifest(),
            "manifest_sha256": MANIFEST_SHA, "seed": SEED,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "labels_attached_after_preselection": True,
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "failure_root_cause": None,
            "next_action": "run internal V1/V2 TrackEval on the frozen L88 strategy outputs",
        }
        with (out / "score_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        write_json(out / "semantic.json", semantic)
        write_json(out / "gate_decision.json", {
            "format": "locatemot-l88-fixed-semantic-gate-v1", "status": gate["decision"],
            "decision": gate["decision"], "checks": gate["checks"], "thresholds": thresholds,
            "checkpoint": checkpoint_info, "selection_frozen_before_fixed_validation": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "candidate_deletion": False,
            "candidate_truncation": False,
        })
        write_json(out / "provenance.json", semantic)
        write_json(out / "status.json", {"format": "locatemot-l88-fixed-semantic-status-v1", "status": "complete",
                                          "decision": gate["decision"], "record_count": len(records),
                                          "calibration_count": 16, "validation_count": 24,
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "no_hota_or_trackeval": True})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 fixed semantic evaluation — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l88-fixed-semantic-status-v1", "status": "incomplete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "no_hota_or_trackeval": True})
        raise
    finally:
        if sidecar is not None:
            del sidecar
        if runtime is not None:
            runtime.close()
        if batch_store is not None:
            del batch_store
        if reader is not None:
            del reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=SELECTION_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
