#!/usr/bin/env python3
"""Fixed 16-calibration/24-validation replay for L88C.

The selected checkpoint and corrected thresholds are frozen from the internal
dev TrackEval selection before labels for this fixed slice are attached.  The
historical L88 records are used only for the immutable original-rule and
pure-gate comparisons; they are never overwritten.
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

from l88_eval_common import (
    ASSET_ROOT, L85_CACHE, L88_CACHE, MANIFEST, MANIFEST_SHA, SEED, THREAD,
    EncoderCacheReader, L88ClipStore, load_checkpoint_into, make_runtime,
    sha256, write_json,
)
from l88_eval_metrics import metric as legacy_metric
from l88c_eval_metrics import metric
from l88_eval_fixed_semantic import (
    FORBIDDEN_LABEL_FIELDS, L29_VALIDATION, load_fixed_key_units, valid_mean,
    make_label_free_record,
)


WORK_ROOT = Path(__file__).resolve().parents[1]
L88_ORIGINAL = WORK_ROOT / "outputs/l88/eval/fixed_semantic_attempt3/score_records.jsonl"
ORIGINAL_SEMANTIC = WORK_ROOT / "outputs/l88/eval/fixed_semantic_attempt3/semantic.json"
PURE_THRESHOLDS = {"candidate_threshold": 0.75, "presence_threshold": 0.0, "null_margin": 0.0}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _attach_labels(store: Any, batch: Any, record: dict[str, Any]) -> None:
    from locatemot.rmot.l80_data import load_full_unit_for_labels

    full = load_full_unit_for_labels(str(batch.unit_key))
    labels = store.attach_labels(batch, full)
    candidate_gt = labels.get("candidate_gt", labels.get("sidecar_candidate_gt"))
    if candidate_gt is None:
        raise AssertionError(f"fixed semantic sidecar field missing: {batch.unit_key}")
    record.update({
        "labels": [bool(value) for value in labels["labels"].tolist()],
        "target_ids": [str(value) for value in labels["target_ids"]],
        "candidate_gt": [None if value is None else str(value) for value in candidate_gt],
        "positive_indices": [int(value) for value in labels["positive_indices"]],
        "positive_count": int(labels["positive_count"]),
        "target_present": bool(labels["target_present"]),
        "candidate_present": bool(labels["candidate_present"]),
        "coverage_mask": bool(labels["coverage_mask"]),
        "category": str(labels["category"]),
        "declared_category": str(labels.get("declared_category", "unknown")),
        "label_source": str(labels["label_source"]),
        "labels_attached": True,
        "label_fields_present": sorted(FORBIDDEN_LABEL_FIELDS.intersection(record) | {"labels"}),
    })


def _load_historical_records() -> list[dict[str, Any]]:
    records = _read_jsonl(L88_ORIGINAL.resolve())
    if len(records) != 40:
        raise AssertionError(f"historical fixed record count drift: {len(records)}")
    order = [int(row["fixed_eval_order"]) for row in records]
    if order != list(range(40)):
        raise AssertionError("historical fixed order drift")
    return records


def _method_bundle(records: list[dict[str, Any]], thresholds: dict[str, float], *, legacy: bool = False) -> dict[str, Any]:
    fn = legacy_metric if legacy else metric
    candidate_only = fn(records, float(thresholds["candidate_threshold"]), -1e30, -1e30)
    frozen = fn(records, float(thresholds["candidate_threshold"]),
                float(thresholds["presence_threshold"]), float(thresholds["null_margin"]))
    return {"candidate_only": candidate_only, "frozen_rule": frozen, "thresholds": thresholds}


def _check_record_shape(record: dict[str, Any]) -> None:
    count = int(record["candidate_count"])
    for name in ("score", "r_total", "candidate_prior", "row_keys", "row_offsets", "candidate_indices"):
        if len(record[name]) != count:
            raise AssertionError(f"fixed row length drift: {record['unit_key']} {name}")
    if not bool(record.get("finite_scores")) or bool(record.get("candidate_deletion")) or bool(record.get("candidate_truncation")):
        raise AssertionError(f"invalid fixed row flags: {record['unit_key']}")


def run(args: argparse.Namespace) -> int:
    global batch_store
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88C fixed semantic output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    runtime = reader = batch_store = sidecar = None
    try:
        selection_path = args.selection.resolve()
        selection = json.loads(selection_path.read_text())
        if selection.get("status") != "complete" or not selection.get("selection_frozen_before_fixed_validation"):
            raise AssertionError("corrected selection was not frozen before fixed validation")
        final = selection["final_selection"]
        final_rule = str(final["rule"])
        final_thresholds = {name: float(final["rule_fit"][name])
                            for name in ("candidate_threshold", "presence_threshold", "null_margin")}
        checkpoint_path = Path(str(final["checkpoint_info"]["path"])).resolve()
        if sha256(checkpoint_path) != str(final["checkpoint_info"]["sha256"]):
            raise AssertionError(f"selected checkpoint SHA drift: {checkpoint_path}")

        rows = load_fixed_key_units()
        if [str(row["evaluation_partition"]) for row in rows[:16]] != ["calibration"] * 16:
            raise AssertionError("calibration prefix drift")
        if [str(row["evaluation_partition"]) for row in rows[16:]] != ["validation"] * 24:
            raise AssertionError("validation suffix drift")

        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
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
                    raise AssertionError(f"empty fixed candidate set: {batch.unit_key}")
                if int((batch.history_frame_ids > int(batch.frame_id)).sum()) != 0:
                    raise AssertionError(f"future history: {batch.unit_key}")
                item = reader.read(f"{batch.dataset}|{batch.video}|{int(batch.frame_id):06d}", device)
                from locatemot.rmot.l88_grounding_runtime import forward_l88_z1

                replay = forward_l88_z1(
                    runtime.model, item, batch.boxes, [str(row["sentence"])], device,
                    query_tile=1, autocast_bf16=False,
                )
                text_global = valid_mean(replay["memory_text"].float(), replay.get("text_token_mask"), "fixed_text_global")
                memory_mask = replay.get("memory_mask")
                frame_global = valid_mean(
                    replay["memory"].float(), None if memory_mask is None else ~memory_mask.bool(), "fixed_frame_global"
                )
                output = sidecar(
                    replay["z1"].float(), text_global, frame_global,
                    batch.observations.float().to(device), batch.history_observations.float().to(device),
                    batch.history_mask.to(device), batch.history_frame_ids.to(device),
                    batch.frame_id, temporal_enabled=True,
                )
                record = make_label_free_record(batch, output, row, checkpoint_info)
                _check_record_shape(record)
                records.append(record)
                batches.append(batch)
                del replay, output, item, text_global, frame_global
                batch_store._store.load_video(str(batch.video))
                gc.collect()
        if len(records) != 40:
            raise AssertionError(f"corrected fixed record count drift: {len(records)}")
        if [str(row["unit_key"]) for row in records] != [str(row["unit_key"]) for row in rows]:
            raise AssertionError("corrected fixed key order drift")

        preselection = {
            "format": "locatemot-l88c-fixed-preselection-v1", "status": "complete",
            "record_count": len(records), "order": [str(row["unit_key"]) for row in records],
            "forbidden_label_fields": sorted(FORBIDDEN_LABEL_FIELDS),
            "forbidden_fields_absent": [not bool(FORBIDDEN_LABEL_FIELDS.intersection(row)) for row in records],
            "all_forbidden_fields_absent": all(not FORBIDDEN_LABEL_FIELDS.intersection(row) for row in records),
            "candidate_count_lengths_complete": all(len(row["score"]) == int(row["candidate_count"]) for row in records),
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "selection_path": str(selection_path), "selection_frozen_before_labels": True,
            "thresholds": final_thresholds, "rule": final_rule,
            "labels_attached": False, "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }
        write_json(out / "preselection_label_isolation.json", preselection)
        if not preselection["all_forbidden_fields_absent"]:
            raise AssertionError("preselection label leak")

        for index in range(16):
            _attach_labels(batch_store, batches[index], records[index])
        calibration = records[:16]
        calibration_metrics = _method_bundle(calibration, final_thresholds)
        for index in range(16, 40):
            _attach_labels(batch_store, batches[index], records[index])
        validation = records[16:]
        validation_metrics = _method_bundle(validation, final_thresholds)

        historical = _load_historical_records()
        original_semantic = json.loads(ORIGINAL_SEMANTIC.resolve().read_text())
        original_bundle = _method_bundle(historical, PURE_THRESHOLDS, legacy=True)
        pure_bundle = _method_bundle(historical, PURE_THRESHOLDS, legacy=False)
        l29 = {
            "evidence_type": "immutable accepted L29 control copied from L62/L64 contract",
            "validation": L29_VALIDATION,
        }
        gate_metrics = validation_metrics["frozen_rule"]
        checks = {
            "hard_violation": float(gate_metrics["legacy_row_hard_violation"]) <= 0.8666666666666667,
            "recall": float(gate_metrics["legacy_candidate_recall"]) >= 0.7233333,
            "precision": float(gate_metrics["legacy_candidate_precision"]) >= 0.0830188679,
            "fp_per_frame": float(gate_metrics["legacy_fp_per_frame"]) <= 11.125,
            "predictions_per_positive": float(gate_metrics["legacy_predictions_per_positive"]) <= 4.069,
            "multi_positive": float(gate_metrics["legacy_row_multi_positive_recall"] or 0.0) >= 0.7894444,
            "inactive_false_acceptance": float(gate_metrics["inactive_false_acceptance"]) < 1.0,
            "finite_keys": bool(gate_metrics["finite_scores"] and gate_metrics["candidate_rows_retained"]),
            "no_deletion": not bool(gate_metrics["candidate_deletion"] or gate_metrics["candidate_truncation"]),
        }
        gate = {
            "format": "locatemot-l88c-fixed-semantic-gate-v1", "decision": "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail",
            "checks": checks, "thresholds": final_thresholds, "rule": final_rule,
            "hard_violation_target": 0.8666666666666667, "recall_floor": 0.7233333,
            "precision_floor": 0.0830188679, "fp_per_frame_ceiling": 11.125,
            "predictions_per_positive_ceiling": 4.069, "multi_positive_floor": 0.7894444,
        }
        semantic = {
            "format": "locatemot-l88c-fixed-semantic-v1", "status": "complete",
            "evidence_type": "fixed 16 calibration / 24 validation corrected replay after dev TrackEval selection",
            "checkpoint": checkpoint_info, "selection_source": str(selection_path),
            "selection_frozen_before_fixed_validation": True, "rule": final_rule,
            "thresholds": final_thresholds, "calibration": calibration_metrics,
            "validation": validation_metrics, "gate": gate, "l29_teacher": l29,
            "l88_original_final_rule_b": original_bundle,
            "l88_original_semantic_source": str(ORIGINAL_SEMANTIC.resolve()),
            "l88_original_semantic_source_sha256": sha256(ORIGINAL_SEMANTIC.resolve()),
            "l88c_pure_gate": pure_bundle,
            "l88c_corrected_final": {"calibration": calibration_metrics, "validation": validation_metrics},
            "historical_original_records_sha256": sha256(L88_ORIGINAL.resolve()),
            "record_count": len(records), "calibration_count": 16, "validation_count": 24,
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "base_detector_digest": base_digest, "adapter_target_manifest": injector.manifest(),
            "manifest_sha256": MANIFEST_SHA, "seed": SEED,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "zero_training": True,
            "corrected_candidate_vs_null": True, "backward_called": False,
            "optimizer_step_called": False, "new_checkpoint_written": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "labels_attached_after_preselection": True, "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "failure_root_cause": None,
            "next_action": "write L88C root-cause analysis and final internal report",
        }
        with (out / "score_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        write_json(out / "semantic.json", semantic)
        write_json(out / "gate_decision.json", gate)
        write_json(out / "provenance.json", semantic)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-fixed-semantic-status-v1", "status": "complete",
            "decision": gate["decision"], "record_count": len(records),
            "calibration_count": 16, "validation_count": 24,
            "zero_training": True, "corrected_candidate_vs_null": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True,
        })
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88C fixed semantic evaluation — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-fixed-semantic-status-v1", "status": "incomplete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "zero_training": True, "corrected_candidate_vs_null": True,
            "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True,
        })
        raise
    finally:
        if sidecar is not None:
            del sidecar
        if runtime is not None:
            runtime.close()
        if batch_store is not None:
            try:
                batch_store._store.release_loaded_cache_items()
                batch_store._store.close()
            except Exception:
                pass
            del batch_store
        if reader is not None:
            del reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
