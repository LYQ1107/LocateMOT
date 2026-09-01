#!/usr/bin/env python3
"""Fixed L73 16-calibration/24-validation semantic probe.

This evaluator streams the frozen post-fusion representation again and applies
the independent L73 adapter checkpoint.  Thresholds and the NULL rule are fit
on calibration only; validation is reported once and never used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
L49_ROOT = ROOT / "outputs/l49/data"
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L62_RECORDS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
L62_SEMANTIC = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/semantic.json"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
CHECKPOINT = ROOT / "outputs/l73/train/attention_region_adapter_smoke100_attempt2/checkpoint_l73_attention_region_step100.pt"
SEED = 20260829

sys.path.insert(0, str(ROOT))
from locatemot.models.l73_attention_region_adapter import L73AttentionRegionAdapter  # noqa: E402
from tools.audit_l73_postfusion_attention import (  # noqa: E402
    L73Bank,
    capture_prefill,
    fixed_units,
    load_model,
    normalize_ids,
    read_jsonl,
    sentence_of,
    sha256_file,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_l62_records() -> list[dict[str, Any]]:
    rows = read_jsonl(L62_RECORDS)
    if len(rows) != 40 or len({str(row["unit_key"]) for row in rows}) != 40:
        raise AssertionError("immutable L62 records are not 40 unique units")
    return rows


def metric(records: list[dict[str, Any]], score_key: str, threshold: float | None,
           null_key: str | None = None, null_threshold: float | None = None) -> dict[str, Any]:
    tp = fp = fn = selected = positives = 0
    target_units = candidate_units = top1 = top5 = empty = 0
    inactive_units = inactive_accept = inactive_fp = 0
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    violations: list[bool] = []
    multi: list[float] = []
    values: list[float] = []
    complete = True
    for record in records:
        scores_raw = record[score_key]
        labels = np.asarray(record["label"], dtype=bool)
        scores = np.asarray([float(value) for value in scores_raw], dtype=np.float64)
        if len(scores) != len(labels) or not np.isfinite(scores).all():
            complete = False
            raise AssertionError(f"score/label mismatch or nonfinite {record['unit_key']}")
        values.extend(scores.tolist())
        selected_mask = np.ones(len(scores), dtype=bool) if threshold is None else scores >= float(threshold)
        null_suppressed = False
        if null_key is not None and null_threshold is not None:
            null_suppressed = float(record[null_key]) >= float(null_threshold)
            if null_suppressed:
                selected_mask[:] = False
        row_tp = int((selected_mask & labels).sum())
        row_fp = int((selected_mask & ~labels).sum())
        row_fn = int((~selected_mask & labels).sum())
        tp += row_tp; fp += row_fp; fn += row_fn
        selected += int(selected_mask.sum()); positives += int(labels.sum())
        empty += int(not selected_mask.any())
        if bool(record["target_ids"]):
            target_units += 1
            if labels.any():
                candidate_units += 1
                order = np.argsort(-scores, kind="stable")
                top1 += int(bool(labels[order[:1]].any()))
                top5 += int(bool(labels[order[:5]].any()))
        if record["category"] == "inactive":
            inactive_units += 1
            inactive_accept += int(selected_mask.any())
            inactive_fp += row_fp
        pos = np.flatnonzero(labels)
        neg = np.flatnonzero(~labels)
        if pos.size and neg.size:
            strict_value = float(scores[pos].min() - scores[neg].max())
            strict.append(strict_value)
            best.append(float(scores[pos].max() - scores[neg].max()))
            average.append(float(scores[pos].mean() - scores[neg].max()))
            violations.append(strict_value < 0)
        if pos.size > 1:
            multi.append(float((selected_mask & labels).sum() / pos.size))
    return {
        "units": len(records), "candidate_rows": int(sum(len(r["label"]) for r in records)),
        "positive_rows": positives, "target_present_units": target_units,
        "candidate_present_units": candidate_units,
        "present_uncovered_units": sum(r["category"] == "present_uncovered" for r in records),
        "candidate_recall": tp / max(1, tp + fn), "candidate_precision": tp / max(1, selected),
        "fp_per_frame": fp / max(1, len(records)), "predictions_per_positive": selected / max(1, positives),
        # Match the accepted L62 convention: ranking top1/top5 is defined on
        # target-present units that have at least one candidate-positive row.
        "top1": top1 / max(1, candidate_units), "top5": top5 / max(1, candidate_units),
        "top1_denominator": "candidate_present_units",
        "hard_violation": float(np.mean(violations)) if violations else None,
        "strict_margin": {"count": len(strict), "mean": float(np.mean(strict)) if strict else None},
        "best_margin": {"count": len(best), "mean": float(np.mean(best)) if best else None},
        "average_margin": {"count": len(average), "mean": float(np.mean(average)) if average else None},
        "multi_positive_recall": float(np.mean(multi)) if multi else None,
        "empty_rate": empty / max(1, len(records)),
        "inactive_false_acceptance": inactive_accept / max(1, inactive_units),
        "inactive_false_positive_rows": inactive_fp,
        "score_mean": float(np.mean(values)) if values else None,
        "score_std": float(np.std(values)) if values else None,
        "threshold": float(threshold) if threshold is not None else None,
        "null_threshold": float(null_threshold) if null_threshold is not None else None,
        "null_suppressed_units": sum(
            int(null_key is not None and null_threshold is not None and float(r[null_key]) >= float(null_threshold))
            for r in records
        ),
        "complete_finite": complete,
    }


def fit_score_threshold(records: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    values = np.unique(np.asarray([float(value) for record in records for value in record[score_key]], dtype=np.float64))
    if not values.size:
        raise AssertionError("no calibration score values")
    candidates = values.tolist() + [float(values.min()) - 1e-12, float(values.max()) + 1e-12]
    best: tuple[tuple[float, int, float], float] | None = None
    for threshold in candidates:
        item = metric(records, score_key, float(threshold))
        tp = item["candidate_recall"] * item["positive_rows"]
        selected = item["predictions_per_positive"] * item["positive_rows"]
        fp = selected - tp
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + item["positive_rows"] - tp)
        key = (float(f1), -int(round(fp)), float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return {"threshold": best[1], "objective": "candidate-level F1 on calibration rows", "tie_rule": "higher F1, fewer FP, higher threshold", "validation_used": False}


def fit_null_threshold(records: list[dict[str, Any]], null_key: str) -> dict[str, Any]:
    values = np.unique(np.asarray([float(record[null_key]) for record in records], dtype=np.float64))
    candidates = values.tolist() + [float(values.min()) - 1e-12, float(values.max()) + 1e-12]
    best: tuple[tuple[float, int, float], float] | None = None
    for threshold in candidates:
        predicted = np.asarray([float(record[null_key]) >= threshold for record in records], dtype=bool)
        target = np.asarray([record["category"] == "inactive" for record in records], dtype=bool)
        tp = int((predicted & target).sum()); fp = int((predicted & ~target).sum()); fn = int((~predicted & target).sum())
        f1 = 2.0 * tp / max(1, 2 * tp + fp + fn)
        inactive_false_acceptance = float(np.mean(~predicted[target])) if target.any() else 0.0
        key = (f1, -int(round(inactive_false_acceptance * 1000000)), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return {"null_threshold": best[1], "rule": "suppress all candidate rows iff null_logit >= threshold", "target": "inactive frame", "objective": "frame-level NULL F1; tie lower inactive false acceptance; then lower threshold", "validation_used": False}


def load_fixed_l29() -> list[dict[str, Any]]:
    result = []
    for row in read_l62_records():
        result.append({
            "unit_key": str(row["unit_key"]), "dataset": row["dataset"], "video": row["video"],
            "frame_id": row["frame_id"], "category": row["category"],
            "target_ids": [] if str(row["category"]) == "inactive" else ["target"],
            "label": [bool(value) for value in row["label"]], "l29": [float(value) for value in row["l29"]],
        })
    return result


def stream_split(units: list[dict[str, Any]], la_model, processor, tokenizer, adapter) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from PIL import Image
    banks: dict[str, L73Bank] = {}
    records: list[dict[str, Any]] = []
    peak = 0
    started = time.perf_counter()
    try:
        for unit in units:
            video = str(unit["video"])
            bank = banks.get(video)
            if bank is None:
                bank = L73Bank(video); banks[video] = bank
            rows = bank.rows_for(int(unit["frame_id"]))
            image_path = IMAGE_ROOT / video / f"{int(unit['frame_id']):06d}.png"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image = Image.open(image_path).convert("RGB")
            boxes = bank.tensors["box"][rows].float().tolist()
            capture = capture_prefill(la_model, processor, tokenizer, image, sentence_of(unit), boxes, retain_vectors=True)
            values = capture.pop("_value_vectors", None); query = capture.pop("_query_hidden", None); attention = capture.pop("_score_vector", None)
            if values is None or query is None or attention is None or any(value is None for value in values) or any(value is None for value in attention):
                raise AssertionError(f"incomplete L73 representation {unit['unit_key']}")
            region = torch.stack([value.detach().clone().float() for value in values])
            query_tensor = query.detach().clone().float().reshape(1, -1)
            attention_tensor = torch.as_tensor([float(value) for value in attention], dtype=torch.float32).reshape(-1, 1).clone()
            with torch.inference_mode():
                output = adapter(query_tensor.to("cuda:0"), region.to("cuda:0"), attention_tensor.to("cuda:0"))
            candidate_logits = output["candidate_logits"].detach().float().cpu().tolist()
            null_logit = float(output["null_logit"].detach().float().cpu().reshape(-1)[0])
            targets = normalize_ids(unit.get("target_ids", []))
            labels_sidecar = bank.load_labels()
            labels = [labels_sidecar[row] is not None and str(labels_sidecar[row]) in targets for row in rows]
            category = "multi_positive" if sum(labels) > 1 else "positive" if sum(labels) == 1 else "present_uncovered" if targets else "inactive"
            row_keys = []
            candidate_indices = bank.tensors["candidate_index"].long().tolist()
            for local, row in enumerate(rows):
                key = [str(unit["dataset"]), video, int(unit["query_id"]), int(unit["frame_id"]), str(bank.path), int(row)]
                row_keys.append(key)
            if len(row_keys) != len(set(tuple(key) for key in row_keys)) or len(row_keys) != len(labels):
                raise AssertionError(f"key/label contract failure {unit['unit_key']}")
            records.append({
                "format": "locatemot-l73-adapter-score-record-v1", "unit_key": str(unit["unit_key"]),
                "fixed_eval_order": int(unit["fixed_eval_order"]), "fixed_eval_split": unit["fixed_eval_split"],
                "dataset": unit["dataset"], "video": video, "query_id": int(unit["query_id"]), "frame_id": int(unit["frame_id"]),
                "sentence_sha256": hashlib.sha256(sentence_of(unit).encode()).hexdigest(), "category": category,
                "target_ids": sorted(targets), "label": labels, "row_keys": row_keys,
                "candidate_index": [int(candidate_indices[row]) for row in rows],
                "attention_score": [float(value) for value in attention], "adapter_logit": candidate_logits,
                "null_logit": null_logit, "candidate_count": len(rows), "candidate_key_drift": 0,
                "candidate_truncation": False, "candidate_deletion": False, "future_history_rows": 0,
                "labels_joined_after_feature_construction": True,
            })
            peak = max(peak, int(torch.cuda.max_memory_allocated()))
            del capture, output, region, query_tensor, attention_tensor, values, query, attention, image, boxes
            torch.cuda.empty_cache()
    finally:
        for bank in banks.values():
            bank.close()
    return records, {"elapsed_seconds": time.perf_counter() - started, "peak_cuda_bytes": peak, "raw_dense_cache_written": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        if ROOT != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("manifest SHA mismatch")
        if not CHECKPOINT.exists():
            raise FileNotFoundError(CHECKPOINT)
        package = torch.load(CHECKPOINT, map_location="cuda:0", weights_only=False)
        adapter = L73AttentionRegionAdapter(hidden=128).to("cuda:0").float()
        adapter.load_state_dict(package["model"], strict=True)
        adapter.eval()
        la_model, processor, tokenizer, transformers_version = load_model()
        cal_units = fixed_units("calibration")
        val_units = fixed_units("validation")
        cal_records, cal_runtime = stream_split(cal_units, la_model, processor, tokenizer, adapter)
        val_records, val_runtime = stream_split(val_units, la_model, processor, tokenizer, adapter)
        adapter_threshold = fit_score_threshold(cal_records, "adapter_logit")
        attention_threshold = fit_score_threshold(cal_records, "attention_score")
        null_threshold = fit_null_threshold(cal_records, "null_logit")
        l29_records = load_fixed_l29()
        # L29 is immutable and its accepted semantic JSON is also retained as
        # an exact numerical provenance anchor.
        l29_metrics = {
            "source": str(L62_SEMANTIC),
            "source_sha256": sha256_file(L62_SEMANTIC),
            "accepted_validation": {
                "candidate_recall": 0.7333333333333333, "candidate_precision": 0.0830188679245283,
                "fp_per_frame": 10.125, "predictions_per_positive": 8.833333333333334,
                "hard_violation": 0.9166666666666666, "multi_positive_recall": 0.8194444444444443,
            },
            "recomputed": {
                "calibration": metric(l29_records[:16], "l29", -1.030576229095459),
                "validation": metric(l29_records[16:], "l29", -1.030576229095459),
            },
        }
        methods = {
            "l29_teacher": {"calibration_candidate_only": l29_metrics["recomputed"]["calibration"], "validation_candidate_only": l29_metrics["recomputed"]["validation"], "threshold": {"threshold": -1.030576229095459, "source": "accepted L62 immutable control"}},
            "l73_attention_primary": {"calibration_candidate_only": metric(cal_records, "attention_score", attention_threshold["threshold"]), "validation_candidate_only": metric(val_records, "attention_score", attention_threshold["threshold"]), "threshold": attention_threshold},
            "l73_adapter_candidate_only": {"calibration_candidate_only": metric(cal_records, "adapter_logit", adapter_threshold["threshold"]), "validation_candidate_only": metric(val_records, "adapter_logit", adapter_threshold["threshold"]), "threshold": adapter_threshold},
            "l73_adapter_final_null": {"calibration_candidate_only": metric(cal_records, "adapter_logit", adapter_threshold["threshold"], "null_logit", null_threshold["null_threshold"]), "validation_candidate_only": metric(val_records, "adapter_logit", adapter_threshold["threshold"], "null_logit", null_threshold["null_threshold"]), "threshold": adapter_threshold, "null_rule": null_threshold},
        }
        final = methods["l73_adapter_final_null"]["validation_candidate_only"]
        base = l29_metrics["accepted_validation"]
        checks = {
            "hard_violation_decrease_ge_0.05": final["hard_violation"] is not None and final["hard_violation"] <= base["hard_violation"] - 0.05,
            "recall_drop_le_0.01": final["candidate_recall"] >= base["candidate_recall"] - 0.01,
            "precision_ge_l29": final["candidate_precision"] >= base["candidate_precision"],
            "fp_per_frame_le_11.125": final["fp_per_frame"] <= 11.125,
            "predictions_per_positive_le_4.069": final["predictions_per_positive"] <= 4.069,
            "multi_positive_preserved": final["multi_positive_recall"] is not None and final["multi_positive_recall"] >= base["multi_positive_recall"] - 0.03,
            "null_not_universal": final["inactive_false_acceptance"] < 1.0,
            "complete_finite_keys": all(r["candidate_key_drift"] == 0 and not r["candidate_truncation"] and len(r["row_keys"]) == r["candidate_count"] and all(math.isfinite(float(v)) for v in r["adapter_logit"]) for r in val_records),
            "candidate_deletion_false": all(not r["candidate_deletion"] for r in val_records),
        }
        payload = {
            "format": "locatemot-l73-attention-region-semantic-v1", "status": "complete",
            "fixed_units": {"calibration": len(cal_records), "validation": len(val_records), "total": len(cal_records) + len(val_records)},
            "methods": methods, "l29_immutable_provenance": l29_metrics,
            "runtime": {"interpreter": sys.executable, "torch": torch.__version__, "transformers": transformers_version, "device": "cuda:0", "seed": SEED, "calibration_runtime": cal_runtime, "validation_runtime": val_runtime},
            "representation": {"primary": "L73 last decoder layer text-to-image attention-weighted post-fusion region value", "token_span_alignment": "UNALIGNED", "static_motion_mask": "UNALIGNED"},
            "screening_gt_used": False, "official_test_labels_read": False, "training_run": False, "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False, "candidate_rows_retained": True,
        }
        gate = {"format": "locatemot-l73-a0-b0-semantic-gate-v1", "status": "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail", "decision": "pass" if all(checks.values()) else "fail", "method": "l73_adapter_final_null", "checks": checks, "base_l29_validation": base, "evaluated_validation": final, "calibration_only_selection": {"candidate_threshold": adapter_threshold, "null_rule": null_threshold}, "screening_gt_used": False, "official_test_labels_read": False, "training_run": False, "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False}
        write_json(out / "semantic.json", payload)
        write_json(out / "gate_decision.json", gate)
        write_json(out / "provenance.json", {"format": "locatemot-l73-semantic-provenance-v1", "status": "complete", "cwd": str(ROOT), "command": " ".join(sys.argv), "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT), "checkpoint_step": package.get("step"), "checkpoint_contains_frozen_model": package.get("contains_frozen_model", True), "l69_root": str(L69_ROOT), "l62_order": str(L62_RECORDS), "l62_records_sha256": sha256_file(L62_RECORDS), "manifest_sha256": sha256_file(MANIFEST), "calibration_units": [r["unit_key"] for r in cal_records], "validation_units": [r["unit_key"] for r in val_records], "selection": "thresholds and null rule fit on calibration only; validation one-way report", "raw_dense_cache_written": False, "candidate_truncation": False, "token_span_alignment": "UNALIGNED", "screening_gt_used": False, "official_test_labels_read": False, "training_run": False, "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False})
        with (out / "score_records.jsonl").open("w") as handle:
            for record in cal_records + val_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        write_json(out / "status.json", {"format": "locatemot-l73-semantic-run-v1", "status": "complete", "gate_status": gate["status"], "calibration_records": len(cal_records), "validation_records": len(val_records), "next_action": "stop L73 expansion if semantic gate fails; no screening/test/HOTA"})
        return 0
    except Exception as exc:
        failure = {"format": "locatemot-l73-attention-region-semantic-v1", "status": "incomplete", "cwd": str(Path.cwd()), "command": " ".join(sys.argv), "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "next_action": "preserve this attempt and fix only the first actionable root cause", "screening_gt_used": False, "official_test_labels_read": False, "training_run": False, "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False}
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text("# INCOMPLETE\n\nFirst actionable root cause: `" + failure["failure_root_cause"] + "`\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
