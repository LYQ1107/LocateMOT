#!/usr/bin/env python3
"""Fixed L80 calibration/validation evaluator.

This evaluator is deliberately separate from the older L79/L78 evaluators.  It
reconstructs the complete L69 candidate set from native frame pointers, scores
all rows with one or more frozen L80 checkpoints, and only then attaches the
expression-level labels needed for calibration and validation.  The raw CLIP
frame cache is process-local and is never serialized.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l80_raw_region_correspondence import L80Config, L80RawRegionCorrespondence  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    CATEGORIES,
    EXPECTED_MANIFEST_SHA,
    L49_DATA,
    L62_ROWS,
    L80BankStore,
    MANIFEST,
    load_fixed_key_units,
    load_full_unit_for_labels,
    read_jsonl,
    sha256_file,
)
from locatemot.rmot.l80_runtime import (  # noqa: E402
    CLIP_SHA256,
    CLIP_WEIGHT,
    FrameFeatureCache,
    load_clip,
    raw_inputs_for_unit,
)


SEED = 20260829
L62_SEMANTIC = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/semantic.json"
L29_VALIDATION_CONTROL = {
    "recall": 0.7333333333333333,
    "precision": 0.0830188679245283,
    "fp_per_frame": 10.125,
    "predictions_per_positive": 8.833333333333334,
    "hard_violation": 0.9166666666666666,
    "multi_positive_recall": 0.8194444444444443,
}
FORBIDDEN_PRESELECTION_FIELDS = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "target_present", "candidate_present", "coverage_mask", "sidecar_candidate_gt",
    "null_target", "label_source",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def safe_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def summary(values: Iterable[float]) -> dict[str, Any]:
    values = [float(x) for x in values]
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None,
                "p25": None, "p50": None, "p75": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
        "min": float(array.min()), "max": float(array.max()),
        "p25": float(np.quantile(array, 0.25)), "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
    }


def checkpoint_step(package: dict[str, Any], path: Path) -> int:
    if "step" in package:
        return int(package["step"])
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else 0


def checkpoint_norm(package: dict[str, Any]) -> float:
    value = 0.0
    for tensor in package.get("model_state_dict", {}).values():
        if torch.is_tensor(tensor):
            value += float(tensor.float().pow(2).sum())
    return float(math.sqrt(value))


def fixed_metadata() -> list[dict[str, Any]]:
    metadata = []
    for index, row in enumerate(load_fixed_key_units()):
        item = dict(row)
        item["fixed_eval_order"] = int(index)
        item["fixed_eval_split"] = "calibration" if index < 16 else "validation"
        if set(item) & FORBIDDEN_PRESELECTION_FIELDS:
            raise AssertionError(f"key-only metadata contains labels: {item['unit_key']}")
        metadata.append(item)
    rows = read_jsonl(L62_ROWS)
    expected = [str(row["unit_key"]) for row in rows]
    actual = [str(row["unit_key"]) for row in metadata]
    if len(rows) != 40 or actual != expected:
        raise AssertionError("L80 fixed metadata order is not the immutable L62 order")
    return metadata


def attach_record_labels(record: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    """Attach labels only after all preselection scores have been materialized."""
    sidecar_path = Path(record["bank_path"]).with_suffix(".labels.json")
    sidecar = json.loads(sidecar_path.read_text())
    candidate_gt = sidecar.get("candidate_gt")
    if not isinstance(candidate_gt, list):
        raise AssertionError(f"missing candidate_gt: {sidecar_path}")
    offsets = [int(x) for x in record["row_offsets"]]
    if max(offsets, default=-1) >= len(candidate_gt):
        raise AssertionError(f"sidecar is shorter than candidate rows: {record['unit_key']}")
    targets = {str(x) for x in unit.get("target_ids", [])}
    labels = [bool(candidate_gt[offset] is not None and str(candidate_gt[offset]) in targets)
              for offset in offsets]
    target_present = bool(targets)
    candidate_present = bool(any(labels))
    present_uncovered = target_present and not candidate_present
    category = (
        "inactive" if not target_present else
        "present_uncovered" if present_uncovered else
        "multi_positive" if sum(labels) > 1 else "positive"
    )
    result = dict(record)
    result.update({
        "labels": [int(x) for x in labels],
        "positive_indices": [int(i) for i, value in enumerate(labels) if value],
        "positive_count": int(sum(labels)),
        "target_ids": sorted(targets),
        "target_present": target_present,
        "candidate_present": candidate_present,
        "coverage_mask": not present_uncovered,
        "null_target": not target_present,
        "category": category,
        "declared_category": str(unit.get("category", "unknown")),
        "sidecar_candidate_gt": [None if candidate_gt[offset] is None else str(candidate_gt[offset]) for offset in offsets],
        "label_source": str(sidecar_path.resolve()),
        "sidecar_labels_loaded": True,
    })
    if len(result["labels"]) != int(record["candidate_count"]):
        raise AssertionError(f"label length drift: {record['unit_key']}")
    return result


def build_score_record(meta: dict[str, Any], store: L80BankStore, clip_model: Any,
                       cache: FrameFeatureCache, model: L80RawRegionCorrespondence,
                       device: torch.device, method: str) -> dict[str, Any]:
    # This function is intentionally label-free.  It receives key/text/frame
    # metadata only and never opens a candidate sidecar.
    batch = store.build_unit(meta)
    if not Path(batch.image_path).is_file():
        raise FileNotFoundError(batch.image_path)
    raw = raw_inputs_for_unit(clip_model, batch, device, cache)
    history = batch.history_observations.to(device=device).clone()
    history_mask = batch.history_mask.to(device=device).clone()
    history_frames = batch.history_frame_ids.to(device=device).clone()
    with torch.inference_mode():
        output = model(
            raw["visual_tokens"], raw["text_tokens"], raw["text_mask"], history,
            history_mask, history_frames, int(batch.frame_id),
        )
    score = output["candidate_logits"].float().cpu().tolist()
    track = output["track_logits"].float().cpu().tolist()
    continuation = output["continuation_logits"].float().cpu().tolist()
    quality = output["quality_logits"].float().cpu().tolist()
    null_logit = float(output["null_logit"].float().cpu())
    cardinality = float(output["cardinality_logit"].float().cpu())
    arrays = [score, track, continuation, quality, [null_logit], [cardinality]]
    if any(not np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in arrays):
        raise FloatingPointError(f"nonfinite L80 score: {batch.unit_key}/{method}")
    if len(score) != batch.candidate_count:
        raise AssertionError(f"candidate score length drift: {batch.unit_key}/{method}")
    record = {
        "format": "locatemot-l80-score-record-v1",
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
    }
    del output, raw, history, history_mask, history_frames, batch
    return record


def merge_score_record(base: dict[str, Any], current: dict[str, Any], method: str) -> None:
    for field in ("unit_key", "row_offsets", "row_keys", "candidate_count",
                  "candidate_index_provenance", "pool_id_provenance"):
        if base[field] != current[field]:
            raise AssertionError(f"L80 row drift for {base['unit_key']} field={field} method={method}")
    for field in ("score_fields", "track_score_fields", "continuation_score_fields",
                  "quality_score_fields", "null_logits", "cardinality_logits"):
        base[field].update(current[field])


def metric(records: list[dict[str, Any]], method: str, threshold: float,
           use_null: bool = False, null_threshold: float | None = None,
           _stratify: bool = True) -> dict[str, Any]:
    if use_null and null_threshold is None:
        raise ValueError("NULL threshold is required")
    tp = fp = fn = selected_count = positive_rows = 0
    top1 = top5 = top_units = empty = 0
    hard: list[bool] = []
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    multi_recall: list[float] = []
    minimum_coverage: list[float] = []
    inactive_units = inactive_accept = inactive_fp_rows = 0
    present_units = present_null_accept = present_null_reject = 0
    present_uncovered_units = candidate_present_units = 0
    score_values: list[float] = []
    null_values: list[float] = []
    cardinality_values: list[float] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scores = np.asarray(record["score_fields"][method], dtype=np.float64)
        labels = np.asarray(record["labels"], dtype=bool)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise AssertionError(f"metric score/label contract: {record['unit_key']}/{method}")
        null_value = float(record["null_logits"].get(method, 0.0))
        card_value = float(record["cardinality_logits"].get(method, 0.0))
        if not np.isfinite([null_value, card_value]).all():
            raise AssertionError(f"nonfinite NULL/cardinality: {record['unit_key']}/{method}")
        selected = scores >= float(threshold)
        null_accept = bool(use_null and null_value >= float(null_threshold) and card_value < 0.0)
        if null_accept:
            selected = np.zeros_like(selected, dtype=bool)
        tp += int((selected & labels).sum())
        fp += int((selected & ~labels).sum())
        fn += int((~selected & labels).sum())
        selected_count += int(selected.sum())
        positive_rows += int(labels.sum())
        category = str(record.get("category", "unknown"))
        by_category[category].append(record)
        by_dataset[str(record["dataset"])].append(record)
        score_values.extend(scores.tolist()); null_values.append(null_value); cardinality_values.append(card_value)
        candidate_present_units += int(labels.any())
        present_uncovered_units += int(category == "present_uncovered")
        present_units += int(category != "inactive")
        present_null_accept += int(category != "inactive" and null_accept)
        present_null_reject += int(category != "inactive" and not null_accept)
        if labels.any():
            order = np.argsort(-scores, kind="stable")
            top1 += int(bool(labels[order[:1]].any()))
            top5 += int(bool(labels[order[:5]].any()))
            top_units += 1
        empty += int(not selected.any())
        if category == "inactive":
            inactive_units += 1
            inactive_accept += int(bool(selected.any()))
            inactive_fp_rows += int((selected & ~labels).sum())
        pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
        if len(pos) and len(neg):
            strict_value = float(scores[pos].min() - scores[neg].max())
            strict.append(strict_value)
            best.append(float(scores[pos].max() - scores[neg].max()))
            average.append(float(scores[pos].mean() - scores[neg].max()))
            hard.append(strict_value < 0.0)
        if len(pos) > 1:
            multi_recall.append(float(selected[pos].sum() / len(pos)))
            minimum_coverage.append(float(selected[pos].all()))
    result: dict[str, Any] = {
        "method": method, "use_null": bool(use_null), "threshold": float(threshold),
        "null_threshold": None if null_threshold is None else float(null_threshold),
        "units": len(records), "candidate_rows": int(sum(len(x["labels"]) for x in records)),
        "positive_rows": int(positive_rows), "selected_rows": int(selected_count),
        "true_positive_rows": int(tp), "false_positive_rows": int(fp), "false_negative_rows": int(fn),
        "top1": top1 / max(1, top_units), "top5": top5 / max(1, top_units),
        "candidate_precision": tp / max(1, selected_count),
        "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(records)),
        "predictions_per_positive": selected_count / max(1, positive_rows),
        "hard_violation": float(np.mean(hard)) if hard else None,
        "strict_margin": summary(strict), "best_margin": summary(best), "average_margin": summary(average),
        "multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "minimum_positive_coverage": float(np.mean(minimum_coverage)) if minimum_coverage else None,
        "empty_rate": empty / max(1, len(records)),
        "inactive_units": int(inactive_units),
        "inactive_false_acceptance": inactive_accept / max(1, inactive_units),
        "inactive_false_positive_rows": int(inactive_fp_rows),
        "present_units": int(present_units), "present_uncovered_units": int(present_uncovered_units),
        "candidate_present_units": int(candidate_present_units),
        "null_acceptance_in_present": present_null_accept / max(1, present_units),
        "null_false_rejection_in_present": present_null_accept / max(1, present_units),
        "null_acceptance_in_inactive": inactive_accept / max(1, inactive_units),
        "null_false_acceptance": inactive_accept / max(1, inactive_units),
        "score_distribution": summary(score_values), "null_logit_distribution": summary(null_values),
        "cardinality_logit_distribution": summary(cardinality_values),
        "per_category": {}, "per_dataset": {},
    }
    if _stratify:
        for key, values in by_category.items():
            result["per_category"][key] = metric(values, method, threshold, use_null, null_threshold, False)
        for key, values in by_dataset.items():
            result["per_dataset"][key] = metric(values, method, threshold, use_null, null_threshold, False)
    return result


def fit_candidate_threshold(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = np.unique(np.concatenate([np.asarray(row["score_fields"][method], dtype=np.float64) for row in records]))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best_key: tuple[float, int, float] | None = None
    best: dict[str, Any] | None = None
    for threshold in candidates:
        tp = fp = fn = 0
        for row in records:
            score = np.asarray(row["score_fields"][method], dtype=np.float64)
            label = np.asarray(row["labels"], dtype=bool)
            selected = score >= float(threshold)
            tp += int((selected & label).sum()); fp += int((selected & ~label).sum()); fn += int((~selected & label).sum())
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        key = (float(f1), -int(fp), float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "f1": float(f1)}
    assert best is not None
    return {
        "threshold": best["threshold"],
        "objective": "exact observed candidate-level F1 on 16 calibration units",
        "tie_rule": "higher F1, fewer FP rows, higher threshold",
        "validation_used": False, "counts": best,
    }


def fit_null_threshold(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    usable = [row for row in records if str(row["category"]) in {"inactive", "positive", "multi_positive"}]
    if not usable:
        raise AssertionError("no calibration rows for NULL fit")
    values = np.unique(np.asarray([float(row["null_logits"][method]) for row in usable], dtype=np.float64))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best_key: tuple[float, int, float] | None = None
    best: dict[str, Any] | None = None
    for threshold in candidates:
        tp = fp = fn = 0
        for row in usable:
            target = str(row["category"]) == "inactive"
            predicted = float(row["null_logits"][method]) >= float(threshold) and float(row["cardinality_logits"][method]) < 0.0
            tp += int(predicted and target); fp += int(predicted and not target); fn += int((not predicted) and target)
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        key = (float(f1), -int(fp), float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "f1": float(f1)}
    assert best is not None
    return {
        "threshold": best["threshold"],
        "objective": "inactive-vs-present NULL F1 on calibration; present-uncovered excluded",
        "rule": "null_accept iff null_logit >= threshold and cardinality_logit < 0",
        "tie_rule": "higher F1, fewer false-positive NULL acceptances, higher threshold",
        "validation_used": False, "usable_calibration_units": len(usable), "counts": best,
    }


def make_control_records() -> list[dict[str, Any]]:
    rows = read_jsonl(L62_ROWS)
    if len(rows) != 40:
        raise AssertionError("immutable L62 rows are not 40")
    result = []
    for order, row in enumerate(rows):
        labels = [int(x) for x in row["label"]]
        score_fields = {name: [float(x) for x in row[name]] for name in ("l29", "m0", "m54")}
        if any(len(value) != len(labels) or not np.isfinite(value).all() for value in (np.asarray(x) for x in score_fields.values())):
            raise AssertionError(f"immutable control length/finite drift at {row['unit_key']}")
        result.append({
            "unit_key": str(row["unit_key"]), "dataset": str(row["dataset"]),
            "video": str(row["video"]), "query_id": int(row.get("query_id", -1)),
            "frame_id": int(row["frame_id"]), "fixed_eval_order": order,
            "fixed_eval_split": "calibration" if order < 16 else "validation",
            "labels": labels, "category": str(row.get("category", "unknown")),
            "candidate_count": len(labels), "score_fields": score_fields,
            "null_logits": {name: 0.0 for name in score_fields},
            "cardinality_logits": {name: 0.0 for name in score_fields},
            "row_keys": row.get("key_audit", {}), "candidate_deletion": False,
            "candidate_truncation": False, "finite_scores": True,
        })
    return result


def immutable_control_thresholds() -> dict[str, float]:
    data = json.loads(L62_SEMANTIC.read_text())
    result = {
        "l29_teacher": float(data["methods"]["l29_teacher"]["threshold"]["threshold"]),
        "l53_m0": float(data["methods"]["l53_m0"]["threshold"]["threshold"]),
        "l54_continuous": float(data["methods"]["l54_continuous"]["threshold"]["threshold"]),
    }
    return result


def score_checkpoint(meta: list[dict[str, Any]], checkpoint: Path, method: str,
                     clip_model: Any, cache: FrameFeatureCache, store: L80BankStore,
                     device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    package = safe_load(checkpoint, map_location="cpu")
    config = L80Config(**package["model_config"])
    model = L80RawRegionCorrespondence(config).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict L80 checkpoint load failed: {loaded}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records: list[dict[str, Any]] = []
    for item in meta:
        current = build_score_record(item, store, clip_model, cache, model, device, method)
        if current["history_future_rows"] != 0 or not current["finite_scores"]:
            raise AssertionError(f"L80 score contract failed: {current['unit_key']}")
        if records:
            pass
        records.append(current)
    info = {
        "path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint),
        "method": method, "step": checkpoint_step(package, checkpoint),
        "epoch": int(package.get("epoch", 0)), "parameter_norm": checkpoint_norm(package),
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
        raise FileExistsError(f"refusing nonempty L80 evaluation output: {out}")
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
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    metadata = fixed_metadata()
    checkpoint_specs: list[tuple[str, Path]] = []
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError("--checkpoint must be NAME=PATH")
        name, path = value.split("=", 1)
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_specs.append((str(name), path))
    if not checkpoint_specs:
        raise ValueError("at least one checkpoint is required")
    names = [name for name, _ in checkpoint_specs]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate checkpoint method name")

    clip_model = load_clip(device)
    cache = FrameFeatureCache(max_items=max(64, len(metadata)))
    store = L80BankStore(max_history=8)
    preselection: list[dict[str, Any]] = []
    checkpoint_info: dict[str, Any] = {}
    try:
        # Every checkpoint is scored against exactly the same native L69 rows.
        for method, checkpoint in checkpoint_specs:
            current, info = score_checkpoint(metadata, checkpoint, method, clip_model, cache, store, device)
            checkpoint_info[method] = info
            if not preselection:
                preselection = current
            else:
                for base, value in zip(preselection, current):
                    merge_score_record(base, value, method)
        forbidden = sorted({field for row in preselection for field in FORBIDDEN_PRESELECTION_FIELDS if field in row})
        if forbidden:
            raise AssertionError(f"preselection labels leaked: {forbidden}")
        if len(preselection) != 40 or [x["fixed_eval_order"] for x in preselection] != list(range(40)):
            raise AssertionError("preselection fixed order drift")
        for row in preselection:
            for method in names:
                if len(row["score_fields"][method]) != int(row["candidate_count"]):
                    raise AssertionError(f"candidate score length drift: {row['unit_key']}/{method}")
                if not np.isfinite(np.asarray(row["score_fields"][method], dtype=np.float64)).all():
                    raise AssertionError(f"nonfinite candidate score: {row['unit_key']}/{method}")
        preselection_audit = {
            "format": "locatemot-l80-preselection-label-isolation-v1", "status": "complete",
            "fixed_records": 40, "calibration_records": 16, "validation_records": 24,
            "preselection_schema": sorted(preselection[0].keys()),
            "forbidden_label_fields": sorted(FORBIDDEN_PRESELECTION_FIELDS),
            "forbidden_fields_absent": not forbidden, "forbidden_fields_found": forbidden,
            "candidate_rows_and_scores_complete": all(
                row["candidate_count"] == len(row["row_keys"]) == len(row["row_offsets"]) and
                all(len(row["score_fields"][name]) == row["candidate_count"] for name in names)
                for row in preselection
            ),
            "candidate_deletion": False, "candidate_truncation": False,
            "sidecar_labels_loaded": False, "calibration_labels_attached": False,
            "validation_labels_attached": False, "selection_frozen": False,
            "event": "all checkpoints and all raw candidate scores constructed before label attachment",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        write_json(out / "preselection_label_isolation.json", preselection_audit)

        # Calibration labels are the first label access after preselection.
        cal_rows = []
        for order in range(16):
            cal_rows.append(attach_record_labels(preselection[order], load_full_unit_for_labels(metadata[order]["unit_key"])))
        thresholds: dict[str, Any] = {}
        null_rules: dict[str, Any] = {}
        calibration_metrics: dict[str, Any] = {}
        selection_candidates = []
        for method in names:
            thresholds[method] = fit_candidate_threshold(cal_rows, method)
            null_rules[method] = fit_null_threshold(cal_rows, method)
            calibration_metrics[method] = {
                "candidate_only": metric(cal_rows, method, thresholds[method]["threshold"]),
                "candidate_plus_null": metric(
                    cal_rows, method, thresholds[method]["threshold"], True, null_rules[method]["threshold"]),
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
                    float(selected_metric["false_positive_rows"]),
                    int(checkpoint_info[method]["step"]),
                    float(checkpoint_info[method]["parameter_norm"]),
                ],
            })
        selection_candidates.sort(key=lambda value: tuple(value["lexicographic_key"]))
        selected = selection_candidates[0]
        preselection_audit.update({
            "calibration_labels_attached": True, "validation_labels_attached": False,
            "selection_frozen": True,
        })
        write_json(out / "preselection_label_isolation.json", preselection_audit)
        write_json(out / "checkpoint_selection.json", {
            "format": "locatemot-l80-checkpoint-selection-v1", "status": "complete",
            "selection_source": "fit/internal-calibration only",
            "tuple": "lower calibration hard violation, higher minimum-positive coverage, lower inactive false acceptance, lower false-positive rows, earlier step, smaller parameter norm",
            "candidate_threshold_rule": "calibration-only observed candidate F1; higher F1, fewer FP rows, higher threshold",
            "null_rule": "calibration-only inactive-vs-present F1; present-uncovered excluded; null_logit threshold and fixed cardinality_logit < 0",
            "candidates": selection_candidates, "selected": selected,
            "validation_used": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
        })

        # Validation labels are deliberately loaded only after the tuple and
        # both calibration rules have been frozen above.
        val_rows = []
        for order in range(16, 40):
            val_rows.append(attach_record_labels(preselection[order], load_full_unit_for_labels(metadata[order]["unit_key"])))
        preselection_audit["validation_labels_attached"] = True
        preselection_audit["validation_labels_read"] = True
        write_json(out / "preselection_label_isolation.json", preselection_audit)

        all_method_results: dict[str, Any] = {}
        for method in names:
            all_method_results[method] = {
                "checkpoint": checkpoint_info[method], "threshold": thresholds[method],
                "null_rule": null_rules[method],
                "calibration_candidate_only": calibration_metrics[method]["candidate_only"],
                "calibration_candidate_plus_null": calibration_metrics[method]["candidate_plus_null"],
                "validation_candidate_only": metric(val_rows, method, thresholds[method]["threshold"]),
                "validation_candidate_plus_null": metric(
                    val_rows, method, thresholds[method]["threshold"], True, null_rules[method]["threshold"]),
            }
        selected_method = str(selected["method"])
        selected_validation = all_method_results[selected_method]["validation_candidate_plus_null"]
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
                for row in val_rows
            ),
            "candidate_deletion_false": all(
                row["candidate_rows_retained"] == row["candidate_count"] and
                not row["candidate_deletion"] and not row["candidate_truncation"]
                for row in preselection
            ),
            "finite_scores": all(row["finite_scores"] for row in preselection),
            "both_domains_reported": {row["dataset"] for row in val_rows} == {"refer_kitti_v1", "refer_kitti_v2"},
        }
        decision = "semantic_gate_pass" if all(gate_checks.values()) else "semantic_gate_fail"
        controls = make_control_records()
        control_thresholds = immutable_control_thresholds()
        control_results = {}
        # Display names are intentionally distinct from the immutable L62
        # field names; never write a control score back into those rows.
        for name, field in (("l29_teacher", "l29"), ("l53_m0", "m0"), ("l54_continuous", "m54")):
            control_results[name] = {
                "source": "immutable L62 score_records.jsonl; L80 candidate rows are not paired row-by-row",
                "calibration": metric(controls[:16], field, control_thresholds[name]),
                "validation": metric(controls[16:], field, control_thresholds[name]),
                "fixed_threshold": control_thresholds[name],
            }
        semantic = {
            "format": "locatemot-l80-semantic-evaluation-v1", "status": "complete",
            "decision": decision, "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "seed": SEED,
            "methods": all_method_results, "immutable_controls": control_results,
            "checkpoint_selection": selected, "gate": {
                "final_output": "selected checkpoint candidate_plus_null",
                "checks": gate_checks, "l29_validation_control": L29_VALIDATION_CONTROL,
                "thresholds": {"recall": 0.7233333, "precision": 0.0830188679,
                               "fp_per_frame": 11.125, "predictions_per_positive": 4.069,
                               "hard_violation_delta": 0.05, "multi_positive": 0.7894444},
            },
            "candidate_rows": {
                "calibration": int(sum(row["candidate_count"] for row in cal_rows)),
                "validation": int(sum(row["candidate_count"] for row in val_rows)),
                "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            },
            "label_events": {
                "preselection_scores_complete": True,
                "calibration_labels_attached_before_selection": True,
                "selection_and_thresholds_frozen_before_validation_attach": True,
                "validation_labels_attached_after_selection": True,
            },
            "evidence_type": "fixed calibration/validation semantic probe; not screening, official test, HOTA or TrackEval",
            "elapsed_sec": time.perf_counter() - started,
        }
        gate = {
            "format": "locatemot-l80-semantic-gate-v1", "status": decision, "decision": decision,
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
                labeled = cal_rows[order] if order < 16 else val_rows[order - 16]
                payload = dict(row)
                payload.update({key: labeled[key] for key in (
                    "labels", "positive_indices", "positive_count", "target_ids", "target_present",
                    "candidate_present", "coverage_mask", "null_target", "category", "declared_category",
                    "sidecar_candidate_gt", "label_source", "sidecar_labels_loaded",
                )})
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        input_meta = {
            "manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)},
            "clip_weight": {"path": str(CLIP_WEIGHT), "sha256": sha256_file(CLIP_WEIGHT)},
            "l69_features": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
            "l49_calibration_units": str(L49_DATA / "calibration_units.jsonl"),
            "l49_validation_units": str(L49_DATA / "validation_units.jsonl"),
            "l62_fixed_rows": {"path": str(L62_ROWS), "sha256": sha256_file(L62_ROWS)},
            "l62_semantic": {"path": str(L62_SEMANTIC), "sha256": sha256_file(L62_SEMANTIC)},
        }
        write_json(out / "provenance.json", {
            "format": "locatemot-l80-evaluation-provenance-v1", "status": "complete", "command": command,
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "inputs": input_meta,
            "checkpoints": checkpoint_info, "fixed_order": [row["unit_key"] for row in metadata],
            "preselection_label_isolation": str(out / "preselection_label_isolation.json"),
            "calibration_labels_attached_only_after_scores": True,
            "validation_labels_attached_only_after_selection": True,
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "same_class_hard_negative_metadata": "unavailable; all-negative candidate diagnostics",
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "raw_dense_cache_written": False, "process_local_frame_cache_serialized": False,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        })
        write_json(out / "config.json", {
            "format": "locatemot-l80-eval-config-v1", "fixed_order_units": 40,
            "calibration_units": 16, "validation_units": 24, "seed": SEED,
            "checkpoint_specs": [[name, str(path)] for name, path in checkpoint_specs],
            "threshold_rule": "calibration-only observed candidate F1; higher F1, fewer FP rows, higher threshold",
            "checkpoint_rule": "calibration-only lower hard violation, higher minimum-positive coverage, lower inactive acceptance, lower FP rows, earlier step, smaller parameter norm",
            "null_rule": "calibration-only null_logit F1 with fixed cardinality_logit < 0; present-uncovered excluded",
            "candidate_set": "all native L69 rows; no top-k/NMS/sampling/deletion",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        })
        write_json(out / "status.json", {
            "format": "locatemot-l80-status-v1", "status": decision,
            "stage": "fixed-calibration-validation-semantic-gate", "command": command,
            "inputs": [str(MANIFEST), str(L62_ROWS)],
            "outputs": [str(out / name) for name in ("semantic.json", "gate_decision.json", "score_records.jsonl")],
            "failure_root_cause": None if decision == "semantic_gate_pass" else "decompose selected validation gate checks",
            "next_action": "request supervisor authorization for P5 only if gate passes" if decision == "semantic_gate_pass" else "stop R0 semantic branch and write evidence-based failure decomposition",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        })
        return {"status": decision, "selected_method": selected_method, "selected_step": int(selected["step"]),
                "selected_validation": selected_validation, "checks": gate_checks, "output": str(out)}
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# L80 evaluation — INCOMPLETE\n\n" + __import__("traceback").format_exc() +
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
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH; repeat for fixed checkpoints")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
