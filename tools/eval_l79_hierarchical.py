#!/usr/bin/env python3
"""L79 fixed calibration/validation evaluator.

The evaluator has an explicit label-isolation boundary: all 40 L69 candidate
row sets and all checkpoint scores are constructed from key-only metadata first;
calibration labels are attached only for the pre-registered tuple/threshold
selection, and validation labels are attached only after that tuple is frozen.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l79_hierarchical_correspondence import L79Config, L79HierarchicalCorrespondence  # noqa: E402
from locatemot.rmot.l79_data import (  # noqa: E402
    L62_ROWS,
    L79BankStore,
    MANIFEST,
    file_meta,
    key_only_unit,
    load_fixed_key_units,
    load_jsonl,
    sha256_file,
)
from locatemot.rmot.l79_runtime import (  # noqa: E402
    CLIP_SHA256,
    CLIP_WEIGHT,
    load_clip_visual,
    load_lora_state_dict,
    preprocess_full_frame,
    visual_pyramid,
)


SEED = 20260829
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
FORBIDDEN_LABEL_FIELDS = {"target_ids", "positive_indices", "positive_count", "category", "labels", "target_present", "candidate_gt"}
L49_DATA = ROOT / "outputs/l49/data"
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L29_VALIDATION_CONTROL = {
    "recall": 0.7333333333333333,
    "precision": 0.0830188679245283,
    "fp_per_frame": 10.125,
    "predictions_per_positive": 8.833333333333334,
    "hard_violation": 0.9166666666666666,
    "multi_positive_recall": 0.8194444444444443,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def summary(values: Iterable[float]) -> dict[str, Any]:
    values = list(float(x) for x in values)
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p25": None, "p50": None, "p75": None}
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), "p25": float(np.quantile(a, .25)),
            "p50": float(np.quantile(a, .50)), "p75": float(np.quantile(a, .75))}


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def label_source(partition: str) -> Path:
    if partition == "calibration":
        return L49_DATA / "calibration_units.jsonl"
    if partition == "validation":
        return L49_DATA / "validation_units.jsonl"
    raise ValueError(partition)


def authorized_labels(fixed_keys: list[str], orders: Iterable[int], partition: str) -> dict[int, dict[str, Any]]:
    orders = sorted(int(x) for x in orders)
    expected = {int(x) for x in orders}
    if partition == "calibration" and any(x >= 16 for x in expected):
        raise AssertionError("calibration loader was asked for validation order")
    if partition == "validation" and any(x < 16 for x in expected):
        raise AssertionError("validation loader was asked for calibration order")
    wanted = {fixed_keys[x] for x in orders}
    found: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(label_source(partition)):
        key = str(row.get("unit_key"))
        if key in wanted:
            found[key] = row
    if set(found) != wanted:
        raise AssertionError(f"authorized {partition} label mismatch: {sorted(wanted - set(found))}")
    result = {order: found[fixed_keys[order]] for order in orders}
    if set(result) != expected:
        raise AssertionError(f"authorized {partition} order mismatch")
    return result


def immutable_l29_rows() -> list[dict[str, Any]]:
    rows = load_jsonl(L62_ROWS)
    if len(rows) != 40 or len({str(x["unit_key"]) for x in rows}) != 40:
        raise AssertionError("L62 immutable rows are not the 40-key slice")
    return rows


def attach_l69_labels(record: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    """Read sidecar and expression labels only at an explicit post-selection boundary."""
    sidecar_path = Path(record["bank_path"]).with_suffix(".labels.json")
    payload = json.loads(sidecar_path.read_text())
    candidate_gt = payload.get("candidate_gt")
    if not isinstance(candidate_gt, list):
        raise AssertionError(f"missing candidate_gt: {sidecar_path}")
    rows = [int(x) for x in record["row_offsets"]]
    if max(rows, default=-1) >= len(candidate_gt):
        raise AssertionError(f"sidecar shorter than candidate rows: {record['unit_key']}")
    targets = {str(x) for x in unit.get("target_ids", [])}
    labels = [bool(candidate_gt[offset] is not None and str(candidate_gt[offset]) in targets) for offset in rows]
    result = dict(record)
    result.update({
        "labels": [int(x) for x in labels], "target_ids": sorted(targets),
        "positive_indices": [i for i, value in enumerate(labels) if value],
        "positive_count": int(sum(labels)), "target_present": bool(targets),
        "candidate_present": bool(any(labels)), "coverage_mask": not (bool(targets) and not bool(any(labels))),
        "category": "inactive" if not targets else "present_uncovered" if not any(labels) else "multi_positive" if sum(labels) > 1 else "positive",
        "sidecar_candidate_gt": [None if candidate_gt[offset] is None else str(candidate_gt[offset]) for offset in rows],
        "label_source": str(sidecar_path), "sidecar_labels_loaded": True,
    })
    if len(result["labels"]) != int(record["candidate_count"]):
        raise AssertionError(f"label length drift: {record['unit_key']}")
    return result


def build_label_free_record(meta: dict[str, Any], bank: dict[str, Any], store: L79BankStore,
                            model: L79HierarchicalCorrespondence, clip_model: torch.nn.Module,
                            device: torch.device, method: str, lora_enabled: bool) -> dict[str, Any]:
    # Build from key-only metadata.  No sidecar or target/category field is
    # touched here.
    batch = store.build_unit(meta)
    if not Path(batch.image_path).is_file():
        raise FileNotFoundError(batch.image_path)
    image = preprocess_full_frame(batch.image_path, device, clip_model.visual.conv1.weight.dtype)
    with torch.inference_mode():
        pyramid = visual_pyramid(clip_model, image, with_grad=False)
        output = model(batch.observations.to(device), batch.history_observations.to(device), batch.history_mask.to(device),
                       batch.text_tokens.to(device), batch.text_mask.to(device), batch.boxes_norm.to(device), pyramid)
    scores = output["frame_membership_logits"].float().cpu().tolist()
    track_scores = output["track_relevance_logits"].float().cpu().tolist()
    quality_scores = output["observation_quality_logits"].float().cpu().tolist()
    continuation_scores = output["continuation_logits"].float().cpu().tolist()
    null_logit = float(output["null_logit"].float().cpu())
    if len(scores) != batch.candidate_count or not np.isfinite(np.asarray(scores)).all():
        raise AssertionError(f"score length/finite drift: {batch.unit_key}/{method}")
    if not all(np.isfinite(np.asarray(x, dtype=float)).all() for x in (track_scores, quality_scores, continuation_scores, [null_logit])):
        raise AssertionError(f"nonfinite auxiliary scores: {batch.unit_key}/{method}")
    record = {
        "format": "locatemot-l79-score-record-v1", "unit_key": batch.unit_key,
        "dataset": batch.dataset, "video": batch.video, "query_id": batch.query_id, "frame_id": batch.frame_id,
        "fixed_eval_order": int(meta["fixed_eval_order"]), "fixed_eval_split": str(meta["fixed_eval_split"]),
        "sentence": batch.sentence, "bank_path": batch.bank_path, "row_offsets": batch.row_offsets,
        "row_keys": [list(x) for x in batch.row_keys], "candidate_index_provenance": batch.candidate_indices,
        "track_id_provenance": batch.track_ids, "pool_id_provenance": batch.pool_ids,
        "candidate_count": batch.candidate_count, "boxes": batch.boxes.tolist(), "image_path": batch.image_path,
        "image_size": list(batch.image_size), "score_fields": {method: scores},
        "track_score_fields": {method: track_scores}, "quality_score_fields": {method: quality_scores},
        "continuation_score_fields": {method: continuation_scores}, "null_logits": {method: null_logit},
        "history_future_rows": int((batch.history_frame_ids > batch.frame_id).sum()),
        "text_valid_tokens": int(batch.text_mask.sum()), "candidate_rows_retained": batch.candidate_count,
        "candidate_deletion": False, "candidate_truncation": False, "sidecar_labels_loaded": False,
        "finite_scores": True, "source_pool_ids_provenance_only": True,
    }
    del output, pyramid, image, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def merge_method_record(base: dict[str, Any], current: dict[str, Any], method: str) -> None:
    for name in ("unit_key", "row_offsets", "row_keys", "candidate_count", "candidate_index_provenance"):
        if base[name] != current[name]:
            raise AssertionError(f"candidate/key drift while merging {method}: {base['unit_key']} field {name}")
    base["score_fields"].update(current["score_fields"])
    base["track_score_fields"].update(current["track_score_fields"])
    base["quality_score_fields"].update(current["quality_score_fields"])
    base["continuation_score_fields"].update(current["continuation_score_fields"])
    base["null_logits"].update(current["null_logits"])


def metric(records: list[dict[str, Any]], method: str, threshold: float, use_null: bool = False,
           null_threshold: float | None = None, _stratify: bool = True) -> dict[str, Any]:
    if use_null and null_threshold is None:
        raise ValueError("null threshold required")
    tp = fp = fn = selected_count = positive_rows = 0
    top1 = top5 = top1_units = empty = 0
    hard: list[bool] = []; strict: list[float] = []; best: list[float] = []; average: list[float] = []
    multi_recall: list[float] = []; minimum_coverage: list[float] = []
    inactive_units = inactive_accept = inactive_fp_rows = 0
    target_present_units = candidate_present_units = present_uncovered_units = 0
    scores_all: list[float] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    null_values: list[float] = []
    for record in records:
        scores = np.asarray(record["score_fields"][method], dtype=np.float64)
        labels = np.asarray(record["labels"], dtype=bool)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise AssertionError(f"metric score/label contract {record['unit_key']} {method}")
        null_value = float(record["null_logits"][method]); null_values.append(null_value)
        selected = scores >= float(threshold)
        null_rejected = bool(use_null and null_value >= float(null_threshold))
        if null_rejected:
            selected[:] = False
        tp += int((selected & labels).sum()); fp += int((selected & ~labels).sum())
        fn += int((~selected & labels).sum()); selected_count += int(selected.sum()); positive_rows += int(labels.sum())
        scores_all.extend(scores.tolist()); by_category[str(record["category"])].append(record); by_dataset[str(record["dataset"])].append(record)
        category = str(record["category"])
        target_present_units += int(category != "inactive")
        candidate_present_units += int(bool(labels.any()))
        present_uncovered_units += int(category == "present_uncovered")
        if labels.any():
            order = np.argsort(-scores, kind="stable")
            top1 += int(bool(labels[order[:1]].any())); top5 += int(bool(labels[order[:5]].any())); top1_units += 1
        if not selected.any():
            empty += 1
        if category == "inactive":
            inactive_units += 1; inactive_accept += int(bool(selected.any())); inactive_fp_rows += int((selected & ~labels).sum())
        pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
        if len(pos) and len(neg):
            strict_margin = float(scores[pos].min() - scores[neg].max())
            strict.append(strict_margin); best.append(float(scores[pos].max() - scores[neg].max())); average.append(float(scores[pos].mean() - scores[neg].max()))
            hard.append(strict_margin < 0)
        if len(pos) > 1:
            multi_recall.append(float((selected[pos]).sum() / len(pos)))
            minimum_coverage.append(float(bool(selected[pos].all())))
    result = {
        "method": method, "use_null": bool(use_null), "threshold": float(threshold),
        "null_threshold": None if null_threshold is None else float(null_threshold), "units": len(records),
        "target_present_units": target_present_units, "candidate_present_units": candidate_present_units,
        "present_uncovered_units": present_uncovered_units, "candidate_rows": sum(len(x["labels"]) for x in records),
        "positive_rows": positive_rows, "selected_rows": selected_count, "true_positive_rows": tp,
        "false_positive_rows": fp, "false_negative_rows": fn,
        "top1": top1 / max(1, top1_units), "top5": top5 / max(1, top1_units),
        "candidate_precision": tp / max(1, selected_count), "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(records)), "predictions_per_positive": selected_count / max(1, positive_rows),
        "hard_violation": float(np.mean(hard)) if hard else None,
        "strict_margin": summary(strict), "best_margin": summary(best), "average_margin": summary(average),
        "multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "minimum_positive_coverage": float(np.mean(minimum_coverage)) if minimum_coverage else None,
        "empty_rate": empty / max(1, len(records)), "inactive_false_acceptance": inactive_accept / max(1, inactive_units),
        "inactive_false_positive_rows": inactive_fp_rows, "inactive_units": inactive_units,
        "null_false_acceptance": inactive_accept / max(1, inactive_units),
        "null_logit_distribution": summary(null_values), "score_distribution": summary(scores_all),
        "per_category": {}, "per_dataset": {},
    }
    if _stratify:
        for key, values in by_category.items():
            result["per_category"][key] = metric(values, method, threshold, use_null, null_threshold, False)
        for key, values in by_dataset.items():
            result["per_dataset"][key] = metric(values, method, threshold, use_null, null_threshold, False)
    return result


def fit_candidate_threshold(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = np.unique(np.concatenate([np.asarray(x["score_fields"][method], dtype=np.float64) for x in records]))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best_key = None; best = None
    for threshold in candidates:
        tp = fp = fn = 0
        for record in records:
            score = np.asarray(record["score_fields"][method]); label = np.asarray(record["labels"], dtype=bool); selected = score >= threshold
            tp += int((selected & label).sum()); fp += int((selected & ~label).sum()); fn += int((~selected & label).sum())
        f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
        key = (float(f1), -int(fp), float(threshold))
        if best_key is None or key > best_key:
            best_key = key; best = {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "f1": float(f1)}
    return {"threshold": best["threshold"], "objective": "candidate-level observed frame-membership F1 on 16 calibration units", "tie_rule": "higher F1, fewer FP, higher threshold", "validation_used": False, "counts": best}


def fit_null_threshold(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    # Present-uncovered is excluded: it has no candidate-positive row and is
    # not an inactive target.  The rule is fitted only on calibration rows.
    usable = [x for x in records if str(x["category"]) in {"inactive", "positive", "multi_positive"}]
    values = np.unique(np.asarray([float(x["null_logits"][method]) for x in usable], dtype=np.float64))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best_key = None; best = None
    for threshold in candidates:
        tp = fp = fn = 0
        for record in usable:
            target = str(record["category"]) == "inactive"; predicted = float(record["null_logits"][method]) >= threshold
            tp += int(predicted and target); fp += int(predicted and not target); fn += int((not predicted) and target)
        f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
        key = (float(f1), -int(fp), float(threshold))
        if best_key is None or key > best_key:
            best_key = key; best = {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "f1": float(f1)}
    return {"threshold": best["threshold"], "objective": "inactive-vs-candidate NULL F1 on calibration; present-uncovered excluded", "tie_rule": "higher F1, fewer false positives, higher threshold", "validation_used": False, "counts": best, "usable_calibration_units": len(usable)}


def adapter_norm(package: dict[str, Any]) -> float:
    total = 0.0
    for value in package.get("model_state_dict", {}).values():
        if torch.is_tensor(value):
            total += float(value.float().pow(2).sum())
    for value in package.get("lora_state_dict", {}).values():
        if torch.is_tensor(value):
            total += float(value.float().pow(2).sum())
    return math.sqrt(total)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-epoch01", required=True, type=Path)
    parser.add_argument("--checkpoint-epoch03", required=True, type=Path)
    parser.add_argument("--checkpoint-epoch05", required=True, type=Path)
    parser.add_argument("--checkpoint-epoch10", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty evaluator output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    checkpoint_paths = {
        "epoch01": args.checkpoint_epoch01.resolve(), "epoch03": args.checkpoint_epoch03.resolve(),
        "epoch05": args.checkpoint_epoch05.resolve(), "epoch10": args.checkpoint_epoch10.resolve(),
    }
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA changed")
        if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
            raise AssertionError("CLIP SHA changed")
        torch.manual_seed(SEED); np.random.seed(SEED)
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        metadata = []
        for index, source_meta in enumerate(load_fixed_key_units()):
            normalized_meta = dict(source_meta)
            normalized_meta["fixed_eval_order"] = int(index)
            normalized_meta["fixed_eval_split"] = str(source_meta.get("evaluation_partition"))
            metadata.append(normalized_meta)
        fixed_keys = [str(x["unit_key"]) for x in metadata]
        if len(metadata) != 40 or [int(x["fixed_eval_order"]) for x in metadata] != list(range(40)):
            raise AssertionError("fixed 40 unit order drift")
        # All model scores are built before any sidecar/target/category fields
        # are attached.  The base records carry only key/text/frame metadata.
        preselection: list[dict[str, Any]] = []
        for meta in metadata:
            key_meta = key_only_unit(meta, str(meta["fixed_eval_split"]))
            if FORBIDDEN_LABEL_FIELDS.intersection(key_meta):
                raise AssertionError(f"label field leaked before selection: {sorted(FORBIDDEN_LABEL_FIELDS.intersection(key_meta))}")
            preselection.append({
                "format": "locatemot-l79-preselection-record-v1", "unit_key": key_meta["unit_key"],
                "dataset": key_meta["dataset"], "video": key_meta["video"], "query_id": key_meta["query_id"],
                "frame_id": key_meta["frame_id"], "sentence": key_meta["sentence"], "fixed_eval_order": int(meta["fixed_eval_order"]),
                "fixed_eval_split": key_meta["evaluation_partition"], "sidecar_labels_loaded": False,
                "candidate_deletion": False, "candidate_truncation": False,
                "score_fields": {}, "track_score_fields": {}, "quality_score_fields": {},
                "continuation_score_fields": {}, "null_logits": {},
            })
        methods: dict[str, dict[str, Any]] = {}
        adapter_norms: dict[str, float] = {}
        checkpoint_hashes: dict[str, str] = {}
        for method, path in checkpoint_paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            package = safe_torch_load(path, "cpu")
            checkpoint_hashes[method] = sha256_file(path); adapter_norms[method] = adapter_norm(package)
            config = L79Config(**package["model_config"])
            model = L79HierarchicalCorrespondence(config).to(device=device, dtype=torch.float32)
            result = model.load_state_dict(package["model_state_dict"], strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise AssertionError(f"{method} model reload mismatch: {result}")
            lora_enabled = bool(package.get("lora_enabled", False))
            clip_model = load_clip_visual(device, enable_lora=lora_enabled)
            if set(package.get("lora_state_dict", {})):
                load_lora_state_dict(clip_model, package["lora_state_dict"])
            model.eval(); clip_model.eval()
            store = L79BankStore(max_history=16)
            for index, meta in enumerate(metadata):
                raw = build_label_free_record(meta, {}, store, model, clip_model, device, method, lora_enabled)
                if not preselection[index].get("candidate_count"):
                    preselection[index].update({key: raw[key] for key in ("bank_path", "row_offsets", "row_keys", "candidate_count", "candidate_index_provenance", "track_id_provenance", "pool_id_provenance", "boxes", "image_path", "image_size", "history_future_rows", "text_valid_tokens", "candidate_rows_retained", "finite_scores")})
                merge_method_record(preselection[index], raw, method)
            del store, clip_model, model, package
            gc.collect(); torch.cuda.empty_cache()
        forbidden_present = sorted({field for row in preselection for field in FORBIDDEN_LABEL_FIELDS if field in row})
        if forbidden_present:
            raise AssertionError(f"preselection forbidden fields found: {forbidden_present}")
        if any(len(row["row_keys"]) != int(row["candidate_count"]) for row in preselection):
            raise AssertionError("preselection candidate/key count drift")
        preselection_audit = {
            "format": "locatemot-l79-preselection-label-isolation-v1", "status": "complete",
            "fixed_order_count": len(preselection), "calibration_count": 16, "validation_count": 24,
            "fixed_order": [row["unit_key"] for row in preselection],
            "preselection_forbidden_fields_absent": not forbidden_present, "forbidden_fields_found": forbidden_present,
            "candidate_rows_and_all_method_scores_complete": all(
                len(row["row_keys"]) == int(row["candidate_count"]) and all(len(x) == int(row["candidate_count"]) for x in row["score_fields"].values()) for row in preselection),
            "finite_all_scores": all(bool(row["finite_scores"]) for row in preselection), "candidate_deletion": False,
            "candidate_truncation": False, "sidecar_labels_loaded": False, "validation_labels_read": False,
            "selection_frozen": False, "event": "all four checkpoint score fields built before authorized calibration label load",
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        write_json(out / "preselection_label_isolation.json", preselection_audit)

        # Calibration is the first authorized label boundary.
        cal_label_rows = authorized_labels(fixed_keys, range(16), "calibration")
        cal_records = []
        for order in range(16):
            cal_records.append(attach_l69_labels(preselection[order], cal_label_rows[order]))
        thresholds: dict[str, Any] = {}; null_rules: dict[str, Any] = {}; calibration_metrics: dict[str, Any] = {}
        selection_candidates = []
        for method in checkpoint_paths:
            thresholds[method] = fit_candidate_threshold(cal_records, method)
            null_rules[method] = fit_null_threshold(cal_records, method)
            cm = metric(cal_records, method, thresholds[method]["threshold"], False)
            nm = metric(cal_records, method, thresholds[method]["threshold"], True, null_rules[method]["threshold"])
            calibration_metrics[method] = {"candidate_only": cm, "candidate_plus_null": nm}
            selection_candidates.append({
                "method": method, "step": int(safe_torch_load(checkpoint_paths[method], "cpu").get("step", 0)),
                "candidate_threshold": thresholds[method], "null_rule": null_rules[method],
                "candidate_only_metrics": cm, "candidate_plus_null_metrics": nm,
                "adapter_plus_marker_norm": adapter_norms[method],
                "lexicographic_key": [
                    float(cm["hard_violation"] if cm["hard_violation"] is not None else 1.0),
                    -float(cm["minimum_positive_coverage"] if cm["minimum_positive_coverage"] is not None else 0.0),
                    float(cm["inactive_false_acceptance"]), float(cm["false_positive_rows"]),
                    int(safe_torch_load(checkpoint_paths[method], "cpu").get("step", 0)), adapter_norms[method],
                ],
            })
        selected = min(selection_candidates, key=lambda x: tuple(x["lexicographic_key"]))
        selection = {
            "format": "locatemot-l79-calibration-selection-v1", "status": "frozen_before_validation",
            "objective": "lower calibration hard violation; higher minimum-positive coverage; lower inactive false acceptance; lower false-positive rows; earlier step; smaller adapter+marker norm",
            "tie_rule": "lexicographic tuple exactly as listed; no validation access",
            "candidates": selection_candidates, "selected_method": selected["method"],
            "selected_candidate_threshold": selected["candidate_threshold"], "selected_null_rule": selected["null_rule"],
            "calibration_labels_attached": True, "validation_labels_read": False, "selection_frozen": True,
        }
        write_json(out / "checkpoint_selection.json", selection)
        preselection_audit["selection_frozen"] = True; preselection_audit["calibration_labels_attached"] = True
        write_json(out / "preselection_label_isolation.json", preselection_audit)

        # Only now attach validation labels.  No validation value can influence
        # thresholds, checkpoint choice or branch choice above.
        val_label_rows = authorized_labels(fixed_keys, range(16, 40), "validation")
        labeled_records = []
        for order in range(40):
            if order < 16:
                labeled_records.append(cal_records[order])
            else:
                labeled_records.append(attach_l69_labels(preselection[order], val_label_rows[order]))
        validation_records = labeled_records[16:]
        results: dict[str, Any] = {}
        for method in checkpoint_paths:
            threshold = thresholds[method]["threshold"]; null_threshold = null_rules[method]["threshold"]
            results[method] = {
                "calibration": calibration_metrics[method],
                "validation_candidate_only": metric(validation_records, method, threshold, False),
                "validation_candidate_plus_null": metric(validation_records, method, threshold, True, null_threshold),
            }
        selected_method = selected["method"]
        selected_validation = results[selected_method]["validation_candidate_plus_null"]
        checks = {
            "hard_violation_improvement": float(L29_VALIDATION_CONTROL["hard_violation"] - (selected_validation["hard_violation"] if selected_validation["hard_violation"] is not None else 1.0)),
            "hard_violation_pass": selected_validation["hard_violation"] is not None and selected_validation["hard_violation"] <= L29_VALIDATION_CONTROL["hard_violation"] - 0.05,
            "recall_pass": selected_validation["candidate_recall"] >= 0.7233333,
            "precision_pass": selected_validation["candidate_precision"] >= 0.0830188679,
            "fp_per_frame_pass": selected_validation["fp_per_frame"] <= 11.125,
            "predictions_per_positive_pass": selected_validation["predictions_per_positive"] <= 4.069,
            "multi_positive_pass": selected_validation["multi_positive_recall"] is not None and selected_validation["multi_positive_recall"] >= 0.7894444,
            "null_false_acceptance_pass": selected_validation["inactive_false_acceptance"] < 1.0,
            "complete_finite_keys_pass": all(len(x["row_keys"]) == x["candidate_count"] and x["finite_scores"] and not x["candidate_deletion"] and not x["candidate_truncation"] for x in labeled_records),
        }
        decision = "semantic_gate_pass" if all(bool(x) for x in checks.values()) else "semantic_gate_fail"
        semantic = {
            "format": "locatemot-l79-semantic-v1", "status": decision, "stage": "fixed-16-calibration-24-validation",
            "selected_method": selected_method, "selection_frozen_before_validation": True,
            "thresholds": thresholds, "null_rules": null_rules, "calibration_metrics": calibration_metrics,
            "validation_metrics": results, "l29_immutable_validation_control": L29_VALIDATION_CONTROL,
            "gate_checks": checks, "candidate_set_note": "L69 budget-40 rows; not paired row-for-row with old L29 rows",
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "text_span_alignment": "UNALIGNED", "l79_frame_probe_not_final_persistent_sequence_decoder": True,
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }
        write_json(out / "semantic.json", semantic)
        gate = {
            "format": "locatemot-l79-gate-decision-v1", "status": decision, "selected_method": selected_method,
            "gate_checks": checks, "calibration_only_selection": True, "validation_used_for_selection": False,
            "failure_root_cause": None if decision == "semantic_gate_pass" else "to_be_decomposed_from_validation_correspondence_and_volume_metrics",
            "next_action": "freeze method and request separate screening authorization" if decision == "semantic_gate_pass" else "stop L79 before screening; write one evidence-based failure decomposition",
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }
        write_json(out / "gate_decision.json", gate)
        with (out / "score_records.jsonl").open("w") as handle:
            for record in labeled_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        provenance = {
            "format": "locatemot-l79-evaluation-provenance-v1", "status": "complete", "project_root": str(ROOT),
            "cwd": str(Path.cwd().resolve()), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "command": command,
            "python": sys.executable, "python_version": sys.version, "torch_version": torch.__version__, "cuda": torch.version.cuda,
            "device": str(device), "visible_cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "seed": SEED,
            "inputs": {"manifest": file_meta(MANIFEST), "l69_feature_root": str(L69_ROOT), "l49_calibration": file_meta(L49_DATA / "calibration_units.jsonl"),
                       "l49_validation": file_meta(L49_DATA / "validation_units.jsonl"), "l62_fixed_rows": file_meta(L62_ROWS),
                       "clip_weight": {"path": str(CLIP_WEIGHT), "sha256": sha256_file(CLIP_WEIGHT), "expected": CLIP_SHA256}},
            "checkpoints": {name: {"path": str(path), "sha256": checkpoint_hashes[name], "adapter_norm": adapter_norms[name]} for name, path in checkpoint_paths.items()},
            "fixed_order": fixed_keys, "calibration_orders": list(range(16)), "validation_orders": list(range(16, 40)),
            "preselection_label_isolation": str(out / "preselection_label_isolation.json"),
            "label_boundary": "all 40 score arrays built first; calibration labels then selection; validation labels only after frozen tuple",
            "candidate_keys": "L69 native frame_ptr/row offsets; duplicate candidate_index retained",
            "same_class_hard_negative_metadata": "unavailable; all-negative fallback in training",
            "token_span_alignment": "UNALIGNED", "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(out / "provenance.json", provenance)
        write_json(out / "config.json", {"format": "locatemot-l79-eval-config-v1", "fixed_units": 40, "calibration_units": 16, "validation_units": 24,
                                          "candidate_threshold_rule": thresholds[selected_method], "null_rule": null_rules[selected_method],
                                          "checkpoint_selection_rule": selection["objective"], "score_field": "frame_membership_logits",
                                          "candidate_set": "all L69 rows; no top-k/NMS/deletion/truncation", "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
        write_json(out / "status.json", {"format": "locatemot-l79-evaluation-status-v1", "status": decision, "stage": "P4-fixed-semantic-gate",
                                          "command": command, "failure_root_cause": None if decision == "semantic_gate_pass" else "see gate_decision.json and reports/l79_failure_decomposition.md",
                                          "next_action": gate["next_action"], "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        print(json.dumps({"status": decision, "selected_method": selected_method, "selected_validation": selected_validation,
                          "checks": checks, "out": str(out)}, indent=2), flush=True)
        return 0
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        write_json(out / "status.json", {"format": "locatemot-l79-evaluation-status-v1", "status": "incomplete", "command": command,
                                          "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": tb,
                                          "next_action": "preserve this evaluator attempt; fix only the first actionable error and rerun in a new output directory",
                                          "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        (out / "INCOMPLETE.md").write_text("# L79 evaluator incomplete\n\n```text\n" + tb + "```\n")
        print(tb, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
