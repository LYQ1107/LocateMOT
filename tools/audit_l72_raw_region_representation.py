#!/usr/bin/env python3
"""L72-A1: streaming LocateAnything raw image--expression representation audit.

The detector is frozen and loaded only in the verified local environment.  A
unit is processed as one original frame and one original expression.  The
script reconstructs L69 rows from the L69 bank's own frame pointers, keeps all
rows (including duplicate candidate indices), and writes only compact vector
summaries and scores.  Raw tensors are released after each unit; no feature
cache is written.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_ROOT = ROOT / "outputs/l49/data"
L62_RECORDS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
MODEL_DIR = ROOT / "models/LocateAnything-3B"
SEED = 20260829
MERGE_KERNEL = (2, 2)
PATCH_SIZE = 14
NATIVE_IOU_THRESHOLD = 0.30
UNRELATED_SENTENCE = "a completely unrelated purple elephant"

sys.path.insert(0, str(ROOT))
from locatemot.models.object_tokens.generation_trace import (  # noqa: E402
    InstrumentedLocateAnythingGeneration,
)
from locatemot.models.l72_raw_region import (  # noqa: E402
    image_token_positions,
    map_box_to_token_indices,
    masked_mean,
    pooled_vector,
    vector_summary,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def short_hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-12, aa + bb - inter)


def normalize_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def unit_key(row: dict[str, Any]) -> str:
    return str(row.get("unit_key") or "{}|{}|{}|{}".format(
        row["dataset"], row["video"], int(row["query_id"]), int(row["frame_id"])
    ))


def fixed_units() -> list[dict[str, Any]]:
    """Join the immutable L62 order to the L49 calibration/validation rows."""
    order = read_jsonl(L62_RECORDS)
    if len(order) != 40 or len({unit_key(row) for row in order}) != 40:
        raise AssertionError("immutable L62 order is not exactly 40 unique units")
    calibration = {unit_key(row): row for row in read_jsonl(L49_ROOT / "calibration_units.jsonl")}
    validation = {unit_key(row): row for row in read_jsonl(L49_ROOT / "validation_units.jsonl")}
    result: list[dict[str, Any]] = []
    for index, old in enumerate(order):
        key = unit_key(old)
        source = calibration if index < 16 else validation
        if key not in source:
            raise KeyError(f"fixed L62 key missing from L49 split: {key}")
        row = dict(source[key])
        if unit_key(row) != key:
            raise AssertionError(f"fixed key mismatch: {key}")
        row["fixed_eval_order"] = index
        row["fixed_eval_split"] = "calibration" if index < 16 else "validation"
        result.append(row)
    return result


class L72Bank:
    """Read-only L69 feature-bank index for one video."""

    required = {
        "box", "frame", "frame_ids", "frame_ptr", "candidate_index", "track_id",
        "pool_id", "raw_rank", "clip", "history_clip", "uidm_h", "geometry",
        "motion", "lifecycle", "objectness",
    }

    def __init__(self, video: str):
        self.video = str(video)
        self.path = L69_ROOT / f"{self.video}.pt"
        self.label_path = self.path.with_suffix(".labels.json")
        if not self.path.exists() or not self.label_path.exists():
            raise FileNotFoundError(f"missing L69 bank/sidecar for {self.video}")
        import torch

        self.blob = torch.load(self.path, map_location="cpu", weights_only=False)
        self.tensors = self.blob["tensors"]
        missing = self.required.difference(self.tensors)
        if missing:
            raise KeyError(f"{self.path}: missing {sorted(missing)}")
        self.labels = json.loads(self.label_path.read_text())["candidate_gt"]
        self.count = int(self.tensors["track_id"].numel())
        if len(self.labels) != self.count:
            raise AssertionError(f"{self.path}: sidecar/count mismatch")
        frame_ids = self.tensors["frame_ids"].long().tolist()
        frame_ptr = self.tensors["frame_ptr"].long().tolist()
        if len(frame_ptr) != len(frame_ids) + 1 or int(frame_ptr[-1]) != self.count:
            raise AssertionError(f"{self.path}: frame pointer contract failed")
        self.frame_ranges: dict[int, tuple[int, int, int]] = {}
        for frame_index, frame_id in enumerate(frame_ids):
            begin, end = int(frame_ptr[frame_index]), int(frame_ptr[frame_index + 1])
            frame_values = self.tensors["frame"].long()[begin:end]
            if not bool((frame_values == int(frame_id)).all()):
                raise AssertionError(f"{self.path}: frame row mismatch at {frame_id}")
            self.frame_ranges[int(frame_id)] = (begin, end, frame_index)
        self.track_rows: dict[int, list[int]] = defaultdict(list)
        for row, track in enumerate(self.tensors["track_id"].long().tolist()):
            self.track_rows[int(track)].append(int(row))
        finite_fields = (
            "box", "clip", "history_clip", "uidm_h", "geometry", "motion",
            "lifecycle", "objectness",
        )
        for name in finite_fields:
            if not bool(torch.isfinite(self.tensors[name].float()).all()):
                raise AssertionError(f"{self.path}: nonfinite {name}")
        self.sha256 = sha256_file(self.path)

    def rows_for(self, frame_id: int) -> list[int]:
        if int(frame_id) not in self.frame_ranges:
            raise KeyError(f"{self.video}: missing frame {frame_id}")
        begin, end, _ = self.frame_ranges[int(frame_id)]
        return list(range(begin, end))

    def history_check(self, row: int, frame_id: int, length: int = 8) -> dict[str, Any]:
        import torch

        track = int(self.tensors["track_id"][row])
        frame_values = self.tensors["frame"].long()
        eligible = [old for old in self.track_rows.get(track, []) if int(frame_values[old]) <= int(frame_id)]
        eligible = sorted(eligible, key=lambda old: (int(frame_values[old]), old))[-int(length):]
        future = sum(int(frame_values[old]) > int(frame_id) for old in eligible)
        return {"history_count": len(eligible), "future_rows": future,
                "history_frame_ids": [int(frame_values[old]) for old in eligible]}

    def close(self) -> None:
        self.blob = None
        self.tensors = {}
        self.labels = []
        self.frame_ranges = {}
        self.track_rows = {}


def build_inputs(processor, image, sentence: str) -> dict[str, Any]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": sentence},
        ],
    }]
    prompt = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    return processor(text=[prompt], images=images, videos=videos, return_tensors="pt")


def find_subsequence(values: list[int], query: list[int]) -> list[int]:
    if not query:
        return []
    for start in range(0, len(values) - len(query) + 1):
        if values[start:start + len(query)] == query:
            return list(range(start, start + len(query)))
    return []


def text_positions(tokenizer, sentence: str, input_ids: Any, attention_mask: Any,
                   image_positions: list[int]) -> tuple[list[int], str]:
    full = input_ids.detach().cpu().reshape(-1).tolist()
    valid = attention_mask.detach().cpu().reshape(-1).bool().tolist()
    candidates: list[tuple[str, list[int]]] = []
    for label, text in (("exact", sentence), ("leading_space", " " + sentence)):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        candidates.append((label, find_subsequence(full, [int(v) for v in encoded])))
    for label, positions in candidates:
        if positions and all(valid[pos] and pos not in set(image_positions) for pos in positions):
            return positions, label
    image_set = set(image_positions)
    fallback = [index for index, flag in enumerate(valid) if flag and index not in image_set]
    return fallback, "whole_text_minus_image_unresolved"


def compact_event(event: Any) -> dict[str, Any]:
    normalized = getattr(event, "normalized_box", None)
    return {
        "block_type": str(getattr(event, "block_type", "")),
        "accepted": bool(getattr(event, "accepted", False)),
        "normalized_box": [float(v) for v in normalized] if normalized is not None else None,
        "generation_score": float(getattr(event, "generation_score", 0.0) or 0.0),
        "output_order": int(getattr(event, "output_order", -1)),
    }


def capture_once(model, runner, processor, tokenizer, image, sentence: str,
                 boxes: list[list[float]], original_size: tuple[int, int],
                 retain_vectors: bool = False) -> dict[str, Any]:
    """Capture one expression, returning summaries and optional in-memory vectors."""
    import torch
    import torch.nn.functional as F

    inputs = build_inputs(processor, image, sentence)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_positions = image_token_positions(input_ids, int(model.config.image_token_index))
    grid_hws = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
    if grid_hws.ndim != 2 or tuple(grid_hws.shape) != (1, 2):
        raise AssertionError(f"unexpected image grid shape {tuple(grid_hws.shape)}")

    with torch.inference_mode():
        trace = runner.run(
            image,
            sentence,
            generation_mode="fast",
            max_new_tokens=6,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            prebuilt_inputs=inputs,
            keep_layers=(-1,),
        )
    raw = trace["raw_vision_features"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 1:
        raise AssertionError(f"expected one raw visual feature list, got {type(raw).__name__}")
    raw_cpu = raw[0].detach().float().cpu()
    if raw_cpu.ndim != 2 or not bool(torch.isfinite(raw_cpu).all()):
        raise AssertionError(f"raw visual feature shape/finite failure: {tuple(raw_cpu.shape)}")
    if len(image_positions) != int(raw_cpu.shape[0]):
        raise AssertionError(
            f"image token/raw feature count mismatch: {len(image_positions)} vs {raw_cpu.shape[0]}"
        )
    first_hidden = trace["hidden_slices"][0] if trace.get("hidden_slices") else {}
    if not first_hidden:
        raise AssertionError("instrumented generation returned no first-step language hidden slice")
    hidden_key = max(first_hidden)
    hidden_cpu = first_hidden[hidden_key]
    if hidden_cpu.ndim != 3:
        raise AssertionError(f"unexpected hidden slice shape {tuple(hidden_cpu.shape)}")
    hidden_cpu = hidden_cpu[0].float().cpu()
    if not bool(torch.isfinite(hidden_cpu).all()):
        raise AssertionError("unmasked first-step language hidden contains nonfinite values")
    if max(image_positions) >= hidden_cpu.shape[0]:
        raise AssertionError("image position exceeds first-step hidden sequence")
    q_positions, q_position_method = text_positions(
        tokenizer, sentence, input_ids, attention_mask, image_positions
    )
    query_vector = masked_mean(hidden_cpu, q_positions)
    if query_vector is None:
        raise AssertionError("no valid expression/text positions for query vector")
    image_hidden = hidden_cpu.index_select(
        0, torch.as_tensor(image_positions, dtype=torch.long)
    )
    processed_image = processor.image_processor.rescale(image, MERGE_KERNEL)
    processed_size = (int(processed_image.width), int(processed_image.height))
    region_rows: list[dict[str, Any]] = []
    raw_vectors: list[torch.Tensor | None] = []
    context_vectors: list[torch.Tensor | None] = []
    native_events = [compact_event(event) for event in trace.get("events", [])]
    native_boxes = [
        [float(v) for v in event["normalized_box"]]
        for event in native_events if event.get("accepted") and event.get("normalized_box") is not None
    ]
    native_scores = [float(event["generation_score"]) for event in native_events
                     if event.get("accepted") and event.get("normalized_box") is not None]
    native_pixel_boxes = [
        [box[0] * original_size[0], box[1] * original_size[1],
         box[2] * original_size[0], box[3] * original_size[1]]
        for box in native_boxes
    ]
    grid_hw = [int(v) for v in grid_hws[0].tolist()]
    for box in boxes:
        mapping = map_box_to_token_indices(
            box, original_size, processed_size, grid_hw,
            patch_size=PATCH_SIZE, merge_kernel=MERGE_KERNEL,
        )
        indices = [int(v) for v in mapping["indices"]]
        raw_region = pooled_vector(raw_cpu, indices)
        context_region = pooled_vector(image_hidden, indices)
        if raw_region is None or context_region is None:
            score = None
        else:
            score = float(F.cosine_similarity(
                context_region.reshape(1, -1), query_vector.reshape(1, -1), dim=1
            )[0])
            if not math.isfinite(score):
                raise AssertionError("nonfinite query/context cosine")
        matched_native = []
        for native_box, native_score in zip(native_pixel_boxes, native_scores):
            matched_native.append((iou_xyxy(box, native_box), native_score))
        native_score = max((score_value for overlap, score_value in matched_native
                            if overlap >= NATIVE_IOU_THRESHOLD), default=-20.0)
        region_rows.append({
            "box": [float(v) for v in box],
            "token_indices": indices,
            "token_count": len(indices),
            "region_available": bool(raw_region is not None and context_region is not None),
            "mapping": mapping,
            "raw_region_summary": vector_summary(raw_region),
            "context_region_summary": vector_summary(context_region),
            "score": score,
            "native_score": float(native_score),
            "native_matched": bool(native_score > -20.0),
        })
        raw_vectors.append(raw_region)
        context_vectors.append(context_region)
    answer = trace.get("answer", "")
    result = {
        "prompt_sha256": short_hash_text(
            processor.py_apply_chat_template([{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sentence},
                ],
            }], tokenize=False, add_generation_prompt=True)
        ),
        "sentence_sha256": short_hash_text(sentence),
        "input_ids_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "image_token_positions": image_positions,
        "image_token_count": len(image_positions),
        "image_grid_hws": grid_hw,
        "raw_visual_shape": list(raw_cpu.shape),
        "raw_visual_finite": bool(torch.isfinite(raw_cpu).all()),
        "first_hidden_key": int(hidden_key),
        "first_hidden_shape": [1, int(hidden_cpu.shape[0]), int(hidden_cpu.shape[1])],
        "first_hidden_finite": bool(torch.isfinite(hidden_cpu).all()),
        "query_position_method": q_position_method,
        "query_positions": [int(v) for v in q_positions],
        "query_vector_summary": vector_summary(query_vector),
        "processed_image_size": list(processed_size),
        "original_image_size": list(original_size),
        "native_events": native_events,
        "native_answer_sha256": short_hash_text(str(answer)),
        "native_proposal_count": len(native_pixel_boxes),
        "candidate_rows": region_rows,
        "_raw_vectors": raw_vectors if retain_vectors else None,
        "_context_vectors": context_vectors if retain_vectors else None,
        "_query_vector": query_vector if retain_vectors else None,
    }
    del trace, inputs, raw, raw_cpu, hidden_cpu, image_hidden, query_vector
    return result


def representation_metrics(records: list[dict[str, Any]], score_field: str,
                            threshold: float | None = None) -> dict[str, Any]:
    values: list[float] = []
    tp = fp = fn = selected = positives = 0
    top1 = top5 = target_present = empty = 0
    inactive_accept = inactive_units = inactive_fp = 0
    unscored_rows = unscored_positive_rows = 0
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    violations: list[bool] = []
    multi_recall: list[float] = []
    per_query_recall: list[float] = []
    target_candidate_present = 0
    target_units = 0
    present_uncovered = 0
    for record in records:
        rows = record["candidate_rows"]
        score_values = [row[score_field] for row in rows]
        finite = np.asarray([
            value is not None and math.isfinite(float(value)) for value in score_values
        ], dtype=bool)
        # Missing token coverage remains an explicit unscored row.  It is not
        # replaced by a zero/sentinel in the artifact; -inf is used only in
        # this local metric computation so it cannot be selected or win a rank.
        score = np.asarray([
            float(value) if valid else -np.inf
            for value, valid in zip(score_values, finite)
        ], dtype=np.float64)
        label = np.asarray([bool(row["label"]) for row in rows], dtype=bool)
        if score.size != label.size:
            raise AssertionError(f"length score record: {record['unit_key']}")
        unscored_rows += int((~finite).sum())
        unscored_positive_rows += int((~finite & label).sum())
        values.extend(score[finite].tolist())
        category = str(record["category"])
        has_target = bool(record["target_ids"])
        if has_target:
            target_units += 1
        if has_target and label.any():
            target_candidate_present += 1
        if category == "present_uncovered":
            present_uncovered += 1
        if threshold is None:
            selected_mask = finite.copy()
        else:
            selected_mask = finite & (score >= float(threshold))
        row_tp = int((selected_mask & label).sum())
        row_fp = int((selected_mask & ~label).sum())
        row_fn = int((~selected_mask & label).sum())
        tp += row_tp
        fp += row_fp
        fn += row_fn
        selected += int(selected_mask.sum())
        positives += int(label.sum())
        empty += int(not selected_mask.any())
        if category == "inactive":
            inactive_units += 1
            inactive_accept += int(selected_mask.any())
            inactive_fp += row_fp
        if has_target:
            target_present += 1
            if label.any():
                order = np.argsort(-score, kind="stable")
                top1 += int(bool(label[order[:1]].any()))
                top5 += int(bool(label[order[:5]].any()))
        if label.any():
            per_query_recall.append(row_tp / float(label.sum()))
        pos = np.flatnonzero(label)
        neg = np.flatnonzero(~label)
        scored_pos = pos[finite[pos]] if pos.size else pos
        scored_neg = neg[finite[neg]] if neg.size else neg
        if scored_pos.size and scored_neg.size:
            gap = float(score[scored_pos].min() - score[scored_neg].max())
            strict.append(gap)
            best.append(float(score[scored_pos].max() - score[scored_neg].max()))
            average.append(float(score[scored_pos].mean() - score[scored_neg].max()))
            violations.append(gap < 0)
        if pos.size > 1:
            multi_recall.append(float((selected_mask & label).sum() / pos.size))
    return {
        "units": len(records),
        "candidate_rows": int(sum(len(record["candidate_rows"]) for record in records)),
        "positive_rows": positives,
        "target_present_units": target_units,
        "candidate_present_units": target_candidate_present,
        "present_uncovered_units": present_uncovered,
        "scored_candidate_rows": int(len(values)),
        "unscored_candidate_rows": int(unscored_rows),
        "unscored_positive_rows": int(unscored_positive_rows),
        "top1": top1 / max(1, target_present),
        "top5": top5 / max(1, target_present),
        "candidate_recall": tp / max(1, tp + fn),
        "candidate_precision": tp / max(1, selected),
        "fp_per_frame": fp / max(1, len(records)),
        "predictions_per_positive": selected / max(1, positives),
        "hard_violation": float(np.mean(violations)) if violations else None,
        "strict_margin": {
            "count": len(strict), "mean": float(np.mean(strict)) if strict else None,
            "min": float(np.min(strict)) if strict else None,
        },
        "best_margin": {
            "count": len(best), "mean": float(np.mean(best)) if best else None,
            "min": float(np.min(best)) if best else None,
        },
        "average_margin": {
            "count": len(average), "mean": float(np.mean(average)) if average else None,
            "min": float(np.min(average)) if average else None,
        },
        "multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "query_recall": {
            "count": len(per_query_recall),
            "mean": float(np.mean(per_query_recall)) if per_query_recall else None,
            "p10": float(np.quantile(per_query_recall, 0.10)) if per_query_recall else None,
            "p50": float(np.quantile(per_query_recall, 0.50)) if per_query_recall else None,
            "p90": float(np.quantile(per_query_recall, 0.90)) if per_query_recall else None,
        },
        "empty_rate": empty / max(1, len(records)),
        "inactive_false_acceptance": inactive_accept / max(1, inactive_units),
        "inactive_false_positive_rows": inactive_fp,
        "score_mean": float(np.mean(values)) if values else None,
        "score_std": float(np.std(values)) if values else None,
        "threshold": float(threshold) if threshold is not None else None,
    }


def fit_threshold(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = np.unique(np.asarray([
        row[field] for record in records for row in record["candidate_rows"]
        if row[field] is not None and math.isfinite(float(row[field]))
    ], dtype=np.float64))
    if values.size == 0 or not np.isfinite(values).all():
        raise AssertionError("cannot fit threshold from nonfinite/empty calibration scores")
    candidates = values.tolist() + [float(values.min()) - 1e-7, float(values.max()) + 1e-7]
    best: tuple[tuple[float, int, float], float] | None = None
    for threshold in candidates:
        metric = representation_metrics(records, field, float(threshold))
        selected = sum(
            int(row[field] is not None and float(row[field]) >= threshold)
            for record in records for row in record["candidate_rows"]
        )
        tp = metric["candidate_recall"] * max(1, metric["positive_rows"])
        fp = selected - tp
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + metric["positive_rows"] - tp)
        key = (float(f1), -int(round(fp)), float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return {
        "threshold": best[1],
        "objective": "candidate-level F1 on this fixed calibration slice",
        "tie_rule": "higher F1, fewer FP, then higher threshold",
        "validation_used": False,
    }


def per_dataset_metrics(records: list[dict[str, Any]], field: str,
                        threshold: float | None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["dataset"])].append(record)
        groups[str(record["category"])].append(record)
    return {name: representation_metrics(group, field, threshold)
            for name, group in sorted(groups.items())}


def model_file_manifest() -> dict[str, Any]:
    prior = ROOT / "outputs/l72/audit/api_smoke_attempt2/api_contract.json"
    if prior.exists():
        try:
            return {
                "source": str(prior),
                "sha256": sha256_file(prior),
                "manifest": json.loads(prior.read_text()).get("model_manifest", {}),
            }
        except Exception:
            pass
    return {"source": "not_prehashed_in_this_attempt"}


def base_payload(args: argparse.Namespace, out: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": "locatemot-l72-raw-region-representation-v1",
        "status": "running",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "seed": SEED,
        "split": args.split,
        "unit_count": len(units),
        "fixed_eval_order": [unit_key(row) for row in units],
        "inputs": {
            "l69_feature_root": str(L69_ROOT),
            "l49_fit_units": str(L49_ROOT / "train_units.jsonl"),
            "l49_calibration_units": str(L49_ROOT / "calibration_units.jsonl"),
            "l49_validation_units": str(L49_ROOT / "validation_units.jsonl"),
            "l62_fixed_order": str(L62_RECORDS),
            "manifest": str(MANIFEST),
            "image_root": str(IMAGE_ROOT),
            "model_dir": str(MODEL_DIR),
        },
        "outputs": {
            "representation": str(out / (
                "representation_calibration.json" if args.split == "calibration"
                else "representation_fixed_eval.json"
            )),
            "unit_records": str(out / "unit_records.jsonl"),
        },
        "model_file_manifest": model_file_manifest(),
        "representation_contract": {
            "raw_visual_tokens_persisted": False,
            "raw_visual_shape_expected": "[merged_image_tokens, 4608]",
            "first_step_language_hidden_used": True,
            "candidate_region_rule": "fixed image-token-center coverage mean",
            "query_rule": "masked mean of exact expression token span when recoverable; otherwise whole-text-minus-image control",
            "merge_kernel": list(MERGE_KERNEL),
            "patch_size": PATCH_SIZE,
            "native_iou_threshold": NATIVE_IOU_THRESHOLD,
            "token_span_alignment": "UNALIGNED",
        },
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "raw_dense_feature_cache_written": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    all_fixed = fixed_units()
    units = all_fixed[:16] if args.split == "calibration" else all_fixed[16:]
    base = base_payload(args, out, units)
    write_json(out / "status.json", base)
    started = time.perf_counter()
    bank_cache: dict[str, L72Bank] = {}
    model = processor = tokenizer = runner = None
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        manifest_sha = sha256_file(MANIFEST)
        if manifest_sha != EXPECTED_MANIFEST_SHA:
            raise AssertionError(f"manifest SHA mismatch: {manifest_sha}")
        if not units or len(units) != (16 if args.split == "calibration" else 24):
            raise AssertionError("fixed calibration/validation unit count mismatch")

        import torch
        import transformers
        from PIL import Image
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        torch.manual_seed(SEED)
        if not torch.cuda.is_available():
            raise RuntimeError("L72 representation capture requires CUDA on GPU0")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
            raise RuntimeError("L72 capture must use CUDA_VISIBLE_DEVICES=0")
        device = torch.device("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True, local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            str(MODEL_DIR), torch_dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, attn_implementation="sdpa",
        ).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if not all(not parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError("detector parameter freeze failed")
        runner = InstrumentedLocateAnythingGeneration(
            model, tokenizer, processor, str(MODEL_DIR), seed=SEED
        )
        base["runtime"] = {
            "interpreter": sys.executable,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "model_dtype": str(torch.bfloat16),
        }

        records: list[dict[str, Any]] = []
        control_delta: dict[str, Any] | None = None
        peak_bytes = 0
        per_video: Counter[str] = Counter()
        for unit_index, unit in enumerate(units):
            video = str(unit["video"])
            bank = bank_cache.get(video)
            if bank is None:
                bank = L72Bank(video)
                bank_cache[video] = bank
            rows = bank.rows_for(int(unit["frame_id"]))
            image_path = IMAGE_ROOT / video / f"{int(unit['frame_id']):06d}.png"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image = Image.open(image_path).convert("RGB")
            expected_size = [int(v) for v in unit.get("image_size", [])]
            if expected_size and expected_size != [image.width, image.height]:
                raise AssertionError(
                    f"image size mismatch for {unit_key(unit)}: {expected_size} vs {image.size}"
                )
            boxes = bank.tensors["box"][rows].float().tolist()
            if len(boxes) != len(rows):
                raise AssertionError("candidate box row count mismatch")
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            capture = capture_once(
                model, runner, processor, tokenizer, image, str(unit["sentence"]),
                boxes, (image.width, image.height), retain_vectors=(unit_index == 0),
            )
            elapsed = time.perf_counter() - t0
            peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated(device)))
            if len(capture["candidate_rows"]) != len(rows):
                raise AssertionError("candidate rows changed during representation capture")
            empty_count = sum(
                int(int(row["token_count"]) == 0) for row in capture["candidate_rows"]
            )
            if empty_count == len(capture["candidate_rows"]):
                # Empty mappings are explicit evidence.  Do not replace them
                # with zero; only an all-empty frame blocks the interface.
                raise AssertionError(
                    f"all candidate image-token regions empty in {unit_key(unit)}"
                )

            # Labels are joined only after the full representation and all row mappings exist.
            target_ids = normalize_ids(unit.get("target_ids", []))
            sidecar = [None if bank.labels[row] is None else str(bank.labels[row]) for row in rows]
            labels = [value is not None and value in target_ids for value in sidecar]
            positive_rows = [index for index, value in enumerate(labels) if value]
            category = (
                "multi_positive" if len(positive_rows) > 1 else
                "positive" if positive_rows else
                "present_uncovered" if target_ids else "inactive"
            )
            # L49's category describes the historical L19 candidate view.  A
            # larger L69 bank can legitimately rescue such a unit, so retain
            # that source label for provenance but derive the current category
            # from the L69 sidecar only.  Do not reject or rewrite either one.
            source_category = str(unit.get("category", "unavailable"))
            category_mismatch = source_category != category
            history_checks = [bank.history_check(row, int(unit["frame_id"])) for row in rows]
            future_rows = sum(int(item["future_rows"]) for item in history_checks)
            row_keys = []
            candidate_indices = bank.tensors["candidate_index"].long().tolist()
            track_ids = bank.tensors["track_id"].long().tolist()
            pool_ids = bank.tensors["pool_id"].long().tolist()
            raw_ranks = bank.tensors["raw_rank"].long().tolist()
            for local, row in enumerate(rows):
                row_keys.append(["{}".format(unit["dataset"]), video, int(unit["query_id"]),
                                 int(unit["frame_id"]), str(bank.path), int(row)])
                capture["candidate_rows"][local].update({
                    "row_key": row_keys[-1],
                    "row_offset": int(row),
                    "candidate_index": int(candidate_indices[row]),
                    "track_id": int(track_ids[row]),
                    "pool_id": int(pool_ids[row]),
                    "raw_rank": int(raw_ranks[row]),
                    "label": bool(labels[local]),
                })
            if [int(row["row_offset"]) for row in capture["candidate_rows"]] != rows:
                raise AssertionError(f"row order drift for {unit_key(unit)}")
            if len(row_keys) != len(set(tuple(key) for key in row_keys)):
                raise AssertionError(f"immutable row key duplicate for {unit_key(unit)}")
            if future_rows:
                raise AssertionError(f"future history rows for {unit_key(unit)}: {future_rows}")

            record = {
                "format": "locatemot-l72-representation-unit-v1",
                "status": "complete",
                "unit_key": unit_key(unit),
                "fixed_eval_order": int(unit["fixed_eval_order"]),
                "fixed_eval_split": str(unit["fixed_eval_split"]),
                "dataset": str(unit["dataset"]),
                "video": video,
                "query_id": int(unit["query_id"]),
                "frame_id": int(unit["frame_id"]),
                "sentence": str(unit["sentence"]),
                "sentence_sha256": short_hash_text(str(unit["sentence"])),
                "image_path": str(image_path),
                "bank_path": str(bank.path),
                "bank_sha256": bank.sha256,
                "category": category,
                "source_unit_category": source_category,
                "category_mismatch_from_l49_view": category_mismatch,
                "target_ids": sorted(target_ids),
                "candidate_present": bool(positive_rows),
                "coverage_mask": not (bool(target_ids) and not bool(positive_rows)),
                "candidate_count": len(rows),
                "positive_count": len(positive_rows),
                "row_keys": row_keys,
                "candidate_rows": capture["candidate_rows"],
                "history_future_rows": future_rows,
                "input_contract": {key: value for key, value in capture.items()
                                    if not key.startswith("_") and key != "candidate_rows"},
                "runtime": {
                    "elapsed_seconds": elapsed,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
                },
                "labels_joined_after_feature_construction": True,
                "raw_vectors_persisted": False,
            }
            records.append(record)
            per_video[video] += len(rows)
            internal_context = capture.pop("_context_vectors", None)
            internal_query = capture.pop("_query_vector", None)
            internal_raw = capture.pop("_raw_vectors", None)
            if unit_index == 0:
                # One fixed unrelated sentence is a label-free conditioning diagnostic.
                control = capture_once(
                    model, runner, processor, tokenizer, image, UNRELATED_SENTENCE,
                    boxes, (image.width, image.height), retain_vectors=True,
                )
                exact_context = internal_context
                control_context = control.pop("_context_vectors")
                exact_query = internal_query
                control_query = control.pop("_query_vector")
                raw_control = control.pop("_raw_vectors")
                paired = [
                    (a, b) for a, b in zip(exact_context or [], control_context or [])
                    if a is not None and b is not None
                ]
                changes = [float((b - a).norm() / (a.norm() + 1e-6)) for a, b in paired]
                raw_diffs = [float((b - a).abs().max()) for a, b in zip(internal_raw or [], raw_control or [])
                             if a is not None and b is not None]
                q_change = float((control_query - exact_query).norm() / (exact_query.norm() + 1e-6))
                control_delta = {
                    "control_sentence": UNRELATED_SENTENCE,
                    "exact_sentence_sha256": short_hash_text(str(unit["sentence"])),
                    "control_sentence_sha256": short_hash_text(UNRELATED_SENTENCE),
                    "candidate_pairs": len(paired),
                    "context_region_relative_l2_mean": float(np.mean(changes)) if changes else None,
                    "context_region_relative_l2_max": float(np.max(changes)) if changes else None,
                    "context_region_changed_fraction_gt_1e-4": (
                        float(np.mean(np.asarray(changes) > 1e-4)) if changes else None
                    ),
                    "query_vector_relative_l2": q_change,
                    "raw_visual_max_abs_diffs": {
                        "count": len(raw_diffs),
                        "max": float(np.max(raw_diffs)) if raw_diffs else None,
                        "mean": float(np.mean(raw_diffs)) if raw_diffs else None,
                    },
                    "control_labels_used": False,
                    "spatial_localization_claim": "diagnostic_only",
                }
                del control, control_context, control_query, raw_control
            del capture, internal_context, internal_query, internal_raw, image
            import torch
            torch.cuda.empty_cache()

        if any(record["candidate_count"] != len(record["candidate_rows"]) for record in records):
            raise AssertionError("candidate count drift in final records")
        finite_score_count = sum(
            int(row["score"] is not None and math.isfinite(float(row["score"])))
            for record in records for row in record["candidate_rows"]
        )
        if finite_score_count == 0:
            raise AssertionError("no finite candidate representation scores")
        threshold = None
        threshold_payload = None
        if args.split == "calibration":
            threshold_payload = fit_threshold(records, "score")
            threshold = float(threshold_payload["threshold"])
        else:
            calibration_candidates = sorted(out.parent.glob("representation_calibration_attempt*/representation_calibration.json"))
            if not calibration_candidates:
                raise FileNotFoundError("no completed L72 calibration representation for frozen validation threshold")
            calibration_payload = json.loads(calibration_candidates[-1].read_text())
            threshold_payload = calibration_payload.get("candidate_threshold_fit")
            if not threshold_payload or "threshold" not in threshold_payload:
                raise AssertionError("calibration threshold missing")
            threshold = float(threshold_payload["threshold"])
            threshold_payload = {**threshold_payload, "validation_used": False,
                                 "source": str(calibration_candidates[-1])}

        metrics = representation_metrics(records, "score", threshold)
        no_threshold_metrics = representation_metrics(records, "score", None)
        grouped = per_dataset_metrics(records, "score", threshold)
        categories = per_dataset_metrics(records, "score", threshold)
        all_scores = [float(row["score"]) for record in records for row in record["candidate_rows"]
                      if row["score"] is not None and math.isfinite(float(row["score"]))]
        raw_token_counts = [int(row["token_count"]) for record in records for row in record["candidate_rows"]]
        native_rows = [row for record in records for row in record["candidate_rows"]]
        payload = {
            **base,
            "status": "complete",
            "format": "locatemot-l72-raw-region-representation-v1",
            "manifest_sha256": manifest_sha,
            "candidate_row_count": sum(len(record["candidate_rows"]) for record in records),
            "finite_candidate_scores": bool(all_scores and np.isfinite(np.asarray(all_scores)).all()),
            "nonempty_region_rows": sum(int(value > 0) for value in raw_token_counts),
            "region_nonempty_fraction": float(np.mean(np.asarray(raw_token_counts) > 0)),
            "image_token_count_consistent": True,
            "candidate_key_drift": 0,
            "duplicate_candidate_indices_retained": sum(
                len([value for value, count in Counter(
                    int(row["candidate_index"]) for row in record["candidate_rows"]
                ).items() if count > 1])
                for record in records
            ),
            "future_history_rows": sum(int(record["history_future_rows"]) for record in records),
            "candidate_threshold_fit": threshold_payload,
            "score_metrics": metrics,
            "score_metrics_no_threshold": no_threshold_metrics,
            "by_dataset_and_category": grouped,
            "control_sentence_diagnostic": control_delta,
            "native_control": {
                "iou_threshold": NATIVE_IOU_THRESHOLD,
                "proposal_rows_matched": sum(int(row["native_matched"]) for row in native_rows),
                "proposal_row_fraction": float(np.mean(np.asarray([row["native_matched"] for row in native_rows], dtype=float))),
                "is_independent_control": True,
                "correspondence_success_claim": False,
            },
            "representation_signal_rule": {
                "usable_signal": {
                    "interface_complete": True,
                    "control_relative_l2_gt": 1e-4,
                    "minimum_nonempty_fraction": 0.90,
                    "calibration_score_finite_non_degenerate": bool(
                        np.isfinite(np.asarray(all_scores)).all() and np.std(np.asarray(all_scores)) > 1e-8
                    ),
                },
                "decision_is_representation_audit_not_semantic_or_hota": True,
            },
            "per_video_candidate_rows": dict(sorted(per_video.items())),
            "runtime_summary": {
                "peak_cuda_bytes": peak_bytes,
                "elapsed_seconds": time.perf_counter() - started,
                "raw_feature_cache_written": False,
            },
            "records": records,
        }
        write_json(out / ("representation_calibration.json" if args.split == "calibration"
                          else "representation_fixed_eval.json"), payload)
        with (out / "unit_records.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        write_json(out / "provenance.json", {
            **base,
            "status": "complete",
            "manifest_sha256": manifest_sha,
            "bank_shas": {video: bank.sha256 for video, bank in bank_cache.items()},
            "model_file_manifest": model_file_manifest(),
            "labels_used": "post_hoc_after_feature_construction",
            "expression_level_gt_only": True,
            "token_span_alignment": "UNALIGNED",
            "persistent_raw_or_dense_cache": False,
            "no_test_or_screening_labels": True,
        })
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        failure = {
            **base,
            "status": "incomplete",
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "preserve this attempt and perform at most one minimal targeted fix",
            "elapsed_seconds": time.perf_counter() - started,
            "traceback": traceback.format_exc(),
        }
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text(
            "# INCOMPLETE\n\n"
            f"First actionable root cause: `{failure['failure_root_cause']}`\n\n"
            "No zero-vector fallback or persistent raw/dense feature cache was used.\n"
        )
        return 1
    finally:
        for bank in bank_cache.values():
            bank.close()
        if model is not None:
            del model
        if processor is not None:
            del processor
        if tokenizer is not None:
            del tokenizer
        if runner is not None:
            del runner
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
