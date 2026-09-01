#!/usr/bin/env python3
"""L73 label-free post-fusion text-to-image attention audit.

The script calls the local LocateAnything language model directly on a full
image/expression prefill with ``use_cache=False`` and ``output_attentions=True``.
Only the last decoder-layer attention and final hidden output are retained long
enough to make compact candidate summaries.  No raw attention or feature cache
is persisted.
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
CONTROL_BASE = "small red cars moving toward front"

sys.path.insert(0, str(ROOT))
from locatemot.models.l73_postfusion import (  # noqa: E402
    center_indices,
    finite_vector_summary,
    masked_mean,
    overlap_indices,
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


def unit_key(row: dict[str, Any]) -> str:
    return str(row.get("unit_key") or "{}|{}|{}|{}".format(
        row["dataset"], row["video"], int(row["query_id"]), int(row["frame_id"])
    ))


def fixed_units(split: str) -> list[dict[str, Any]]:
    """Join only the requested L49 split to the immutable L62 order."""
    order = read_jsonl(L62_RECORDS)
    if len(order) != 40 or len({unit_key(row) for row in order}) != 40:
        raise AssertionError("L62 fixed order is not 40 unique units")
    source_path = L49_ROOT / (
        "calibration_units.jsonl" if split == "calibration" else "validation_units.jsonl"
    )
    source = {unit_key(row): row for row in read_jsonl(source_path)}
    indices = range(0, 16) if split == "calibration" else range(16, 40)
    result: list[dict[str, Any]] = []
    for index in indices:
        key = unit_key(order[index])
        if key not in source:
            raise KeyError(f"fixed unit missing from {source_path}: {key}")
        row = dict(source[key])
        row["fixed_eval_order"] = index
        row["fixed_eval_split"] = split
        result.append(row)
    return result


def normalize_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def sentence_of(unit: dict[str, Any]) -> str:
    sentence = str(unit.get("sentence") or "")
    if not sentence:
        sentence = str(unit.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty expression for {unit_key(unit)}")
    return sentence


def make_equal_length_control(tokenizer, sentence: str) -> tuple[str, int, bool]:
    """Create a deterministic unrelated control with the same token count."""
    target = tokenizer(sentence, add_special_tokens=False)["input_ids"]
    if target and isinstance(target[0], list):
        target = target[0]
    target_count = len(target)
    base = tokenizer(CONTROL_BASE, add_special_tokens=False)["input_ids"]
    if base and isinstance(base[0], list):
        base = base[0]
    if target_count == 0 or not base:
        return CONTROL_BASE, target_count, False
    repeats = (target_count + len(base) - 1) // len(base)
    control_ids = (list(base) * repeats)[:target_count]
    control = tokenizer.decode(
        control_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    encoded = tokenizer(control, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return control, target_count, len(encoded) == target_count


def find_subsequence(values: list[int], query: list[int]) -> list[int]:
    if not query:
        return []
    for start in range(0, len(values) - len(query) + 1):
        if values[start:start + len(query)] == query:
            return list(range(start, start + len(query)))
    return []


def image_positions(input_ids, image_token_index: int) -> list[int]:
    values = input_ids.detach().cpu().reshape(-1).tolist()
    return [index for index, value in enumerate(values) if int(value) == int(image_token_index)]


def text_positions(tokenizer, sentence: str, input_ids, attention_mask,
                   image_pos: list[int]) -> tuple[list[int], str]:
    full = input_ids.detach().cpu().reshape(-1).tolist()
    valid = attention_mask.detach().cpu().reshape(-1).bool().tolist()
    image_set = set(image_pos)
    for label, text in (("exact", sentence), ("leading_space", " " + sentence)):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        positions = find_subsequence(full, [int(value) for value in encoded])
        if positions and all(valid[pos] and pos not in image_set for pos in positions):
            return positions, label
    fallback = [idx for idx, flag in enumerate(valid) if flag and idx not in image_set]
    return fallback, "whole_text_minus_image_unresolved"


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-12, aa + bb - inter)


def compact_event(event: Any) -> dict[str, Any]:
    box = getattr(event, "normalized_box", None)
    return {
        "accepted": bool(getattr(event, "accepted", False)),
        "normalized_box": [float(value) for value in box] if box is not None else None,
        "generation_score": float(getattr(event, "generation_score", 0.0) or 0.0),
    }


class L73Bank:
    required = {
        "box", "frame", "frame_ids", "frame_ptr", "candidate_index", "track_id",
        "pool_id", "raw_rank", "clip", "history_clip", "uidm_h", "geometry",
        "motion", "lifecycle", "objectness",
    }

    def __init__(self, video: str):
        import torch

        self.video = str(video)
        self.path = L69_ROOT / f"{self.video}.pt"
        self.label_path = self.path.with_suffix(".labels.json")
        if not self.path.exists() or not self.label_path.exists():
            raise FileNotFoundError(f"missing L69 bank or sidecar for {self.video}")
        self.blob = torch.load(self.path, map_location="cpu", weights_only=False)
        self.tensors = self.blob["tensors"]
        missing = self.required.difference(self.tensors)
        if missing:
            raise KeyError(f"{self.path}: missing {sorted(missing)}")
        count = int(self.tensors["track_id"].numel())
        # Keep label contents lazy.  L73 feature construction must finish
        # before expression-level GT is read; callers explicitly invoke
        # load_labels() only after the frozen prefill and row mapping.
        self.labels: list[Any] | None = None
        self._row_count = count
        frame_ids = self.tensors["frame_ids"].long().tolist()
        frame_ptr = self.tensors["frame_ptr"].long().tolist()
        if len(frame_ptr) != len(frame_ids) + 1 or int(frame_ptr[-1]) != count:
            raise AssertionError(f"{self.path}: frame_ptr contract failed")
        self.frame_ranges: dict[int, tuple[int, int]] = {}
        frame_tensor = self.tensors["frame"].long()
        for frame_index, frame_id in enumerate(frame_ids):
            begin, end = int(frame_ptr[frame_index]), int(frame_ptr[frame_index + 1])
            if not bool((frame_tensor[begin:end] == int(frame_id)).all()):
                raise AssertionError(f"{self.path}: frame rows mismatch at {frame_id}")
            self.frame_ranges[int(frame_id)] = (begin, end)
        for name in ("box", "clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness"):
            if not bool(torch.isfinite(self.tensors[name].float()).all()):
                raise AssertionError(f"{self.path}: nonfinite {name}")
        self.metadata = dict(self.blob.get("metadata", {}))
        self.sha256 = sha256_file(self.path)

    def load_labels(self) -> list[Any]:
        if self.labels is None:
            sidecar = json.loads(self.label_path.read_text())
            labels = sidecar["candidate_gt"]
            if len(labels) != self._row_count:
                raise AssertionError(
                    f"{self.path}: sidecar length {len(labels)} != {self._row_count}"
                )
            self.labels = labels
        return self.labels

    def rows_for(self, frame_id: int) -> list[int]:
        if int(frame_id) not in self.frame_ranges:
            raise KeyError(f"{self.video}: missing frame {frame_id}")
        begin, end = self.frame_ranges[int(frame_id)]
        return list(range(begin, end))

    def future_rows(self, row: int, frame_id: int, length: int = 8) -> int:
        track = int(self.tensors["track_id"][row])
        frame_values = self.tensors["frame"].long()
        all_rows = (self.tensors["track_id"].long() == track).nonzero().reshape(-1).tolist()
        eligible = sorted(
            [old for old in all_rows if int(frame_values[old]) <= int(frame_id)],
            key=lambda old: (int(frame_values[old]), int(old)),
        )[-int(length):]
        return sum(int(frame_values[old]) > int(frame_id) for old in eligible)

    def close(self) -> None:
        self.blob = None
        self.tensors = {}
        self.labels = None
        self.frame_ranges = {}


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


def local_model_manifest() -> dict[str, Any]:
    path = ROOT / "outputs/l72/audit/api_smoke_attempt2/api_contract.json"
    if not path.exists():
        return {"source": str(path), "available": False}
    payload = json.loads(path.read_text())
    return {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "model_manifest": payload.get("model_manifest", {}),
    }


def capture_prefill(model, processor, tokenizer, image, sentence: str,
                    boxes: list[list[float]] | None = None,
                    retain_vectors: bool = False) -> dict[str, Any]:
    """Run one full-image multimodal prefill and summarize final-layer attention."""
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": sentence},
        ],
    }]
    inputs = build_inputs(processor, image, sentence)
    device = next(model.parameters()).device
    input_ids_cpu = inputs["input_ids"].detach().cpu()
    attention_mask_cpu = inputs["attention_mask"].detach().cpu()
    grid_cpu = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
    if grid_cpu.ndim != 2 or grid_cpu.shape[0] != 1:
        raise AssertionError(f"unexpected image_grid_hws {tuple(grid_cpu.shape)}")
    image_pos = image_positions(input_ids_cpu, int(model.config.image_token_index))
    if not image_pos:
        raise AssertionError("no image token positions in prefill input")
    q_pos, q_method = text_positions(tokenizer, sentence, input_ids_cpu,
                                     attention_mask_cpu, image_pos)
    if not q_pos:
        raise AssertionError("no valid expression positions")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.vision_model.dtype)
    grid = grid_cpu.to(device=device, dtype=torch.int32)

    captured: dict[str, Any] = {}

    def attention_hook(module, args, kwargs, output):
        del module, args
        attn = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if attn is not None:
            captured["attention"] = attn.detach().float().cpu()
        mask = kwargs.get("attention_mask") if isinstance(kwargs, dict) else None
        if mask is not None:
            captured["attention_mask"] = mask.detach().float().cpu()

    def norm_hook(module, args, output):
        del module, args
        hidden = output[0] if isinstance(output, tuple) else output
        captured["final_hidden"] = hidden.detach().float().cpu()

    lm = model.language_model
    decoder = getattr(lm, "model", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise AssertionError(f"cannot locate Qwen decoder layers on {type(lm).__name__}")
    attn_handle = decoder.layers[-1].self_attn.register_forward_hook(
        attention_hook, with_kwargs=True
    )
    norm_handle = decoder.norm.register_forward_hook(norm_hook)
    try:
        with torch.inference_mode():
            raw_list = model.extract_feature(pixel_values, grid)
            if not isinstance(raw_list, (list, tuple)) or len(raw_list) != 1:
                raise AssertionError(f"unexpected raw feature container {type(raw_list).__name__}")
            raw = torch.cat(raw_list, dim=0)
            projected = model.mlp1(raw)
            if projected.ndim != 2 or projected.shape[0] != len(image_pos):
                raise AssertionError(
                    f"projected visual/image position mismatch {tuple(projected.shape)} vs {len(image_pos)}"
                )
            if not bool(torch.isfinite(projected.float()).all()):
                raise AssertionError("projected visual values are nonfinite")
            outputs = lm(
                input_ids=inputs["input_ids"].to(device),
                visual_features=projected,
                image_token_index=int(model.config.image_token_index),
                attention_mask=inputs["attention_mask"].to(device),
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
            # Keep the projected image values from this same frozen visual
            # forward for the diagnostic value aggregation below.  Re-running
            # extract_feature here would duplicate the detector/backbone
            # forward for every expression/control sentence.
            image_values = projected.detach().float().cpu()
            del outputs, raw_list, raw, projected
    finally:
        attn_handle.remove()
        norm_handle.remove()

    attention = captured.get("attention")
    final_hidden = captured.get("final_hidden")
    if attention is None or final_hidden is None:
        raise AssertionError("last-layer attention or final hidden hook captured nothing")
    if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[2] != attention.shape[3]:
        raise AssertionError(f"attention orientation/shape failed: {tuple(attention.shape)}")
    if final_hidden.ndim != 3 or final_hidden.shape[0] != 1:
        raise AssertionError(f"final hidden shape failed: {tuple(final_hidden.shape)}")
    if attention.shape[2] != input_ids_cpu.shape[1] or final_hidden.shape[1] != input_ids_cpu.shape[1]:
        raise AssertionError(
            f"attention/hidden sequence mismatch {tuple(attention.shape)} / {tuple(final_hidden.shape)} "
            f"vs {tuple(input_ids_cpu.shape)}"
        )
    if not bool(torch.isfinite(attention).all()):
        raise AssertionError("captured attention probabilities contain nonfinite values")
    if not bool(torch.isfinite(final_hidden).all()):
        raise AssertionError("captured final hidden contains nonfinite values")
    attention_mean = attention[0].mean(dim=0)
    hidden = final_hidden[0]
    query_hidden = masked_mean(hidden, q_pos)
    if query_hidden is None:
        raise AssertionError("query hidden mean is empty")
    # The direct prefill's returned attention is [head, query, key].  This
    # orientation check uses the probability row invariant, not a label.
    row_sums = attention_mean.sum(dim=-1)
    if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-3, rtol=2e-3)):
        raise AssertionError("attention orientation check failed: rows do not sum to one")

    processed = processor.image_processor.rescale(image, list(MERGE_KERNEL))
    original_size = (int(image.width), int(image.height))
    processed_size = (int(processed.width), int(processed.height))
    grid_hw = [int(value) for value in grid_cpu[0].tolist()]
    if boxes is None:
        boxes = []
    rows: list[dict[str, Any]] = []
    score_values: list[float | None] = []
    value_vectors: list[torch.Tensor | None] = []
    for box in boxes:
        overlap = overlap_indices(box, original_size, processed_size, grid_hw,
                                  patch_size=PATCH_SIZE, merge_kernel=MERGE_KERNEL)
        center = center_indices(box, original_size, processed_size, grid_hw,
                                patch_size=PATCH_SIZE, merge_kernel=MERGE_KERNEL)
        indices = [int(value) for value in overlap["indices"]]
        if indices:
            q_index = torch.as_tensor(q_pos, dtype=torch.long)
            image_index = torch.as_tensor(indices, dtype=torch.long)
            attention_slice = attention_mean.index_select(0, q_index).index_select(1, image_index)
            score = float(attention_slice.mean())
            mass = float(attention_slice.sum())
            area = torch.as_tensor(overlap["overlap_areas"], dtype=torch.float32)
            area_score = float((attention_slice.mean(dim=0) * area).sum() / area.sum().clamp_min(1e-6))
            weights = attention_slice.mean(dim=0)
            weights = weights / weights.sum().clamp_min(1e-8)
            value = weights @ image_values.index_select(0, image_index)
            value = value.float()
            value_summary = finite_vector_summary(value)
            center_score = None
            if center["indices"]:
                center_index = torch.as_tensor(center["indices"], dtype=torch.long)
                center_score = float(
                    attention_mean.index_select(0, q_index).index_select(1, center_index).mean()
                )
        else:
            score = None
            mass = None
            area_score = None
            value = None
            value_summary = finite_vector_summary(None)
            center_score = None
        if score is not None and not math.isfinite(score):
            raise AssertionError("nonfinite candidate attention score")
        score_values.append(score)
        value_vectors.append(value)
        rows.append({
            "box": [float(value) for value in box],
            "overlap_indices": indices,
            "overlap_areas": [float(value) for value in overlap["overlap_areas"]],
            "token_count": int(overlap["token_count"]),
            "region_available": bool(indices),
            "mapping": overlap,
            "center_token_count": int(center["token_count"]),
            "center_score": center_score,
            "attention_score": score,
            "attention_mass": mass,
            "area_weighted_attention_score": area_score,
            "value_summary": value_summary,
        })
    mask = captured.get("attention_mask")
    result: dict[str, Any] = {
        "input_ids_shape": [int(value) for value in input_ids_cpu.shape],
        "attention_mask_shape": [int(value) for value in attention_mask_cpu.shape],
        "attention_mask_valid_tokens": int(attention_mask_cpu.bool().sum()),
        "image_token_positions": [int(value) for value in image_pos],
        "image_token_count": len(image_pos),
        "expression_positions": [int(value) for value in q_pos],
        "expression_position_method": q_method,
        "image_grid_hws": grid_hw,
        "merged_grid_shape": [int(value) for value in (grid_hw[0] // 2, grid_hw[1] // 2)],
        "processed_image_size": list(processed_size),
        "original_image_size": list(original_size),
        "attention_shape": [int(value) for value in attention.shape],
        "attention_orientation": "[batch, heads, query_position, key_position]",
        "attention_head_count": int(attention.shape[1]),
        "attention_rows_sum_to_one": True,
        "attention_finite": True,
        "attention_mask_contract_shape": list(mask.shape) if mask is not None else None,
        "attention_mask_supplied_to_last_layer": mask is not None,
        "padding_mask_explicit": False,
        "projected_visual_dim": int(image_values.shape[1]),
        "projected_visual_shape": [int(value) for value in image_values.shape],
        "projected_visual_finite": bool(torch.isfinite(image_values).all()),
        "final_hidden_shape": [int(value) for value in final_hidden.shape],
        "final_hidden_finite": True,
        "query_hidden_summary": finite_vector_summary(query_hidden),
        "candidate_rows": rows,
        "_score_vector": score_values if retain_vectors else None,
        "_value_vectors": value_vectors if retain_vectors else None,
        "_query_hidden": query_hidden if retain_vectors else None,
    }
    del captured, attention, attention_mean, final_hidden, hidden, image_values, inputs
    return result


def label_rows(unit: dict[str, Any], bank: L73Bank, rows: list[int], candidate_rows: list[dict[str, Any]]) -> tuple[str, set[str], list[bool]]:
    target_ids = normalize_ids(unit.get("target_ids", []))
    labels_sidecar = bank.load_labels()
    sidecar = [None if labels_sidecar[row] is None else str(labels_sidecar[row]) for row in rows]
    labels = [value is not None and value in target_ids for value in sidecar]
    positives = [idx for idx, value in enumerate(labels) if value]
    category = (
        "multi_positive" if len(positives) > 1 else
        "positive" if positives else
        "present_uncovered" if target_ids else "inactive"
    )
    if len(candidate_rows) != len(labels):
        raise AssertionError("candidate row/label length mismatch")
    for index, value in enumerate(labels):
        candidate_rows[index]["label"] = bool(value)
    return category, target_ids, labels


def metric_payload(records: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    values: list[float] = []
    tp = fp = fn = selected = positives = 0
    top1 = top5 = target_units = candidate_units = empty = 0
    inactive_units = inactive_accept = inactive_fp = 0
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    violations: list[bool] = []
    multi: list[float] = []
    unscored = unscored_positive = 0
    for record in records:
        scores_raw = [row["attention_score"] for row in record["candidate_rows"]]
        finite = np.asarray([
            value is not None and math.isfinite(float(value)) for value in scores_raw
        ], dtype=bool)
        scores = np.asarray([
            float(value) if valid else -np.inf
            for value, valid in zip(scores_raw, finite)
        ], dtype=np.float64)
        labels = np.asarray([bool(row["label"]) for row in record["candidate_rows"]], dtype=bool)
        if len(scores) != len(labels):
            raise AssertionError(f"metric length mismatch {record['unit_key']}")
        values.extend(scores[finite].tolist())
        unscored += int((~finite).sum())
        unscored_positive += int((~finite & labels).sum())
        if threshold is None:
            selected_mask = finite.copy()
        else:
            selected_mask = finite & (scores >= float(threshold))
        row_tp = int((selected_mask & labels).sum())
        row_fp = int((selected_mask & ~labels).sum())
        row_fn = int((~selected_mask & labels).sum())
        tp += row_tp; fp += row_fp; fn += row_fn
        selected += int(selected_mask.sum()); positives += int(labels.sum())
        empty += int(not selected_mask.any())
        target_present = bool(record["target_ids"])
        if target_present:
            target_units += 1
            if labels.any():
                candidate_units += 1
                order = np.argsort(-scores, kind="stable")
                top1 += int(bool(labels[order[:1]].any()))
                top5 += int(bool(labels[order[:5]].any()))
        if record["category"] == "inactive":
            inactive_units += 1; inactive_accept += int(selected_mask.any()); inactive_fp += row_fp
        pos = np.flatnonzero(labels)
        neg = np.flatnonzero(~labels)
        scored_pos = pos[finite[pos]] if pos.size else pos
        scored_neg = neg[finite[neg]] if neg.size else neg
        if scored_pos.size and scored_neg.size:
            strict_value = float(scores[scored_pos].min() - scores[scored_neg].max())
            strict.append(strict_value)
            best.append(float(scores[scored_pos].max() - scores[scored_neg].max()))
            average.append(float(scores[scored_pos].mean() - scores[scored_neg].max()))
            violations.append(strict_value < 0)
        if pos.size > 1:
            multi.append(float((selected_mask & labels).sum() / pos.size))
    return {
        "units": len(records),
        "candidate_rows": int(sum(len(record["candidate_rows"]) for record in records)),
        "positive_rows": positives,
        "target_present_units": target_units,
        "candidate_present_units": candidate_units,
        "present_uncovered_units": sum(record["category"] == "present_uncovered" for record in records),
        "scored_candidate_rows": len(values),
        "unscored_candidate_rows": unscored,
        "unscored_positive_rows": unscored_positive,
        "top1": top1 / max(1, target_units),
        "top5": top5 / max(1, target_units),
        "candidate_recall": tp / max(1, tp + fn),
        "candidate_precision": tp / max(1, selected),
        "fp_per_frame": fp / max(1, len(records)),
        "predictions_per_positive": selected / max(1, positives),
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
    }


def fit_threshold(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.unique(np.asarray([
        row["attention_score"] for record in records for row in record["candidate_rows"]
        if row["attention_score"] is not None and math.isfinite(float(row["attention_score"]))
    ], dtype=np.float64))
    if values.size == 0:
        raise AssertionError("no finite calibration attention scores")
    candidates = values.tolist() + [float(values.min()) - 1e-12, float(values.max()) + 1e-12]
    best: tuple[tuple[float, int, float], float] | None = None
    for threshold in candidates:
        metric = metric_payload(records, float(threshold))
        selected = sum(
            int(row["attention_score"] is not None and float(row["attention_score"]) >= threshold)
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
        "objective": "candidate-level F1 on fixed calibration rows",
        "tie_rule": "higher F1, fewer FP, then higher threshold",
        "validation_used": False,
    }


def grouped_metrics(records: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["dataset"])].append(record)
        groups[str(record["category"])].append(record)
    return {key: metric_payload(value, threshold) for key, value in sorted(groups.items())}


def base_payload(mode: str, out: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": "locatemot-l73-postfusion-attention-v1",
        "status": "running",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "mode": mode,
        "seed": SEED,
        "unit_count": len(units),
        "fixed_eval_order": [unit_key(row) for row in units],
        "inputs": {
            "l69_feature_root": str(L69_ROOT),
            "l49_split": str(L49_ROOT / ("calibration_units.jsonl" if mode == "calibration" else "validation_units.jsonl")),
            "l62_fixed_order": str(L62_RECORDS),
            "manifest": str(MANIFEST),
            "image_root": str(IMAGE_ROOT),
            "model_dir": str(MODEL_DIR),
        },
        "model_file_manifest": local_model_manifest(),
        "attention_contract": {
            "prefill": True,
            "use_cache": False,
            "output_attentions": True,
            "last_layer_only_retained": True,
            "primary_score": "mean over heads and expression query rows of attention to positive-area-overlap image cells",
            "region_mapping": "all merged 2x2 patch cells with positive box overlap",
            "l72_center_mapping_control": True,
            "token_span_alignment": "UNALIGNED",
        },
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_dense_feature_cache_written": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
    }


def load_model():
    import torch
    import transformers
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, attn_implementation="sdpa",
    ).to(torch.device("cuda:0")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not all(not parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("LocateAnything parameters are not frozen")
    return model, processor, tokenizer, transformers.__version__


def process_units(mode: str, out: Path, units: list[dict[str, Any]],
                  model, processor, tokenizer) -> dict[str, Any]:
    import torch
    from PIL import Image

    bank_cache: dict[str, L73Bank] = {}
    records: list[dict[str, Any]] = []
    control_delta: dict[str, Any] | None = None
    peak_bytes = 0
    started = time.perf_counter()
    for unit_index, unit in enumerate(units):
        video = str(unit["video"])
        bank = bank_cache.get(video)
        if bank is None:
            bank = L73Bank(video)
            bank_cache[video] = bank
        rows = bank.rows_for(int(unit["frame_id"]))
        image_path = IMAGE_ROOT / video / f"{int(unit['frame_id']):06d}.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        expected_size = [int(value) for value in unit.get("image_size", [])]
        if expected_size and expected_size != [image.width, image.height]:
            raise AssertionError(f"image size mismatch {unit_key(unit)}: {expected_size} vs {image.size}")
        boxes = bank.tensors["box"][rows].float().tolist()
        if len(boxes) != len(rows):
            raise AssertionError("candidate row/box count mismatch")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        capture = capture_prefill(model, processor, tokenizer, image, sentence_of(unit), boxes,
                                  retain_vectors=(unit_index == 0))
        elapsed = time.perf_counter() - t0
        peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated()))
        if len(capture["candidate_rows"]) != len(rows):
            raise AssertionError("candidate rows changed during postfusion capture")
        candidate_indices = bank.tensors["candidate_index"].long().tolist()
        track_ids = bank.tensors["track_id"].long().tolist()
        pool_ids = bank.tensors["pool_id"].long().tolist()
        raw_ranks = bank.tensors["raw_rank"].long().tolist()
        row_keys = []
        for local, row in enumerate(rows):
            row_key = [str(unit["dataset"]), video, int(unit["query_id"]), int(unit["frame_id"]), str(bank.path), int(row)]
            row_keys.append(row_key)
            capture["candidate_rows"][local].update({
                "row_key": row_key,
                "row_offset": int(row),
                "candidate_index": int(candidate_indices[row]),
                "track_id": int(track_ids[row]),
                "pool_id": int(pool_ids[row]),
                "raw_rank": int(raw_ranks[row]),
            })
        if len(row_keys) != len(set(tuple(value) for value in row_keys)):
            raise AssertionError(f"duplicate immutable row key {unit_key(unit)}")
        future_rows = sum(bank.future_rows(row, int(unit["frame_id"])) for row in rows)
        if future_rows:
            raise AssertionError(f"future history rows for {unit_key(unit)}: {future_rows}")

        # Labels are joined only after raw prefill, attention, and all row
        # mappings have succeeded.
        category, target_ids, labels = label_rows(unit, bank, rows, capture["candidate_rows"])
        source_category = str(unit.get("category", "unavailable"))
        internal_scores = capture.pop("_score_vector", None)
        internal_values = capture.pop("_value_vectors", None)
        internal_query = capture.pop("_query_hidden", None)
        record = {
            "format": "locatemot-l73-attention-unit-v1",
            "status": "complete",
            "unit_key": unit_key(unit),
            "fixed_eval_order": int(unit["fixed_eval_order"]),
            "fixed_eval_split": str(unit["fixed_eval_split"]),
            "dataset": str(unit["dataset"]),
            "video": video,
            "query_id": int(unit["query_id"]),
            "frame_id": int(unit["frame_id"]),
            "sentence": sentence_of(unit),
            "sentence_sha256": hashlib.sha256(sentence_of(unit).encode()).hexdigest(),
            "image_path": str(image_path),
            "bank_path": str(bank.path),
            "bank_sha256": bank.sha256,
            "category": category,
            "source_unit_category": source_category,
            "category_mismatch_from_l49_view": source_category != category,
            "target_ids": sorted(target_ids),
            "candidate_present": bool(any(labels)),
            "coverage_mask": not (bool(target_ids) and not bool(any(labels))),
            "candidate_count": len(rows),
            "positive_count": int(sum(labels)),
            "row_keys": row_keys,
            "candidate_rows": capture["candidate_rows"],
            "history_future_rows": future_rows,
            "attention_input_contract": {key: value for key, value in capture.items()},
            "runtime": {"elapsed_seconds": elapsed, "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())},
            "labels_joined_after_feature_construction": True,
            "raw_attention_persisted": False,
        }
        records.append(record)

        if unit_index == 0:
            control_sentence, target_token_count, control_len_ok = make_equal_length_control(tokenizer, sentence_of(unit))
            control = capture_prefill(model, processor, tokenizer, image, control_sentence, boxes, retain_vectors=True)
            control_scores = control.pop("_score_vector", None)
            control_values = control.pop("_value_vectors", None)
            control_query = control.pop("_query_hidden", None)
            if internal_scores is None or control_scores is None:
                raise AssertionError("control score vectors unavailable")
            exact_scores = np.asarray([
                float(value) if value is not None else np.nan for value in internal_scores
            ], dtype=np.float64)
            other_scores = np.asarray([
                float(value) if value is not None else np.nan for value in control_scores
            ], dtype=np.float64)
            valid = np.isfinite(exact_scores) & np.isfinite(other_scores)
            score_delta = other_scores[valid] - exact_scores[valid]
            paired_values = [
                (a, b) for a, b in zip(internal_values or [], control_values or [])
                if a is not None and b is not None
            ]
            value_changes = [float((b - a).norm() / (a.norm() + 1e-6)) for a, b in paired_values]
            query_change = None
            if internal_query is not None and control_query is not None:
                query_change = float((control_query - internal_query).norm() / (internal_query.norm() + 1e-6))
            control_delta = {
                "control_sentence": control_sentence,
                "control_sentence_sha256": hashlib.sha256(control_sentence.encode()).hexdigest(),
                "exact_sentence_sha256": record["sentence_sha256"],
                "exact_expression_token_count": int(target_token_count),
                "control_expression_token_count_equal": bool(control_len_ok),
                "exact_input_shape": capture.get("input_ids_shape"),
                "control_input_shape": control.get("input_ids_shape"),
                "candidate_pairs": int(valid.sum()),
                "attention_score_relative_l2": float(np.linalg.norm(score_delta) / (np.linalg.norm(exact_scores[valid]) + 1e-6)) if valid.any() else None,
                "attention_score_changed_fraction_gt_1e-4": float(np.mean(np.abs(score_delta) > 1e-4)) if valid.any() else None,
                "attention_score_delta_mean": float(np.mean(score_delta)) if valid.any() else None,
                "attention_value_relative_l2_mean": float(np.mean(value_changes)) if value_changes else None,
                "attention_value_relative_l2_max": float(np.max(value_changes)) if value_changes else None,
                "query_hidden_relative_l2": query_change,
                "labels_used": False,
                "diagnostic_only": True,
            }
            del control, control_scores, control_values, control_query
        del capture, internal_scores, internal_values, internal_query, image
        torch.cuda.empty_cache()

    if not records:
        raise AssertionError("no records")
    threshold_payload = None
    threshold = None
    if mode == "calibration":
        threshold_payload = fit_threshold(records)
        threshold = float(threshold_payload["threshold"])
    else:
        prior = sorted(out.parent.glob("attention_calibration_attempt*/attention_calibration.json"))
        if not prior:
            raise FileNotFoundError("no completed calibration attention artifact")
        calibration = json.loads(prior[-1].read_text())
        threshold_payload = calibration.get("candidate_threshold_fit")
        if not threshold_payload or "threshold" not in threshold_payload:
            raise AssertionError("calibration threshold missing")
        threshold = float(threshold_payload["threshold"])
        threshold_payload = {**threshold_payload, "validation_used": False, "source": str(prior[-1])}
    all_rows = [row for record in records for row in record["candidate_rows"]]
    primary_nonempty = sum(int(row["region_available"]) for row in all_rows)
    center_nonempty = sum(int(row["center_token_count"] > 0) for row in all_rows)
    metrics = metric_payload(records, threshold)
    no_threshold = metric_payload(records, None)
    payload = {
        "format": "locatemot-l73-postfusion-attention-v1",
        "status": "complete",
        "mode": mode,
        "manifest_sha256": sha256_file(MANIFEST),
        "candidate_row_count": len(all_rows),
        "candidate_key_drift": 0,
        "duplicate_candidate_indices_retained": sum(
            len([value for value, count in Counter(int(row["candidate_index"]) for row in record["candidate_rows"]).items() if count > 1])
            for record in records
        ),
        "primary_overlap_nonempty_rows": primary_nonempty,
        "primary_overlap_nonempty_fraction": primary_nonempty / max(1, len(all_rows)),
        "l72_center_nonempty_rows": center_nonempty,
        "l72_center_nonempty_fraction": center_nonempty / max(1, len(all_rows)),
        "all_candidate_rows_retained": True,
        "finite_primary_scores_among_available": all(
            row["attention_score"] is None or math.isfinite(float(row["attention_score"])) for row in all_rows
        ),
        "future_history_rows": sum(int(record["history_future_rows"]) for record in records),
        "candidate_threshold_fit": threshold_payload,
        "score_metrics": metrics,
        "score_metrics_no_threshold": no_threshold,
        "by_dataset_and_category": grouped_metrics(records, threshold),
        "control_sentence_diagnostic": control_delta,
        "representation_signal_rule": {
            "interface_complete": True,
            "control_score_relative_l2_gt": 1e-4,
            "minimum_primary_overlap_nonempty_fraction": 0.90,
            "calibration_score_finite_non_degenerate": bool(metrics["score_std"] is not None and metrics["score_std"] > 1e-8),
            "decision_is_not_semantic_or_hota": True,
        },
        "records": records,
        "runtime_summary": {
            "peak_cuda_bytes": peak_bytes,
            "elapsed_seconds": time.perf_counter() - started,
            "raw_or_dense_cache_written": False,
        },
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
    }
    out_file = out / ("attention_calibration.json" if mode == "calibration" else "attention_fixed_eval.json")
    write_json(out_file, payload)
    with (out / "unit_records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(out / "provenance.json", {
        "format": "locatemot-l73-postfusion-attention-provenance-v1",
        "status": "complete",
        "source_payload": str(out_file),
        "manifest_sha256": payload["manifest_sha256"],
        "bank_shas": {video: bank.sha256 for video, bank in bank_cache.items()},
        "labels": "expression-level labels joined after feature construction for descriptive audit",
        "token_span_alignment": "UNALIGNED",
        "attention_source": "last decoder layer self-attention from direct multimodal prefill",
        "prefill_contract": {"use_cache": False, "output_attentions": True, "last_layer_hook": True},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_dense_feature_cache_written": False,
    })
    write_json(out / "status.json", payload)
    for bank in bank_cache.values():
        bank.close()
    return payload


def run_api_smoke(out: Path) -> dict[str, Any]:
    import torch
    from PIL import Image

    fit_rows = read_jsonl(L49_ROOT / "train_units.jsonl")
    if not fit_rows:
        raise AssertionError("no fit units for API smoke")
    unit = next(row for row in fit_rows if str(row.get("dataset", "")).startswith("refer_kitti_v"))
    video = str(unit["video"])
    image_path = IMAGE_ROOT / video / f"{int(unit['frame_id']):06d}.png"
    image = Image.open(image_path).convert("RGB")
    model, processor, tokenizer, transformers_version = load_model()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    capture = capture_prefill(model, processor, tokenizer, image, sentence_of(unit), None, retain_vectors=False)
    elapsed = time.perf_counter() - started
    payload = {
        "format": "locatemot-l73-postfusion-api-smoke-v1",
        "status": "complete",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "interpreter": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "device": "cuda:0",
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "model_dtype": str(next(model.parameters()).dtype),
        "job": {
            "dataset": str(unit["dataset"]),
            "video": video,
            "query_id": int(unit["query_id"]),
            "frame_id": int(unit["frame_id"]),
            "image_path": str(image_path),
            "sentence_sha256": hashlib.sha256(sentence_of(unit).encode()).hexdigest(),
            "labels_used": False,
        },
        "prefill": {key: value for key, value in capture.items() if not key.startswith("_")},
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "model_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "no_gradient_through_frozen_model": True,
        "raw_attention_persisted": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
        "model_file_manifest": local_model_manifest(),
        "token_span_alignment": "UNALIGNED",
    }
    write_json(out / "api_contract.json", payload)
    write_json(out / "environment.json", {
        "format": "locatemot-l73-environment-v1",
        "status": "complete",
        "interpreter": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda:0",
        "offline_huggingface": True,
        "model_file_manifest": local_model_manifest(),
    })
    write_json(out / "status.json", payload)
    del capture, model, processor, tokenizer, image
    torch.cuda.empty_cache()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("api", "calibration", "validation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("immutable manifest SHA mismatch")
        if not os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "0"):
            raise RuntimeError("L73 requires CUDA_VISIBLE_DEVICES=0")
        if args.mode == "api":
            run_api_smoke(out)
        else:
            units = fixed_units(args.mode)
            model, processor, tokenizer, _ = load_model()
            base = base_payload(args.mode, out, units)
            write_json(out / "status.json", base)
            try:
                process_units(args.mode, out, units, model, processor, tokenizer)
            finally:
                del model, processor, tokenizer
                import torch
                torch.cuda.empty_cache()
        return 0
    except Exception as exc:
        failure = {
            "format": "locatemot-l73-postfusion-attention-v1",
            "status": "incomplete",
            "mode": args.mode,
            "project_root": str(ROOT),
            "cwd": os.getcwd(),
            "command": " ".join(sys.argv),
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "preserve this attempt and perform at most one minimal targeted fix",
            "traceback": traceback.format_exc(),
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "training_run": False,
            "hota_trackeval_run": False,
            "ordinary_mot_ovmot_touched": False,
            "raw_dense_feature_cache_written": False,
        }
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text(
            "# INCOMPLETE\n\n"
            f"First actionable root cause: `{failure['failure_root_cause']}`\n\n"
            "No labels were used to construct features and no raw/dense feature cache was written.\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
