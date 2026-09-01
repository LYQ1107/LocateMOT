#!/usr/bin/env python3
"""L74 post-fusion attention localization and diffuseness audit.

The ``label_free`` phase runs the frozen multimodal prefill on the fixed
16-calibration/24-validation units without loading expression labels.  It
captures only compact summaries from the last four decoder self-attention
layers.  The ``label_audit`` phase is intentionally separate and may run only
after the label-free decision has been written.
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
PEAK_RATIO_THRESHOLD = 1.25
MAJORITY_UNITS = 20
PRIMARY_NONEMPTY_THRESHOLD = 0.90
EPS = 1e-12

sys.path.insert(0, str(ROOT))
from locatemot.models.l74_attention_diagnostics import (  # noqa: E402
    candidate_attention_summary,
    finite_stats,
    safe_corr,
)
from tools.audit_l73_postfusion_attention import (  # noqa: E402
    L73Bank,
    build_inputs,
    center_indices,
    fixed_units,
    image_positions,
    load_model,
    make_equal_length_control,
    overlap_indices,
    sentence_of,
    text_positions,
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


def metadata_units() -> list[dict[str, Any]]:
    """Return fixed-unit fields needed for a label-free image prefill only."""
    order = read_jsonl(L62_RECORDS)
    if len(order) != 40 or len({unit_key(row) for row in order}) != 40:
        raise AssertionError("L62 fixed order must contain 40 unique units")
    sources: dict[str, dict[str, Any]] = {}
    for split, name in (("calibration", "calibration_units.jsonl"),
                        ("validation", "validation_units.jsonl")):
        for row in read_jsonl(L49_ROOT / name):
            key = unit_key(row)
            # Deliberately discard target/category/label fields before any
            # representation is built.  A1 reloads the full rows later.
            sources[key] = {
                "unit_key": key,
                "dataset": str(row["dataset"]),
                "video": str(row["video"]),
                "query_id": int(row["query_id"]),
                "frame_id": int(row["frame_id"]),
                "sentence": str(row.get("sentence") or row.get("expression") or ""),
                "image_size": [int(value) for value in row.get("image_size", [])],
                "source_split": split,
            }
    result = []
    for index, order_row in enumerate(order):
        key = unit_key(order_row)
        if key not in sources:
            raise KeyError(f"missing fixed unit metadata: {key}")
        row = dict(sources[key])
        row["fixed_eval_order"] = index
        row["fixed_eval_split"] = "calibration" if index < 16 else "validation"
        if not row["sentence"]:
            raise AssertionError(f"empty expression in label-free unit {key}")
        result.append(row)
    return result


def layer_name(layer_id: int) -> str:
    return f"layer_{int(layer_id)}"


def panel_names(layer_ids: list[int], head_count: int) -> list[str]:
    return ["primary", "query_row_max", "query_row_mass_weighted"] + \
        [f"head_{head}" for head in range(head_count)] + \
        [layer_name(layer_id) for layer_id in layer_ids]


def _as_int_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def capture_attention(
    model,
    processor,
    tokenizer,
    image,
    sentence: str,
    boxes: list[list[float]],
) -> dict[str, Any]:
    """Capture only compact candidate summaries from the last four layers."""
    import torch

    inputs = build_inputs(processor, image, sentence)
    device = next(model.parameters()).device
    input_ids_cpu = inputs["input_ids"].detach().cpu()
    attention_mask_cpu = inputs["attention_mask"].detach().cpu()
    grid_cpu = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
    if grid_cpu.ndim != 2 or grid_cpu.shape[0] != 1:
        raise AssertionError(f"unexpected image_grid_hws {tuple(grid_cpu.shape)}")
    image_pos = image_positions(input_ids_cpu, int(model.config.image_token_index))
    if not image_pos:
        raise AssertionError("no image token positions")
    q_pos, q_method = text_positions(tokenizer, sentence, input_ids_cpu,
                                     attention_mask_cpu, image_pos)
    if not q_pos:
        raise AssertionError("no valid expression token positions")
    decoder = getattr(model.language_model, "model", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise AssertionError("cannot locate decoder layers")
    total_layers = len(decoder.layers)
    layer_ids = list(range(max(0, total_layers - 4), total_layers))
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_id: int):
        def hook(module, args, kwargs, output):
            del module, args, kwargs
            value = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if value is not None:
                captured[layer_id] = value.detach().float().cpu()
        return hook

    handles = [decoder.layers[layer_id].self_attn.register_forward_hook(
        make_hook(layer_id), with_kwargs=True
    ) for layer_id in layer_ids]
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.vision_model.dtype)
    grid = grid_cpu.to(device=device, dtype=torch.int32)
    try:
        with torch.inference_mode():
            raw_list = model.extract_feature(pixel_values, grid)
            if not isinstance(raw_list, (list, tuple)) or len(raw_list) != 1:
                raise AssertionError("unexpected raw feature container")
            raw = torch.cat(raw_list, dim=0)
            projected = model.mlp1(raw)
            if projected.ndim != 2 or projected.shape[0] != len(image_pos):
                raise AssertionError(
                    f"projected/image position mismatch {tuple(projected.shape)} vs {len(image_pos)}"
                )
            if not bool(torch.isfinite(projected.float()).all()):
                raise AssertionError("projected visual values are nonfinite")
            outputs = model.language_model(
                input_ids=inputs["input_ids"].to(device=device),
                visual_features=projected,
                image_token_index=int(model.config.image_token_index),
                attention_mask=inputs["attention_mask"].to(device=device),
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
            del outputs, raw_list, raw, projected
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layer_ids):
        raise AssertionError(f"missing layer attention hooks: {sorted(set(layer_ids) - set(captured))}")

    sequence_length = int(input_ids_cpu.shape[1])
    # Local positions are retained only for the explicit L73 reproduction
    # control.  The corrected primary always uses absolute decoder positions.
    local_image_indices = torch.arange(len(image_pos), dtype=torch.long)
    query_indices = torch.as_tensor(q_pos, dtype=torch.long)
    layer_vectors: dict[int, dict[str, np.ndarray]] = {}
    for layer_id in layer_ids:
        attention = captured[layer_id]
        if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[2] != attention.shape[3]:
            raise AssertionError(f"layer {layer_id} attention shape/orientation failed: {tuple(attention.shape)}")
        if attention.shape[2] != sequence_length:
            raise AssertionError(f"layer {layer_id} sequence length mismatch")
        if not bool(torch.isfinite(attention).all()):
            raise AssertionError(f"layer {layer_id} attention has nonfinite values")
        row_sums = attention[0].sum(dim=-1)
        if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-3, rtol=2e-3)):
            raise AssertionError(f"layer {layer_id} attention rows do not sum to one")
        # ``indices`` below are local positions in the flattened image-token
        # lattice.  Decoder attention uses absolute sequence positions, so
        # first select the actual image-token key columns.  The legacy L73
        # path used local indices directly; retain that route only as an
        # explicit reproduction control, never as the corrected primary.
        absolute_image_positions = torch.as_tensor(image_pos, dtype=torch.long)
        image_attention = attention[0].index_select(1, query_indices).index_select(2, absolute_image_positions)
        legacy_image_attention = attention[0].index_select(1, query_indices).index_select(2, local_image_indices)
        # [heads, expression_rows, image_tokens]
        per_head_mean = image_attention.mean(dim=1)
        per_head_max = image_attention.max(dim=1).values
        query_mass = image_attention.sum(dim=2)
        query_weights = query_mass / query_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
        per_head_weighted = (image_attention * query_weights.unsqueeze(-1)).sum(dim=1)
        layer_mean = image_attention.mean(dim=(0, 1))
        layer_max = per_head_mean.mean(dim=0).clone()
        # The registered row-max is max over expression rows after head mean.
        layer_max = image_attention.mean(dim=0).max(dim=0).values
        layer_weighted = per_head_weighted.mean(dim=0)
        layer_vectors[layer_id] = {
            "mean": layer_mean.numpy(),
            "max": layer_max.numpy(),
            "mass_weighted": layer_weighted.numpy(),
            "legacy_mean": legacy_image_attention.mean(dim=(0, 1)).numpy(),
            "per_head_mean": per_head_mean.numpy(),
            "per_head_max": per_head_max.numpy(),
            "per_head_mass_weighted": per_head_weighted.numpy(),
        }
        del attention, image_attention, legacy_image_attention, per_head_mean, per_head_max, query_mass
    final_id = layer_ids[-1]
    final = layer_vectors[final_id]
    processed = processor.image_processor.rescale(image, list(MERGE_KERNEL))
    original_size = (int(image.width), int(image.height))
    processed_size = (int(processed.width), int(processed.height))
    grid_hw = _as_int_list(grid_cpu[0])
    if grid_hw[0] % MERGE_KERNEL[0] or grid_hw[1] % MERGE_KERNEL[1]:
        raise AssertionError(f"non-divisible image grid {grid_hw}")
    merged_shape = (grid_hw[0] // MERGE_KERNEL[0], grid_hw[1] // MERGE_KERNEL[1])
    rows: list[dict[str, Any]] = []
    primary_scores: list[float | None] = []
    for box in boxes:
        overlap = overlap_indices(box, original_size, processed_size, grid_hw,
                                  patch_size=PATCH_SIZE, merge_kernel=MERGE_KERNEL)
        center = center_indices(box, original_size, processed_size, grid_hw,
                                patch_size=PATCH_SIZE, merge_kernel=MERGE_KERNEL)
        indices = [int(value) for value in overlap["indices"]]
        areas = [float(value) for value in overlap["overlap_areas"]]
        mapping = {
            "overlap_indices": indices,
            "overlap_areas": areas,
            "scaled_box": [float(value) for value in overlap["scaled_box"]],
            "grid_shape": [int(value) for value in overlap["grid_shape"]],
            "processed_size": [int(value) for value in processed_size],
            "original_size": [int(value) for value in original_size],
            "area_fraction_processed": float(overlap["area_fraction_processed"]),
            "center_indices": [int(value) for value in center["indices"]],
            "center_token_count": int(center["token_count"]),
        }
        def summarise(vector: np.ndarray) -> dict[str, Any]:
            summary = candidate_attention_summary(
                vector, indices, areas, merged_shape, processed_size, overlap["scaled_box"]
            )
            summary["candidate_area_fraction_processed"] = float(overlap["area_fraction_processed"])
            return summary

        primary = summarise(final["mean"])
        legacy_primary = summarise(final["legacy_mean"])
        row_max = summarise(final["max"])
        row_weighted = summarise(final["mass_weighted"])
        headwise = []
        for head in range(final["per_head_mean"].shape[0]):
            summary = summarise(final["per_head_mean"][head])
            summary["head"] = int(head)
            headwise.append(summary)
        layerwise = []
        for layer_id in layer_ids:
            summary = summarise(layer_vectors[layer_id]["mean"])
            summary["layer"] = int(layer_id)
            layerwise.append(summary)
        primary_scores.append(primary.get("candidate_mean"))
        rows.append({
            "box": [float(value) for value in box],
            "mapping": mapping,
            "overlap_token_count": int(len(indices)),
            "region_available": bool(indices),
            "primary": primary,
            "legacy_l73_primary": legacy_primary,
            "query_row_max": row_max,
            "query_row_mass_weighted": row_weighted,
            "headwise": headwise,
            "layerwise": layerwise,
            "finite": all(bool(item.get("finite", False)) for item in [primary, row_max, row_weighted]),
        })
    result = {
        "input_ids_shape": [int(value) for value in input_ids_cpu.shape],
        "attention_mask_shape": [int(value) for value in attention_mask_cpu.shape],
        "attention_mask_valid_tokens": int(attention_mask_cpu.bool().sum()),
        "image_token_positions": [int(value) for value in image_pos],
        "image_token_count": int(len(image_pos)),
        "expression_positions": [int(value) for value in q_pos],
        "expression_position_method": q_method,
        "image_grid_hws": [int(value) for value in grid_hw],
        "merged_grid_shape": [int(value) for value in merged_shape],
        "original_image_size": [int(value) for value in original_size],
        "processed_image_size": [int(value) for value in processed_size],
        "attention_orientation": "[batch, heads, query_position, key_position]",
        "attention_layers_captured": [int(value) for value in layer_ids],
        "attention_layer_count": int(total_layers),
        "attention_head_count": int(final["per_head_mean"].shape[0]),
        "attention_shapes": {layer_name(layer_id): [1, int(final["per_head_mean"].shape[0]), sequence_length, sequence_length]
                             for layer_id in layer_ids},
        "attention_rows_sum_to_one": True,
        "attention_finite": True,
        "projected_visual_token_count": int(len(image_pos)),
        "projected_visual_dim": int(model.config.hidden_size) if hasattr(model.config, "hidden_size") else None,
        "padding_mask_explicit": False,
        "candidate_rows": rows,
        "_primary_scores": primary_scores,
    }
    del captured, layer_vectors, inputs, input_ids_cpu, attention_mask_cpu, local_image_indices, query_indices
    return result


def row_key(unit: dict[str, Any], bank: L73Bank, row: int) -> list[Any]:
    return [str(unit["dataset"]), str(unit["video"]), int(unit["query_id"]),
            int(unit["frame_id"]), str(bank.path), int(row)]


def add_row_provenance(unit: dict[str, Any], bank: L73Bank, rows: list[int], capture: dict[str, Any]) -> None:
    candidate_indices = bank.tensors["candidate_index"].long().tolist()
    track_ids = bank.tensors["track_id"].long().tolist()
    pool_ids = bank.tensors["pool_id"].long().tolist()
    raw_ranks = bank.tensors["raw_rank"].long().tolist()
    if len(capture["candidate_rows"]) != len(rows):
        raise AssertionError(f"candidate count drift for {unit_key(unit)}")
    keys = []
    for local, row in enumerate(rows):
        key = row_key(unit, bank, row)
        keys.append(tuple(key))
        capture["candidate_rows"][local].update({
            "row_key": key,
            "row_offset": int(row),
            "candidate_index": int(candidate_indices[row]),
            "track_id": int(track_ids[row]),
            "pool_id": int(pool_ids[row]),
            "raw_rank": int(raw_ranks[row]),
        })
    if len(keys) != len(set(keys)):
        raise AssertionError(f"duplicate immutable row key in {unit_key(unit)}")


def pairwise_relative(left: list[float | None], right: list[float | None]) -> dict[str, Any]:
    a = np.asarray([np.nan if value is None else float(value) for value in left])
    b = np.asarray([np.nan if value is None else float(value) for value in right])
    valid = np.isfinite(a) & np.isfinite(b)
    delta = b[valid] - a[valid]
    if not valid.any():
        return {"pairs": 0, "relative_l2": None, "changed_fraction_gt_1e-4": None}
    return {
        "pairs": int(valid.sum()),
        "relative_l2": float(np.linalg.norm(delta) / (np.linalg.norm(a[valid]) + 1e-6)),
        "changed_fraction_gt_1e-4": float(np.mean(np.abs(delta) > 1e-4)),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
    }


def neighborhood_margins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in rows if row["primary"].get("candidate_mean") is not None]
    if len(finite) < 2:
        return {"all_adjacent": finite_stats([]), "spatial_nearest": finite_stats([]), "top_gap": None}
    scores = np.asarray([float(row["primary"]["candidate_mean"]) for row in finite])
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    adjacent = np.abs(np.diff(sorted_scores)).tolist()
    centers = []
    for row in finite:
        center = row["primary"].get("spatial_centroid_processed")
        if center is None:
            center = row["primary"].get("box_center_processed")
        centers.append(center)
    spatial = []
    if all(center is not None for center in centers):
        xy = np.asarray(centers, dtype=np.float64)
        for index in range(len(finite)):
            distance = np.linalg.norm(xy - xy[index], axis=1)
            distance[index] = np.inf
            nearest = int(np.argmin(distance))
            spatial.append(abs(float(scores[index] - scores[nearest])))
    return {
        "all_adjacent": finite_stats(adjacent),
        "spatial_nearest": finite_stats(spatial),
        "top_gap": float(sorted_scores[0] - sorted_scores[1]),
    }


def get_panel_summary(row: dict[str, Any], panel: str) -> dict[str, Any]:
    if panel == "legacy_l73_primary":
        return row["legacy_l73_primary"]
    if panel == "primary":
        return row["primary"]
    if panel == "query_row_max":
        return row["query_row_max"]
    if panel == "query_row_mass_weighted":
        return row["query_row_mass_weighted"]
    if panel.startswith("head_"):
        head = int(panel.split("_", 1)[1])
        return row["headwise"][head]
    if panel.startswith("layer_"):
        layer = int(panel.split("_", 1)[1])
        for summary in row["layerwise"]:
            if int(summary["layer"]) == layer:
                return summary
    raise KeyError(panel)


def summarize_panel(records: list[dict[str, Any]], panel: str) -> dict[str, Any]:
    rows = [row for record in records for row in record["candidate_rows"]]
    summaries = [get_panel_summary(row, panel) for row in rows]
    available = [summary for summary in summaries if not summary.get("empty", False)]
    peak = [summary.get("candidate_concentration", {}).get("peak_to_uniform") for summary in available]
    entropy = [summary.get("candidate_concentration", {}).get("normalized_entropy") for summary in available]
    candidate_mass = [summary.get("candidate_mass_fraction") for summary in available]
    mean_score = [summary.get("candidate_mean") for summary in available]
    centroid_distance = [summary.get("box_center_distance_normalized") for summary in available]
    unit_peak_medians = []
    unit_margin = []
    for record in records:
        unit_summaries = [get_panel_summary(row, panel) for row in record["candidate_rows"]]
        unit_peak = [summary.get("candidate_concentration", {}).get("peak_to_uniform")
                     for summary in unit_summaries
                     if not summary.get("empty", False)
                     and summary.get("candidate_concentration", {}).get("peak_to_uniform") is not None]
        if unit_peak:
            unit_peak_medians.append(float(np.median(unit_peak)))
        scores = [summary.get("candidate_mean") for summary in unit_summaries
                  if summary.get("candidate_mean") is not None]
        if len(scores) >= 2:
            ordered = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
            unit_margin.append(float(ordered[0] - ordered[1]))
    return {
        "candidate_count": len(summaries),
        "available_candidate_count": len(available),
        "peak_to_uniform": finite_stats(peak),
        "normalized_entropy": finite_stats(entropy),
        "candidate_mass_fraction": finite_stats(candidate_mass),
        "candidate_score": finite_stats(mean_score),
        "box_center_distance_normalized": finite_stats(centroid_distance),
        "unit_median_peak_to_uniform": finite_stats(unit_peak_medians),
        "unit_median_margin": finite_stats(unit_margin),
        "units_with_median_peak_gt_1.25": int(sum(value > PEAK_RATIO_THRESHOLD for value in unit_peak_medians)),
        "unit_denominator": len(records),
    }


def unit_label_free_summary(record: dict[str, Any]) -> dict[str, Any]:
    rows = record["candidate_rows"]
    primary = [row["primary"] for row in rows]
    available = [item for item in primary if not item.get("empty", False)]
    centers = [item.get("box_center_distance_normalized") for item in available]
    areas = [item.get("candidate_area_fraction_processed") for item in available]
    scores = [item.get("candidate_mean") for item in available]
    margins = neighborhood_margins(rows)
    return {
        "unit_key": record["unit_key"],
        "fixed_eval_order": record["fixed_eval_order"],
        "fixed_eval_split": record["fixed_eval_split"],
        "dataset": record["dataset"],
        "video": record["video"],
        "candidate_count": len(rows),
        "primary_nonempty_count": len(available),
        "primary_nonempty_fraction": len(available) / max(1, len(rows)),
        "primary_score": finite_stats(scores),
        "primary_box_center_distance": finite_stats(centers),
        "primary_score_area_correlation": safe_corr(scores, areas),
        "neighboring_candidate_margins": margins,
        "duplicate_candidate_index_count": sum(
            count > 1 for count in Counter(int(row["candidate_index"]) for row in rows).values()
        ),
        "row_key_count": len({tuple(row["row_key"]) for row in rows}),
        "labels_used": False,
    }


def make_record(unit: dict[str, Any], bank: L73Bank, rows: list[int], capture: dict[str, Any], elapsed: float, peak: int) -> dict[str, Any]:
    add_row_provenance(unit, bank, rows, capture)
    all_keys = [tuple(row["row_key"]) for row in capture["candidate_rows"]]
    if len(all_keys) != len(set(all_keys)):
        raise AssertionError(f"row key drift for {unit_key(unit)}")
    future_rows = sum(bank.future_rows(row, int(unit["frame_id"])) for row in rows)
    if future_rows:
        raise AssertionError(f"future history rows: {unit_key(unit)}={future_rows}")
    return {
        "format": "locatemot-l74-label-free-unit-v1",
        "status": "complete",
        "unit_key": unit_key(unit),
        "fixed_eval_order": int(unit["fixed_eval_order"]),
        "fixed_eval_split": str(unit["fixed_eval_split"]),
        "dataset": str(unit["dataset"]),
        "video": str(unit["video"]),
        "query_id": int(unit["query_id"]),
        "frame_id": int(unit["frame_id"]),
        "sentence_sha256": hashlib.sha256(sentence_of(unit).encode()).hexdigest(),
        "image_path": str(IMAGE_ROOT / str(unit["video"]) / f"{int(unit['frame_id']):06d}.png"),
        "bank_path": str(bank.path),
        "bank_sha256": bank.sha256,
        "candidate_count": len(rows),
        "candidate_rows": capture["candidate_rows"],
        "attention_contract": {key: value for key, value in capture.items() if key != "candidate_rows" and not key.startswith("_")},
        "history_future_rows": int(future_rows),
        "elapsed_seconds": float(elapsed),
        "peak_cuda_bytes": int(peak),
        "labels_used": False,
        "raw_attention_persisted": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
    }


def provenance_payload(out: Path, mode: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": "locatemot-l74-attention-localization-provenance-v1",
        "status": "running",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "mode": mode,
        "seed": SEED,
        "runtime": {
            "interpreter": sys.executable,
            "device": "cuda:0",
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "offline_huggingface": os.environ.get("HF_HUB_OFFLINE") == "1",
        },
        "manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)},
        "inputs": {
            "l69_root": str(L69_ROOT),
            "l62_records": str(L62_RECORDS),
            "l49_metadata_sources": [str(L49_ROOT / "calibration_units.jsonl"), str(L49_ROOT / "validation_units.jsonl")],
            "image_root": str(IMAGE_ROOT),
            "model_dir": str(MODEL_DIR),
        },
        "fixed_units": [unit_key(unit) for unit in units],
        "fixed_unit_count": len(units),
        "candidate_protocol": {
            "complete_candidate_rows": True,
            "immutable_row_key": "(dataset,video,query,frame,bank_path,row_offset)",
            "duplicate_candidate_index_retained": True,
            "primary_mapping": "positive processed-area overlap with merged 2x2 cells",
            "center_mapping_control": "L72 center-cell rule",
            "no_top_k": True,
            "no_nms": True,
            "no_candidate_filter": True,
        },
        "attention_contract": {
            "use_cache": False,
            "output_attentions": True,
            "orientation": "[batch,heads,query_position,key_position]",
            "layers": "last four available decoder self-attention layers",
            "query_rows": "tokenizer exact/leading-space expression span; fixed whole-text fallback only if unresolved",
            "key_columns": "all image-token positions",
            "labels_read_in_label_free": False,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_mask": "UNALIGNED",
        },
        "outputs": {"directory": str(out)},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_attention_persisted": False,
    }


def label_free_phase(out: Path) -> dict[str, Any]:
    import torch
    from PIL import Image

    units = metadata_units()
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("immutable manifest SHA mismatch")
    model, processor, tokenizer, transformers_version = load_model()
    records: list[dict[str, Any]] = []
    bank_cache: dict[str, L73Bank] = {}
    peak_bytes = 0
    started = time.perf_counter()
    control = None
    try:
        for index, unit in enumerate(units):
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
            expected = [int(value) for value in unit.get("image_size", [])]
            if expected and expected != [image.width, image.height]:
                raise AssertionError(f"image size mismatch {unit_key(unit)}")
            boxes = bank.tensors["box"][rows].float().tolist()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            capture = capture_attention(model, processor, tokenizer, image, sentence_of(unit), boxes)
            elapsed = time.perf_counter() - t0
            peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated()))
            records.append(make_record(unit, bank, rows, capture, elapsed, int(torch.cuda.max_memory_allocated())))
            if index == 0:
                control_sentence, target_count, length_ok = make_equal_length_control(tokenizer, sentence_of(unit))
                control_capture = capture_attention(model, processor, tokenizer, image, control_sentence, boxes)
                control = {
                    "control_sentence": control_sentence,
                    "control_sentence_sha256": hashlib.sha256(control_sentence.encode()).hexdigest(),
                    "exact_sentence_sha256": records[-1]["sentence_sha256"],
                    "exact_token_count": int(len(records[-1]["attention_contract"]["expression_positions"])),
                    "control_token_count_target": int(target_count),
                    "control_token_count_equal": bool(length_ok),
                    "attention_score": pairwise_relative(
                        [row["primary"].get("candidate_mean") for row in records[-1]["candidate_rows"]],
                        [row["primary"].get("candidate_mean") for row in control_capture["candidate_rows"]],
                    ),
                    "diagnostic_only": True,
                    "labels_used": False,
                }
                del control_capture
            del capture, image
            torch.cuda.empty_cache()
    finally:
        for bank in bank_cache.values():
            bank.close()
        del model, processor, tokenizer
        torch.cuda.empty_cache()
    if len(records) != 40:
        raise AssertionError(f"expected 40 records, got {len(records)}")
    layer_ids = records[0]["attention_contract"]["attention_layers_captured"]
    head_count = int(records[0]["attention_contract"]["attention_head_count"])
    panels = panel_names([int(value) for value in layer_ids], head_count)
    unit_summaries = [unit_label_free_summary(record) for record in records]
    panel_summary = {panel: summarize_panel(records, panel) for panel in panels}
    legacy_summary = summarize_panel(records, "legacy_l73_primary")
    all_rows = [row for record in records for row in record["candidate_rows"]]
    primary_available = [row["primary"] for row in all_rows if not row["primary"].get("empty", False)]
    primary_nonempty = len(primary_available) / max(1, len(all_rows))
    complete = (
        all(len(record["candidate_rows"]) == record["candidate_count"] for record in records)
        and all(record["row_key_count"] if "row_key_count" in record else True for record in records)
        and all(row.get("finite", False) for row in all_rows)
        and all(record["history_future_rows"] == 0 for record in records)
    )
    # Unit-level score gap is diagnostic only; it is never used as a model
    # selector.  The same fixed threshold is applied to every registered panel.
    concentration_candidates = {
        panel: summary["units_with_median_peak_gt_1.25"] >= MAJORITY_UNITS
        for panel, summary in panel_summary.items()
    }
    any_concentration = any(concentration_candidates.values())
    center_distances = [value for unit in unit_summaries
                        for value in [unit["primary_box_center_distance"].get("std")]
                        if value is not None]
    margin_stds = [unit["neighboring_candidate_margins"]["all_adjacent"].get("std")
                   for unit in unit_summaries
                   if unit["neighboring_candidate_margins"]["all_adjacent"].get("std") is not None]
    spatial_nonconstant = bool(
        primary_available and all(math.isfinite(float(value)) for value in center_distances)
        and (float(np.std(center_distances)) > 1e-6 if center_distances else False)
        and (float(np.std(margin_stds)) > 1e-8 if margin_stds else False)
    )
    primary_score_std = panel_summary["primary"]["candidate_score"].get("std")
    decision_checks = {
        "complete_finite_keys": bool(complete),
        "primary_nonempty_fraction_ge_0.90": bool(primary_nonempty >= PRIMARY_NONEMPTY_THRESHOLD),
        "fixed_panel_concentration_majority": bool(any_concentration),
        "spatial_centroid_and_neighbor_margins_nonconstant": spatial_nonconstant,
        "primary_score_finite_non_degenerate": bool(primary_score_std is not None and primary_score_std > 1e-8),
        "inactive_presence_check": "deferred_to_A1_labels",
    }
    if not complete:
        decision = "localization_interface_blocked"
    elif not all((value is True) for key, value in decision_checks.items()
                 if key != "inactive_presence_check"):
        decision = "no_localization_signal"
    else:
        decision = "localization_signal_present"
    decision_payload = {
        "format": "locatemot-l74-a0-decision-v1",
        "status": decision,
        "decision_stage": "label_free_40_units_before_A1_labels",
        "pre_registered_rule": {
            "localization_interface_blocked": "prefill, attention orientation/positions, finite, key/order, row count, or repeatability failure",
            "no_localization_signal": "complete contract but primary coverage <0.90, no fixed panel has median peak-to-uniform >1.25 on at least 20/40 units, spatial diagnostics are constant/nonfinite, or scores are degenerate",
            "localization_signal_present": "all label-free checks pass",
            "peak_to_uniform_threshold": PEAK_RATIO_THRESHOLD,
            "majority_unit_threshold": MAJORITY_UNITS,
        },
        "checks": decision_checks,
        "evidence": {
            "unit_count": len(records),
            "candidate_row_count": len(all_rows),
            "primary_nonempty_rows": len(primary_available),
            "primary_nonempty_fraction": primary_nonempty,
            "panel_concentration": concentration_candidates,
            "panel_summary": panel_summary,
            "primary_score_std": primary_score_std,
            "labels_used": False,
            "validation_labels_read": False,
        },
        "control_diagnostic": control,
        "next_action": "Run A1 fixed-label explanatory audit only if status=localization_signal_present; otherwise stop and authorize a new raw representation hypothesis.",
    }
    payload = {
        "format": "locatemot-l74-label-free-diagnostics-v1",
        "status": "complete",
        "mode": "label_free",
        "decision": decision,
        "manifest_sha256": sha256_file(MANIFEST),
        "fixed_units": 40,
        "calibration_units": 16,
        "validation_units": 24,
        "candidate_row_count": len(all_rows),
        "candidate_key_drift": 0,
        "duplicate_candidate_indices_retained": sum(
            sum(count > 1 for count in Counter(int(row["candidate_index"]) for row in record["candidate_rows"]).values())
            for record in records
        ),
        "candidate_rows_all_retained": True,
        "candidate_truncation": False,
        "future_history_rows": sum(record["history_future_rows"] for record in records),
        "layer_ids": [int(value) for value in layer_ids],
        "head_count": head_count,
        "panel_names": panels,
        "panel_summary": panel_summary,
        "legacy_l73_primary_summary": legacy_summary,
        "unit_summaries": unit_summaries,
        "decision_payload": decision_payload,
        "control_diagnostic": control,
        "runtime": {
            "interpreter": sys.executable,
            "torch": torch.__version__,
            "transformers": transformers_version,
            "device": "cuda:0",
            "seed": SEED,
            "peak_cuda_bytes": peak_bytes,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "labels_used": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_attention_persisted": False,
    }
    write_json(out / "label_free_diagnostics.json", payload)
    with (out / "unit_records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    provenance = provenance_payload(out, "label_free", units)
    provenance.update({
        "status": "complete",
        "runtime": {**provenance["runtime"], "torch": torch.__version__, "transformers": transformers_version},
        "outputs": {"label_free_diagnostics": str(out / "label_free_diagnostics.json"),
                    "unit_records": str(out / "unit_records.jsonl"),
                    "labels_read": False},
    })
    write_json(out / "provenance.json", provenance)
    write_json(out / "decision.json", decision_payload)
    write_json(out / "status.json", {"format": "locatemot-l74-status-v1", "status": "complete", "decision": decision,
                                      "labels_used": False, "failure_root_cause": None,
                                      "next_action": decision_payload["next_action"]})
    return payload


def score_value(row: dict[str, Any], panel: str) -> float | None:
    summary = get_panel_summary(row, panel)
    value = summary.get("candidate_mean")
    return None if value is None else float(value)


def metric_for_control(records: list[dict[str, Any]], panel: str, subset_name: str = "all") -> dict[str, Any]:
    selected_records = records
    rows_count = 0
    positive_rows = 0
    finite_rows = 0
    covered_positive_rows = 0
    total_positive_rows = 0
    top1 = top5 = candidate_present_units = target_present_units = 0
    hard_flags: list[bool] = []
    strict, best, average, top_min_positive_rank = [], [], [], []
    multi_min_coverage, multi_full_recall = [], []
    inactive_accept = inactive_total = inactive_fp = 0
    unit_records = []
    for record in selected_records:
        rows = record["candidate_rows"]
        scores = np.asarray([score_value(row, panel) if score_value(row, panel) is not None else -np.inf for row in rows])
        finite = np.isfinite(scores)
        labels = np.asarray([bool(row.get("label", False)) for row in rows], dtype=bool)
        rows_count += len(rows); finite_rows += int(finite.sum())
        total_positive_rows += int(labels.sum())
        positive_rows += int(labels.sum())
        target_present = bool(record.get("target_ids"))
        candidate_present = bool(labels.any())
        if target_present:
            target_present_units += 1
        if candidate_present:
            candidate_present_units += 1
            order = np.argsort(-scores, kind="stable")
            top1 += int(bool(labels[order[:1]].any()))
            top5 += int(bool(labels[order[:5]].any()))
        pos = np.flatnonzero(labels)
        neg = np.flatnonzero(~labels)
        scored_pos = pos[finite[pos]] if pos.size else pos
        scored_neg = neg[finite[neg]] if neg.size else neg
        if scored_pos.size:
            covered_positive_rows += int(scored_pos.size)
            total_positive_rows += 0
        if scored_pos.size and scored_neg.size:
            strict_value = float(scores[scored_pos].min() - scores[scored_neg].max())
            strict.append(strict_value)
            best.append(float(scores[scored_pos].max() - scores[scored_neg].max()))
            average.append(float(scores[scored_pos].mean() - scores[scored_neg].max()))
            hard_flags.append(strict_value < 0)
        if pos.size:
            ranks = {int(index): int(rank + 1) for rank, index in enumerate(np.argsort(-scores, kind="stable"))}
            top_min_positive_rank.append(min(ranks[int(index)] for index in pos))
        if pos.size > 1:
            min_rank = top_min_positive_rank[-1] if top_min_positive_rank else None
            multi_min_coverage.append(float(min_rank <= pos.size) if min_rank is not None else 0.0)
            multi_full_recall.append(float((finite & labels).sum() / pos.size))
        category = str(record.get("category", "unavailable"))
        if category == "inactive":
            inactive_total += 1
            accept = bool(finite.any())
            inactive_accept += int(accept)
            inactive_fp += int((finite & ~labels).sum())
        unit_records.append({
            "unit_key": record["unit_key"],
            "category": category,
            "dataset": record["dataset"],
            "candidate_count": len(rows),
            "positive_count": int(labels.sum()),
            "top_positive_rank": top_min_positive_rank[-1] if pos.size and top_min_positive_rank else None,
            "strict_margin": strict[-1] if scored_pos.size and scored_neg.size else None,
            "hard_violation": bool(hard_flags[-1]) if scored_pos.size and scored_neg.size else None,
        })
    # This is intentionally the complete-candidate, no-threshold description.
    # It is not a deployable selection metric and does not delete rows.
    full_precision = covered_positive_rows / max(1, finite_rows)
    full_recall = covered_positive_rows / max(1, total_positive_rows)
    return {
        "panel": panel,
        "subset": subset_name,
        "threshold_policy": "all finite candidate rows; no threshold/filter; descriptive only",
        "units": len(selected_records),
        "candidate_rows": rows_count,
        "finite_candidate_rows": finite_rows,
        "positive_rows": positive_rows,
        "candidate_precision_full_set": full_precision,
        "candidate_recall_full_set": full_recall,
        "target_present_units": target_present_units,
        "candidate_present_units": candidate_present_units,
        "top1": top1 / max(1, candidate_present_units),
        "top5": top5 / max(1, candidate_present_units),
        "hard_negative_violation": float(np.mean(hard_flags)) if hard_flags else None,
        "hard_negative_pairwise_accuracy": float(1.0 - np.mean(hard_flags)) if hard_flags else None,
        "strict_margin": finite_stats(strict),
        "best_margin": finite_stats(best),
        "average_margin": finite_stats(average),
        "multi_positive_min_positive_coverage": float(np.mean(multi_min_coverage)) if multi_min_coverage else None,
        "multi_positive_full_set_recall": float(np.mean(multi_full_recall)) if multi_full_recall else None,
        "multi_positive_min_positive_rank": finite_stats(top_min_positive_rank),
        "inactive_false_acceptance_full_set": inactive_accept / max(1, inactive_total),
        "inactive_false_positive_rows_full_set": inactive_fp,
        "empty_rate": 1.0 - finite_rows / max(1, rows_count),
        "unit_ranking_records": unit_records,
    }


def add_labels_after_decision(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_units = {unit_key(row): row for row in fixed_units("calibration") + fixed_units("validation")}
    if len(full_units) != 40:
        raise AssertionError("fixed labeled unit set is not 40")
    banks: dict[str, L73Bank] = {}
    labeled = []
    try:
        for record in records:
            key = record["unit_key"]
            unit = full_units[key]
            video = str(record["video"])
            bank = banks.get(video)
            if bank is None:
                bank = L73Bank(video)
                banks[video] = bank
            labels_sidecar = bank.load_labels()
            target_ids = {str(value) for value in (unit.get("target_ids") or [])}
            cloned = json.loads(json.dumps(record))
            labels = []
            for row in cloned["candidate_rows"]:
                row_offset = int(row["row_offset"])
                value = labels_sidecar[row_offset]
                label = value is not None and str(value) in target_ids
                row["label"] = bool(label)
                labels.append(bool(label))
            category = (
                "multi_positive" if sum(labels) > 1 else
                "positive" if sum(labels) == 1 else
                "present_uncovered" if target_ids else "inactive"
            )
            cloned.update({
                "target_ids": sorted(target_ids),
                "category": category,
                "candidate_present": bool(any(labels)),
                "coverage_mask": not (bool(target_ids) and not bool(any(labels))),
                "positive_count": int(sum(labels)),
                "labels_used": True,
                "labels_joined_after_label_free_decision": True,
            })
            labeled.append(cloned)
    finally:
        for bank in banks.values():
            bank.close()
    return labeled


def label_audit_phase(out: Path) -> dict[str, Any]:
    decision = json.loads((out / "decision.json").read_text())
    if decision.get("status") != "localization_signal_present":
        raise RuntimeError(f"A1 labels are not authorized for decision {decision.get('status')}")
    records = read_jsonl(out / "unit_records.jsonl")
    if len(records) != 40:
        raise AssertionError("label-free records must contain 40 units")
    labeled = add_labels_after_decision(records)
    panels = json.loads((out / "label_free_diagnostics.json").read_text())["panel_names"]
    calibration = [record for record in labeled if record["fixed_eval_split"] == "calibration"]
    validation = [record for record in labeled if record["fixed_eval_split"] == "validation"]
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for name, subset in (("all", labeled), ("calibration", calibration), ("validation", validation)):
        by_group[name] = {panel: metric_for_control(subset, panel, name) for panel in panels}
    for group_key, group in (("dataset", labeled), ("category", labeled)):
        by_group[group_key] = {}
        values = sorted({str(record["dataset"] if group_key == "dataset" else record["category"]) for record in group})
        for value in values:
            subset = [record for record in group if str(record["dataset"] if group_key == "dataset" else record["category"]) == value]
            by_group[group_key][value] = {panel: metric_for_control(subset, panel, f"{group_key}:{value}") for panel in panels}
    l73_control = json.loads((ROOT / "outputs/l73/audit/attention_calibration_attempt2/attention_calibration.json").read_text())
    primary_l73_threshold = float(l73_control["candidate_threshold_fit"]["threshold"])
    # A1 is explanatory.  A fixed primary threshold is recorded as a reused
    # L73 control; all other panels remain thresholdless to avoid refitting.
    panel_gate = {}
    primary_validation = by_group["validation"]["primary"]
    for panel in panels:
        value = by_group["validation"][panel]
        panel_gate[panel] = {
            "hard_violation_decrease_ge_0.05_vs_l73_primary": bool(
                value["hard_negative_violation"] is not None
                and value["hard_negative_violation"] <= 1.0 - 0.05
            ),
            "candidate_recall_drop_le_0.01_vs_l73_primary": bool(
                value["candidate_recall_full_set"] >= 0.99
            ),
            "multi_positive_preserved": bool(
                value["multi_positive_full_set_recall"] is None
                or value["multi_positive_full_set_recall"] >= 0.0
            ),
            "threshold_or_filter_used": False,
        }
    payload = {
        "format": "locatemot-l74-labeled-localization-audit-v1",
        "status": "complete",
        "decision_input": decision,
        "labels_read_after_a0_decision": True,
        "fixed_units": 40,
        "calibration_units": 16,
        "validation_units": 24,
        "candidate_rows": sum(len(record["candidate_rows"]) for record in labeled),
        "candidate_rows_retained": True,
        "duplicate_candidate_index_retained": True,
        "threshold_policy": {
            "primary_l73_threshold_reused": primary_l73_threshold,
            "other_controls": "thresholdless full-candidate descriptive metrics",
            "validation_threshold_refit": False,
            "top_k": False,
            "nms": False,
            "candidate_deletion": False,
        },
        "controls": panels,
        "metrics": by_group,
        "panel_gate_diagnostic": panel_gate,
        "l73_primary_context": {
            "source": str(ROOT / "outputs/l73/audit/attention_calibration_attempt2/attention_calibration.json"),
            "source_sha256": sha256_file(ROOT / "outputs/l73/audit/attention_calibration_attempt2/attention_calibration.json"),
            "validation_hard_violation": 1.0,
            "validation_candidate_recall": 0.45161290322580644,
            "note": "L73 primary was compared descriptively; no threshold was refit in L74."
        },
        "oracle": {
            "present_uncovered_not_false_negative": True,
            "token_span_alignment": "UNALIGNED",
            "static_motion_mask": "UNALIGNED",
        },
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
        "next_action_rule": "A1 only proposes a single B1 probe if a fixed control passes all registered ranking checks; L74 itself does not train.",
    }
    write_json(out / "labeled_audit.json", payload)
    with (out / "labeled_unit_records.jsonl").open("w") as handle:
        for record in labeled:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(out / "a1_status.json", {"format": "locatemot-l74-a1-status-v1", "status": "complete",
                                        "labels_read_after_a0_decision": True,
                                        "next_action_rule": payload["next_action_rule"]})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("label_free", "label_audit"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if ROOT.resolve() != Path.cwd().resolve():
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("immutable manifest SHA mismatch")
    if args.phase == "label_free":
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"refusing non-empty attempt directory: {out}")
        out.mkdir(parents=True, exist_ok=True)
    else:
        if not (out / "decision.json").exists() or not (out / "unit_records.jsonl").exists():
            raise FileNotFoundError("label-free decision/records missing")
    try:
        if args.phase == "label_free":
            label_free_phase(out)
        else:
            label_audit_phase(out)
        return 0
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "format": "locatemot-l74-status-v1",
            "status": "incomplete",
            "phase": args.phase,
            "project_root": str(ROOT),
            "cwd": os.getcwd(),
            "command": " ".join(sys.argv),
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "next_action": "preserve attempt and fix only the first actionable root cause",
            "screening_gt_used": False,
            "official_test_labels_read": args.phase == "label_audit",
            "training_run": False,
            "hota_trackeval_run": False,
            "ordinary_mot_ovmot_touched": False,
            "raw_attention_persisted": False,
        }
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text(
            "# INCOMPLETE\n\n"
            f"First actionable root cause: `{failure['failure_root_cause']}`\n\n"
            f"Traceback is in `{out / 'status.json'}`.\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
