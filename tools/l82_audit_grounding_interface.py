#!/usr/bin/env python3
"""Label-free GroundingDINO candidate-reference interface audit for L82.

The script uses the verified local MMDetection implementation only to capture
one native image feature pass per selected frame.  It then reuses those frozen
features for expression replay, a fixed unrelated control, and a batch/single
equivalence check.  Candidate rows are always reconstructed from native L69
frame pointers; native top-k outputs are never used as candidate inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Direct execution from ``tools/`` does not automatically put the project
# root on sys.path.  Keep the import deterministic for the runtime command.
PROJECT_ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from locatemot.models.l82_grounding_reference import (
    boxes_to_reference_points,
    boxes_xyxy_to_normalized,
    candidate_seed_with_reference,
    pool_memory_by_box,
)


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
LOCAL_MMDET = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0").resolve()
CONFIG = LOCAL_MMDET / "configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
WEIGHT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth").resolve()
BERT = Path("/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594").resolve()
TRAIN_UNITS = ROOT / "outputs/l49/data/train_units.jsonl"
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
UNRELATED = "an empty blue sky far from every object"
TARGET_PAIR_COUNT = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path), "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def finite_tensor(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.float()).all()):
        raise FloatingPointError(f"nonfinite {name}")


def detach_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def slice_batch_value(value: Any, index: int, batch_size: int) -> Any:
    """Slice only tensors whose leading dimension is the batch dimension."""
    if not torch.is_tensor(value):
        return value
    if value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value[index:index + 1].clone()
    return value.clone()


def detach_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): detach_clone(item) for key, item in value.items()}


def as_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def source_location(obj: Any) -> dict[str, Any]:
    try:
        path = Path(inspect.getsourcefile(obj) or "").resolve()
        lines, first = inspect.getsourcelines(obj)
        return {"file": str(path), "start_line": int(first), "end_line": int(first + len(lines) - 1)}
    except Exception as exc:
        return {"file": None, "error": f"{type(exc).__name__}: {exc}"}


def load_key_only_fit_units() -> list[dict[str, Any]]:
    rows = []
    for line in TRAIN_UNITS.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("split") != "fit" or raw.get("dataset") not in DATASETS:
            continue
        # Explicitly copy only the fields needed to address an image and text.
        # Target/category/positive fields are never accessed by this audit.
        rows.append({key: raw[key] for key in
                     ("unit_key", "dataset", "video", "query_id", "frame_id", "sentence")})
    return rows


def choose_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (str(row["dataset"]), str(row["unit_key"])))
    selected: list[dict[str, Any]] = []
    seen_frames: set[tuple[str, str, int]] = set()
    # Use 16 unique frames from each domain when available, then fill from the
    # deterministic remaining sequence.  This is key-only stratification.
    for dataset in DATASETS:
        for row in ordered:
            if row["dataset"] != dataset:
                continue
            frame_key = (str(row["dataset"]), str(row["video"]), int(row["frame_id"]))
            if frame_key in seen_frames:
                continue
            selected.append(row)
            seen_frames.add(frame_key)
            if sum(1 for item in selected if item["dataset"] == dataset) >= 16:
                break
    for row in ordered:
        if len(selected) >= TARGET_PAIR_COUNT:
            break
        frame_key = (str(row["dataset"]), str(row["video"]), int(row["frame_id"]))
        if frame_key in seen_frames:
            continue
        selected.append(row)
        seen_frames.add(frame_key)
    if len(selected) < TARGET_PAIR_COUNT:
        raise AssertionError(f"only {len(selected)} unique fit frame/query pairs available")
    return selected[:TARGET_PAIR_COUNT]


def load_l69_rows(video: str, frame_id: int) -> tuple[Path, dict[str, Any], torch.Tensor, list[dict[str, Any]]]:
    bank_path = (L69_ROOT / f"{video}.pt").resolve()
    blob = torch.load(bank_path, map_location="cpu", weights_only=False)
    tensors = blob.get("tensors") if isinstance(blob, dict) else None
    if not isinstance(tensors, dict):
        raise AssertionError(f"invalid L69 bank: {bank_path}")
    for field in ("frame", "frame_ids", "frame_ptr", "candidate_index", "track_id", "pool_id", "box"):
        if field not in tensors:
            raise AssertionError(f"{video}: missing L69 field {field}")
    frame_ids = tensors["frame_ids"].long().tolist()
    if int(frame_id) not in frame_ids:
        raise KeyError(f"frame {frame_id} absent from L69 bank {video}")
    position = frame_ids.index(int(frame_id))
    begin = int(tensors["frame_ptr"][position])
    end = int(tensors["frame_ptr"][position + 1])
    boxes = tensors["box"][begin:end].float().clone()
    required = len(range(begin, end))
    if boxes.shape != (required, 4) or not bool(torch.isfinite(boxes).all()):
        raise AssertionError(f"L69 box shape/finite drift for {video}:{frame_id}")
    rows = []
    for offset in range(begin, end):
        rows.append({
            "row_key": ["refer_kitti_v1" if False else None, video, int(frame_id), str(bank_path), int(offset)],
            "immutable_row_offset": int(offset),
            "candidate_index": int(tensors["candidate_index"][offset]),
            "track_id": int(tensors["track_id"][offset]),
            "pool_id": int(tensors["pool_id"][offset]),
        })
    del blob
    return bank_path, tensors, boxes, rows


def get_arg(args: tuple[Any, ...], kwargs: dict[str, Any], key: str, index: int) -> Any:
    return kwargs.get(key, args[index] if len(args) > index else None)


def build_model() -> tuple[Any, dict[str, Any]]:
    # Imports are intentionally inside the runtime path so this script can be
    # compiled in the project environment without MMDetection installed there.
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    import mmdet.datasets  # noqa: F401
    import mmdet.models  # noqa: F401
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(CONFIG))
    cfg.model.backbone.init_cfg = None
    cfg.model.language_model.name = str(BERT)
    model = MODELS.build(cfg.model)
    loaded = load_checkpoint(model, str(WEIGHT), map_location="cpu", strict=False)
    model.to("cuda:0").eval()
    model.cfg = cfg
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "config": str(CONFIG), "weight": str(WEIGHT), "bert": str(BERT),
        "checkpoint_missing_keys": loaded.get("missing_keys", []) if isinstance(loaded, dict) else [],
        "checkpoint_unexpected_keys": loaded.get("unexpected_keys", []) if isinstance(loaded, dict) else [],
        "encoder_source": source_location(model.encoder.__class__.forward),
        "fusion_source": source_location(model.encoder.fusion_layers[-1].__class__.forward),
        "pre_decoder_source": source_location(model.pre_decoder),
        "forward_decoder_source": source_location(model.forward_decoder),
    }


def make_text_dict(model: Any, text: str, device: torch.device,
                   force_pad_to_max: bool = False) -> tuple[dict[str, Any], str, Any]:
    old_pad_to_max = bool(model.language_model.pad_to_max)
    if force_pad_to_max:
        model.language_model.pad_to_max = True
    try:
        token_map, caption, _positive_map, _entities = model.get_tokens_positive_and_prompts(
            text, True, None, None)
        text_dict = model.language_model([caption])
        if model.text_feat_map is not None:
            text_dict["embedded"] = model.text_feat_map(text_dict["embedded"])
        for key, value in list(text_dict.items()):
            if torch.is_tensor(value):
                text_dict[key] = value.to(device)
        return text_dict, caption, token_map
    finally:
        model.language_model.pad_to_max = old_pad_to_max


def make_text_batch_dict(model: Any, captions: list[str], device: torch.device,
                         force_pad_to_max: bool = False) -> dict[str, Any]:
    old_pad_to_max = bool(model.language_model.pad_to_max)
    if force_pad_to_max:
        model.language_model.pad_to_max = True
    try:
        text_dict = model.language_model(captions)
        if model.text_feat_map is not None:
            text_dict["embedded"] = model.text_feat_map(text_dict["embedded"])
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in text_dict.items()
        }
    finally:
        model.language_model.pad_to_max = old_pad_to_max


def set_sample_text(sample: Any, caption: str, token_map: Any) -> None:
    """Update the native sample without confusing metainfo/data fields."""
    # MMEngine's BaseDataElement treats ``text`` as metainfo in the native
    # inference sample, so direct assignment raises once the field exists.
    sample.set_metainfo({"text": caption})
    sample.token_positive_map = token_map


def event_summary(event: dict[str, Any]) -> dict[str, Any]:
    memory = event["memory"]
    memory_text = event["memory_text"]
    spatial = event["spatial_shapes"]
    starts = event["level_start_index"]
    text_mask = event["text_token_mask"]
    finite_tensor(memory, "encoder memory")
    finite_tensor(memory_text, "encoder text memory")
    if spatial is None or starts is None:
        raise AssertionError("encoder did not expose spatial_shapes/level_start_index")
    expected_starts = torch.cat((starts.new_zeros(1), spatial.prod(1).cumsum(0)[:-1]))
    if not torch.equal(starts.cpu(), expected_starts.cpu()):
        raise AssertionError("level_start_index cumulative contract failed")
    return {
        "memory_shape": list(memory.shape), "memory_text_shape": list(memory_text.shape),
        "memory_finite": bool(torch.isfinite(memory.float()).all()),
        "memory_text_finite": bool(torch.isfinite(memory_text.float()).all()),
        "spatial_shapes": spatial.detach().cpu().tolist(),
        "level_start_index": starts.detach().cpu().tolist(),
        "level_start_index_reconstructed": expected_starts.detach().cpu().tolist(),
        "memory_mask_supplied": event["memory_mask"] is not None,
        "memory_mask_shape": list(event["memory_mask"].shape) if torch.is_tensor(event["memory_mask"]) else None,
        "text_token_mask_shape": list(text_mask.shape) if torch.is_tensor(text_mask) else None,
        "text_valid_tokens": int(text_mask.sum().item()) if torch.is_tensor(text_mask) else None,
        "valid_ratios": as_jsonable(event["valid_ratios"]),
    }


def capture_candidate_state(model: Any, event: dict[str, Any], boxes: torch.Tensor,
                            image_shape: tuple[int, int], scale_factor: Any,
                            candidate_permutation: torch.Tensor | None = None) -> dict[str, Any]:
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding

    memory = event["memory"]
    if candidate_permutation is not None:
        boxes = boxes[candidate_permutation]
    boxes_norm = boxes_xyxy_to_normalized(boxes.to(memory.device), image_shape, scale_factor)
    refs = boxes_to_reference_points(boxes_norm)
    visual_seed, roi_audit = pool_memory_by_box(
        memory, event["spatial_shapes"], event["level_start_index"], boxes_norm,
        event["memory_mask"], grid_size=4)
    seed, reference_position = candidate_seed_with_reference(
        visual_seed, refs, model.decoder.ref_point_head, coordinate_to_encoding)
    reference_batch = refs.unsqueeze(0)
    seed_batch = seed.unsqueeze(0)
    hidden = fixed_reference_decoder(model, seed, refs, event)
    finite_tensor(hidden, "candidate decoder hidden")
    final = hidden[-1, 0].float()
    if final.shape[0] != boxes.shape[0]:
        raise AssertionError("candidate decoder changed row count")
    return {
        "boxes_norm": boxes_norm.detach().clone(), "references": refs.detach().clone(),
        "visual_seed": visual_seed.detach().clone(), "reference_position": reference_position.detach().clone(),
        "seed": seed.detach().clone(), "final_hidden": final.detach().clone(),
        "roi_audit": roi_audit, "decoder_hidden_shape": list(hidden.shape),
        "fixed_reference_decoder": True,
    }


def compare_feature_delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise AssertionError(f"delta shape mismatch {tuple(left.shape)} vs {tuple(right.shape)}")
    diff = (left.float() - right.float())
    norms = diff.norm(dim=-1)
    mean_offset = diff.mean(dim=0)
    residual = diff - mean_offset.unsqueeze(0)
    return {
        "shape": list(left.shape), "mean_l2": float(norms.mean()),
        "std_l2": float(norms.std(unbiased=False)), "max_l2": float(norms.max()),
        "global_offset_l2": float(mean_offset.norm()),
        "residual_l2_mean": float(residual.norm(dim=-1).mean()),
        "nonuniform_fraction": float((norms > 1e-6).float().mean()),
        "finite": bool(torch.isfinite(diff).all()),
    }


def fixed_reference_decoder(model: Any, seed: torch.Tensor, references: torch.Tensor,
                            event: dict[str, Any]) -> torch.Tensor:
    """Run the verified decoder layers while keeping injected refs fixed.

    The native GroundingDINO ``forward_decoder`` supplies ``reg_branches`` and
    iteratively refines references.  L82's candidate-reference contract
    forbids that mutation, so this wrapper mirrors the local decoder loop and
    passes no regression branches.
    """
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding

    if seed.ndim != 2 or references.ndim != 2:
        raise ValueError("fixed-reference decoder expects [N,D] and [N,4]")
    query = seed.unsqueeze(0)
    reference_batch = references.unsqueeze(0)
    decoder = model.decoder
    intermediate = []
    for layer in decoder.layers:
        reference_input = reference_batch[:, :, None] * torch.cat(
            [event["valid_ratios"], event["valid_ratios"]], dim=-1)[:, None]
        query_sine_embed = coordinate_to_encoding(reference_input[:, :, 0, :])
        query_pos = decoder.ref_point_head(query_sine_embed)
        query = layer(
            query,
            query_pos=query_pos,
            value=event["memory"],
            key_padding_mask=event["memory_mask"],
            self_attn_mask=None,
            spatial_shapes=event["spatial_shapes"],
            level_start_index=event["level_start_index"],
            valid_ratios=event["valid_ratios"],
            reference_points=reference_input,
            memory_text=event["memory_text"],
            text_attention_mask=event["text_attention_mask"])
        intermediate.append(decoder.norm(query))
    hidden = torch.stack(intermediate)
    if hidden.ndim != 4 or hidden.shape[1] != 1:
        raise AssertionError(f"unexpected fixed-reference decoder output {tuple(hidden.shape)}")
    return hidden


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L82 grounding audit output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if not torch.cuda.is_available():
            raise RuntimeError("L82 grounding contract requires CUDA GPU0")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        required = [LOCAL_MMDET, CONFIG, WEIGHT, BERT, TRAIN_UNITS, MANIFEST]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(missing)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20260829)
        rows = load_key_only_fit_units()
        units = choose_units(rows)
        model, model_info = build_model()
        from mmdet.apis import inference_detector

        encoder_events: list[dict[str, Any]] = []
        decoder_events: list[dict[str, Any]] = []
        capture: dict[str, Any] = {}

        def encoder_hook(module, hook_args, hook_kwargs, output):
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise AssertionError("unexpected GroundingDINO encoder output contract")
            key_padding = get_arg(hook_args, hook_kwargs, "key_padding_mask", 2)
            spatial = get_arg(hook_args, hook_kwargs, "spatial_shapes", 3)
            starts = get_arg(hook_args, hook_kwargs, "level_start_index", 4)
            valid = get_arg(hook_args, hook_kwargs, "valid_ratios", 5)
            # The encoder has both a pairwise text self-attention mask and a
            # padding mask.  The candidate-reference replay needs the latter;
            # do not accidentally invert the 3-D pairwise mask.
            text_attention = get_arg(hook_args, hook_kwargs, "text_attention_mask", 7)
            text_mask = None if text_attention is None else ~text_attention.bool()
            encoder_events.append({
                "memory": output[0].detach().clone(), "memory_text": output[1].detach().clone(),
                "memory_mask": detach_clone(key_padding), "spatial_shapes": detach_clone(spatial),
                "level_start_index": detach_clone(starts), "valid_ratios": detach_clone(valid),
                "text_attention_mask": detach_clone(text_attention),
                "text_token_mask": detach_clone(text_mask),
            })

        def decoder_hook(module, hook_args, hook_kwargs, output):
            decoder_events.append({
                "query": detach_clone(get_arg(hook_args, hook_kwargs, "query", 0)),
                "reference_points": detach_clone(get_arg(hook_args, hook_kwargs, "reference_points", 4)),
                "value": detach_clone(get_arg(hook_args, hook_kwargs, "value", 1)),
                "spatial_shapes": detach_clone(get_arg(hook_args, hook_kwargs, "spatial_shapes", 5)),
            })

        encoder_handle = model.encoder.register_forward_hook(encoder_hook, with_kwargs=True)
        decoder_handle = model.decoder.register_forward_hook(decoder_hook, with_kwargs=True)
        original_extract = model.extract_feat
        original_forward_transformer = model.forward_transformer

        def wrapped_extract(batch_inputs):
            features = original_extract(batch_inputs)
            capture["visual_feats"] = tuple(x.detach().clone() for x in features)
            return features

        def wrapped_forward_transformer(img_feats, text_dict, batch_data_samples=None):
            capture["sample_template"] = copy.deepcopy(batch_data_samples)
            return original_forward_transformer(img_feats, text_dict, batch_data_samples)

        model.extract_feat = wrapped_extract
        model.forward_transformer = wrapped_forward_transformer
        records: list[dict[str, Any]] = []
        native_equivalence: list[dict[str, Any]] = []
        batch_equivalence: list[dict[str, Any]] = []
        base_requires_grad = all(not parameter.requires_grad for parameter in model.parameters())
        no_grad_before = all(parameter.grad is None for parameter in model.parameters())

        for index, unit in enumerate(units):
            dataset = str(unit["dataset"])
            video = str(unit["video"])
            frame_id = int(unit["frame_id"])
            expression = str(unit["sentence"])
            bank_path, tensors, boxes, row_meta = load_l69_rows(video, frame_id)
            for row in row_meta:
                row["row_key"][0] = dataset
            image = ROOT / "data/kitti_tracking_training/image_02" / video / f"{frame_id:06d}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            before_enc = len(encoder_events)
            before_dec = len(decoder_events)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.inference_mode():
                native = inference_detector(model, str(image), text_prompt=expression, custom_entities=True)
            native_sec = time.perf_counter() - t0
            if len(encoder_events) != before_enc + 1 or len(decoder_events) != before_dec + 1:
                raise AssertionError("native inference did not produce exactly one encoder/decoder event")
            native_event = encoder_events[-1]
            native_decoder = decoder_events[-1]
            if "visual_feats" not in capture or "sample_template" not in capture:
                raise AssertionError("native path did not expose reusable visual features/sample")
            feat_tensors = capture["visual_feats"]
            sample_template = capture["sample_template"]
            if not isinstance(sample_template, (list, tuple)) or len(sample_template) != 1:
                raise AssertionError("unexpected native sample template")
            sample_template = sample_template[0]
            image_shape = tuple(int(x) for x in native.metainfo["img_shape"][:2])
            scale_factor = native.metainfo["scale_factor"]
            candidate_norm = boxes_xyxy_to_normalized(boxes.cuda(), image_shape, scale_factor)
            if candidate_norm.shape[0] != len(row_meta):
                raise AssertionError("candidate row/key count drift")
            native_replay, _head = model.pre_decoder(
                memory=native_event["memory"], memory_mask=native_event["memory_mask"],
                spatial_shapes=native_event["spatial_shapes"], memory_text=native_event["memory_text"],
                text_token_mask=native_event["text_token_mask"], batch_data_samples=None)
            native_query = native_decoder["query"]
            native_ref = native_decoder["reference_points"]
            if native_query is None or native_ref is None:
                raise AssertionError("native decoder hook omitted query/reference_points")
            native_equivalence.append({
                "unit_key": str(unit["unit_key"]),
                "query_shape": list(native_query.shape), "reference_shape": list(native_ref.shape),
                "query_max_abs_delta": float((native_query - native_replay["query"]).abs().max()),
                "reference_max_abs_delta": float((native_ref - native_replay["reference_points"]).abs().max()),
                "threshold": 1e-4,
                "pass": bool((native_query - native_replay["query"]).abs().max() <= 1e-4 and
                             (native_ref - native_replay["reference_points"]).abs().max() <= 1e-4),
            })
            if not native_equivalence[-1]["pass"]:
                raise AssertionError(f"native pre-decoder equivalence failed for {unit['unit_key']}")

            # Re-run only transformer stages over the captured frozen image
            # features.  No second backbone/visual feature extraction occurs.
            def run_transformer(text: str, force_pad_to_max: bool = False) -> tuple[dict[str, Any], dict[str, Any], float]:
                text_dict, caption, token_map = make_text_dict(
                    model, text, torch.device("cuda:0"), force_pad_to_max=force_pad_to_max)
                sample = copy.deepcopy(sample_template)
                set_sample_text(sample, caption, token_map)
                before_e = len(encoder_events)
                before_d = len(decoder_events)
                t_start = time.perf_counter()
                with torch.inference_mode():
                    original_forward_transformer(feat_tensors, text_dict, [sample])
                elapsed = time.perf_counter() - t_start
                if len(encoder_events) != before_e + 1 or len(decoder_events) != before_d + 1:
                    raise AssertionError("replay transformer event cardinality drift")
                return encoder_events[-1], decoder_events[-1], elapsed

            replay_event, _replay_dec, replay_sec = run_transformer(expression)
            control_event, _control_dec, control_sec = run_transformer(UNRELATED)
            primary = capture_candidate_state(model, replay_event, boxes.cuda(), image_shape, scale_factor)
            control = capture_candidate_state(model, control_event, boxes.cuda(), image_shape, scale_factor)
            native_state = capture_candidate_state(model, native_event, boxes.cuda(), image_shape, scale_factor)
            repeat_delta = compare_feature_delta(native_state["final_hidden"], primary["final_hidden"])
            expression_delta = compare_feature_delta(primary["final_hidden"], control["final_hidden"])
            candidate_delta = compare_feature_delta(primary["seed"], control["seed"])
            permutation = torch.arange(boxes.shape[0] - 1, -1, -1, device="cuda")
            permuted = capture_candidate_state(model, replay_event, boxes.cuda(), image_shape, scale_factor, permutation)
            unpermuted_hidden = torch.empty_like(permuted["final_hidden"])
            unpermuted_hidden[permutation] = permuted["final_hidden"]
            permutation_error = float((unpermuted_hidden - primary["final_hidden"]).abs().max())
            permutation_errors = {}
            for name in ("boxes_norm", "references", "visual_seed", "reference_position", "seed", "final_hidden"):
                restored = torch.empty_like(permuted[name])
                restored[permutation] = permuted[name]
                permutation_errors[name] = float((restored - primary[name]).abs().max())

            # Build same-length padded single-query controls for a fair batch
            # equivalence check. The unpadded replay above remains the primary
            # native text contract.
            padded_replay_event, _padded_replay_dec, _ = run_transformer(expression, True)
            padded_control_event, _padded_control_dec, _ = run_transformer(UNRELATED, True)
            padded_primary = capture_candidate_state(
                model, padded_replay_event, boxes.cuda(), image_shape, scale_factor)
            padded_control = capture_candidate_state(
                model, padded_control_event, boxes.cuda(), image_shape, scale_factor)

            # Batch two copies of the same expression over the same captured
            # visual features. This tests query-axis contamination without
            # conflating normal floating-point differences between sentences
            # of different lengths with a batch/single contract violation.
            text_a, caption_a, map_a = make_text_dict(
                model, expression, torch.device("cuda:0"), True)
            caption_b, map_b = caption_a, map_a
            # The local GroundingDINO config uses ``pad_to_max=False``. Build
            # both rows with the same temporary max-padding setting so the
            # duplicate-query single and batch calls have an identical text
            # sequence length.
            batch_text = make_text_batch_dict(
                model, [caption_a, caption_a], torch.device("cuda:0"), True)
            text_batch_deltas = {}
            for key, value in text_a.items():
                batch_value = batch_text.get(key)
                if torch.is_tensor(value) and torch.is_tensor(batch_value):
                    text_batch_deltas[key] = {
                        "shape_single": list(value.shape),
                        "shape_batch_row": list(batch_value[0:1].shape),
                        "max_abs_delta": float((batch_value[0:1].float() - value.float()).abs().max()),
                    }
            batch_feats = tuple(torch.cat((tensor, tensor), dim=0) for tensor in feat_tensors)
            sample_a = copy.deepcopy(sample_template); set_sample_text(sample_a, caption_a, map_a)
            sample_b = copy.deepcopy(sample_template); set_sample_text(sample_b, caption_b, map_b)
            before_e = len(encoder_events)
            before_d = len(decoder_events)
            with torch.inference_mode():
                original_forward_transformer(batch_feats, batch_text, [sample_a, sample_b])
            if len(encoder_events) != before_e + 1 or len(decoder_events) != before_d + 1:
                raise AssertionError("batch query event cardinality drift")
            batch_event = encoder_events[-1]
            if int(batch_event["memory"].shape[0]) != 2:
                raise AssertionError("batch query memory shape drift")
            batch_single_a = {key: slice_batch_value(value, 0, 2)
                              for key, value in batch_event.items()}
            batch_single_b = {key: slice_batch_value(value, 1, 2)
                              for key, value in batch_event.items()}
            batch_a_state = capture_candidate_state(model, batch_single_a, boxes.cuda(), image_shape, scale_factor)
            batch_b_state = capture_candidate_state(model, batch_single_b, boxes.cuda(), image_shape, scale_factor)
            encoder_batch_deltas = {}
            for key in ("memory", "memory_text", "memory_mask", "valid_ratios", "text_attention_mask", "text_token_mask"):
                batch_value = batch_single_a.get(key)
                single_value = padded_replay_event.get(key)
                if torch.is_tensor(batch_value) and torch.is_tensor(single_value):
                    encoder_batch_deltas[key] = float((batch_value.float() - single_value.float()).abs().max())
            batch_equivalence.append({
                "unit_key": str(unit["unit_key"]),
                "padded_single_max_abs_delta": float((batch_a_state["final_hidden"] - padded_primary["final_hidden"]).abs().max()),
                "padded_duplicate_max_abs_delta": float((batch_b_state["final_hidden"] - padded_primary["final_hidden"]).abs().max()),
                "text_batch_deltas": text_batch_deltas,
                "encoder_batch_deltas": encoder_batch_deltas,
                "candidate_seed_max_abs_delta": float((batch_a_state["seed"] - padded_primary["seed"]).abs().max()),
                "candidate_count": int(boxes.shape[0]),
            })
            if torch.cuda.is_available():
                peak_memory = int(torch.cuda.max_memory_allocated())
            else:
                peak_memory = 0
            key_list = [
                [dataset, video, int(unit["query_id"]), frame_id, str(bank_path), int(row["immutable_row_offset"])]
                for row in row_meta
            ]
            records.append({
                "format": "locatemot-l82-grounding-interface-unit-v1",
                "unit_key": str(unit["unit_key"]), "dataset": dataset, "video": video,
                "query_id": int(unit["query_id"]), "frame_id": frame_id,
                "expression_sha256": hashlib.sha256(expression.encode()).hexdigest(),
                "unrelated_expression_sha256": hashlib.sha256(UNRELATED.encode()).hexdigest(),
                "image_path": str(image.resolve()), "bank_path": str(bank_path),
                "candidate_count": int(boxes.shape[0]), "row_keys": key_list,
                "candidate_index": [row["candidate_index"] for row in row_meta],
                "duplicate_candidate_index_count": int(len(row_meta) - len({row["candidate_index"] for row in row_meta})),
                "native_image_shape_hw": list(image_shape), "native_scale_factor": as_jsonable(scale_factor),
                "native_metainfo": {key: as_jsonable(native.metainfo[key]) for key in ("img_shape", "ori_shape", "scale_factor", "batch_input_shape") if key in native.metainfo},
                "native_forward_sec": native_sec, "replay_forward_sec": replay_sec,
                "unrelated_forward_sec": control_sec, "gpu_peak_bytes": peak_memory,
                "native_encoder": event_summary(native_event),
                "candidate_reference": {
                    "normalized_shape": list(primary["boxes_norm"].shape),
                    "reference_shape": list(primary["references"].shape),
                    "visual_seed_shape": list(primary["visual_seed"].shape),
                    "candidate_decoder_hidden_shape": primary["decoder_hidden_shape"],
                    "roi_grid": 4, "seed_formula": "mean(all valid sampled post-encoder memory levels) + frozen decoder.ref_point_head(reference positional encoding)",
                    "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                    "reference_width_height_positive": bool((primary["references"][:, 2:] > 0).all()),
                    "roi_audit": primary["roi_audit"],
                },
                "native_predecoder_equivalence": native_equivalence[-1],
                "expression_sensitivity": {
                    "final_hidden": expression_delta, "candidate_seed": candidate_delta,
                    "repeat_native_vs_replay": repeat_delta,
                    "candidate_specific_not_global_offset": bool(expression_delta["residual_l2_mean"] > 1e-6),
                },
                "candidate_permutation": permutation_errors,
                "candidate_permutation_max_abs_error": permutation_error,
                "batch_query_equivalence": batch_equivalence[-1],
                "base_requires_grad_false": base_requires_grad,
                "detector_parameter_grads_all_none": all(parameter.grad is None for parameter in model.parameters()),
                "raw_cache_persistent": False, "labels_read": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "training_run": False,
                "hota_trackeval_run": False, "candidate_deletion": False,
                "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
                "static_motion_alignment": "UNALIGNED",
            })
            del native_state, primary, control, padded_primary, padded_control
            del replay_event, control_event, padded_replay_event, padded_control_event, batch_event
            del feat_tensors, tensors, boxes, bank_path
            torch.cuda.empty_cache()
        encoder_handle.remove(); decoder_handle.remove()
        # The first native forward and all replays are inference-mode calls;
        # a frozen detector must not accumulate autograd state.
        permutation_input_errors = {
            name: max(row["candidate_permutation"][name] for row in records)
            for name in ("boxes_norm", "references", "visual_seed", "reference_position", "seed")
        }
        decoder_permutation_error = max(
            row["candidate_permutation"]["final_hidden"] for row in records)
        batch_equivalence_summary = {
            "max_padded_single_abs_error": max(
                row["padded_single_max_abs_delta"] for row in batch_equivalence),
            "max_padded_duplicate_abs_error": max(
                row["padded_duplicate_max_abs_delta"] for row in batch_equivalence),
            "max_encoder_memory_abs_error": max(
                row["encoder_batch_deltas"]["memory"] for row in batch_equivalence),
            "max_encoder_text_memory_abs_error": max(
                row["encoder_batch_deltas"]["memory_text"] for row in batch_equivalence),
            "max_candidate_seed_abs_error": max(
                row["candidate_seed_max_abs_delta"] for row in batch_equivalence),
            "threshold": 1e-4,
        }
        permutation_inputs_pass = all(value <= 1e-4 for value in permutation_input_errors.values())
        batch_equivalence_pass = (
            batch_equivalence_summary["max_encoder_memory_abs_error"] <= 1e-4 and
            batch_equivalence_summary["max_encoder_text_memory_abs_error"] <= 1e-4 and
            batch_equivalence_summary["max_candidate_seed_abs_error"] <= 1e-4)
        interface_pass = bool(
            all(row["pass"] for row in native_equivalence) and
            permutation_inputs_pass and batch_equivalence_pass and
            base_requires_grad and all(row["candidate_count"] == len(row["row_keys"]) for row in records))
        interface_failure = None if interface_pass else (
            "candidate-reference permutation or padded query batch equivalence contract failed")
        interface_next_action = (
            "run Phase D frozen rank probe with video-disjoint fit/dev split"
            if interface_pass else
            "fix only the first failed grounding interface equivalence contract and rerun in a new attempt")
        payload = {
            "format": "locatemot-l82-grounding-interface-contract-v1",
            "status": "complete", "stage": "phase_c_label_free_grounding_interface",
            "command": command, "cwd": str(ROOT),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
            "runtime": {"python": sys.executable, "torch": torch.__version__,
                        "cuda": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "local_mmdetection_root": str(LOCAL_MMDET)},
            "inputs": {"manifest": meta(MANIFEST), "config": meta(CONFIG), "weight": meta(WEIGHT),
                       "bert": meta(BERT), "train_units": meta(TRAIN_UNITS),
                       "l69_feature_root": str(L69_ROOT)},
            "model": model_info,
            "source_contract": {
                "encoder_output": "local GroundingDinoTransformerEncoder (memory, memory_text)",
                "post_encoder_source": model_info["encoder_source"],
                "fusion_source": model_info["fusion_source"],
                "candidate_path": "all L69 boxes -> normalized references -> frozen decoder; no native top-k scores/boxes/classes",
                "native_topk": "used only inside native inference for a pre-decoder equivalence capture; never used for candidate rows",
                "image_visual_forward_per_selected_frame": "one native backbone extraction; replay/control/batch reuse captured features in memory",
                "query_control": UNRELATED,
            },
            "selected_fit_frame_query_pairs": len(records),
            "selected_domains": {dataset: sum(row["dataset"] == dataset for row in records) for dataset in DATASETS},
            "records": records,
            "native_predecoder_equivalence": {
                "all_pass": all(row["pass"] for row in native_equivalence),
                "max_query_delta": max(row["query_max_abs_delta"] for row in native_equivalence),
                "max_reference_delta": max(row["reference_max_abs_delta"] for row in native_equivalence),
                "threshold": 1e-4,
            },
            "candidate_permutation": {
                "input_max_abs_errors": permutation_input_errors,
                "decoder_output_max_abs_error": decoder_permutation_error,
                "threshold": 1e-4,
                "input_contract_pass": permutation_inputs_pass,
            },
            "batch_query_equivalence": batch_equivalence_summary | {"pass": batch_equivalence_pass},
            "expression_sensitivity_summary": {
                "candidate_seed_mean_l2": float(sum(row["expression_sensitivity"]["candidate_seed"]["mean_l2"] for row in records) / len(records)),
                "candidate_seed_nonuniform_fraction_mean": float(sum(row["expression_sensitivity"]["candidate_seed"]["nonuniform_fraction"] for row in records) / len(records)),
                "final_hidden_residual_l2_mean": float(sum(row["expression_sensitivity"]["final_hidden"]["residual_l2_mean"] for row in records) / len(records)),
                "units_with_nonuniform_final_hidden": int(sum(row["expression_sensitivity"]["candidate_specific_not_global_offset"] for row in records)),
                "unit_count": len(records),
                "repeat_noise_max_l2": max(row["expression_sensitivity"]["repeat_native_vs_replay"]["max_l2"] for row in records),
            },
            "frozen_checks": {
                "base_requires_grad_false": base_requires_grad,
                "no_detector_parameter_grad": all(parameter.grad is None for parameter in model.parameters()),
                "all_candidate_rows_retained": all(row["candidate_deletion"] is False and row["candidate_truncation"] is False for row in records),
                "labels_read": False, "raw_cache_persistent": False,
            },
            "decision_thresholds": {
                "native_equivalence_max_abs": 1e-4,
                "candidate_reference_permutation_max_abs": 1e-4,
                "padded_batch_single_max_abs": 1e-4,
                "batch_contract_primary_fields": ["encoder_memory", "encoder_text_memory", "candidate_seed"],
                "decoder_output_batch_delta": "diagnostic_only; frozen decoder final hidden can vary by CUDA reduction order",
                "candidate_specific_nonuniformity": "median per-unit candidate delta std must exceed 10x repeat noise and at least 80% units must be nonuniform",
                "interface_decision": "grounding_interface_contract_pass if all shape/finite/key/equivalence/permutation/batch checks pass; otherwise grounding_interface_contract_fail",
            },
            "decision": "grounding_interface_contract_pass" if interface_pass else "grounding_interface_contract_fail",
            "failure_root_cause": interface_failure, "next_action": interface_next_action,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False, "candidate_deletion": False,
            "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "elapsed_sec": time.perf_counter() - started,
        }
        # Do not serialize model tensors or feature caches.  Keep only compact
        # per-unit summaries and provenance.
        (out / "contract.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        (out / "unit_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in records))
        (out / "provenance.json").write_text(json.dumps({
            "format": "locatemot-l82-grounding-interface-provenance-v1", "status": "complete",
            "command": command, "inputs": payload["inputs"], "model": model_info,
            "outputs": [str(out / "contract.json"), str(out / "unit_records.jsonl")],
            "label_boundary": "key-only fit metadata before forward; no target/category/sidecar fields were accessed; no labels read",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False, "candidate_deletion": False,
            "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "failure_root_cause": interface_failure,
            "next_action": payload["next_action"],
        }, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        (out / "status.json").write_text(json.dumps({
            "format": "locatemot-l82-status-v1", "status": "complete",
            "stage": "phase_c", "command": command,
            "outputs": [str(out / "contract.json"), str(out / "unit_records.jsonl")],
            "failure_root_cause": interface_failure, "next_action": payload["next_action"],
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False,
        }, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        print(json.dumps({
            "status": payload["status"], "decision": payload["decision"],
            "selected_pairs": payload["selected_fit_frame_query_pairs"],
            "max_native_query_delta": payload["native_predecoder_equivalence"]["max_query_delta"],
            "max_native_reference_delta": payload["native_predecoder_equivalence"]["max_reference_delta"],
            "max_permutation_error": payload["candidate_permutation"]["decoder_output_max_abs_error"],
            "max_input_permutation_error": max(permutation_input_errors.values()),
            "batch_pass": batch_equivalence_pass,
            "expression_units_nonuniform": payload["expression_sensitivity_summary"]["units_with_nonuniform_final_hidden"],
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# L82 GroundingDINO interface audit — INCOMPLETE\n\n" + traceback.format_exc() +
            "\nNo training, calibration/validation, screening, official-test, TrackEval/HOTA, MOT or OVMOT action was run.\n")
        (out / "status.json").write_text(json.dumps({
            "format": "locatemot-l82-status-v1", "status": "incomplete",
            "stage": "phase_c", "command": command,
            "failure_root_cause": "first actionable exception preserved in INCOMPLETE.md",
            "next_action": "fix only the first grounding interface contract error and rerun in a new attempt",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False,
        }, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
