"""Differentiable, query-tiled GroundingDINO runtime for L88.

The frozen backbone is run once per unique frame to create the compact
query-independent ``pre_transformer`` cache.  Expressions then replay the
canonical local GroundingDINO text path, run the adapted encoder, and use the
complete L69 candidate rows in the fixed-reference decoder layer zero.  Native
``pre_decoder``/top-k/refinement branches are intentionally never called.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
LOCAL_MMDET = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0").resolve()
CONFIG = LOCAL_MMDET / "configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
WEIGHT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth").resolve()
BERT = Path("/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594").resolve()
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
MMDET_REFERENCE = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    path = path.resolve()
    item: dict[str, Any] = {"path": str(path), "exists": path.exists(),
                            "bytes": path.stat().st_size if path.exists() else None,
                            "mtime_ns": path.stat().st_mtime_ns if path.exists() else None}
    if include_hash and path.is_file():
        item["sha256"] = sha256_file(path)
    return item


def _ensure_mmdet_path() -> None:
    if str(LOCAL_MMDET) not in sys.path:
        sys.path.insert(0, str(LOCAL_MMDET))


def build_groundingdino(device: torch.device) -> tuple[Any, dict[str, Any]]:
    """Build the exact local GroundingDINO model and freeze all base params."""
    _ensure_mmdet_path()
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
    model.to(device)
    model.eval()
    model.cfg = cfg
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "config": file_meta(CONFIG),
        "weight": file_meta(WEIGHT),
        "bert": file_meta(BERT, include_hash=False),
        "mmdetection_root": str(LOCAL_MMDET),
        "mmdetection_reference": MMDET_REFERENCE,
        "checkpoint_missing_keys": list(loaded.get("missing_keys", [])) if isinstance(loaded, dict) else [],
        "checkpoint_unexpected_keys": list(loaded.get("unexpected_keys", [])) if isinstance(loaded, dict) else [],
        "checkpoint_warning_expected": "language_model...position_ids load warning may be present; strict=False is the verified local contract",
        "device": str(device),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "parameters_frozen": all(not value.requires_grad for value in model.parameters()),
    }


def _clone_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clone_cpu(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _move_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, tuple):
        return tuple(_move_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_device(item, device) for key, item in value.items()}
    return value


def _repeat_batch(value: Any, count: int, device: torch.device) -> Any:
    value = _move_device(value, device)
    if not torch.is_tensor(value):
        return value
    if value.ndim == 0:
        return value
    if int(value.shape[0]) == 1:
        return value.expand(count, *value.shape[1:])
    if int(value.shape[0]) == count:
        return value
    raise AssertionError(f"L88 cache batch dimension drift: {tuple(value.shape)} count={count}")


def _meta_json(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _meta_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_meta_json(item) for item in value]
    return value


def _make_pipeline(model: Any) -> Any:
    _ensure_mmdet_path()
    from mmdet.apis.inference import get_test_pipeline_cfg
    from mmengine.dataset import Compose
    cfg = model.cfg.copy()
    return Compose(get_test_pipeline_cfg(cfg))


def capture_encoder_inputs(model: Any, image_path: Path, device: torch.device) -> dict[str, Any]:
    """Run only image preprocessing/backbone/pre-transformer for one frame."""
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    pipeline = _make_pipeline(model)
    data = pipeline({"img_path": str(image_path), "img_id": 0})
    if data is None or "inputs" not in data or "data_samples" not in data:
        raise AssertionError("MMDetection test pipeline did not return inputs/data_samples")
    prepared = {"inputs": [data["inputs"]], "data_samples": [data["data_samples"]]}
    processed = model.data_preprocessor(prepared, False)
    with torch.no_grad():
        features = model.extract_feat(processed["inputs"])
        enc_inputs, _decoder_inputs = model.pre_transformer(features, processed["data_samples"])
    if not isinstance(enc_inputs, dict):
        raise AssertionError("GroundingDINO pre_transformer did not return a dict")
    required = {"feat", "feat_mask", "feat_pos", "spatial_shapes", "level_start_index", "valid_ratios"}
    if not required.issubset(enc_inputs):
        raise AssertionError(f"L88 encoder input keys missing: {sorted(required - set(enc_inputs))}")
    result = {key: _clone_cpu(enc_inputs[key]) for key in required}
    sample = processed["data_samples"][0]
    metainfo = _meta_json(getattr(sample, "metainfo", {}))
    result["image_path"] = str(image_path)
    result["metainfo"] = metainfo
    result["format"] = "locatemot-l88-query-independent-encoder-input-v1"
    result["labels_in_cache"] = False
    result["query_independent"] = True
    result["candidate_deletion"] = False
    result["candidate_truncation"] = False
    for key in required:
        value = result[key]
        if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value.float()).all()):
            raise FloatingPointError(f"nonfinite cached encoder input {key}")
    del pipeline, data, prepared, processed, features, enc_inputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cache_key(dataset: str, video: str, frame: int) -> str:
    return f"{dataset}|{video}|{int(frame):06d}"


def cache_filename(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:24] + ".pt"


def save_cache_item(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite cache item {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(item, temporary)
    os.replace(temporary, path)
    return {"file": str(path.name), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_cache_item(cache_root: Path, key: str, device: torch.device) -> dict[str, Any]:
    manifest = cache_root / "manifest.jsonl"
    filename = None
    if manifest.is_file():
        with manifest.open() as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    if str(item.get("cache_key")) == str(key):
                        filename = str(item["file"])
                        break
    if filename is None:
        filename = cache_filename(key)
    path = cache_root / filename
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("labels_in_cache") or not item.get("query_independent"):
        raise AssertionError(f"invalid L88 cache flags: {path}")
    required = {"feat", "feat_mask", "feat_pos", "spatial_shapes", "level_start_index", "valid_ratios"}
    if not required.issubset(item):
        raise AssertionError(f"cache item missing keys: {sorted(required - set(item))}")
    for name in required:
        value = item[name]
        if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value.float()).all()):
            raise FloatingPointError(f"nonfinite cache item {key}:{name}")
    return item


def make_text_batch(model: Any, sentences: list[str], device: torch.device) -> tuple[dict[str, Any], list[str], list[Any]]:
    if not sentences:
        raise ValueError("empty L88 expression tile")
    old_pad = bool(model.language_model.pad_to_max)
    model.language_model.pad_to_max = True
    try:
        captions: list[str] = []
        token_maps: list[Any] = []
        for sentence in sentences:
            token_map, caption, _positive_map, _entities = model.get_tokens_positive_and_prompts(
                str(sentence), True, None, None)
            captions.append(caption)
            token_maps.append(token_map)
        with torch.no_grad():
            text_dict = model.language_model(captions)
            if model.text_feat_map is not None:
                text_dict["embedded"] = model.text_feat_map(text_dict["embedded"])
        text_dict = {key: _move_device(value, device) for key, value in text_dict.items()}
        return text_dict, captions, token_maps
    finally:
        model.language_model.pad_to_max = old_pad


def _normalise_boxes(boxes: torch.Tensor, meta: dict[str, Any], device: torch.device) -> torch.Tensor:
    from locatemot.models.l82_grounding_reference import boxes_xyxy_to_normalized
    image_shape = meta.get("img_shape")
    scale_factor = meta.get("scale_factor")
    if image_shape is None or scale_factor is None:
        raise AssertionError(f"L88 cache metadata missing img_shape/scale_factor: {sorted(meta)}")
    shape = tuple(int(value) for value in image_shape[:2])
    return boxes_xyxy_to_normalized(boxes.to(device), shape, scale_factor)


def _fixed_reference_z1(model: Any, memory: torch.Tensor, memory_mask: torch.Tensor | None,
                        spatial_shapes: torch.Tensor, starts: torch.Tensor,
                        valid_ratios: torch.Tensor, memory_text: torch.Tensor,
                        text_mask: torch.Tensor, boxes_norm: torch.Tensor) -> torch.Tensor:
    """Run the original-content fixed-reference decoder layer zero only."""
    from locatemot.models.l82_grounding_reference import boxes_to_reference_points, pool_memory_by_box
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding

    qcount = int(memory.shape[0])
    refs = boxes_to_reference_points(boxes_norm)
    seeds = []
    for index in range(qcount):
        seed, _audit = pool_memory_by_box(
            memory[index:index + 1], spatial_shapes, starts, boxes_norm,
            None if memory_mask is None else memory_mask[index:index + 1], grid_size=4)
        seeds.append(seed)
    visual_seed = torch.stack(seeds, dim=0)
    reference_encoding = coordinate_to_encoding(refs.unsqueeze(0), num_feats=128)
    reference_position = model.decoder.ref_point_head(reference_encoding).squeeze(0)
    seed = visual_seed + reference_position.unsqueeze(0)
    reference_batch = refs.unsqueeze(0).expand(qcount, -1, -1)
    reference_input = reference_batch[:, :, None] * torch.cat((valid_ratios, valid_ratios), dim=-1)[:, None]
    query_sine_embed = coordinate_to_encoding(reference_input[:, :, 0, :])
    query_pos = model.decoder.ref_point_head(query_sine_embed)
    layer = model.decoder.layers[0]
    query = layer(
        seed,
        query_pos=query_pos,
        value=memory,
        key_padding_mask=memory_mask,
        self_attn_mask=None,
        spatial_shapes=spatial_shapes,
        level_start_index=starts,
        valid_ratios=valid_ratios,
        reference_points=reference_input,
        memory_text=memory_text,
        text_attention_mask=~text_mask.bool(),
    )
    result = model.decoder.norm(query)
    if result.shape != seed.shape or not bool(torch.isfinite(result.float()).all()):
        raise AssertionError(f"L88 Z1 output shape/finite drift: {tuple(result.shape)}")
    return result


def forward_l88_z1(model: Any, cache_item: dict[str, Any], boxes: torch.Tensor,
                    sentences: list[str], device: torch.device, *, query_tile: int = 4,
                    autocast_bf16: bool = False,
                    prepared_text: tuple[dict[str, Any], list[str], list[Any]] | None = None) -> dict[str, Any]:
    """Forward a complete expression tile through adapted encoder and Z1."""
    if int(query_tile) != len(sentences):
        raise AssertionError("forward_l88_z1 receives exactly one query tile")

    # The local MMCV multi-scale deformable-attention kernel is not invariant
    # to the leading query batch size: a batched replay can produce a
    # materially different Z1 from the immutable L85 one-query reference
    # path.  Keep the public tile contract, but evaluate each expression in a
    # separate one-query call and stack the complete results.  This preserves
    # row/query order and the differentiable LoRA path while making the
    # zero-update parity contract meaningful.
    if len(sentences) > 1:
        parts = [forward_l88_z1(
            model, cache_item, boxes, [sentence], device,
            query_tile=1, autocast_bf16=autocast_bf16)
                 for sentence in sentences]
        memory_masks = [part.get("memory_mask") for part in parts]
        if all(torch.is_tensor(value) for value in memory_masks):
            memory_mask = torch.cat([value for value in memory_masks if torch.is_tensor(value)], dim=0)
        elif any(value is not None for value in memory_masks):
            raise AssertionError("inconsistent L88 memory-mask replay contract")
        else:
            memory_mask = None
        result: dict[str, Any] = {
            "z1": torch.cat([part["z1"] for part in parts], dim=0),
            "memory_text": torch.cat([part["memory_text"] for part in parts], dim=0),
            "memory": torch.cat([part["memory"] for part in parts], dim=0),
            "memory_mask": memory_mask,
            "text_token_mask": torch.cat([part["text_token_mask"] for part in parts], dim=0),
            "captions": [caption for part in parts for caption in part["captions"]],
            "token_maps": [token_map for part in parts for token_map in part["token_maps"]],
            "cache_key": cache_item.get("cache_key"),
            "candidate_count": int(boxes.shape[0]),
            "candidate_deletion": False,
            "candidate_truncation": False,
            "query_execution": "serial_one_query_for_exact_L85_parity",
        }
        for name, value in (("z1", result["z1"]), ("memory_text", result["memory_text"]),
                            ("memory", result["memory"])):
            if not bool(torch.isfinite(value.float()).all()):
                raise FloatingPointError(f"nonfinite serial L88 {name}")
        if result["memory_mask"] is not None and not bool(torch.isfinite(result["memory_mask"].float()).all()):
            raise FloatingPointError("nonfinite serial L88 memory_mask")
        return result
    if prepared_text is None:
        text_dict, captions, token_maps = make_text_batch(model, sentences, device)
    else:
        if len(sentences) != 1 or len(prepared_text[1]) != 1 or len(prepared_text[2]) != 1:
            raise AssertionError("prepared L88 text is only valid for one sentence")
        text_dict, captions, token_maps = prepared_text
        if any(torch.is_tensor(value) and value.device != device for value in text_dict.values()):
            raise AssertionError("prepared L88 text device drift")
    qcount = len(sentences)
    enc = {key: _repeat_batch(cache_item[key], qcount, device)
           for key in ("feat", "feat_mask", "feat_pos", "valid_ratios")}
    enc["spatial_shapes"] = _move_device(cache_item["spatial_shapes"], device)
    enc["level_start_index"] = _move_device(cache_item["level_start_index"], device)
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                         enabled=bool(autocast_bf16 and device.type == "cuda"))
    with amp:
        encoded = model.forward_encoder(**enc, text_dict=text_dict)
        z1 = _fixed_reference_z1(
            model, encoded["memory"], encoded.get("memory_mask"), encoded["spatial_shapes"],
            enc["level_start_index"], encoded["memory"].new_tensor(enc["valid_ratios"]),
            encoded["memory_text"], encoded["text_token_mask"],
            _normalise_boxes(boxes, cache_item["metainfo"], device),
        )
    if tuple(z1.shape[:2]) != (qcount, int(boxes.shape[0])):
        raise AssertionError(f"L88 Z1 query/candidate shape drift: {tuple(z1.shape)}")
    for name, value in (("z1", z1), ("memory_text", encoded["memory_text"]), ("memory", encoded["memory"])):
        if not bool(torch.isfinite(value.float()).all()):
            raise FloatingPointError(f"nonfinite L88 {name}")
    return {
        "z1": z1,
        "memory_text": encoded["memory_text"],
        "memory": encoded["memory"],
        "memory_mask": encoded.get("memory_mask"),
        "text_dict": text_dict,
        "captions": captions,
        "token_maps": token_maps,
        "text_token_mask": encoded["text_token_mask"],
        "cache_key": cache_item.get("cache_key"),
        "candidate_count": int(boxes.shape[0]),
        "candidate_deletion": False,
        "candidate_truncation": False,
    }


class L88GroundingRuntime:
    """Lifecycle wrapper that keeps one frozen/adapted detector on one device."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model, self.model_info = build_groundingdino(device)
        self.injector = None
        self.cache_reads = 0
        self.cache_builds = 0

    def inject(self, injector: Any) -> None:
        self.injector = injector

    def cache_frame(self, image_path: Path) -> dict[str, Any]:
        item = capture_encoder_inputs(self.model, image_path, self.device)
        self.cache_builds += 1
        return item

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = [
    "BERT", "CONFIG", "IMAGE_ROOT", "L69_ROOT", "LOCAL_MMDET", "MANIFEST", "MANIFEST_SHA",
    "MMDET_REFERENCE", "ROOT", "THREAD", "WEIGHT", "L88GroundingRuntime", "build_groundingdino",
    "cache_filename", "cache_key", "capture_encoder_inputs", "file_meta", "forward_l88_z1",
    "load_cache_item", "make_text_batch", "save_cache_item", "sha256_file",
]
