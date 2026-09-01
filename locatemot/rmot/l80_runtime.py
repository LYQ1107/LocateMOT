"""L80 private raw-image CLIP runtime and differentiable ROI construction."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
CLIP_WEIGHT = Path("/home/lwr/.cache/clip/ViT-B-16.pt").resolve()
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VISUAL_TAPS = (3, 7, 11)
IMAGE_SIZE = 224
ROI_GRID = 4
CONTEXT_GRID = 2
SCALE_COUNT = len(VISUAL_TAPS)
TOKENS_PER_SCALE = ROI_GRID * ROI_GRID + CONTEXT_GRID * CONTEXT_GRID + 1
REGION_TOKEN_COUNT = SCALE_COUNT * TOKENS_PER_SCALE


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def letterbox_geometry(width: int, height: int, size: int = IMAGE_SIZE) -> dict[str, float | int]:
    scale = min(float(size) / max(1, int(width)), float(size) / max(1, int(height)))
    resized_width = max(1, int(round(int(width) * scale)))
    resized_height = max(1, int(round(int(height) * scale)))
    return {
        "original_width": int(width), "original_height": int(height), "scale": float(scale),
        "resized_width": resized_width, "resized_height": resized_height,
        "offset_x": (int(size) - resized_width) // 2, "offset_y": (int(size) - resized_height) // 2,
        "output_size": int(size),
    }


def padding_box(box: list[float] | tuple[float, ...], width: int, height: int, padding: float = 0.10) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    result = [
        max(0.0, x1 - padding * bw), max(0.0, y1 - padding * bh),
        min(float(width), x2 + padding * bw), min(float(height), y2 + padding * bh),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"empty padded box: {box}")
    return result


def map_boxes_to_letterbox(boxes: torch.Tensor, image_size: tuple[int, int], padding: float = 0.10) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    width, height = int(image_size[0]), int(image_size[1])
    geometry = letterbox_geometry(width, height)
    scale = float(geometry["scale"])
    ox, oy, size = float(geometry["offset_x"]), float(geometry["offset_y"]), float(geometry["output_size"])
    values, details = [], []
    for raw in boxes.float().cpu().tolist():
        padded = padding_box(raw, width, height, padding)
        resized = [padded[0] * scale + ox, padded[1] * scale + oy,
                   padded[2] * scale + ox, padded[3] * scale + oy]
        normalized = [max(0.0, min(1.0, x / size)) for x in resized]
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            raise AssertionError(f"empty normalized ROI for box {raw}")
        values.append(normalized)
        details.append({"original_box": raw, "padded_clipped_box": padded,
                        "resized_box": resized, "normalized_box": normalized,
                        "padding": padding, "empty": False})
    result = torch.tensor(values, dtype=torch.float32, device=boxes.device)
    if result.shape != (int(boxes.shape[0]), 4) or not bool(torch.isfinite(result).all()):
        raise AssertionError("L80 box mapping shape/finite contract")
    return result, details


def preprocess_frame(path: Path, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        geometry = letterbox_geometry(width, height)
        resized = image.resize((int(geometry["resized_width"]), int(geometry["resized_height"])), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), tuple(int(round(x * 255.0)) for x in CLIP_MEAN))
        canvas.paste(resized, (int(geometry["offset_x"]), int(geometry["offset_y"])))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
    value = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(CLIP_STD, dtype=torch.float32)[:, None, None]
    value = ((value - mean) / std).unsqueeze(0).to(device=device, dtype=dtype)
    geometry.update({"image_path": str(path.resolve()), "pixel_shape": list(value.shape),
                     "preprocess": "letterbox_to_224_bicubic_clip_normalization"})
    return value, geometry


def load_clip(device: torch.device) -> Any:
    if not CLIP_WEIGHT.is_file():
        raise FileNotFoundError(CLIP_WEIGHT)
    if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
        raise AssertionError("L80 CLIP SHA mismatch")
    import clip
    model, _ = clip.load(str(CLIP_WEIGHT), device=device, jit=False)
    model.to(device=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def visual_pyramid(model: Any, pixel: torch.Tensor) -> torch.Tensor:
    """Frozen CLIP taps with shape ``[3,B,196,768]`` before projection."""
    visual = model.visual
    x = visual.conv1(pixel.to(dtype=visual.conv1.weight.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = visual.ln_pre(torch.cat((cls, x), dim=1) + visual.positional_embedding.to(x.dtype))
    x = x.permute(1, 0, 2)
    taps = []
    use_amp = x.device.type == "cuda"
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.autocast(device_type="cpu", enabled=False)
    with amp:
        for index, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            if index in VISUAL_TAPS:
                taps.append(x.permute(1, 0, 2)[:, 1:, :])
    if len(taps) != SCALE_COUNT:
        raise AssertionError(f"CLIP tap count drift: {len(taps)}")
    result = torch.stack(taps, dim=0).float()
    if tuple(result.shape) != (SCALE_COUNT, int(pixel.shape[0]), 196, 768):
        raise AssertionError(f"unexpected L80 visual pyramid {tuple(result.shape)}")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite L80 visual pyramid")
    return result.detach()


def _context_boxes(boxes: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    half = (boxes[:, 2:] - boxes[:, :2]) * 0.5 * float(factor)
    return torch.cat((center - half, center + half), dim=-1).clamp(0.0, 1.0)


def _sample_map(feature_map: torch.Tensor, boxes: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Bilinear ``align_corners=False`` samples, returning ``[N,G*G,C]``."""
    if feature_map.ndim != 2 or feature_map.shape != (196, 768):
        raise AssertionError(f"feature map shape {tuple(feature_map.shape)}")
    count = int(boxes.shape[0])
    if count == 0:
        return feature_map.new_zeros((0, grid_size * grid_size, feature_map.shape[-1]))
    fractions = (torch.arange(grid_size, device=boxes.device, dtype=boxes.dtype) + 0.5) / float(grid_size)
    gy, gx = torch.meshgrid(fractions, fractions, indexing="ij")
    x = boxes[:, 0, None, None] + (boxes[:, 2] - boxes[:, 0])[:, None, None] * gx[None, :, :]
    y = boxes[:, 1, None, None] + (boxes[:, 3] - boxes[:, 1])[:, None, None] * gy[None, :, :]
    grid = torch.stack((x, y), dim=-1) * 2.0 - 1.0
    fmap = feature_map.transpose(0, 1).reshape(1, 768, 14, 14).expand(count, -1, -1, -1)
    sampled = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled.permute(0, 2, 3, 1).reshape(count, grid_size * grid_size, 768)


def candidate_region_tokens(pyramid: torch.Tensor, boxes_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ROI, context and scene tokens for every candidate row."""
    if pyramid.ndim != 4 or tuple(pyramid.shape[1:]) != (1, 196, 768):
        raise AssertionError(f"L80 pyramid must be [3,1,196,768], got {tuple(pyramid.shape)}")
    count = int(boxes_norm.shape[0])
    context = _context_boxes(boxes_norm)
    per_level, counts = [], []
    for level in range(SCALE_COUNT):
        fmap = pyramid[level, 0]
        roi = _sample_map(fmap, boxes_norm, ROI_GRID)
        ctx = _sample_map(fmap, context, CONTEXT_GRID)
        scene = fmap.mean(dim=0, keepdim=True).expand(count, -1, -1)
        per_level.append(torch.cat((roi, ctx, scene), dim=1))
        counts.append({"roi_tokens": int(roi.shape[1]), "context_tokens": int(ctx.shape[1]), "scene_tokens": 1,
                       "roi_valid_samples": int(roi.shape[0] * roi.shape[1]),
                       "context_valid_samples": int(ctx.shape[0] * ctx.shape[1])})
    tokens = torch.cat(per_level, dim=1).float().detach()
    expected = (count, REGION_TOKEN_COUNT, 768)
    if tuple(tokens.shape) != expected or not bool(torch.isfinite(tokens).all()):
        raise AssertionError(f"L80 ROI token contract {tuple(tokens.shape)} != {expected}")
    return tokens, {"levels": counts, "region_token_count": REGION_TOKEN_COUNT,
                    "roi_grid": ROI_GRID, "context_grid": CONTEXT_GRID,
                    "context_scale": 1.5, "padding": 0.10,
                    "candidate_rows": count, "roi_empty_rows": 0,
                    "candidate_deletion": False, "candidate_truncation": False}


class FrameFeatureCache:
    """Bounded process-local cache; it is never serialized as an artifact."""
    def __init__(self, max_items: int = 16) -> None:
        self.max_items = int(max_items)
        self._items: OrderedDict[tuple[str, int], tuple[torch.Tensor, dict[str, Any]]] = OrderedDict()
        self.visual_forward_count = 0

    def get(self, key: tuple[str, int]) -> tuple[torch.Tensor, dict[str, Any]] | None:
        value = self._items.pop(key, None)
        if value is not None:
            self._items[key] = value
        return value

    def put(self, key: tuple[str, int], value: tuple[torch.Tensor, dict[str, Any]]) -> None:
        self._items.pop(key, None)
        self._items[key] = (value[0].detach(), dict(value[1]))
        while len(self._items) > self.max_items:
            _key, old = self._items.popitem(last=False)
            del old

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def raw_inputs_for_unit(model: Any, batch: Any, device: torch.device, cache: FrameFeatureCache) -> dict[str, Any]:
    key = (str(batch.video), int(batch.frame_id))
    cached = cache.get(key)
    if cached is None:
        pixel, geometry = preprocess_frame(Path(batch.image_path), device, model.visual.conv1.weight.dtype)
        pyramid = visual_pyramid(model, pixel)
        cache.visual_forward_count += 1
        cached = (pyramid, geometry)
        cache.put(key, cached)
        del pixel
    pyramid, geometry = cached
    boxes_norm, mapping = map_boxes_to_letterbox(batch.boxes, batch.image_size)
    region_tokens, roi_audit = candidate_region_tokens(pyramid, boxes_norm.to(pyramid.device))
    text_tokens = batch.text_tokens.float().to(device=device).clone()
    text_mask = batch.text_mask.bool().to(device=device).clone()
    if not bool(torch.isfinite(text_tokens).all()) or not bool(text_mask.any()):
        raise AssertionError(f"invalid text cache values for {batch.unit_key}")
    if region_tokens.shape[0] != batch.candidate_count:
        raise AssertionError(f"raw row count drift for {batch.unit_key}")
    return {
        "visual_tokens": region_tokens.clone(), "text_tokens": text_tokens,
        "text_mask": text_mask, "boxes_norm": boxes_norm.detach().clone(),
        "box_mapping": mapping, "geometry": geometry, "pyramid_shape": list(pyramid.shape),
        "roi_audit": roi_audit, "visual_forward_count": cache.visual_forward_count,
        "frame_cache_items": len(cache), "raw_cache_persistent": False,
    }


__all__ = [
    "CLIP_SHA256", "CLIP_WEIGHT", "CONTEXT_GRID", "FrameFeatureCache", "IMAGE_SIZE",
    "REGION_TOKEN_COUNT", "SCALE_COUNT", "TOKENS_PER_SCALE", "VISUAL_TAPS", "candidate_region_tokens",
    "letterbox_geometry", "load_clip", "map_boxes_to_letterbox", "preprocess_frame",
    "raw_inputs_for_unit", "sha256_file", "visual_pyramid",
]
