"""R1 high-resolution ROI interface over the immutable L80 CLIP runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from locatemot.rmot.l80_runtime import (
    CLIP_SHA256, CLIP_WEIGHT, FrameFeatureCache, SCALE_COUNT, _context_boxes,
    _sample_map, load_clip, map_boxes_to_letterbox, preprocess_frame, sha256_file,
    visual_pyramid,
)


R1_ROI_GRID = 8
R1_CONTEXT_GRID = 4
R1_TOKENS_PER_SCALE = R1_ROI_GRID * R1_ROI_GRID + R1_CONTEXT_GRID * R1_CONTEXT_GRID + 1
R1_REGION_TOKEN_COUNT = SCALE_COUNT * R1_TOKENS_PER_SCALE


def candidate_region_tokens_r1(pyramid: torch.Tensor, boxes_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if pyramid.ndim != 4 or tuple(pyramid.shape[1:]) != (1, 196, 768):
        raise AssertionError(f"R1 pyramid shape drift: {tuple(pyramid.shape)}")
    count = int(boxes_norm.shape[0])
    context = _context_boxes(boxes_norm, factor=1.5)
    levels = []
    audits = []
    for level in range(SCALE_COUNT):
        fmap = pyramid[level, 0]
        roi = _sample_map(fmap, boxes_norm, R1_ROI_GRID)
        ctx = _sample_map(fmap, context, R1_CONTEXT_GRID)
        scene = fmap.mean(dim=0, keepdim=True).expand(count, -1, -1)
        levels.append(torch.cat((roi, ctx, scene), dim=1))
        audits.append({
            "level": int(level), "roi_tokens": int(roi.shape[1]),
            "context_tokens": int(ctx.shape[1]), "scene_tokens": 1,
            "roi_samples": int(count * roi.shape[1]), "context_samples": int(count * ctx.shape[1]),
        })
    result = torch.cat(levels, dim=1).float().detach()
    expected = (count, R1_REGION_TOKEN_COUNT, 768)
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise AssertionError(f"R1 region token contract {tuple(result.shape)} != {expected}")
    return result, {
        "levels": audits, "region_token_count": R1_REGION_TOKEN_COUNT,
        "roi_grid": R1_ROI_GRID, "context_grid": R1_CONTEXT_GRID,
        "context_scale": 1.5, "padding": 0.10,
        "candidate_rows": count, "candidate_deletion": False, "candidate_truncation": False,
    }


def raw_inputs_for_unit_r1(model: Any, batch: Any, device: torch.device,
                           cache: FrameFeatureCache) -> dict[str, Any]:
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
    region_tokens, roi_audit = candidate_region_tokens_r1(pyramid, boxes_norm.to(pyramid.device))
    text_tokens = batch.text_tokens.float().to(device=device).clone()
    text_mask = batch.text_mask.bool().to(device=device).clone()
    if not bool(torch.isfinite(text_tokens).all()) or not bool(text_mask.any()):
        raise AssertionError(f"R1 text contract failed: {batch.unit_key}")
    if int(region_tokens.shape[0]) != batch.candidate_count:
        raise AssertionError(f"R1 candidate count drift: {batch.unit_key}")
    return {
        "visual_tokens": region_tokens.clone(), "text_tokens": text_tokens,
        "text_mask": text_mask, "boxes_norm": boxes_norm.detach().clone(),
        "box_mapping": mapping, "geometry": geometry, "pyramid_shape": list(pyramid.shape),
        "roi_audit": roi_audit, "visual_forward_count": cache.visual_forward_count,
        "frame_cache_items": len(cache), "raw_cache_persistent": False,
    }


__all__ = [
    "CLIP_SHA256", "CLIP_WEIGHT", "FrameFeatureCache", "R1_CONTEXT_GRID", "R1_REGION_TOKEN_COUNT",
    "R1_ROI_GRID", "R1_TOKENS_PER_SCALE", "candidate_region_tokens_r1", "load_clip",
    "raw_inputs_for_unit_r1", "sha256_file",
]
