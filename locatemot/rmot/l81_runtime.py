"""L81 adapter around the audited L80 frozen CLIP runtime.

The wrapper changes only the returned representation: L81 receives the three
full-frame CLIP taps, the existing complete-candidate local tokens, and the
normalized native boxes.  The process-local cache is bounded and never
serialized.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from locatemot.rmot.l80_runtime import (
    CLIP_SHA256,
    CLIP_WEIGHT,
    FrameFeatureCache,
    candidate_region_tokens,
    load_clip,
    map_boxes_to_letterbox,
    preprocess_frame,
    sha256_file,
    visual_pyramid,
)


def raw_inputs_for_l81(clip_model: Any, batch: Any, device: torch.device,
                       cache: FrameFeatureCache) -> dict[str, Any]:
    """Build one complete L81 frame/query representation without labels."""
    key = (str(batch.video), int(batch.frame_id))
    cached = cache.get(key)
    if cached is None:
        pixel, geometry = preprocess_frame(
            Path(batch.image_path), device, clip_model.visual.conv1.weight.dtype)
        pyramid = visual_pyramid(clip_model, pixel)
        cache.visual_forward_count += 1
        cached = (pyramid, geometry)
        cache.put(key, cached)
        del pixel
    pyramid, geometry = cached
    boxes_norm, mapping = map_boxes_to_letterbox(batch.boxes, batch.image_size)
    local_tokens, roi_audit = candidate_region_tokens(
        pyramid, boxes_norm.to(device=pyramid.device))
    # Inference-mode tensors are cloned before entering the trainable L81 head.
    visual_copy = pyramid.detach().clone()
    local_copy = local_tokens.detach().clone()
    text_tokens = batch.text_tokens.float().to(device=device).clone()
    text_mask = batch.text_mask.bool().to(device=device).clone()
    boxes_copy = boxes_norm.to(device=device).detach().clone()
    if visual_copy.shape != (3, 1, 196, 768):
        raise AssertionError(f"L81 full visual shape drift: {tuple(visual_copy.shape)}")
    if local_copy.shape != (batch.candidate_count, 63, 768):
        raise AssertionError(f"L81 local visual shape drift: {tuple(local_copy.shape)}")
    if boxes_copy.shape != (batch.candidate_count, 4):
        raise AssertionError("L81 box shape drift")
    if not bool(torch.isfinite(visual_copy).all() and torch.isfinite(local_copy).all() and
                torch.isfinite(text_tokens).all() and torch.isfinite(boxes_copy).all()):
        raise FloatingPointError(f"nonfinite raw L81 input: {batch.unit_key}")
    if not bool(text_mask.any()):
        raise AssertionError(f"empty L81 text mask: {batch.unit_key}")
    return {
        "visual_pyramid": visual_copy,
        "local_tokens": local_copy,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "boxes_norm": boxes_copy,
        "box_mapping": mapping,
        "geometry": geometry,
        "pyramid_shape": list(visual_copy.shape),
        "local_shape": list(local_copy.shape),
        "roi_audit": roi_audit,
        "visual_forward_count": int(cache.visual_forward_count),
        "frame_cache_items": len(cache),
        "raw_cache_persistent": False,
    }


__all__ = [
    "CLIP_SHA256", "CLIP_WEIGHT", "FrameFeatureCache", "candidate_region_tokens",
    "load_clip", "map_boxes_to_letterbox", "preprocess_frame", "raw_inputs_for_l81",
    "sha256_file", "visual_pyramid",
]
