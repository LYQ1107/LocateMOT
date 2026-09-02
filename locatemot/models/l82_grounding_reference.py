"""Small, dependency-light helpers for the L82 GroundingDINO probe.

The helpers do not construct a detector and do not perform proposal selection.
They turn every immutable L69 box into a normalized reference and pool the
frozen post-encoder visual memory at that box.  The actual decoder module is
owned by the verified local MMDetection runtime and is passed in by the audit.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def expand_scale_factor(scale_factor: Any) -> torch.Tensor:
    """Return ``[sx, sy, sx, sy]`` and reject ambiguous metadata."""
    value = torch.as_tensor(scale_factor, dtype=torch.float32).flatten()
    if value.numel() == 2:
        value = value.repeat(2)
    if value.numel() != 4:
        raise ValueError(f"scale_factor must have length 2 or 4, got {value.numel()}")
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError(f"invalid scale_factor: {value.tolist()}")
    return value


def boxes_xyxy_to_normalized(
    boxes_xyxy: torch.Tensor,
    image_shape_hw: tuple[int, int] | list[int],
    scale_factor: Any,
) -> torch.Tensor:
    """Map original-pixel xyxy boxes into the processed image frame."""
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError(f"expected boxes [N,4], got {tuple(boxes_xyxy.shape)}")
    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid processed image shape: {image_shape_hw}")
    scale = expand_scale_factor(scale_factor).to(device=boxes_xyxy.device, dtype=torch.float32)
    boxes = boxes_xyxy.float() * scale
    divisor = boxes.new_tensor([float(w), float(h), float(w), float(h)])
    normalized = (boxes / divisor).clamp(0.0, 1.0)
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("nonfinite normalized candidate boxes")
    if bool((normalized[:, 2:] <= normalized[:, :2]).any()):
        raise ValueError("candidate box became empty after native rescaling")
    return normalized


def boxes_to_reference_points(boxes_normalized: torch.Tensor) -> torch.Tensor:
    """Convert normalized xyxy boxes to GroundingDINO cxcywh references."""
    if boxes_normalized.ndim != 2 or boxes_normalized.shape[-1] != 4:
        raise ValueError("expected normalized xyxy boxes [N,4]")
    x1, y1, x2, y2 = boxes_normalized.unbind(-1)
    reference = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5,
                             (x2 - x1).clamp_min(1e-5),
                             (y2 - y1).clamp_min(1e-5)), dim=-1)
    if not bool(torch.isfinite(reference).all()) or bool((reference <= 0).any()):
        raise ValueError("invalid candidate reference points")
    return reference.clamp(1e-4, 1.0 - 1e-4)


def pool_memory_by_box(
    memory: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    boxes_normalized: torch.Tensor,
    memory_mask: torch.Tensor | None = None,
    grid_size: int = 4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pool all levels of flattened encoder memory at every candidate box.

    ``memory`` is the post-encoder visual memory ``[B,L,D]``.  A fixed 4x4
    bilinear lattice per level is used only to build a candidate seed; no row
    is filtered or selected.  The returned seed is ``[N,D]`` for batch size 1.
    """
    if memory.ndim != 3 or spatial_shapes.ndim != 2 or spatial_shapes.shape[-1] != 2:
        raise ValueError("invalid memory/spatial-shape dimensions")
    if memory.shape[0] != 1 or boxes_normalized.ndim != 2 or boxes_normalized.shape[-1] != 4:
        raise ValueError("L82 contract requires batch=1 and boxes [N,4]")
    if int(level_start_index.numel()) != int(spatial_shapes.shape[0]):
        raise ValueError("level start/shape count mismatch")
    count = int(boxes_normalized.shape[0])
    if count == 0:
        return memory.new_zeros((0, memory.shape[-1])), {"candidate_count": 0, "levels": []}
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    fractions = (torch.arange(grid_size, device=memory.device, dtype=memory.dtype) + 0.5) / float(grid_size)
    gy, gx = torch.meshgrid(fractions, fractions, indexing="ij")
    grid_x = boxes_normalized[:, 0, None, None] + (boxes_normalized[:, 2] - boxes_normalized[:, 0])[:, None, None] * gx
    grid_y = boxes_normalized[:, 1, None, None] + (boxes_normalized[:, 3] - boxes_normalized[:, 1])[:, None, None] * gy
    sample_grid = torch.stack((grid_x, grid_y), dim=-1) * 2.0 - 1.0
    level_tokens = []
    level_audit = []
    for level, (shape, start) in enumerate(zip(spatial_shapes.tolist(), level_start_index.tolist())):
        height, width = int(shape[0]), int(shape[1])
        start = int(start)
        end = start + height * width
        if end > int(memory.shape[1]):
            raise ValueError(f"level {level} exceeds flattened memory: {start}:{end}")
        fmap = memory[:, start:end].transpose(1, 2).reshape(1, memory.shape[-1], height, width)
        fmap = fmap.expand(count, -1, -1, -1)
        sampled = F.grid_sample(fmap, sample_grid.float(), mode="bilinear",
                                padding_mode="border", align_corners=False)
        pooled = sampled.mean(dim=(2, 3))
        level_tokens.append(pooled)
        valid_samples = count * grid_size * grid_size
        invalid_samples = 0
        if memory_mask is not None:
            if memory_mask.ndim != 2 or memory_mask.shape[1] != memory.shape[1]:
                raise ValueError("memory mask shape mismatch")
            # A mask is not silently treated as validity: report the nearest
            # lattice cell mask explicitly.  Bilinear values remain unchanged.
            ix = ((sample_grid[..., 0] + 1.0) * 0.5 * width).floor().long().clamp(0, width - 1)
            iy = ((sample_grid[..., 1] + 1.0) * 0.5 * height).floor().long().clamp(0, height - 1)
            nearest = start + iy * width + ix
            invalid_samples = int(memory_mask[0, nearest.reshape(-1)].bool().sum().item())
            valid_samples -= invalid_samples
        level_audit.append({"level": level, "height": height, "width": width,
                            "start": start, "end": end, "grid_size": grid_size,
                            "valid_samples": int(valid_samples),
                            "invalid_samples": int(invalid_samples),
                            "mask_supplied": memory_mask is not None})
    seed = torch.stack(level_tokens, dim=0).mean(dim=0)
    if seed.shape != (count, memory.shape[-1]) or not bool(torch.isfinite(seed).all()):
        raise FloatingPointError("nonfinite or malformed candidate seed")
    return seed, {"candidate_count": count, "levels": level_audit,
                   "grid_size": grid_size, "all_rows_retained": True,
                   "candidate_deletion": False, "candidate_truncation": False}


def candidate_seed_with_reference(
    visual_seed: torch.Tensor,
    reference_points: torch.Tensor,
    reference_position_head,
    coordinate_to_encoding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine frozen visual ROI memory and the pretrained reference PE."""
    if visual_seed.ndim != 2 or reference_points.ndim != 2:
        raise ValueError("visual seed and references must be [N,D]/[N,4]")
    encoded = coordinate_to_encoding(reference_points.unsqueeze(0), num_feats=visual_seed.shape[-1] // 2)
    position = reference_position_head(encoded).squeeze(0)
    if position.shape != visual_seed.shape:
        raise ValueError(f"reference PE shape {tuple(position.shape)} != {tuple(visual_seed.shape)}")
    seed = visual_seed + position
    if not bool(torch.isfinite(seed).all()):
        raise FloatingPointError("nonfinite candidate seed")
    return seed, position


__all__ = [
    "boxes_xyxy_to_normalized", "boxes_to_reference_points", "candidate_seed_with_reference",
    "expand_scale_factor", "pool_memory_by_box",
]
