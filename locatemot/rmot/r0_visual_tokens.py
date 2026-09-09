"""Query-independent GroundingDINO visual token construction for R0.

Only the frozen ``extract_feat`` maps enter this module.  It performs no
proposal selection and retains a fixed inner/context lattice for every native
L69 row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class R0VisualTokenConfig:
    dim: int = 256
    expected_levels: int = 4
    grid_size: int = 3
    context_scale: float = 1.75


def _check_boxes(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 2 or tuple(boxes.shape[-1:]) != (4,):
        raise ValueError(f"expected normalized boxes [N,4], got {tuple(boxes.shape)}")
    boxes = boxes.float()
    if not bool(torch.isfinite(boxes).all()):
        raise FloatingPointError("nonfinite normalized boxes")
    if boxes.numel() and bool((boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("empty normalized candidate box")
    if boxes.numel() and bool((boxes < 0).any() or (boxes > 1).any()):
        raise ValueError("normalized boxes outside [0,1]")
    return boxes


def expand_normalized_boxes(boxes: torch.Tensor, scale: float) -> torch.Tensor:
    """Expand xyxy boxes around their centers and clamp to the image."""
    boxes = _check_boxes(boxes)
    if not float(scale) >= 1.0:
        raise ValueError(f"context scale must be >=1, got {scale}")
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    half = (boxes[:, 2:] - boxes[:, :2]) * (float(scale) * 0.5)
    expanded = torch.cat((center - half, center + half), dim=-1).clamp(0.0, 1.0)
    if expanded.numel() and bool((expanded[:, 2:] <= expanded[:, :2]).any()):
        raise ValueError("context expansion produced empty box")
    return expanded


def sample_box_tokens_from_level(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    grid_size: int = 3,
) -> torch.Tensor:
    """Sample a fixed bilinear lattice from one ``[1,D,H,W]`` level.

    The returned order is row-major grid order and has shape ``[N,g*g,D]``.
    ``align_corners=False`` and border padding match the audited L82 mapping.
    """
    if feature.ndim != 4 or int(feature.shape[0]) != 1:
        raise ValueError(f"feature must be [1,D,H,W], got {tuple(feature.shape)}")
    if int(grid_size) < 1:
        raise ValueError("grid_size must be positive")
    boxes = _check_boxes(boxes).to(device=feature.device)
    if int(feature.shape[1]) <= 0 or int(feature.shape[2]) <= 0 or int(feature.shape[3]) <= 0:
        raise ValueError("feature has an empty spatial dimension")
    if not bool(torch.isfinite(feature.float()).all()):
        raise FloatingPointError("nonfinite visual feature level")
    n = int(boxes.shape[0])
    fractions = (torch.arange(int(grid_size), device=feature.device, dtype=torch.float32) + 0.5) / float(grid_size)
    gy, gx = torch.meshgrid(fractions, fractions, indexing="ij")
    grid_x = boxes[:, 0, None, None] + (boxes[:, 2] - boxes[:, 0])[:, None, None] * gx
    grid_y = boxes[:, 1, None, None] + (boxes[:, 3] - boxes[:, 1])[:, None, None] * gy
    grid = torch.stack((grid_x, grid_y), dim=-1) * 2.0 - 1.0
    expanded = feature.expand(n, -1, -1, -1)
    sampled = F.grid_sample(
        expanded, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    tokens = sampled.flatten(2).transpose(1, 2).contiguous()
    expected = (n, int(grid_size) * int(grid_size), int(feature.shape[1]))
    if tuple(tokens.shape) != expected:
        raise AssertionError(f"ROI token shape drift: {tuple(tokens.shape)} != {expected}")
    if not bool(torch.isfinite(tokens.float()).all()):
        raise FloatingPointError("nonfinite sampled visual tokens")
    return tokens


def sample_track_visual_tokens(
    visual_feats: Iterable[torch.Tensor],
    boxes_normalized: torch.Tensor,
    config: R0VisualTokenConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Return inner/context tokens for all rows across all four levels."""
    config = config or R0VisualTokenConfig()
    levels = tuple(visual_feats)
    if len(levels) != int(config.expected_levels):
        raise ValueError(f"expected exactly {config.expected_levels} visual levels, got {len(levels)}")
    boxes = _check_boxes(boxes_normalized)
    inner: list[torch.Tensor] = []
    context: list[torch.Tensor] = []
    shapes: list[list[int]] = []
    for level, feature in enumerate(levels):
        if not torch.is_tensor(feature) or feature.ndim != 4:
            raise ValueError(f"level {level} is not [1,D,H,W]")
        if tuple(feature.shape[:2]) != (1, int(config.dim)):
            raise ValueError(f"level {level} shape {tuple(feature.shape)} != [1,{config.dim},H,W]")
        shapes.append([int(feature.shape[2]), int(feature.shape[3])])
        inner.append(sample_box_tokens_from_level(feature, boxes, config.grid_size))
        expanded = expand_normalized_boxes(boxes, config.context_scale)
        context.append(sample_box_tokens_from_level(feature, expanded, config.grid_size))
    inner_tokens = torch.cat(inner, dim=1)
    context_tokens = torch.cat(context, dim=1)
    expected = (int(boxes.shape[0]), int(config.expected_levels) * int(config.grid_size) ** 2, int(config.dim))
    if tuple(inner_tokens.shape) != expected or tuple(context_tokens.shape) != expected:
        raise AssertionError(f"track token shape drift: {tuple(inner_tokens.shape)}, {tuple(context_tokens.shape)}")
    if not bool(torch.isfinite(inner_tokens.float()).all()) or not bool(torch.isfinite(context_tokens.float()).all()):
        raise FloatingPointError("nonfinite track visual tokens")
    return {
        "inner_tokens": inner_tokens,
        "context_tokens": context_tokens,
        "boxes_normalized": boxes,
        "level_shapes": torch.tensor(shapes, dtype=torch.int64, device=boxes.device),
    }


__all__ = [
    "R0VisualTokenConfig", "expand_normalized_boxes", "sample_box_tokens_from_level",
    "sample_track_visual_tokens",
]
