"""L73 post-fusion attention and candidate-cell helpers.

This module contains no model-weight changes.  It only makes the image-token
lattice and box-to-cell overlap rule explicit for the label-free audit.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch


def merged_grid(image_grid_hw: Iterable[int], merge_kernel: tuple[int, int] = (2, 2)) -> tuple[int, int]:
    grid_h, grid_w = (int(value) for value in image_grid_hw)
    merge_h, merge_w = (int(value) for value in merge_kernel)
    if grid_h % merge_h or grid_w % merge_w:
        raise ValueError(
            f"pre-merge image grid {grid_h}x{grid_w} is not divisible by "
            f"merge kernel {merge_h}x{merge_w}"
        )
    return grid_h // merge_h, grid_w // merge_w


def center_indices(
    box_xyxy: Iterable[float],
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
    image_grid_hw: Iterable[int],
    patch_size: int = 14,
    merge_kernel: tuple[int, int] = (2, 2),
) -> dict[str, Any]:
    """Return the L72-style merged-cell-center control mapping."""
    ow, oh = (float(value) for value in original_size)
    pw, ph = (float(value) for value in processed_size)
    feat_h, feat_w = merged_grid(image_grid_hw, merge_kernel)
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    sx, sy = pw / max(1.0, ow), ph / max(1.0, oh)
    scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    scaled[0] = max(0.0, min(pw, scaled[0]))
    scaled[1] = max(0.0, min(ph, scaled[1]))
    scaled[2] = max(0.0, min(pw, scaled[2]))
    scaled[3] = max(0.0, min(ph, scaled[3]))
    cell_h = float(patch_size * int(merge_kernel[0]))
    cell_w = float(patch_size * int(merge_kernel[1]))
    indices: list[int] = []
    for row in range(feat_h):
        cy = (row + 0.5) * cell_h
        for col in range(feat_w):
            cx = (col + 0.5) * cell_w
            if scaled[0] <= cx <= scaled[2] and scaled[1] <= cy <= scaled[3]:
                indices.append(row * feat_w + col)
    return {
        "indices": indices,
        "token_count": len(indices),
        "scaled_box": scaled,
        "grid_shape": [feat_h, feat_w],
        "scale_factor": [sx, sy],
        "empty": not indices,
    }


def overlap_indices(
    box_xyxy: Iterable[float],
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
    image_grid_hw: Iterable[int],
    patch_size: int = 14,
    merge_kernel: tuple[int, int] = (2, 2),
) -> dict[str, Any]:
    """Map a box to every merged patch cell with positive area overlap.

    The returned indices are row-major in the same order as the processor's
    flattened image tokens.  ``overlap_areas`` is only a deterministic area
    diagnostic; no score, label, or candidate filtering participates.
    """
    ow, oh = (float(value) for value in original_size)
    pw, ph = (float(value) for value in processed_size)
    feat_h, feat_w = merged_grid(image_grid_hw, merge_kernel)
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    sx, sy = pw / max(1.0, ow), ph / max(1.0, oh)
    scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    scaled[0] = max(0.0, min(pw, scaled[0]))
    scaled[1] = max(0.0, min(ph, scaled[1]))
    scaled[2] = max(0.0, min(pw, scaled[2]))
    scaled[3] = max(0.0, min(ph, scaled[3]))
    cell_h = float(patch_size * int(merge_kernel[0]))
    cell_w = float(patch_size * int(merge_kernel[1]))
    indices: list[int] = []
    areas: list[float] = []
    for row in range(feat_h):
        cy1, cy2 = row * cell_h, min(ph, (row + 1) * cell_h)
        for col in range(feat_w):
            cx1, cx2 = col * cell_w, min(pw, (col + 1) * cell_w)
            area = max(0.0, min(scaled[2], cx2) - max(scaled[0], cx1)) * max(
                0.0, min(scaled[3], cy2) - max(scaled[1], cy1)
            )
            if area > 0.0:
                indices.append(row * feat_w + col)
                areas.append(float(area))
    box_area = max(0.0, scaled[2] - scaled[0]) * max(0.0, scaled[3] - scaled[1])
    return {
        "indices": indices,
        "overlap_areas": areas,
        "token_count": len(indices),
        "scaled_box": scaled,
        "grid_shape": [feat_h, feat_w],
        "scale_factor": [sx, sy],
        "box_area_processed": float(box_area),
        "area_fraction_processed": float(box_area / max(1.0, pw * ph)),
        "empty": not indices,
    }


def finite_vector_summary(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {"present": False, "dim": 0, "finite": True}
    flat = value.detach().float().reshape(-1).cpu()
    return {
        "present": True,
        "dim": int(flat.numel()),
        "finite": bool(torch.isfinite(flat).all()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "l2": float(flat.norm()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def masked_mean(value: torch.Tensor, positions: list[int]) -> torch.Tensor | None:
    if not positions:
        return None
    index = torch.as_tensor(positions, dtype=torch.long, device=value.device)
    selected = value.index_select(0, index)
    if not bool(torch.isfinite(selected.float()).all()):
        raise ValueError("nonfinite values in masked mean")
    return selected.float().mean(dim=0)

