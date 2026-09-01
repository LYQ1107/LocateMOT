"""Small, dependency-free L72 image-token lattice and region helpers.

This module contains no LocateAnything model modification.  It maps original
pixel boxes to the processor's merged image-token lattice and summarizes
already-captured tensors while preserving explicit empty-region diagnostics.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch


def merged_grid(image_grid_hw: Iterable[int], patch_size: int = 14,
                merge_kernel: tuple[int, int] = (2, 2)) -> tuple[int, int]:
    grid_h, grid_w = (int(value) for value in image_grid_hw)
    mh, mw = (int(value) for value in merge_kernel)
    if grid_h % mh or grid_w % mw:
        raise ValueError(f"pre-merge grid {grid_h}x{grid_w} is not divisible by {mh}x{mw}")
    return grid_h // mh, grid_w // mw


def image_token_positions(input_ids: torch.Tensor, image_token_index: int) -> list[int]:
    values = input_ids.detach().cpu().reshape(-1).tolist()
    return [index for index, value in enumerate(values) if int(value) == int(image_token_index)]


def map_box_to_token_indices(
    box_xyxy: Iterable[float],
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
    image_grid_hw: Iterable[int],
    patch_size: int = 14,
    merge_kernel: tuple[int, int] = (2, 2),
) -> dict[str, Any]:
    """Return merged-token indices whose patch centers lie in a pixel box.

    The processor first resizes the source image and then patchifies it.  A
    merged token corresponds to ``merge_kernel`` by ``merge_kernel`` patches;
    its center is used as the fixed, non-learned membership rule.  No GT or
    proposal score participates in this mapping.
    """
    ow, oh = (int(value) for value in original_size)
    pw, ph = (int(value) for value in processed_size)
    h_feat, w_feat = merged_grid(image_grid_hw, patch_size, merge_kernel)
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    sx, sy = pw / max(1.0, float(ow)), ph / max(1.0, float(oh))
    scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    scaled[0] = max(0.0, min(float(pw), scaled[0]))
    scaled[1] = max(0.0, min(float(ph), scaled[1]))
    scaled[2] = max(0.0, min(float(pw), scaled[2]))
    scaled[3] = max(0.0, min(float(ph), scaled[3]))
    mh, mw = (int(value) for value in merge_kernel)
    cell_h, cell_w = patch_size * mh, patch_size * mw
    indices: list[int] = []
    centers: list[list[float]] = []
    for row in range(h_feat):
        cy = (row + 0.5) * cell_h
        for col in range(w_feat):
            cx = (col + 0.5) * cell_w
            if scaled[0] <= cx <= scaled[2] and scaled[1] <= cy <= scaled[3]:
                indices.append(row * w_feat + col)
                centers.append([float(cx), float(cy)])
    return {
        "indices": indices,
        "grid_shape": [h_feat, w_feat],
        "original_box": [x1, y1, x2, y2],
        "scaled_box": scaled,
        "processed_size": [pw, ph],
        "scale_factor": [sx, sy],
        "token_center_count": len(indices),
        "token_centers": centers,
        "empty": not indices,
        "boundary_box": bool(x1 <= 0 or y1 <= 0 or x2 >= ow or y2 >= oh),
    }


def pooled_vector(tokens: torch.Tensor, indices: list[int]) -> torch.Tensor | None:
    if not indices:
        return None
    index = torch.as_tensor(indices, dtype=torch.long, device=tokens.device)
    selected = tokens.index_select(0, index)
    if not torch.isfinite(selected.float()).all():
        raise ValueError("nonfinite selected image-token values")
    return selected.float().mean(dim=0)


def masked_mean(hidden: torch.Tensor, positions: list[int]) -> torch.Tensor | None:
    if not positions:
        return None
    index = torch.as_tensor(positions, dtype=torch.long, device=hidden.device)
    selected = hidden.index_select(0, index)
    if not torch.isfinite(selected.float()).all():
        raise ValueError("nonfinite selected language hidden values")
    return selected.float().mean(dim=0)


def vector_summary(value: torch.Tensor | None) -> dict[str, Any]:
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
