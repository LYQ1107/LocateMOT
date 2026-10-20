"""Official-geometry region token extraction from one GroundAnything feature map."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _validate_boxes(boxes: Sequence[Sequence[float]], width: int, height: int) -> torch.Tensor:
    values = torch.as_tensor(boxes, dtype=torch.float32)
    if values.numel() == 0:
        return values.reshape(0, 4)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("boxes must have shape [N,4]")
    if not torch.isfinite(values).all():
        raise ValueError("boxes contain non-finite coordinates")
    values = values.clone()
    values[:, 0::2].clamp_(0, max(width - 1, 0))
    values[:, 1::2].clamp_(0, max(height - 1, 0))
    if (values[:, 2] <= values[:, 0]).any() or (values[:, 3] <= values[:, 1]).any():
        raise ValueError("boxes must have positive area")
    return values


def boxes_to_feature_rois(
    boxes: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int],
    spatial_layout: tuple[int, int],
) -> torch.Tensor:
    """Map image-pixel boxes to ``(batch,x1,y1,x2,y2)`` feature coordinates."""
    width, height = image_size
    gh, gw = spatial_layout
    values = _validate_boxes(boxes, width, height)
    if values.numel() == 0:
        return values.new_empty((0, 5))
    scale_x = float(gw) / float(width)
    scale_y = float(gh) / float(height)
    rois = values.clone()
    rois[:, [0, 2]] *= scale_x
    rois[:, [1, 3]] *= scale_y
    return torch.cat((torch.zeros((len(rois), 1), dtype=rois.dtype), rois), dim=1)


def pool_region_tokens(
    spatial: torch.Tensor,
    boxes: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int],
    output_size: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool fixed-size regions without a second visual encoder.

    ``torchvision.ops.roi_align`` is used when available, matching the native
    bridge.  The small ``grid_sample`` fallback keeps CPU Phase-A tests
    runnable in environments without compiled torchvision operators.
    """
    if spatial.ndim == 3:
        spatial = spatial.unsqueeze(0)
    if spatial.ndim != 4:
        raise ValueError("spatial must have shape [1,C,H,W] or [C,H,W]")
    if spatial.shape[0] != 1:
        raise ValueError("shared region pooling expects one current frame")
    _, channels, gh, gw = spatial.shape
    rois = boxes_to_feature_rois(boxes, image_size=image_size,
                                 spatial_layout=(gh, gw)).to(spatial.device)
    if rois.numel() == 0:
        empty = spatial.new_empty((0, channels, output_size, output_size))
        return empty.mean(dim=(-1, -2)), empty
    try:
        from torchvision.ops import roi_align

        local = roi_align(spatial.float(), rois.float(), output_size=output_size,
                          spatial_scale=1.0, sampling_ratio=2, aligned=True)
    except Exception:
        # A deterministic equivalent for the contract test, not a replacement
        # for the native CUDA ROIAlign path.
        rows = []
        for _, x1, y1, x2, y2 in rois:
            xx1, yy1 = float(x1 / max(gw, 1) * 2 - 1), float(y1 / max(gh, 1) * 2 - 1)
            xx2, yy2 = float(x2 / max(gw, 1) * 2 - 1), float(y2 / max(gh, 1) * 2 - 1)
            ys = torch.linspace(yy1, yy2, output_size, device=spatial.device)
            xs = torch.linspace(xx1, xx2, output_size, device=spatial.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
            rows.append(F.grid_sample(spatial.float(), grid, mode="bilinear",
                                      align_corners=False)[0])
        local = torch.stack(rows, dim=0)
    pooled = local.mean(dim=(-1, -2))
    return pooled.to(dtype=spatial.dtype), local.to(dtype=spatial.dtype)
