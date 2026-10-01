"""Shared visual foundation interface.

The U1 implementation will inject MM-GroundingDINO-B.  The tiny fallback is
only a shape-checking skeleton and is never a research checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class FoundationConfig:
    name: str = "mm-groundingdino-swin-b"
    checkpoint: str = "weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
    feature_dim: int = 256
    levels: int = 4


class SharedVisualFoundation(nn.Module):
    """One injected foundation used by every task in the unified model."""

    def __init__(self, config: FoundationConfig | None = None, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config or FoundationConfig()
        self.backbone = backbone if backbone is not None else nn.Sequential(
            nn.Conv2d(3, self.config.feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.config.feature_dim),
            nn.GELU(),
        )

    def forward(self, frames: torch.Tensor) -> dict[str, Any]:
        if frames.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got {tuple(frames.shape)}")
        batch, time = frames.shape[:2]
        flattened = frames.reshape(batch * time, *frames.shape[2:])
        encoded = self.backbone(flattened)
        features = encoded.reshape(batch, time, self.config.feature_dim, *encoded.shape[-2:])
        return {"multi_scale": (features,), "frame_features": features}
