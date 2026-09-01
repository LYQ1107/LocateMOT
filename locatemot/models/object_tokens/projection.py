"""Randomly-initialized projection layers for ObjectToken interface testing.

These layers are NOT trained in Stage L0-B and must not be interpreted as
learned fusion. They only validate shapes and the downstream interface.
"""
from __future__ import annotations

import torch
from torch import nn


class ObjectTokenProjection(nn.Module):
    def __init__(
        self,
        pbd_dim: int = 2048,
        region_dim: int = 4608,
        geom_dim: int = 5,
        fused_dim: int = 256,
        gen_dim: int = 16,
    ):
        super().__init__()
        self.pbd_proj = nn.Linear(pbd_dim, fused_dim)
        self.region_proj = nn.Linear(region_dim, fused_dim)
        self.geom_proj = nn.Linear(geom_dim, 32)
        self.gen_proj = nn.Linear(gen_dim, 32)
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim * 2 + 32 + 32, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(
        self,
        pbd_feature: torch.Tensor,
        region_feature: torch.Tensor,
        geometry: torch.Tensor,
        generation_feature: torch.Tensor,
    ) -> torch.Tensor:
        p = self.pbd_proj(pbd_feature)
        r = self.region_proj(region_feature)
        g = self.geom_proj(geometry)
        s = self.gen_proj(generation_feature)
        return self.fuse(torch.cat([p, r, g, s], dim=-1))
