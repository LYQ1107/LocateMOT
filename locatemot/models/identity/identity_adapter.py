"""Lightweight universal Identity Adapter.

Inputs: LocateAnything ObjectToken parts (frozen): PBD box-end, PBD
coordinate mean, MoonViT region, geometry, generation score.
Output: L2-normalized IdentityToken (d=256).
One shared checkpoint; no dataset-specific heads.
"""
from __future__ import annotations

import torch
from torch import nn


class IdentityAdapter(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1,
                 input_mode: str = "full"):
        super().__init__()
        self.d_model = d_model
        self.input_mode = input_mode
        self.pbd_proj = nn.Sequential(
            nn.Linear(2048, d_model), nn.LayerNorm(d_model),
            nn.Dropout(dropout))
        if input_mode == "full":
            self.region_proj = nn.Sequential(
                nn.Linear(4608, d_model), nn.LayerNorm(d_model),
                nn.Dropout(dropout))
            self.geom_proj = nn.Sequential(
                nn.Linear(5, 64), nn.GELU())
            self.gen_proj = nn.Sequential(
                nn.Linear(1, 16), nn.GELU())
            self.fuse = nn.Sequential(
                nn.Linear(d_model * 2 + 64 + 16, d_model),
                nn.GELU(), nn.LayerNorm(d_model), nn.Dropout(dropout))
        else:
            self.region_proj = self.geom_proj = self.gen_proj = None
            self.fuse = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model))

    def forward(self, pbd_box_end, pbd_coord, region, geometry, gen_score):
        p = self.pbd_proj(pbd_box_end)
        if self.input_mode == "full":
            r = self.region_proj(region)
            g = self.geom_proj(geometry)
            z = self.gen_proj(gen_score)
            x = torch.cat([p, r, g, z], dim=-1)
        else:
            x = p
        x = self.fuse(x)
        ztok = self.head(x)
        return torch.nn.functional.normalize(ztok, dim=-1)


def infonce_loss(anchor, positive, negatives, temperature=0.1):
    """InfoNCE with one positive per anchor plus explicit negatives."""
    sim_pos = (anchor * positive).sum(-1) / temperature
    sim_neg = torch.einsum("bd,bnd->bn", anchor, negatives) / temperature
    logits = torch.cat([sim_pos.unsqueeze(-1), sim_neg], dim=-1)
    labels = torch.zeros(anchor.shape[0], dtype=torch.long,
                         device=anchor.device)
    return torch.nn.functional.cross_entropy(logits, labels)
