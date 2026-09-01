"""Feature projection to a unified d_model=256 space."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

_EMBED_CACHE = {}


def category_hash_embedding(category: str, dim: int = 32, seed: int = 20260806) -> torch.Tensor:
    """Deterministic, non-learned embedding from a category string (no GT leak
    at inference; the embedding is a fixed hash function, not trained)."""
    key = (category, dim, seed)
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    rng = np.random.RandomState(seed)
    base = rng.randn(1_000_000, dim).astype(np.float32)
    h = int(hashlib.sha256(category.encode()).hexdigest()[:8], 16) % 1_000_000
    emb = torch.from_numpy(base[h])
    _EMBED_CACHE[key] = emb
    return emb


class FeatureProjector(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pbd_proj = nn.Sequential(nn.Linear(2048, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.region_proj = nn.Sequential(nn.Linear(4608, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.geom_proj = nn.Sequential(nn.Linear(5, 64), nn.GELU())
        self.gen_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.cat_proj = nn.Sequential(nn.Linear(32, 32), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2 + 64 + 16 + 32, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        pbd: torch.Tensor,
        region: torch.Tensor,
        geometry: torch.Tensor,
        gen_score: torch.Tensor,
        category_embed: torch.Tensor,
    ) -> torch.Tensor:
        p = self.pbd_proj(pbd)
        r = self.region_proj(region)
        g = self.geom_proj(geometry)
        s = self.gen_proj(gen_score.unsqueeze(-1))
        c = self.cat_proj(category_embed)
        return self.fuse(torch.cat([p, r, g, s, c], dim=-1))
