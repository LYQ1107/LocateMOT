"""Relation encoder producing pairwise embedding, scalar score and attention bias."""
from __future__ import annotations

import torch
from torch import nn

from .relation_features import RelationMLP, build_relation_features


class RelationEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        rel_dim: int = 128,
        n_heads: int = 8,
        use_region_geom: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_region_geom = use_region_geom
        # feature dim: 9 base fields + 1 region cos + 5 geom delta (if enabled) + 2 gen + 2 gap
        in_dim = 9 + (6 if use_region_geom else 0) + 2 + 2
        self.mlp = RelationMLP(in_dim=in_dim, hidden=rel_dim, out_dim=rel_dim, dropout=dropout)
        self.bias_head = nn.Linear(rel_dim, n_heads)

    def forward(self, batch):
        feat = build_relation_features(batch, use_region_geom=self.use_region_geom)
        emb, score = self.mlp(feat)
        per_head_bias = self.bias_head(emb)  # B,M,N,H
        return emb, score, per_head_bias
