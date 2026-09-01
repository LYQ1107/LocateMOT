"""Source-blind static-only scorer for the Stage L22 candidate bank v2."""
from __future__ import annotations

import torch
from torch import nn


class L22StaticCandidateScorer(nn.Module):
    """Query/crop/context/geometry/real-motion scorer with no tracker state."""

    source_in_score = False
    uses_grouping = False
    uses_membership = False
    uses_null_scalar = False
    uses_source_acceptance = False
    uses_temporal_gru = False

    def __init__(self, query_dim: int = 512, crop_dim: int = 512,
                 geometry_dim: int = 10, motion_dim: int = 16,
                 hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.query_dim = query_dim; self.crop_dim = crop_dim
        self.geometry_dim = geometry_dim; self.motion_dim = motion_dim
        self.input_dim = query_dim + 2 * crop_dim + geometry_dim + motion_dim + 1
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, query: torch.Tensor, crop_tight: torch.Tensor,
                crop_context: torch.Tensor, geometry: torch.Tensor,
                motion: torch.Tensor, objectness: torch.Tensor) -> torch.Tensor:
        values = [query, crop_tight, crop_context, geometry, motion, objectness]
        batch = query.shape[0]
        values = [torch.nan_to_num(torch.as_tensor(v, device=query.device).float()).reshape(batch, -1)
                  for v in values]
        value = torch.cat(values, dim=-1)
        if value.shape[1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} input features, got {value.shape[1]}")
        return self.net(value).squeeze(-1)


__all__ = ["L22StaticCandidateScorer"]
