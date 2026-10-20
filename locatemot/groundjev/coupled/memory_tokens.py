"""Projection of causal identity state into an opt-in G2 input space."""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import nn


class IdentityMemoryTokenProjector(nn.Module):
    """Lightweight memory-token projection for the future C2 path.

    The module is intentionally standalone until the official GroundAnything
    DLM input hook is enabled.  It never reads future labels and returns one
    token per causal identity in caller-provided order.
    """

    def __init__(self, input_dim: int, g2_embedding_dim: int, *, hidden_dim: int = 512) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.g2_embedding_dim = int(g2_embedding_dim)
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, g2_embedding_dim),
        )

    def forward(self, memories: Mapping[int, torch.Tensor], *,
                ordered_ids: Optional[list[int]] = None) -> torch.Tensor:
        ids = ordered_ids if ordered_ids is not None else sorted(memories)
        if not ids:
            return torch.empty((0, self.g2_embedding_dim),
                               device=next(self.parameters()).device,
                               dtype=next(self.parameters()).dtype)
        values = []
        for track_id in ids:
            value = memories[int(track_id)]
            if value.ndim != 1 or value.shape[0] != self.input_dim:
                raise ValueError(f"memory {track_id} has invalid shape {tuple(value.shape)}")
            values.append(value)
        return self.project(torch.stack(values).to(next(self.parameters()).device,
                                                     dtype=next(self.parameters()).dtype))
