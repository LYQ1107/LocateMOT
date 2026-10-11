"""Causal alive/lifecycle head."""
from __future__ import annotations

import torch
from torch import nn


class LifecycleHead(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_dim + 5, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, track_tokens: torch.Tensor, ages: torch.Tensor, hits: torch.Tensor, lost_age: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        if track_tokens.shape[0] == 0:
            return track_tokens.new_empty((0,))
        stats = torch.stack(
            [
                ages.float().clamp_max(1000) / 100.0,
                hits.float().clamp_max(1000) / 100.0,
                lost_age.float().clamp_max(30) / 30.0,
                confidence.clamp(0.0, 1.0),
                (lost_age == 0).float(),
            ],
            dim=-1,
        )
        return self.head(torch.cat([track_tokens, stats], dim=-1)).squeeze(-1)
