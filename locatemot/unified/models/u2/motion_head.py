"""Causal normalized box-motion prediction."""
from __future__ import annotations

import torch
from torch import nn


class MotionHead(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 4))

    def forward(self, track_tokens: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        if track_tokens.shape[0] == 0:
            return track_tokens.new_empty((0, 4))
        return 0.25 * torch.tanh(self.head(torch.cat([track_tokens, velocity], dim=-1)))
