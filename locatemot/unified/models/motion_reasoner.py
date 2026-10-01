"""Motion branch kept separate from static specification routing."""
from __future__ import annotations

import torch
from torch import nn


class MotionReasoner(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, state: torch.Tensor, motion_spec: torch.Tensor) -> torch.Tensor:
        return self.net(state + motion_spec[:, None])
