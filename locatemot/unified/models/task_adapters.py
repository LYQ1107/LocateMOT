"""Small task adapters; shared identity state remains outside these modules."""
from __future__ import annotations

import torch
from torch import nn


class TaskAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.adapter(value)
