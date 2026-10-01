"""Reliability-weighted fusion of appearance, motion and relation cues."""
from __future__ import annotations

import torch
from torch import nn


class CueFusion(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.reliability = nn.Linear(hidden_dim * 3, 3)

    def forward(self, appearance: torch.Tensor, motion: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack((appearance, motion, relation), dim=-2)
        weights = self.reliability(torch.cat((appearance, motion, relation), dim=-1)).softmax(dim=-1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=-2)
