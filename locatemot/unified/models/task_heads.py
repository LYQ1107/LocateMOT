"""Task-specific output heads over shared track state."""
from __future__ import annotations

import torch
from torch import nn


class TaskHeads(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in ("mot", "ovmot", "rmot", "point", "box", "mask", "grounding")})

    def forward(self, state: torch.Tensor, task_type: str) -> torch.Tensor:
        if task_type not in self.heads:
            raise KeyError(task_type)
        return self.heads[task_type](state).squeeze(-1)
