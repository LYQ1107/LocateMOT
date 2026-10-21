"""Task-specific readouts that stay on the shared GroundJEV state.

The category heads are deliberately small linear readouts.  They consume the
same candidate/context hidden state as the typed JEV decision heads and do not
instantiate a second visual encoder or a temporal association network.
"""
from __future__ import annotations

import torch
from torch import nn


class GroundJEVTaskHeads(nn.Module):
    """OVMOT category readouts attached to a shared GroundJEV hidden state."""

    def __init__(self, hidden_dim: int, *, group_classes: int = 8, leaf_classes: int = 128):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.group_classes = int(group_classes)
        self.leaf_classes = int(leaf_classes)
        if self.hidden_dim <= 0 or self.group_classes <= 1 or self.leaf_classes <= 1:
            raise ValueError("task-head dimensions must be positive and have at least two classes")
        self.category_group = nn.Linear(self.hidden_dim, self.group_classes)
        self.category_leaf = nn.Linear(self.hidden_dim, self.leaf_classes)

    def forward(self, candidate_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if candidate_hidden.ndim != 2 or candidate_hidden.shape[-1] != self.hidden_dim:
            raise ValueError("candidate_hidden must have shape [N, hidden_dim]")
        return {
            "category_group_logits": self.category_group(candidate_hidden),
            "category_leaf_logits": self.category_leaf(candidate_hidden),
        }
