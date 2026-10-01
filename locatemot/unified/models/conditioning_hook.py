"""Language-conditioned multi-scale visual sampling hook skeleton."""
from __future__ import annotations

import torch
from torch import nn


class ConditioningHook(nn.Module):
    def __init__(self, hidden_dim: int = 256, samples: int = 8) -> None:
        super().__init__()
        self.samples = int(samples)
        self.offsets = nn.Linear(hidden_dim, self.samples * 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, track_state: torch.Tensor, static_spec: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = track_state + static_spec[:, None]
        return {"sampling_offsets": self.offsets(fused).view(*fused.shape[:2], self.samples, 2), "conditioned_state": self.proj(fused)}
