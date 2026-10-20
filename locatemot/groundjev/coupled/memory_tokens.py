"""Memory-token projection; no image encoder is hidden here."""
from __future__ import annotations

import torch
from torch import nn


class MemoryTokenProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0:
            return tokens.new_empty((0, self.hidden_dim), dtype=torch.float32)
        if tokens.ndim != 2 or tokens.shape[-1] != self.input_dim:
            raise ValueError("memory tokens have the wrong feature dimension")
        return self.norm(self.projection(tokens.float()))
