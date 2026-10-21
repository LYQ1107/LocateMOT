"""Small typed readout projection over shared GroundAnything region evidence."""
from __future__ import annotations

import torch
from torch import nn


class RegionTokenProjector(nn.Module):
    """Project frozen G2 region evidence into the decision hidden space.

    This is an interface inside the shared backbone, not an independent
    appearance encoder: it consumes only the already-computed G2 token.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[-1] != self.input_dim:
            raise ValueError(f"expected [N,{self.input_dim}] region tokens, got {tuple(tokens.shape)}")
        return self.norm(torch.tanh(self.projection(tokens.float())))
