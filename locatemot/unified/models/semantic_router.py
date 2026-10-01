"""Static/motion/relation semantic routing from a common specification."""
from __future__ import annotations

import torch
from torch import nn


class SemanticRouter(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(4, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)

    def forward(self, spec_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        queries = self.queries[None].expand(spec_tokens.shape[0], -1, -1)
        routed, _ = self.attn(queries, spec_tokens, spec_tokens)
        return {"static_spec": routed[:, 0], "motion_spec": routed[:, 1], "relation_spec": routed[:, 2], "global_spec": routed[:, 3]}
