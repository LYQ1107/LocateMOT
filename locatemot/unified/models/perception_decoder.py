"""Universal object/rescue perception decoder skeleton."""
from __future__ import annotations

import torch
from torch import nn


class UniversalPerceptionDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_queries: int = 100) -> None:
        super().__init__()
        self.query = nn.Embedding(num_queries, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, frame_features: torch.Tensor, spec_global: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch = frame_features.shape[0]
        pooled = frame_features.mean(dim=(-1, -2))
        queries = self.proj(self.query.weight)[None].expand(batch, -1, -1)
        if spec_global is not None:
            queries = queries + spec_global[:, None]
        return {"object_queries": queries, "frame_context": pooled}
