"""Task-independent visual query helper kept for the U2 integration API."""
from __future__ import annotations

import torch
from torch import nn


class UniversalPerceptionDecoder(nn.Module):
    """Produce generic visual queries without adding specification features.

    U2 identity candidates come from ``PerceptionBridge`` in normal use.  This
    small helper remains useful for shape-level foundation integration and has
    no language-conditioned query path.
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 100) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, frame_features: torch.Tensor, spec_global: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if spec_global is not None:
            raise ValueError("generic perception queries cannot consume specification tokens")
        if frame_features.ndim != 4:
            raise ValueError("frame_features must be [B,C,H,W]")
        pooled = frame_features.mean(dim=(-1, -2))
        queries = self.proj(self.query)[None].expand(frame_features.shape[0], -1, -1)
        return {"object_queries": queries, "frame_context": pooled}
