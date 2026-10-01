"""Universal track-query decoder skeleton."""
from __future__ import annotations

import torch
from torch import nn


class UniversalTrackDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.update = nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True)
        self.box = nn.Linear(hidden_dim, 4)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, queries: torch.Tensor, frame_context: torch.Tensor) -> dict[str, torch.Tensor]:
        updated = self.update(queries + frame_context[:, None])
        return {"track_queries": updated, "boxes": self.box(updated).sigmoid(), "scores": self.score(updated).sigmoid().squeeze(-1)}
