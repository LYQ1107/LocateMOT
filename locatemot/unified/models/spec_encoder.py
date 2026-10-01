"""Specification encoder shared by language, class and prompt tasks."""
from __future__ import annotations

import torch
from torch import nn


class SpecificationEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, vocab_size: int = 32768) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.token = nn.Embedding(vocab_size, hidden_dim)
        self.task = nn.Embedding(7, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: torch.Tensor, task_id: torch.Tensor) -> dict[str, torch.Tensor]:
        if token_ids.ndim != 2 or task_id.ndim != 1 or token_ids.shape[0] != task_id.shape[0]:
            raise ValueError("specification inputs must be [B,L] and [B]")
        tokens = self.norm(self.token(token_ids) + self.task(task_id)[:, None])
        return {"spec_tokens": tokens, "spec_global": tokens.mean(dim=1), "task_embedding": self.task(task_id)}
