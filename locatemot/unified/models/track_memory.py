"""Persistent online track memory shared by all task heads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrackMemoryState:
    embeddings: torch.Tensor
    boxes: torch.Tensor
    ids: torch.Tensor
    ages: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class TrackMemory:
    def __init__(self, hidden_dim: int = 256) -> None:
        self.hidden_dim = int(hidden_dim)
        self.state: TrackMemoryState | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, embeddings: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> TrackMemoryState:
        count = embeddings.shape[0]
        ids = torch.arange(count, device=embeddings.device, dtype=torch.long)
        self.state = TrackMemoryState(embeddings, boxes, ids, torch.zeros(count, device=embeddings.device), {"scores": scores})
        return self.state
