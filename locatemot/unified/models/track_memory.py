"""Compatibility facade over the U2 persistent track core.

Legacy callers may still import ``TrackMemory`` from this path.  The old
``torch.arange`` per-frame ID placeholder is intentionally gone: IDs now come
from ``SharedTrackCore`` and persist in ``UnifiedTrackState``.
"""
from __future__ import annotations

import torch

from .u2.shared_track_core import SharedTrackCore
from .u2.track_state import UnifiedTrackState

TrackMemoryState = UnifiedTrackState


class TrackMemory:
    def __init__(self, hidden_dim: int = 256, max_age: int = 30) -> None:
        self.hidden_dim = int(hidden_dim)
        self.core = SharedTrackCore(hidden_dim=hidden_dim, max_age=max_age).eval()
        self.state: UnifiedTrackState | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, embeddings: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> UnifiedTrackState:
        candidates = {
            "id_visual": embeddings,
            "semantic_visual": embeddings,
            "decoder_query": embeddings,
            "boxes": boxes,
            "scores": scores,
        }
        with torch.no_grad():
            output = self.core(candidates, self.state, task_type="mot")
        self.state = output["state"]
        return self.state


__all__ = ["TrackMemory", "TrackMemoryState", "UnifiedTrackState"]
