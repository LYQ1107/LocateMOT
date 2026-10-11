"""Persistent memory facade used by U2 runtimes.

The actual state transition lives in :mod:`shared_track_core`; this module
exists so the architecture has an explicit track-memory component and a
stable import for future U3 adapters.
"""
from __future__ import annotations

from .track_state import UnifiedTrackState


class TrackMemory:
    """State container contract; updates are performed by ``SharedTrackCore``."""

    state_type = UnifiedTrackState

    @staticmethod
    def empty(hidden_dim: int = 256, **kwargs) -> UnifiedTrackState:
        return UnifiedTrackState.empty(hidden_dim=hidden_dim, **kwargs)


__all__ = ["TrackMemory", "UnifiedTrackState"]
