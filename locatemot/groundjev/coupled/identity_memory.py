"""Finite causal identity memory used by the coupled runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrackMemory:
    persistent_track_id: int
    identity_visual_token: torch.Tensor
    long_term_visual_token: torch.Tensor
    recent_visual_token: torch.Tensor
    last_box: tuple[float, float, float, float]
    predicted_motion: tuple[float, float, float, float] | None = None
    last_seen_timestamp: int = 0
    lost_duration: int = 0
    semantic_evidence: str | None = None
    confidence: float = 0.0
    identity_uncertainty: float = 1.0
    memory_update_status: str = "HOLD"
    state: str = "ACTIVE"

    def token(self) -> torch.Tensor:
        return self.recent_visual_token

    def detached(self) -> "TrackMemory":
        return TrackMemory(
            persistent_track_id=self.persistent_track_id,
            identity_visual_token=self.identity_visual_token.detach().cpu().clone(),
            long_term_visual_token=self.long_term_visual_token.detach().cpu().clone(),
            recent_visual_token=self.recent_visual_token.detach().cpu().clone(),
            last_box=tuple(float(x) for x in self.last_box),
            predicted_motion=None if self.predicted_motion is None else tuple(float(x) for x in self.predicted_motion),
            last_seen_timestamp=int(self.last_seen_timestamp), lost_duration=int(self.lost_duration),
            semantic_evidence=self.semantic_evidence, confidence=float(self.confidence),
            identity_uncertainty=float(self.identity_uncertainty),
            memory_update_status=self.memory_update_status, state=self.state,
        )

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        for key in ("identity_visual_token", "long_term_visual_token", "recent_visual_token"):
            out[key] = getattr(self, key).detach().cpu().tolist()
        return out


@dataclass
class IdentityMemoryState:
    tracks: dict[int, TrackMemory] = field(default_factory=dict)
    next_track_id: int = 1
    timestamp: int = -1

    def ordered(self) -> list[TrackMemory]:
        return [self.tracks[k] for k in sorted(self.tracks)]

    def create(self, token: torch.Tensor, box: tuple[float, float, float, float], *, timestamp: int,
               confidence: float = 0.0, semantic_evidence: str | None = None) -> TrackMemory:
        track_id = self.next_track_id
        self.next_track_id += 1
        value = token.detach().clone()
        track = TrackMemory(track_id, value.clone(), value.clone(), value.clone(), tuple(box),
                            last_seen_timestamp=int(timestamp), confidence=float(confidence),
                            semantic_evidence=semantic_evidence)
        self.tracks[track_id] = track
        return track

    def snapshot(self) -> dict[str, Any]:
        return {"next_track_id": self.next_track_id, "timestamp": self.timestamp,
                "tracks": {str(k): v.to_dict() for k, v in self.tracks.items()}}

    def assert_causal(self, timestamp: int) -> None:
        if int(timestamp) <= self.timestamp:
            raise ValueError("identity memory timestamp must advance causally")
