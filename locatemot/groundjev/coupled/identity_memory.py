"""Finite causal identity memory used by the single-backbone runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import torch


@dataclass
class IdentityMemory:
    persistent_track_id: int
    identity_visual_token: Optional[torch.Tensor] = None
    long_term_visual_token: Optional[torch.Tensor] = None
    recent_visual_token: Optional[torch.Tensor] = None
    last_box: Optional[list[float]] = None
    predicted_motion: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    last_seen_timestamp: float = -1.0
    lost_duration: float = 0.0
    semantic_evidence: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    identity_uncertainty: float = 1.0
    state: str = "ACTIVE"
    memory_update_status: str = "HOLD"
    age: int = 0

    def feature(self, *, device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
        value = self.recent_visual_token
        if value is None:
            value = self.long_term_visual_token
        if value is None:
            value = self.identity_visual_token
        if value is None:
            return None
        return value.to(device=device, dtype=dtype)

    def predict_box(self, timestamp: float) -> Optional[list[float]]:
        if self.last_box is None or self.last_seen_timestamp < 0:
            return None
        dt = float(timestamp) - self.last_seen_timestamp
        return [float(x + v * dt) for x, v in zip(self.last_box, self.predicted_motion)]

    def mark_unseen(self, timestamp: float) -> None:
        if self.last_seen_timestamp >= 0 and timestamp < self.last_seen_timestamp:
            raise ValueError("identity memory cannot receive a future-to-past timestamp")
        self.lost_duration = max(0.0, float(timestamp) - self.last_seen_timestamp)
        self.state = "LOST"
        self.memory_update_status = "HOLD"

    def update_observation(self, *, token: torch.Tensor, box: list[float],
                           timestamp: float, confidence: float,
                           write: bool, ema_alpha: float = 0.2,
                           semantic_label: Optional[str] = None) -> None:
        if timestamp < self.last_seen_timestamp:
            raise ValueError("identity memory update is not causal")
        if token.ndim != 1 or not torch.isfinite(token).all():
            raise ValueError("identity visual token must be finite and one-dimensional")
        if len(box) != 4:
            raise ValueError("identity memory box must be xyxy")
        if self.last_box is not None and timestamp > self.last_seen_timestamp:
            dt = max(float(timestamp - self.last_seen_timestamp), 1e-6)
            self.predicted_motion = [(float(new - old) / dt)
                                     for new, old in zip(box, self.last_box)]
        self.last_box = [float(v) for v in box]
        self.recent_visual_token = token.detach()
        self.last_seen_timestamp = float(timestamp)
        self.lost_duration = 0.0
        self.confidence = float(confidence)
        self.identity_uncertainty = max(0.0, 1.0 - float(confidence))
        self.age += 1
        self.state = "ACTIVE"
        if write:
            if self.long_term_visual_token is None:
                self.long_term_visual_token = token.detach().clone()
            else:
                alpha = float(ema_alpha)
                self.long_term_visual_token = ((1.0 - alpha) * self.long_term_visual_token.detach()
                                               + alpha * token.detach())
            if self.identity_visual_token is None:
                self.identity_visual_token = token.detach().clone()
            self.memory_update_status = "UPDATE"
        else:
            self.memory_update_status = "HOLD"
        if semantic_label:
            self.semantic_evidence[semantic_label] = self.semantic_evidence.get(semantic_label, 0.0) + 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("identity_visual_token", "long_term_visual_token", "recent_visual_token"):
            value = getattr(self, key)
            payload[key] = value.detach().cpu().tolist() if value is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IdentityMemory":
        row = dict(payload)
        for key in ("identity_visual_token", "long_term_visual_token", "recent_visual_token"):
            value = row.get(key)
            row[key] = torch.tensor(value, dtype=torch.float32) if value is not None else None
        return cls(**row)


class IdentityMemoryStore:
    """Serializable finite store; only assigned observations may be written."""

    def __init__(self, *, max_tracks: int = 256, ema_alpha: float = 0.2) -> None:
        self.max_tracks = int(max_tracks)
        self.ema_alpha = float(ema_alpha)
        self.tracks: dict[int, IdentityMemory] = {}
        self.next_id = 1
        self.last_timestamp: Optional[float] = None

    def allocate(self) -> IdentityMemory:
        if len(self.tracks) >= self.max_tracks:
            raise RuntimeError("identity memory capacity reached; terminate explicitly first")
        track = IdentityMemory(persistent_track_id=self.next_id)
        self.next_id += 1
        self.tracks[track.persistent_track_id] = track
        return track

    def get(self, track_id: int) -> Optional[IdentityMemory]:
        return self.tracks.get(int(track_id))

    def mark_unseen(self, seen_ids: Iterable[int], timestamp: float) -> None:
        timestamp = float(timestamp)
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("timestamps must increase for causal memory")
        seen = {int(item) for item in seen_ids}
        for track_id, track in self.tracks.items():
            if track_id not in seen and track.state not in ("TERMINATE",):
                track.mark_unseen(timestamp)
        self.last_timestamp = timestamp

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "groundjev-single-backbone-memory-v1",
            "max_tracks": self.max_tracks,
            "ema_alpha": self.ema_alpha,
            "next_id": self.next_id,
            "last_timestamp": self.last_timestamp,
            "tracks": [track.to_dict() for track in self.tracks.values()],
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != "groundjev-single-backbone-memory-v1":
            raise ValueError("unknown GroundJEV memory format")
        self.max_tracks = int(payload["max_tracks"])
        self.ema_alpha = float(payload["ema_alpha"])
        self.next_id = int(payload["next_id"])
        self.last_timestamp = payload.get("last_timestamp")
        self.tracks = {int(row["persistent_track_id"]): IdentityMemory.from_dict(row)
                       for row in payload.get("tracks", [])}
        if len(self.tracks) > self.max_tracks:
            raise ValueError("serialized memory exceeds configured capacity")
