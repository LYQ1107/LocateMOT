"""Safe, reversible lifecycle state transitions."""
from __future__ import annotations

from dataclasses import dataclass

from .identity_memory import TrackMemory


@dataclass
class LifecycleConfig:
    terminate_after_lost: int = 30
    terminate_probability: float = 0.95


class LifecycleController:
    def __init__(self, config: LifecycleConfig | None = None):
        self.config = config or LifecycleConfig()

    def apply(self, track: TrackMemory, action: str, *, timestamp: int,
              confidence: float = 0.0, matched: bool = False) -> str:
        action = str(action).upper()
        if matched and action in {"ACTIVE", "REACTIVATE"}:
            track.state = "ACTIVE"
            track.lost_duration = 0
            track.last_seen_timestamp = int(timestamp)
            return track.state
        if action == "LOST" or (not matched and action in {"ACTIVE", "REACTIVATE"}):
            track.state = "LOST"
            track.lost_duration = max(track.lost_duration + 1, int(timestamp) - track.last_seen_timestamp)
            return track.state
        if action == "TERMINATE":
            # A matched observation is evidence that the identity is alive.  A
            # TERMINATE suggestion on that observation is therefore treated as
            # a recoverable ACTIVE transition instead of deleting a track.
            if matched:
                track.state = "ACTIVE"
                track.lost_duration = 0
                track.last_seen_timestamp = int(timestamp)
                return track.state
            # A single bad lifecycle logit cannot destroy a recoverable track.
            if (track.lost_duration < self.config.terminate_after_lost or
                    float(confidence) < self.config.terminate_probability):
                track.state = "LOST"
                track.lost_duration = max(track.lost_duration + 1, int(timestamp) - track.last_seen_timestamp)
            else:
                track.state = "TERMINATED"
            return track.state
        return track.state

    def advance_unmatched(self, track: TrackMemory, *, timestamp: int) -> str:
        """Advance an absent track, terminating only after the registered timeout."""
        self.apply(track, "LOST", timestamp=timestamp, matched=False)
        if track.lost_duration >= self.config.terminate_after_lost:
            self.apply(track, "TERMINATE", timestamp=timestamp,
                       confidence=1.0, matched=False)
        return track.state
