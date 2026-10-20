"""Safe execution of model lifecycle suggestions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .identity_memory import IdentityMemory, IdentityMemoryStore


@dataclass(frozen=True)
class LifecycleConfig:
    max_lost_seconds: float = 3.0
    terminate_confidence: float = 0.8
    terminate_after_lost_seconds: float = 30.0


class LifecycleController:
    """Separate model suggestions from state transitions and enforce safety."""

    def __init__(self, store: IdentityMemoryStore, config: Optional[LifecycleConfig] = None) -> None:
        self.store = store
        self.config = config or LifecycleConfig()

    def suggest(self, track: IdentityMemory, model_choice: str, *,
                model_probability: float) -> str:
        choice = str(model_choice).upper()
        if choice not in {"ACTIVE", "LOST", "REACTIVATE", "TERMINATE"}:
            choice = "LOST"
        # A model TERMINATE is reversible until explicit evidence/timeout.  It
        # cannot permanently delete an identity on one uncertain frame.
        if choice == "TERMINATE":
            if (model_probability < self.config.terminate_confidence and
                    track.lost_duration < self.config.terminate_after_lost_seconds):
                return "LOST"
        if choice == "REACTIVATE" and track.state != "LOST":
            return "ACTIVE"
        return choice

    def apply_unseen(self, *, seen_ids: Iterable[int], timestamp: float,
                     lifecycle_outputs: Optional[dict[int, dict[str, Any]]] = None) -> list[dict[str, Any]]:
        outputs = []
        seen = {int(item) for item in seen_ids}
        lifecycle_outputs = lifecycle_outputs or {}
        self.store.mark_unseen(seen, timestamp)
        for track_id, track in list(self.store.tracks.items()):
            if track_id in seen:
                continue
            raw = lifecycle_outputs.get(track_id, {})
            proposed = raw.get("choice", "LOST")
            probability = float(raw.get("probability", 0.0))
            choice = self.suggest(track, proposed, model_probability=probability)
            if choice == "TERMINATE":
                track.state = "TERMINATE"
                # Keep the event in the serialized report before removal; the
                # store deletes only after the safe timeout/threshold gate.
                self.store.tracks.pop(track_id, None)
            else:
                track.state = choice
            outputs.append({"track_id": track_id, "proposed": proposed,
                            "applied": choice, "probability": probability})
        return outputs

    def validate_assignment(self, assignment: dict[int, str], *, num_candidates: int) -> None:
        used = set()
        for candidate, token in assignment.items():
            if int(candidate) < 0 or int(candidate) >= int(num_candidates):
                raise ValueError("assignment candidate index out of range")
            if token.isdigit():
                track_id = int(token)
                if track_id in used:
                    raise ValueError("one persistent identity assigned twice in one frame")
                if track_id not in self.store.tracks:
                    raise ValueError("assignment references an unknown persistent identity")
                used.add(track_id)
            elif token not in {"NEW", "NO_MATCH", "REJECT"}:
                raise ValueError(f"unsupported assignment action {token!r}")
