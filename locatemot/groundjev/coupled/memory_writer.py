"""Assignment-conditioned causal memory writes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from .identity_memory import IdentityMemoryStore


@dataclass(frozen=True)
class MemoryWriteConfig:
    threshold: float = 0.5
    ema_alpha: float = 0.2


class MemoryWriter:
    """Write only observations that received the final global assignment."""

    def __init__(self, store: IdentityMemoryStore,
                 config: Optional[MemoryWriteConfig] = None) -> None:
        self.store = store
        self.config = config or MemoryWriteConfig(ema_alpha=store.ema_alpha)

    def apply(self, *, assignments: Mapping[int, str],
              candidate_tokens: torch.Tensor,
              boxes: list[list[float]], timestamp: float,
              update_probabilities: Mapping[int, float],
              confidences: Optional[Mapping[int, float]] = None,
              labels: Optional[Mapping[int, str]] = None) -> dict[str, Any]:
        if candidate_tokens.ndim != 2 or candidate_tokens.shape[0] != len(boxes):
            raise ValueError("candidate_tokens and boxes disagree")
        confidences = confidences or {}
        labels = labels or {}
        writes, holds, ignored, new_assignments = [], [], [], {}
        for candidate_index, token in assignments.items():
            index = int(candidate_index)
            action = str(token)
            if index < 0 or index >= len(boxes):
                raise ValueError("candidate index out of range")
            if action in {"REJECT", "NO_MATCH"}:
                ignored.append(index)
                continue
            if action == "NEW":
                track = self.store.allocate()
                track_id = track.persistent_track_id
                new_assignments[index] = track_id
            elif action.isdigit():
                track_id = int(action)
                track = self.store.get(track_id)
                if track is None:
                    raise ValueError(f"unknown identity {track_id}")
            else:
                raise ValueError(f"unsupported assignment action {action!r}")
            probability = float(update_probabilities.get(index, 0.0))
            write = probability >= self.config.threshold
            track.update_observation(
                token=candidate_tokens[index],
                box=list(boxes[index]),
                timestamp=float(timestamp),
                confidence=float(confidences.get(index, probability)),
                write=write,
                ema_alpha=self.config.ema_alpha,
                semantic_label=labels.get(index),
            )
            row = {"candidate": index, "track_id": track_id,
                   "action": "UPDATE" if write else "HOLD",
                   "probability": probability}
            (writes if write else holds).append(row)
        return {"writes": writes, "holds": holds, "ignored": ignored,
                "new_assignments": new_assignments,
                "assigned": len(writes) + len(holds),
                "unassigned_memory": len(ignored)}
