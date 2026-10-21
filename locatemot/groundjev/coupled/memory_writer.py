"""Assignment-conditioned memory writes."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .identity_memory import IdentityMemoryState
from .shared_ground_state import SharedCandidate


class MemoryWriter:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def write(self, memory: IdentityMemoryState, candidates: Sequence[SharedCandidate],
              assignments: Mapping[int, int], update_probabilities: Mapping[int, float],
              *, timestamp: int, labels: Sequence[str] | None = None,
              blocked_track_ids: set[int] | None = None) -> dict[str, int]:
        if int(timestamp) <= memory.timestamp:
            raise ValueError("memory writes must advance causally")
        writes = holds = skipped = 0
        blocked = {int(value) for value in (blocked_track_ids or set())}
        for candidate_index, track_id in assignments.items():
            candidate = candidates[int(candidate_index)]
            probability = float(update_probabilities.get(int(candidate_index), 0.0))
            track = memory.tracks.get(int(track_id))
            if track is None:
                continue
            if int(track_id) in blocked or track.state == "TERMINATED":
                track.memory_update_status = "HOLD"
                skipped += 1
                continue
            if probability < self.threshold:
                track.memory_update_status = "HOLD"
                holds += 1
                continue
            token = candidate.region_token.detach().clone()
            track.recent_visual_token = token
            if track.long_term_visual_token.shape == token.shape:
                track.long_term_visual_token = 0.9 * track.long_term_visual_token + 0.1 * token
            else:
                track.long_term_visual_token = token.clone()
            track.identity_visual_token = track.long_term_visual_token.clone()
            track.last_box = tuple(float(v) for v in candidate.box)
            track.last_seen_timestamp = int(timestamp)
            track.lost_duration = 0
            track.state = "ACTIVE"
            track.memory_update_status = "UPDATE"
            if labels is not None and int(candidate_index) < len(labels):
                track.semantic_evidence = str(labels[int(candidate_index)])
            writes += 1
        memory.timestamp = int(timestamp)
        return {"writes": writes, "holds": holds, "skipped": skipped,
                "assigned": len(assignments)}
