"""Training query augmentation of the actual persistent state.

Only previous-frame unmatched, non-ignored proposals are supplied by the
temporal trainer. No current/future candidate or GT identity enters this module.
"""
from __future__ import annotations

import torch

from .track_state import UnifiedTrackState


class QueryInteraction:
    def __init__(self, drop_probability: float = 0.10, false_positive_probability: float = 0.20) -> None:
        self.drop_probability = float(drop_probability)
        self.false_positive_probability = float(false_positive_probability)

    def augment_state(
        self,
        state: UnifiedTrackState,
        past_negatives: dict[str, torch.Tensor] | None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[UnifiedTrackState, dict[str, list[int]]]:
        device = state.track_ids.device
        keep = torch.rand(state.num_tracks, device=device, generator=generator) >= self.drop_probability
        fields = {name: value[keep] if isinstance(value, torch.Tensor) else value
                  for name, value in state.__dict__.items()}
        fields["metadata"] = dict(state.metadata)
        injected_ids: list[int] = []
        chosen: list[int] = []
        if (state.num_tracks and past_negatives is not None and len(past_negatives["boxes"])
                and torch.rand((), device=device, generator=generator) < self.false_positive_probability):
            count = max(1, round(state.num_tracks * .10))
            indices = torch.randperm(len(past_negatives["boxes"]), device=device, generator=generator)[:count]
            count = len(indices)
            chosen = indices.tolist()
            new_ids = torch.arange(state.next_track_id, state.next_track_id + count, device=device)
            injected_ids = new_ids.tolist()
            id_tokens = past_negatives["candidate_id_tokens"][indices]
            additions = {
                "track_ids": new_ids, "id_memory": id_tokens, "identity_anchor": id_tokens,
                "appearance_ema": id_tokens, "semantic_memory": past_negatives["candidate_sem_tokens"][indices],
                "boxes": past_negatives["boxes"][indices], "velocity": state.velocity.new_zeros((count, 4)),
                "age": state.age.new_ones(count), "hits": state.hits.new_ones(count),
                "lost_age": state.lost_age.new_zeros(count),
                "last_frame": state.last_frame.new_full((count,), state.frame_index),
                "confidence": past_negatives["scores"][indices], "alive": state.alive.new_ones(count),
            }
            for name, addition in additions.items():
                fields[name] = torch.cat([fields[name], addition], dim=0)
            fields["next_track_id"] = state.next_track_id + count
        return UnifiedTrackState(**fields), {
            "dropped_track_ids": state.track_ids[~keep].tolist(), "injected_track_ids": injected_ids,
            "injected_past_negative_indices": chosen,
        }
