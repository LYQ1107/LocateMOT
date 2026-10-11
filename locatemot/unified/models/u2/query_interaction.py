"""MOTR-inspired training-time query drop and real false-positive injection."""
from __future__ import annotations

import torch


class QueryInteraction:
    def __init__(self, drop_probability: float = 0.10, false_positive_probability: float = 0.20) -> None:
        self.drop_probability = float(drop_probability)
        self.false_positive_probability = float(false_positive_probability)

    def __call__(
        self,
        track_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        *,
        training: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not training or track_tokens.numel() == 0:
            return track_tokens, candidate_tokens, {
                "dropped_tracks": torch.zeros(track_tokens.shape[0], dtype=torch.bool, device=track_tokens.device),
                "injected_candidates": torch.zeros(candidate_tokens.shape[0], dtype=torch.bool, device=candidate_tokens.device),
            }
        keep = torch.rand(track_tokens.shape[0], device=track_tokens.device, generator=generator) >= self.drop_probability
        kept_tracks = track_tokens[keep]
        injected = torch.zeros(candidate_tokens.shape[0], dtype=torch.bool, device=candidate_tokens.device)
        if candidate_tokens.shape[0] and torch.rand((), device=candidate_tokens.device, generator=generator) < self.false_positive_probability:
            count = max(1, int(round(candidate_tokens.shape[0] * 0.10)))
            chosen = torch.randperm(candidate_tokens.shape[0], device=candidate_tokens.device, generator=generator)[:count]
            injected[chosen] = True
            kept_tracks = torch.cat([kept_tracks, candidate_tokens[chosen]], dim=0)
        return kept_tracks, candidate_tokens, {"dropped_tracks": ~keep, "injected_candidates": injected}
