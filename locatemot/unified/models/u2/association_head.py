"""Task-independent pair, no-match and birth logits."""
from __future__ import annotations

import torch
from torch import nn


class AssociationHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, pair_dim: int = 13) -> None:
        super().__init__()
        self.pair = nn.Sequential(
            nn.Linear(hidden_dim * 2 + pair_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.no_match = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.new = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(
        self, track_tokens: torch.Tensor, candidate_tokens: torch.Tensor, pair_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        t, n = track_tokens.shape[0], candidate_tokens.shape[0]
        if t == 0 or n == 0:
            # Birth and no-match decisions still have to be real head outputs
            # when the opposite side of the bipartite set is empty.
            return {
                "identity_pair_logits": track_tokens.new_empty((t, n)),
                "no_match_logits": self.no_match(track_tokens).squeeze(-1) if t else track_tokens.new_empty((0,)),
                "new_logits": self.new(candidate_tokens).squeeze(-1) if n else candidate_tokens.new_empty((0,)),
            }
        trk = track_tokens[:, None, :].expand(t, n, -1)
        cand = candidate_tokens[None, :, :].expand(t, n, -1)
        pair_logits = self.pair(torch.cat([trk, cand, pair_features], dim=-1)).squeeze(-1)
        return {
            "identity_pair_logits": pair_logits,
            "no_match_logits": self.no_match(track_tokens).squeeze(-1),
            "new_logits": self.new(candidate_tokens).squeeze(-1),
        }
