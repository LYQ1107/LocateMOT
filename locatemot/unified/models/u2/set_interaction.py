"""Identity-only track/candidate set interaction."""
from __future__ import annotations

import torch
from torch import nn


class SetInteraction(nn.Module):
    def __init__(self, hidden_dim: int = 256, pair_dim: int = 13, layers: int = 4, heads: int = 8) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.track_role = nn.Parameter(torch.zeros(1, hidden_dim))
        self.candidate_role = nn.Parameter(torch.zeros(1, hidden_dim))
        self.pair_to_track = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.pair_to_candidate = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=1024, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(
        self,
        track_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        pair_features: torch.Tensor,
        *,
        track_mask: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if track_tokens.ndim != 2 or candidate_tokens.ndim != 2:
            raise ValueError("tokens must be [count,hidden]")
        t, n = track_tokens.shape[0], candidate_tokens.shape[0]
        if pair_features.shape[:2] != (t, n):
            raise ValueError("pair feature shape mismatch")
        if t == 0 or n == 0:
            return track_tokens, candidate_tokens
        if track_mask is None:
            track_mask = torch.ones(t, dtype=torch.bool, device=track_tokens.device)
        if candidate_mask is None:
            candidate_mask = torch.ones(n, dtype=torch.bool, device=candidate_tokens.device)
        track_message = self.pair_to_track(pair_features).mean(dim=1)
        candidate_message = self.pair_to_candidate(pair_features).mean(dim=0)
        tokens = torch.cat(
            [track_tokens + self.track_role + track_message, candidate_tokens + self.candidate_role + candidate_message], dim=0
        ).unsqueeze(0)
        valid = torch.cat([track_mask, candidate_mask], dim=0).unsqueeze(0)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        return encoded[0, :t], encoded[0, t:]
