"""Expression-level query-to-persistent-track set decoder for Stage L37.

The model consumes frozen L28 observation features and word-level text tokens.
It deliberately does not consume source/pool/group/state identifiers.  The
current-frame membership head is the only emission head; sequence relevance,
continuation and NULL are auxiliary signals.
"""
from __future__ import annotations

import torch
from torch import nn


class L37ExpressionTrackSet(nn.Module):
    def __init__(self, feature_dim: int = 1432, text_dim: int = 768,
                 hidden: int = 128, heads: int = 4, layers: int = 2,
                 history: int = 8, max_text: int = 64, dropout: float = 0.1):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = int(hidden)
        self.history = int(history)
        self.max_text = int(max_text)
        self.query_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.query_pos = nn.Parameter(torch.zeros(1, max_text, hidden))
        self.query_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout,
                                       batch_first=True, norm_first=True,
                                       activation="gelu"), num_layers=layers)
        self.obs_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        self.history_pos = nn.Parameter(torch.zeros(1, history, hidden))
        self.time_proj = nn.Linear(1, hidden)
        self.history_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout,
                                       batch_first=True, norm_first=True,
                                       activation="gelu"), num_layers=layers)
        self.query_to_history = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.track_norm = nn.LayerNorm(hidden)
        self.set_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout,
                                       batch_first=True, norm_first=True,
                                       activation="gelu"), num_layers=layers)
        self.query_norm = nn.LayerNorm(hidden)
        self.current_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.sequence_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.continuation_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.null_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        nn.init.normal_(self.query_pos, std=0.02)
        nn.init.normal_(self.history_pos, std=0.02)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.0)

    def encode_query(self, query_tokens: torch.Tensor,
                     query_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(torch.nan_to_num(query_tokens.float()))
        q = q[:, :self.max_text] + self.query_pos[:, :q.shape[1]]
        mask = query_mask[:, :q.shape[1]].bool()
        q = q.masked_fill(~mask.unsqueeze(-1), 0.0)
        q = self.query_encoder(q, src_key_padding_mask=~mask)
        q = q.masked_fill(~mask.unsqueeze(-1), 0.0)
        return q, self._masked_mean(q, mask)

    def encode_history(self, observations: torch.Tensor,
                       observation_mask: torch.Tensor,
                       observation_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.obs_proj(torch.nan_to_num(observations.float()))
        length = x.shape[1]
        x = x + self.history_pos[:, :length] + self.time_proj(observation_time.float().unsqueeze(-1))
        mask = observation_mask.bool()
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = self.history_encoder(x, src_key_padding_mask=~mask)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x, self._masked_mean(x, mask)

    def forward(self, observations: torch.Tensor, observation_mask: torch.Tensor,
                observation_time: torch.Tensor, query_tokens: torch.Tensor,
                query_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # observations [tracks, history, feature_dim], query_tokens [tokens, dim]
        if query_tokens.ndim == 2:
            query_tokens = query_tokens.unsqueeze(0)
        if query_mask.ndim == 1:
            query_mask = query_mask.unsqueeze(0)
        q_tokens, q_global = self.encode_query(query_tokens, query_mask)
        q_tokens = q_tokens.expand(observations.shape[0], -1, -1)
        q_mask = query_mask.expand(observations.shape[0], -1)
        hist, hist_global = self.encode_history(observations, observation_mask, observation_time)
        # Each track gets a query-conditioned representation from the full word
        # sequence attending to its temporally ordered observation tokens.
        cross, _ = self.query_to_history(q_tokens, hist, hist,
                                         key_padding_mask=~observation_mask.bool())
        cross_global = self._masked_mean(cross, q_mask)
        track_base = self.track_norm(hist_global + cross_global)
        # Set competition is within this query/frame's complete persistent track set.
        set_track = self.set_encoder(track_base.unsqueeze(0))[0]
        q_for_tracks = q_global.expand(observations.shape[0], -1)
        current_position = observation_mask.bool().sum(1).clamp_min(1) - 1
        current_hist = hist[torch.arange(hist.shape[0], device=hist.device), current_position]
        current_logits = self.current_head(torch.cat((set_track, current_hist, q_for_tracks), -1)).squeeze(-1)
        sequence_logits = self.sequence_head(torch.cat((set_track, q_for_tracks), -1)).squeeze(-1)
        continuation_logits = self.continuation_head(set_track).squeeze(-1)
        null_context = set_track.mean(0, keepdim=True) if len(set_track) else q_global[:1]
        null_logits = self.null_head(q_global[:1] + null_context).squeeze()
        membership_logits = self.current_head(torch.cat((
            set_track[:, None, :].expand(-1, hist.shape[1], -1),
            hist, q_for_tracks[:, None, :].expand(-1, hist.shape[1], -1)), -1)).squeeze(-1)
        stale_logits = -current_logits
        return {
            "current_membership_logits": current_logits,
            "membership_logits": membership_logits,
            "sequence_logits": sequence_logits,
            "continuation_logits": continuation_logits,
            "null_logit": null_logits,
            "stale_logits": stale_logits,
            "track_embedding": set_track,
        }
