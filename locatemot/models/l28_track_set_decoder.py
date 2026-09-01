"""Small query-conditioned persistent track-set decoder for Stage L28."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class L28TrackSetDecoder(nn.Module):
    """Causal track-history encoder followed by same-query set competition."""

    def __init__(self, feature_dim=1432, hidden=128, heads=4, layers=2,
                 dropout=0.1):
        super().__init__()
        self.hidden = int(hidden)
        self.text_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden))
        self.obs_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        self.time_proj = nn.Linear(1, hidden)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=layers)
        self.query_to_obs = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.track_set = nn.TransformerEncoder(set_layer, num_layers=layers)
        self.track_norm = nn.LayerNorm(hidden)
        self.track_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        self.membership_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.continuation_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.null_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    @staticmethod
    def _causal_mask(length, device):
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _masked_mean(value, mask):
        weight = mask.float().unsqueeze(-1)
        return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)

    def encode_observations(self, observations, observation_mask,
                            observation_time):
        # observations: [tracks, history, feature_dim]; all history entries
        # are selected from observations available at the current cutoff.
        n, length, _ = observations.shape
        obs = self.obs_proj(torch.nan_to_num(observations.float()))
        obs = obs + self.time_proj(observation_time.float().unsqueeze(-1))
        # Left-padded histories can create all-masked causal queries. Keep
        # padding finite and let the subsequent masked pooling ignore it;
        # this avoids NaNs while preserving the causal order of real tokens.
        obs = obs.masked_fill(~observation_mask.bool().unsqueeze(-1), 0.0)
        obs = self.temporal(obs, mask=self._causal_mask(length, obs.device))
        obs = obs.masked_fill(~observation_mask.bool().unsqueeze(-1), 0.0)
        return obs, observation_mask.bool(), self._masked_mean(obs, observation_mask)

    def forward_encoded(self, encoded, observation_mask, query_tokens,
                        query_mask):
        obs, observation_mask, track_base = encoded
        n, length, _ = obs.shape
        qtok = self.text_proj(torch.nan_to_num(query_tokens.float()))
        q = self._masked_mean(qtok, query_mask.bool().unsqueeze(0)
                              if query_mask.ndim == 1 else query_mask.bool())
        if q.shape[0] != n:
            q = q[:1].expand(n, -1)
        cross, _ = self.query_to_obs(
            q.unsqueeze(1), obs, obs, key_padding_mask=~observation_mask.bool())
        track = self.track_norm(track_base + cross[:, 0])
        track = self.track_set(track.unsqueeze(0))[0]
        fused = torch.cat((track, q), dim=-1)
        membership = self.membership_head(
            torch.cat((obs, q[:, None, :].expand(n, length, -1)), dim=-1)).squeeze(-1)
        return {
            "track_logits": self.track_head(fused).squeeze(-1),
            "membership_logits": membership,
            "null_logit": self.null_head(q[:1]).squeeze(),
            "continuation_logits": self.continuation_head(track).squeeze(-1),
            "track_embedding": track,
        }

    def forward(self, observations, observation_mask, observation_time,
                query_tokens, query_mask):
        encoded = self.encode_observations(
            observations, observation_mask, observation_time)
        return self.forward_encoded(encoded, observation_mask, query_tokens,
                                    query_mask)
