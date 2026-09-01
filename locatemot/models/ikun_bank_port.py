"""Controlled iKUN-style selector over the frozen LocateMOT track bank.

Derived conceptually from dyhBUPT/iKUN (MIT, commit 4db56bfa): local/global
fusion, text-guided KUM, temporal tracklet pooling, cosine classification, and
pseudo-frequency calibration.  The implementation is rewritten for cached
L11 observations and is not an official iKUN checkpoint reproduction.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class IKunProjection(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(value.float()))


class IKunBankPort(nn.Module):
    """Causal fixed-bank analogue of iKUN's tracklet KUM classifier."""

    def __init__(self, hidden: int = 256, heads: int = 4,
                 window: int = 8, samples: int = 2):
        super().__init__()
        self.hidden = int(hidden)
        self.window = int(window)
        self.samples = int(samples)
        self.local = IKunProjection(512, hidden)
        self.global_context = IKunProjection(512, hidden)
        self.text = IKunProjection(512, hidden)
        self.local_global = nn.MultiheadAttention(
            hidden, heads, batch_first=True)
        self.local_global_norm = nn.LayerNorm(hidden)
        self.text_guidance = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.logit_scale = nn.Parameter(torch.tensor(3.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _sample_history(values: list[torch.Tensor], count: int) -> torch.Tensor:
        if len(values) <= count:
            return torch.stack(values).mean(0)
        positions = torch.linspace(
            0, len(values) - 1, count, device=values[0].device).round().long()
        return torch.stack([values[int(index)] for index in positions]).mean(0)

    def forward_frame(self, features: dict, query: torch.Tensor,
                      track_ids: torch.Tensor,
                      state: Dict[int, list[torch.Tensor]] | None = None) -> dict:
        state = {} if state is None else state
        count = int(track_ids.numel())
        if not count:
            return {"logits": query.new_zeros(0), "state": state}
        local = self.local(features["clip"])
        frame_context = torch.nan_to_num(features["clip"].float()).mean(0, keepdim=True)
        global_token = self.global_context(frame_context).expand(count, -1)
        tokens = torch.stack((local, global_token), dim=1)
        attended, _ = self.local_global(
            local.unsqueeze(1), tokens, tokens, need_weights=False)
        visual = self.local_global_norm(local + attended[:, 0])
        text = F.normalize(self.text(query.reshape(1, -1))[0], dim=-1)
        guided = self.fusion(torch.cat((
            visual, visual * (1.0 + self.text_guidance(text).unsqueeze(0)),
        ), dim=-1))

        new_state = dict(state)
        pooled = []
        for index, raw_track_id in enumerate(track_ids.detach().cpu().tolist()):
            track_id = int(raw_track_id)
            history = list(state.get(track_id, []))
            history.append(guided[index])
            history = history[-self.window:]
            new_state[track_id] = history
            pooled.append(self._sample_history(history, self.samples))
        tracklet = F.normalize(torch.stack(pooled), dim=-1)
        logits = self.logit_scale.exp().clamp(max=100.0) * \
            (tracklet * text.unsqueeze(0)).sum(-1) + self.logit_bias
        return {"logits": logits, "state": new_state,
                "track_embedding": tracklet, "query_embedding": text}

    def forward(self, features: dict, query: torch.Tensor,
                track_ids: torch.Tensor, state=None) -> dict:
        return self.forward_frame(features, query, track_ids, state)


def pseudo_frequency_offset(query: torch.Tensor, table: dict | None,
                            a: float = 8.0, b: float = -0.1,
                            tau: float = 100.0) -> float:
    """iKUN test-time calibration rewritten for cached CLIP text features."""
    if not table or not table.get("features"):
        return 0.0
    features = torch.as_tensor(table["features"], dtype=torch.float32)
    probabilities = torch.as_tensor(table["probabilities"], dtype=torch.float32)
    value = F.normalize(query.detach().cpu().float().reshape(1, -1), dim=-1)
    features = F.normalize(features, dim=-1)
    similarity = (value @ features.T)[0]
    span = similarity.max() - similarity.min()
    similarity = (similarity - similarity.min()) / span.clamp_min(1e-6)
    weight = torch.softmax(tau * similarity, dim=0)
    probability = float((weight * probabilities).sum())
    return a * probability + b
