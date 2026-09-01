"""A controlled functional iKUN port over frozen RN50 image features.

The official iKUN code consumes spatial RN50 feature maps from a real track
crop and a real full frame.  LocateMOT keeps the same source images and the
same causal eight-frame/two-sample tracklet rule, but stores spatial features
after channel-wise spatial pooling so that the reusable cache remains small.
The learned module is therefore a functional transplant, not an official
iKUN checkpoint reproduction.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class RN50Projection(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(value.float()))


class IKunRN50BankPort(nn.Module):
    """Causal RN50 local/global iKUN-style selector on the L11 bank."""

    def __init__(self, hidden: int = 256, heads: int = 4,
                 window: int = 8, samples: int = 2):
        super().__init__()
        self.hidden = int(hidden)
        self.window = int(window)
        self.samples = int(samples)
        self.local = RN50Projection(2048, hidden)
        self.global_context = RN50Projection(2048, hidden)
        self.text = RN50Projection(512, hidden)
        self.local_global = nn.MultiheadAttention(
            hidden, heads, batch_first=True, dropout=0.0)
        self.local_position = nn.Parameter(torch.zeros(1, 1, hidden))
        self.global_position = nn.Parameter(torch.zeros(1, 1, hidden))
        self.local_global_norm = nn.LayerNorm(hidden)
        self.kum = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        self.logit_scale = nn.Parameter(torch.tensor(3.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _sample_history(values: list[torch.Tensor], count: int) -> torch.Tensor:
        """Match iKUN's np.linspace(0, len, count, endpoint=False) rule."""
        if len(values) <= count:
            return torch.stack(values).mean(0)
        length = len(values)
        positions = (torch.arange(count, device=values[0].device) * length
                     // count).long()
        return torch.stack([values[int(index)] for index in positions]).mean(0)

    def forward_frame(self, features: dict, query: torch.Tensor,
                      track_ids: torch.Tensor,
                      state: Dict[int, list[torch.Tensor]] | None = None,
                      use_global: bool = True,
                      use_kum: bool = True) -> dict:
        state = {} if state is None else state
        count = int(track_ids.numel())
        if not count:
            return {"logits": query.new_zeros(0), "state": state}

        local = self.local(features["ikun_local"])
        if use_global:
            global_token = self.global_context(features["ikun_global"])
        else:
            global_token = torch.zeros_like(local)
        local_token = local.unsqueeze(1) + self.local_position
        global_token = global_token.unsqueeze(1) + self.global_position
        attended, _ = self.local_global(
            query=local_token, key=global_token, value=global_token,
            need_weights=False,
        )
        visual = self.local_global_norm(local + attended[:, 0])
        text = F.normalize(self.text(query.reshape(1, -1))[0], dim=-1)
        if use_kum:
            guidance = self.kum(text).unsqueeze(0)
            visual = visual + visual * guidance
        guided = self.fusion(torch.cat((visual, local), dim=-1))

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
                track_ids: torch.Tensor, state=None,
                use_global: bool = True, use_kum: bool = True) -> dict:
        return self.forward_frame(
            features, query, track_ids, state,
            use_global=use_global, use_kum=use_kum,
        )
