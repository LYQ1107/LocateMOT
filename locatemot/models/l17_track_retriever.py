"""Precision-first causal track-set retrieval for LocateMOT Stage L17."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.l16_track_selector import FAMILY_NAMES


class Projection(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(value.float()))


class L17TrackSetRetriever(nn.Module):
    """Static/dynamic track evidence with an explicit frame-null decision."""

    def __init__(self, hidden: int = 256, dropout: float = 0.10,
                 holistic_query: bool = False):
        super().__init__()
        self.hidden = int(hidden)
        self.holistic_query = bool(holistic_query)
        self.clip = Projection(1024, hidden, dropout)
        self.pbd = Projection(2048, hidden, dropout)
        self.identity = Projection(384, hidden, dropout)
        self.numeric = Projection(32, hidden, dropout)
        self.static_fusion = Projection(2 * hidden, hidden, dropout)
        self.dynamic_fusion = Projection(2 * hidden, hidden, dropout)
        if self.holistic_query:
            self.query_holistic = Projection(
                512 + len(FAMILY_NAMES), hidden, dropout)
        else:
            self.query_static = Projection(
                512 + len(FAMILY_NAMES), hidden, dropout)
            self.query_dynamic = Projection(
                512 + len(FAMILY_NAMES), hidden, dropout)
        self.branch_gate = nn.Sequential(
            nn.Linear(512 + len(FAMILY_NAMES), hidden), nn.GELU(),
            nn.Linear(hidden, 2))
        self.track_gru = nn.GRUCell(hidden, hidden)
        self.match = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden))
        self.membership = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        self.null = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        self.cosine_scale = nn.Parameter(torch.tensor(2.0))
        self.current_clip_scale = nn.Parameter(torch.tensor(2.0))
        self.history_clip_scale = nn.Parameter(torch.tensor(1.0))
        self.membership_bias = nn.Parameter(torch.tensor(-1.0))

    @staticmethod
    def _previous(track_ids: torch.Tensor, state: Dict[int, torch.Tensor],
                  hidden: int, device: torch.device) -> torch.Tensor:
        zero = torch.zeros(hidden, device=device)
        return torch.stack([
            state.get(int(track_id), zero)
            for track_id in track_ids.detach().cpu().tolist()
        ])

    def forward_frame(self, features: dict, query: torch.Tensor,
                      family: torch.Tensor, track_ids: torch.Tensor,
                      state: Dict[int, torch.Tensor] | None = None,
                      use_null: bool = True,
                      use_identity: bool = True) -> dict:
        state = {} if state is None else state
        count = int(track_ids.numel())
        if not count:
            query_input = torch.cat((
                query.float().reshape(-1), family.float().reshape(-1)))
            if self.holistic_query:
                query_static = query_dynamic = F.normalize(
                    self.query_holistic(query_input.reshape(1, -1))[0],
                    dim=-1)
                weights = query_input.new_tensor([0.5, 0.5])
            else:
                query_static = F.normalize(
                    self.query_static(query_input.reshape(1, -1))[0], dim=-1)
                query_dynamic = F.normalize(
                    self.query_dynamic(query_input.reshape(1, -1))[0], dim=-1)
                weights = torch.softmax(
                    self.branch_gate(query_input.reshape(1, -1))[0], dim=-1)
            query_embedding = F.normalize(
                weights[0] * query_static + weights[1] * query_dynamic, dim=-1)
            pooled = query_embedding.new_zeros(2 * self.hidden)
            null_logit = self.null(
                torch.cat((query_embedding, pooled), -1)).squeeze()
            return {"logits": query.new_zeros(0), "membership_logits":
                    query.new_zeros(0), "null_logit": null_logit,
                    "state": state, "query_embedding": query_embedding,
                    "branch_weights": weights}
        numeric = torch.cat((
            features["geometry"], features["motion"], features["context"],
            features["lifecycle"], features["objectness"].reshape(-1, 1)), -1)
        static = self.static_fusion(torch.cat((
            self.clip(torch.cat((features["clip"], features["history_clip"]), -1)),
            self.pbd(features["pbd"])), -1))
        identity = self.identity(features["uidm_h"])
        if not use_identity:
            identity = torch.zeros_like(identity)
        dynamic = self.dynamic_fusion(torch.cat((identity, self.numeric(numeric)), -1))
        query_input = torch.cat((query.float().reshape(-1), family.float().reshape(-1)))
        if self.holistic_query:
            query_static = query_dynamic = F.normalize(
                self.query_holistic(query_input.reshape(1, -1))[0], dim=-1)
            weights = query_input.new_tensor([0.5, 0.5])
        else:
            query_static = F.normalize(
                self.query_static(query_input.reshape(1, -1))[0], dim=-1)
            query_dynamic = F.normalize(
                self.query_dynamic(query_input.reshape(1, -1))[0], dim=-1)
            weights = torch.softmax(
                self.branch_gate(query_input.reshape(1, -1))[0], dim=-1)
        observation = weights[0] * static + weights[1] * dynamic
        previous = self._previous(track_ids, state, self.hidden, track_ids.device)
        track = self.track_gru(observation, previous)
        query_embedding = F.normalize(
            weights[0] * query_static + weights[1] * query_dynamic, dim=-1)
        expanded = query_embedding.unsqueeze(0).expand(count, -1)
        matched = self.match(torch.cat((
            track, expanded, track * expanded, (track - expanded).abs()), -1))
        normalized_track = F.normalize(track, dim=-1)
        membership_logits = self.membership(matched).squeeze(-1) + \
            self.membership_bias + self.cosine_scale * \
            (normalized_track * expanded).sum(-1)
        raw_query = F.normalize(query.float().reshape(1, -1), dim=-1)[0]
        current_clip = F.normalize(
            torch.nan_to_num(features["clip"].float()), dim=-1)
        history_clip = F.normalize(
            torch.nan_to_num(features["history_clip"].float()), dim=-1)
        membership_logits = membership_logits + \
            self.current_clip_scale * (current_clip * raw_query).sum(-1) + \
            self.history_clip_scale * (history_clip * raw_query).sum(-1)
        pooled_max = matched.max(0).values
        pooled_mean = matched.mean(0)
        null_logit = self.null(torch.cat((
            query_embedding, pooled_max, pooled_mean), -1)).squeeze()
        logits = membership_logits - null_logit if use_null else membership_logits
        new_state = dict(state)
        for index, raw_track_id in enumerate(track_ids.detach().cpu().tolist()):
            new_state[int(raw_track_id)] = track[index]
        return {
            "logits": logits, "membership_logits": membership_logits,
            "null_logit": null_logit, "state": new_state,
            "track_embedding": F.normalize(track, dim=-1),
            "query_embedding": query_embedding, "branch_weights": weights,
        }

    def forward(self, features: dict, query: torch.Tensor,
                family: torch.Tensor, track_ids: torch.Tensor, state=None,
                use_null: bool = True, use_identity: bool = True) -> dict:
        return self.forward_frame(features, query, family, track_ids, state,
                                  use_null=use_null, use_identity=use_identity)
