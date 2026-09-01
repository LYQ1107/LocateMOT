"""Controlled FlexHook-on-LocalAnything bank port.

This is an adapter, not a copy of the official raw-image implementation.  It
keeps the auditable parts of FlexHook—language-conditioned token evidence and
pairwise correspondence—while replacing unavailable ROPE-Swin feature maps
with frozen LocateMOT bank observations.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.l16_track_selector import FAMILY_NAMES
from locatemot.models.l18_query_slots import L18QuerySlots


class Projection(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(value.float()))


class FlexHookBankPort(nn.Module):
    """Causal track-set correspondence over frozen bank observations."""

    def __init__(self, hidden: int = 256, heads: int = 4,
                 dropout: float = 0.10, token_dim: int = 512,
                 use_slots: bool = True, holistic_only: bool = False):
        super().__init__()
        self.hidden = int(hidden)
        self.use_slots = bool(use_slots)
        self.holistic_only = bool(holistic_only)
        self.slots = L18QuerySlots(token_dim, hidden, heads, dropout)
        self.clip_current = Projection(512, hidden, dropout)
        self.clip_history = Projection(512, hidden, dropout)
        self.pbd = Projection(2048, hidden, dropout)
        self.identity = Projection(384, hidden, dropout)
        self.numeric = Projection(32, hidden, dropout)
        self.source_embedding = nn.Embedding(2, hidden)
        self.observation = nn.Sequential(
            nn.Linear(6 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.temporal = nn.GRUCell(hidden, hidden)
        self.correspondence = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.corr_norm = nn.LayerNorm(hidden)
        self.pairwise = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.presence_readout = nn.Sequential(
            nn.Linear(2 * hidden + 32, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.family = Projection(len(FAMILY_NAMES), hidden, dropout)
        self.query_fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        self.cosine_scale = nn.Parameter(torch.tensor(2.0))
        self.history_cosine_scale = nn.Parameter(torch.tensor(0.75))
        self.output_bias = nn.Parameter(torch.tensor(-1.0))
        # L18 keeps a detached causal cache for inference and its historical
        # training protocol.  L19 turns this off and detaches only at explicit
        # truncated-BPTT chunk boundaries in the trainer.
        self.detach_state = True

    @staticmethod
    def _previous(track_ids: torch.Tensor, state: Dict[int, torch.Tensor],
                  hidden: int, device: torch.device) -> torch.Tensor:
        zero = torch.zeros(hidden, device=device)
        values = [state.get(int(track_id), zero)
                  for track_id in track_ids.detach().cpu().tolist()]
        return torch.stack(values, dim=0) if values else zero.new_zeros((0, hidden))

    def query_context(self, query_tokens: torch.Tensor | None,
                      query: torch.Tensor, family: torch.Tensor,
                      query_mask: torch.Tensor | None = None) -> dict:
        if not self.use_slots or query_tokens is None:
            query_tokens = query.float().reshape(1, -1)
            # The fallback is one query token, so a cached 77-token mask would
            # have an incompatible shape when L19 intentionally uses ordinary
            # latent slots instead of named specification slots.
            query_mask = None
        context = self.slots(query_tokens, query_mask)
        if self.holistic_only:
            slots = context["holistic"].unsqueeze(0)
        else:
            slots = context["slots"]
        family_h = self.family(family.float().reshape(1, -1))[0]
        fused = self.query_fusion(torch.cat((context["holistic"], family_h), -1))
        holistic = F.normalize(
            context["holistic"] + family_h + 0.10 * fused, dim=-1)
        return {**context, "slots": slots, "holistic": holistic}

    def _encode_tracks(self, features: dict, track_ids: torch.Tensor,
                       state: Dict[int, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        numeric = torch.cat((
            features["geometry"], features["motion"], features["context"],
            features["lifecycle"], features["objectness"].reshape(-1, 1)), -1)
        source = features.get("pool_id", torch.zeros(
            len(track_ids), dtype=torch.long, device=track_ids.device)).long()
        source = source.clamp(0, 1)
        base = self.observation(torch.cat((
            self.clip_current(features["clip"]),
            self.clip_history(features["history_clip"]),
            self.pbd(features["pbd"]),
            self.identity(features["uidm_h"]),
            self.numeric(numeric),
            self.source_embedding(source),
        ), -1))
        previous = self._previous(track_ids, state, self.hidden, track_ids.device)
        temporal = self.temporal(base, previous)
        return temporal, {"source": source, "numeric": numeric, "base": base}

    def forward_frame(self, features: dict, query: torch.Tensor,
                      family: torch.Tensor, track_ids: torch.Tensor,
                      state: Dict[int, torch.Tensor] | None = None,
                      query_tokens: torch.Tensor | None = None,
                      query_mask: torch.Tensor | None = None,
                      query_context: dict | None = None) -> dict:
        state = {} if state is None else state
        context = query_context if query_context is not None else \
            self.query_context(query_tokens, query, family, query_mask)
        n = int(track_ids.numel())
        if not n:
            return {
                "logits": query.new_zeros(0),
                "membership_logits": query.new_zeros(0),
                "presence_logits": query.new_zeros(0),
                "state": state, "track_embedding": query.new_zeros((0, self.hidden)),
                "query_context": context,
            }
        temporal, aux = self._encode_tracks(features, track_ids, state)
        slots = context["slots"].unsqueeze(0)
        attended, _ = self.correspondence(
            temporal.unsqueeze(0), slots, slots, need_weights=False)
        corr = self.corr_norm(temporal + attended[0])
        pair = self.pairwise(torch.cat((
            corr, context["holistic"].unsqueeze(0).expand(n, -1),
            corr * context["holistic"].unsqueeze(0),
            (corr - context["holistic"].unsqueeze(0)).abs(),
        ), -1))
        membership = self.readout(torch.cat((pair, corr), -1)).squeeze(-1)
        crop = F.normalize(torch.nan_to_num(features["clip"].float()), dim=-1)
        history = F.normalize(
            torch.nan_to_num(features["history_clip"].float()), dim=-1)
        raw_query = F.normalize(query.float().reshape(1, -1), dim=-1)[0]
        membership = membership + self.output_bias + \
            self.cosine_scale * (crop * raw_query).sum(-1) + \
            self.history_cosine_scale * (history * raw_query).sum(-1)
        presence_input = torch.cat((
            corr, pair, aux["numeric"],
        ), -1)
        presence = self.presence_readout(presence_input).squeeze(-1)
        new_state = dict(state)
        for index, track_id in enumerate(track_ids.detach().cpu().tolist()):
            value = corr[index].detach() if self.detach_state else corr[index]
            new_state[int(track_id)] = value
        return {
            "logits": membership,
            "membership_logits": membership,
            "presence_logits": presence,
            "state": new_state,
            "track_embedding": F.normalize(corr, dim=-1),
            "query_context": context,
            "track_features": corr,
            "aux": aux,
        }

    def forward(self, features: dict, query: torch.Tensor,
                family: torch.Tensor, track_ids: torch.Tensor,
                state: Dict[int, torch.Tensor] | None = None,
                query_tokens: torch.Tensor | None = None,
                query_mask: torch.Tensor | None = None,
                query_context: dict | None = None) -> dict:
        return self.forward_frame(features, query, family, track_ids, state,
                                  query_tokens=query_tokens,
                                  query_mask=query_mask,
                                  query_context=query_context)
