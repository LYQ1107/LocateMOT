"""Stage L19 source-balanced, ungated semantic track retriever.

This module is separate from :mod:`l18_coverage_retrieve_repair`.  The L18
class and its checkpoints retain the original global coverage gate; L19 uses
the same frozen-bank correspondence trunk but treats coverage as a four-state
auxiliary task.  The emitted score never contains a global ``log(p_source)``
term in the default ``aux_only`` mode.
"""
from __future__ import annotations

import torch
from torch import nn

from locatemot.models.flexhook_bank_port import FlexHookBankPort
from locatemot.models.l18_coverage_retrieve_repair import STATE_NAMES


class L19UngatedRetriever(FlexHookBankPort):
    """Causal membership/presence retriever with auxiliary coverage states."""

    def __init__(self, hidden: int = 256, heads: int = 4,
                 dropout: float = 0.10, token_dim: int = 512,
                 use_slots: bool = False, holistic_only: bool = True,
                 coverage_mode: str = "aux_only"):
        if coverage_mode not in {"aux_only", "soft_residual"}:
            raise ValueError(f"unknown L19 coverage mode: {coverage_mode}")
        super().__init__(hidden=hidden, heads=heads, dropout=dropout,
                         token_dim=token_dim, use_slots=use_slots,
                         holistic_only=holistic_only)
        self.coverage_mode = str(coverage_mode)
        self.detach_state = False
        self.coverage_head = nn.Sequential(
            nn.Linear(5 * hidden + 4, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, len(STATE_NAMES)),
        )
        # This head sees only the query-independent observation trunk and its
        # numeric quality features.  It is validity/presence, not membership.
        self.validity_readout = nn.Sequential(
            nn.Linear(hidden + 32, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.coverage_temperature = nn.Parameter(torch.tensor(1.0))
        self.coverage_scale = nn.Parameter(torch.tensor(0.20))
        self.presence_scale = nn.Parameter(torch.tensor(0.75))
        self.confidence_scale = nn.Parameter(torch.tensor(0.25))
        self.confidence_readout = nn.Sequential(
            nn.Linear(hidden + 9, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # Named for the ablation report, but not applied by default.
        self.reserve_bias = nn.Parameter(torch.tensor(0.0))

    def _coverage(self, track: torch.Tensor, source: torch.Tensor,
                  context: dict) -> torch.Tensor:
        hidden = self.hidden
        zero = track.new_zeros(hidden)
        main = track[source == 0]
        reserve = track[source == 1]
        main_mean = main.mean(0) if len(main) else zero
        reserve_mean = reserve.mean(0) if len(reserve) else zero
        main_max = main.max(0).values if len(main) else zero
        reserve_max = reserve.max(0).values if len(reserve) else zero
        counts = track.new_tensor([
            float(len(main)) / 32.0, float(len(reserve)) / 32.0,
            float(bool(len(main))), float(bool(len(reserve))),
        ])
        inp = torch.cat((context["holistic"], main_mean, reserve_mean,
                         main_max, reserve_max, counts), -1)
        return self.coverage_head(inp)

    def forward_frame(self, features: dict, query: torch.Tensor,
                      family: torch.Tensor, track_ids: torch.Tensor,
                      state: dict | None = None,
                      query_tokens: torch.Tensor | None = None,
                      query_mask: torch.Tensor | None = None,
                      query_context: dict | None = None) -> dict:
        # Call the trunk directly so L18's global source gate is never
        # inherited accidentally.
        base = FlexHookBankPort.forward_frame(
            self, features, query, family, track_ids, state,
            query_tokens=query_tokens, query_mask=query_mask,
            query_context=query_context)
        context = base["query_context"]
        if not len(track_ids):
            zeros = context["holistic"].new_zeros(4)
            base.update({
                "logits": context["holistic"].new_zeros(0),
                "membership_logits": context["holistic"].new_zeros(0),
                "presence_logits": context["holistic"].new_zeros(0),
                "confidence_logits": context["holistic"].new_zeros(0),
                "coverage_logits": zeros,
                "state_probabilities": torch.softmax(zeros, -1),
                "coverage_gate": context["holistic"].new_zeros(0),
                "coverage_contribution": context["holistic"].new_zeros(0),
                "reserve_bias_contribution": context["holistic"].new_zeros(0),
                "score_components": {},
            })
            return base

        track = base["track_features"]
        aux = base["aux"]
        source = aux["source"]
        coverage_logits = self._coverage(track, source, context)
        temperature = self.coverage_temperature.abs().clamp_min(0.25)
        probabilities = torch.softmax(coverage_logits / temperature, -1)

        # Presence is explicitly query independent: only the observation trunk
        # and numeric quality features are used.
        presence = self.validity_readout(torch.cat((aux["base"],
                                                     aux["numeric"]), -1)).squeeze(-1)
        confidence = self.confidence_readout(torch.cat((aux["base"],
                                                         aux["numeric"][:, :9]), -1)).squeeze(-1)
        membership = base["membership_logits"]
        residual = track.new_zeros(len(track))
        if self.coverage_mode == "soft_residual":
            # Bounded per-candidate residual for the named ablation; it is not
            # a source death gate.
            p_main = probabilities[1] + 0.15 * probabilities[2]
            p_reserve = probabilities[2] + 0.10 * probabilities[1]
            source_p = torch.where(source == 0, p_main, p_reserve)
            residual = self.coverage_scale * (source_p - 0.5)
        reserve_bias = torch.where(
            source == 1, self.reserve_bias, source.new_zeros(()).float())
        final = (membership + self.presence_scale * presence +
                 self.confidence_scale * confidence + residual + reserve_bias)
        base.update({
            "logits": final,
            "membership_logits": membership,
            "presence_logits": presence,
            "confidence_logits": confidence,
            "coverage_logits": coverage_logits,
            "state_probabilities": probabilities,
            "coverage_gate": residual,
            "coverage_contribution": residual,
            "reserve_bias_contribution": reserve_bias,
            "score_components": {
                "membership": membership,
                "presence": presence,
                "confidence": confidence,
                "coverage": residual,
                "reserve_bias": reserve_bias,
            },
        })
        return base

    def forward(self, features: dict, query: torch.Tensor,
                family: torch.Tensor, track_ids: torch.Tensor,
                state: dict | None = None,
                query_tokens: torch.Tensor | None = None,
                query_mask: torch.Tensor | None = None,
                query_context: dict | None = None) -> dict:
        return self.forward_frame(features, query, family, track_ids, state,
                                  query_tokens=query_tokens,
                                  query_mask=query_mask,
                                  query_context=query_context)
