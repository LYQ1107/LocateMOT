"""Coverage-aware retrieve-and-repair model for Stage L18."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.flexhook_bank_port import FlexHookBankPort


STATE_NAMES = (
    "ABSENT", "MAIN_COVERED", "RESERVE_COVERED", "PRESENT_UNCOVERED",
)


class L18CARRRetriever(FlexHookBankPort):
    """Three-task causal retriever with explicit four-state coverage."""

    def __init__(self, hidden: int = 256, heads: int = 4,
                 dropout: float = 0.10, token_dim: int = 512,
                 use_slots: bool = True, holistic_only: bool = False,
                 use_coverage: bool = True):
        super().__init__(hidden=hidden, heads=heads, dropout=dropout,
                         token_dim=token_dim, use_slots=use_slots,
                         holistic_only=holistic_only)
        self.use_coverage = bool(use_coverage)
        self.coverage_head = nn.Sequential(
            nn.Linear(5 * hidden + 4, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, len(STATE_NAMES)),
        )
        self.coverage_temperature = nn.Parameter(torch.tensor(1.0))
        self.reserve_bias = nn.Parameter(torch.tensor(-0.25))
        self.coverage_scale = nn.Parameter(torch.tensor(1.0))
        self.presence_scale = nn.Parameter(torch.tensor(0.75))

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
        base = super().forward_frame(
            features, query, family, track_ids, state,
            query_tokens=query_tokens, query_mask=query_mask,
            query_context=query_context)
        context = base["query_context"]
        if not len(track_ids):
            coverage_logits = self.coverage_head(torch.cat((
                context["holistic"], context["holistic"].new_zeros(4 * self.hidden),
                context["holistic"].new_zeros(4),
            ), -1)) if self.use_coverage else context["holistic"].new_zeros(4)
            base["coverage_logits"] = coverage_logits
            base["state_probabilities"] = torch.softmax(coverage_logits, -1)
            return base
        track = base["track_features"]
        source = base["aux"]["source"]
        coverage_logits = self._coverage(track, source, context) \
            if self.use_coverage else track.new_zeros(4)
        temperature = self.coverage_temperature.abs().clamp_min(0.25)
        probabilities = torch.softmax(coverage_logits / temperature, -1)
        p_main = probabilities[1] + 0.15 * probabilities[2]
        p_reserve = probabilities[2] + 0.10 * probabilities[1]
        gate = torch.where(source == 0, p_main.clamp_min(1e-4),
                          p_reserve.clamp_min(1e-4)).log()
        final_logits = base["membership_logits"] + \
            self.presence_scale * base["presence_logits"] + \
            self.coverage_scale * gate + torch.where(
                source == 1, self.reserve_bias, source.new_zeros(()).float())
        base.update({
            "logits": final_logits,
            "coverage_logits": coverage_logits,
            "state_probabilities": probabilities,
            "coverage_gate": gate,
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
