"""Token-level specification slots for Stage L18.

The module deliberately consumes cached token states rather than rebuilding a
dataset-specific lexical taxonomy.  It is small enough to run on the frozen
track bank and exposes slot reliability for the research audit.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


SLOT_NAMES = (
    "holistic", "entity", "appearance", "absolute_space", "relation",
    "motion", "temporal",
)


class L18QuerySlots(nn.Module):
    """Cross-attend learnable semantic slots to token-level text states."""

    def __init__(self, token_dim: int = 512, hidden: int = 256,
                 heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden = int(hidden)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden),
        )
        self.slot_queries = nn.Parameter(
            torch.randn(len(SLOT_NAMES), hidden) * 0.02)
        self.sentence_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(hidden)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, len(SLOT_NAMES)),
        )
        self.holistic_fusion = nn.Sequential(
            nn.Linear(2 * hidden + len(SLOT_NAMES), hidden), nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, token_states: torch.Tensor,
                token_mask: torch.Tensor | None = None) -> dict:
        """Return ``slots``, ``holistic`` and a normalized reliability gate.

        ``token_states`` is ``[tokens, token_dim]`` or ``[1, tokens,
        token_dim]``.  The bank evaluator uses one expression at a time, so a
        batch dimension is accepted only for convenience and removed in the
        returned tensors.
        """
        tokens = torch.nan_to_num(token_states.float())
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError(f"expected [T,D] or [1,T,D], got {tuple(tokens.shape)}")
        if token_mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool,
                              device=tokens.device)
        else:
            mask = token_mask.to(tokens.device).bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != tokens.shape[:2]:
                raise ValueError("token mask/state shape mismatch")
        projected = self.token_projection(tokens)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(projected.dtype)
        pooled = (projected * mask.unsqueeze(-1)).sum(dim=1) / counts
        raw_tokens = torch.nan_to_num(tokens)
        raw_pooled = (raw_tokens * mask.unsqueeze(-1)).sum(dim=1) / counts
        sentence = self.sentence_projection(raw_pooled)
        queries = self.slot_queries.unsqueeze(0) + sentence.unsqueeze(1)
        attended, attention = self.cross_attention(
            queries, projected, projected,
            key_padding_mask=~mask, need_weights=True,
            average_attn_weights=False)
        slots = self.slot_norm(queries + attended)[0]
        pooled_hidden = pooled[0]
        gate_input = torch.cat((slots.mean(dim=0), pooled_hidden), dim=-1)
        reliability = torch.softmax(self.gate(gate_input), dim=-1)
        holistic = self.holistic_fusion(
            torch.cat((slots[0], pooled_hidden, reliability), dim=-1))
        return {
            "slots": slots,
            "holistic": F.normalize(holistic, dim=-1),
            "reliability": reliability,
            "attention": attention[0],
            "token_projection": projected[0],
        }
