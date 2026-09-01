"""Query-conditioned same-frame correspondence probe for Stage L33.

The module scores current-frame candidate observations.  Its visual input is a
frozen candidate/fragment pair descriptor; no source, pool, group, or state
namespace is accepted.  Word-level text tokens condition the visual descriptor
through cross-attention, rather than through a fixed additive association score.
"""
from __future__ import annotations

import torch
from torch import nn


class L33QueryHardNegativeProbe(nn.Module):
    def __init__(self, visual_dim: int = 51, text_dim: int = 768,
                 hidden: int = 128, heads: int = 4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.visual_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden))
        self.cross_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(3 * hidden), nn.Linear(3 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.relevance_head = nn.Linear(hidden, 1)
        self.null_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    @staticmethod
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.bool().float().unsqueeze(-1)
        return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)

    def forward(self, visual_features: torch.Tensor, query_tokens: torch.Tensor,
                query_mask: torch.Tensor):
        """Return candidate logits and a query-conditioned NULL logit.

        visual_features is [candidates, visual_dim], query_tokens is [tokens,
        text_dim] (or [1,tokens,text_dim]).  Each candidate gets the same
        query token sequence, and cross-attention is performed independently
        for that candidate's visual fragment descriptor.
        """
        visual_features = torch.nan_to_num(visual_features.float())
        query_tokens = torch.nan_to_num(query_tokens.float())
        if query_tokens.ndim == 2:
            query_tokens = query_tokens.unsqueeze(0)
        qmask = query_mask.bool()
        if qmask.ndim == 1:
            qmask = qmask.unsqueeze(0)
        q = self.text_proj(query_tokens)
        v = self.visual_proj(visual_features).unsqueeze(1)
        n = v.shape[0]
        q_for_candidates = q[:1].expand(n, -1, -1)
        attended, _ = self.cross_attention(
            q_for_candidates, v, v, need_weights=False)
        q_pool = self.masked_mean(q[:1], qmask[:1]).expand(n, -1)
        attended_pool = self.masked_mean(attended, qmask[:1].expand(n, -1))
        fused = self.fuse(torch.cat((q_pool, attended_pool, v[:, 0]), dim=-1))
        return {
            "relevance_logits": self.relevance_head(fused).squeeze(-1),
            "null_logit": self.null_head(q_pool[:1]).squeeze(),
            "candidate_embedding": fused,
        }
