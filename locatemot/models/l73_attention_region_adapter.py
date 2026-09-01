"""Small RMOT-only adapter for the L73 post-fusion attention representation.

The LocateAnything model is not imported here.  This module consumes only the
detached, per-unit summaries produced by the L73 frozen prefill audit:
projected post-fusion region values, the query hidden summary, and the scalar
text-to-image attention score.  It deliberately has no teacher score, ID,
tracker, or candidate filtering path.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class L73AttentionRegionAdapter(nn.Module):
    """Independent candidate relevance and query-conditioned null head."""

    def __init__(self, region_dim: int = 2048, query_dim: int = 2048,
                 hidden: int = 128) -> None:
        super().__init__()
        if hidden < 1 or region_dim < 1 or query_dim < 1:
            raise ValueError("dimensions must be positive")
        self.region_dim = int(region_dim)
        self.query_dim = int(query_dim)
        self.hidden = int(hidden)
        self.region_proj = nn.Sequential(
            nn.LayerNorm(self.region_dim),
            nn.Linear(self.region_dim, self.hidden),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.query_dim),
            nn.Linear(self.query_dim, self.hidden),
        )
        self.attention_score_proj = nn.Linear(1, self.hidden)
        self.candidate_head = nn.Sequential(
            nn.LayerNorm(self.hidden * 3),
            nn.Linear(self.hidden * 3, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.null_head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.GELU(),
            nn.Linear(self.hidden // 2, 1),
        )

    def forward(self, query_hidden: torch.Tensor,
                region_values: torch.Tensor,
                attention_scores: torch.Tensor) -> dict[str, torch.Tensor]:
        if query_hidden.ndim == 1:
            query_hidden = query_hidden.unsqueeze(0)
        if query_hidden.ndim != 2 or query_hidden.shape[0] != 1:
            raise ValueError(f"query_hidden must be [1,D], got {tuple(query_hidden.shape)}")
        if region_values.ndim != 2 or region_values.shape[1] != self.region_dim:
            raise ValueError(
                f"region_values must be [N,{self.region_dim}], got {tuple(region_values.shape)}"
            )
        if attention_scores.ndim == 1:
            attention_scores = attention_scores.unsqueeze(-1)
        if attention_scores.ndim != 2 or attention_scores.shape != (region_values.shape[0], 1):
            raise ValueError(
                "attention_scores must be [N,1] aligned with region_values, "
                f"got {tuple(attention_scores.shape)}"
            )
        if query_hidden.shape[1] != self.query_dim:
            raise ValueError(f"query dimension mismatch: {query_hidden.shape[1]} != {self.query_dim}")
        if not (torch.isfinite(query_hidden).all() and
                torch.isfinite(region_values).all() and
                torch.isfinite(attention_scores).all()):
            raise ValueError("nonfinite adapter inputs")
        q = self.query_proj(query_hidden.float())
        v = self.region_proj(region_values.float())
        a = self.attention_score_proj(attention_scores.float())
        q_one = q.expand(region_values.shape[0], -1)
        pair = torch.cat((q_one, v, a), dim=-1)
        candidate_logits = self.candidate_head(pair).squeeze(-1)
        null_logit = self.null_head(q).squeeze(-1)
        return {
            "candidate_logits": candidate_logits,
            "null_logit": null_logit,
            "query_vector": F.normalize(q, dim=-1),
            "region_vectors": F.normalize(v, dim=-1),
        }


def adapter_config(model: L73AttentionRegionAdapter) -> dict[str, Any]:
    return {
        "class": type(model).__name__,
        "region_dim": model.region_dim,
        "query_dim": model.query_dim,
        "hidden": model.hidden,
        "inputs": ["postfusion_attention_weighted_region_value", "query_hidden", "attention_score"],
        "outputs": ["candidate_logits", "query_conditioned_null_logit"],
        "forbidden_inputs": [
            "source_id", "pool_id", "group_id", "track_id", "query_id",
            "state_id", "L29_score", "L70_score", "GT_identity",
        ],
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_mask": "UNALIGNED",
    }
