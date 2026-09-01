"""L77 frozen-region / word-token correspondence head.

This is an RMOT-only small adapter.  It receives a complete current-frame
candidate set of frozen L69 region vectors and a masked L48 word-token
sequence.  IDs, old scores and tracker state are intentionally absent from
the interface.  The membership score is bounded before calibration.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class L77RegionCrossAttention(nn.Module):
    def __init__(self, region_dim: int = 512, text_dim: int = 768, hidden: int = 192,
                 heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.region_dim = int(region_dim)
        self.text_dim = int(text_dim)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.temperature = 0.25
        self.region_norm = nn.LayerNorm(self.region_dim)
        self.region_proj = nn.Linear(self.region_dim, self.hidden)
        self.text_norm = nn.LayerNorm(self.text_dim)
        self.text_proj = nn.Linear(self.text_dim, self.hidden)
        self.region_token_norm = nn.LayerNorm(self.hidden)
        self.word_to_region = nn.MultiheadAttention(
            self.hidden, self.heads, dropout=float(dropout), batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.hidden)
        set_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=self.heads,
            dim_feedforward=self.hidden * 2, dropout=float(dropout),
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.set_competition = nn.TransformerEncoder(set_layer, num_layers=1)
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.hidden), nn.Linear(self.hidden, self.hidden // 2),
            nn.GELU(), nn.Linear(self.hidden // 2, 1),
        )
        self.absent_head = nn.Sequential(
            nn.LayerNorm(self.hidden * 2), nn.Linear(self.hidden * 2, self.hidden // 2),
            nn.GELU(), nn.Linear(self.hidden // 2, 1),
        )

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor, dim: int) -> Tensor:
        weight = mask.to(value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        denom = weight.sum(dim=dim).clamp_min(1.0)
        return (value * weight).sum(dim=dim) / denom

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        region = batch["region"]
        text = batch["text"]
        text_mask = batch["text_mask"].bool()
        if region.ndim != 2 or region.shape[1] != self.region_dim or region.shape[0] == 0:
            raise ValueError(f"region must be [N,{self.region_dim}], got {tuple(region.shape)}")
        if text.ndim != 2 or text.shape[1] != self.text_dim or text_mask.shape != text.shape[:1]:
            raise ValueError(f"text/mask shape mismatch: {tuple(text.shape)} / {tuple(text_mask.shape)}")
        if not bool(text_mask.any()):
            raise ValueError("text mask has no valid token")
        if not torch.isfinite(region).all() or not torch.isfinite(text).all():
            raise ValueError("nonfinite L77 input")

        region_token = F.normalize(self.region_proj(self.region_norm(region)), dim=-1)
        text_token = F.normalize(self.text_proj(self.text_norm(text)), dim=-1)
        query = self.region_token_norm(region_token).unsqueeze(0)
        key_value = text_token.unsqueeze(0)
        cross, attention = self.word_to_region(
            query, key_value, key_value,
            key_padding_mask=(~text_mask).unsqueeze(0), need_weights=True,
            average_attn_weights=False,
        )
        candidate_tokens = self.cross_norm(query + cross).squeeze(0)
        # Every candidate is passed to one shared set block.  No row is
        # deleted, ranked, sampled, or NMS-filtered here.
        set_tokens = self.set_competition(candidate_tokens.unsqueeze(0)).squeeze(0)
        raw_score = self.score_head(set_tokens).squeeze(-1)
        match_logits = 2.0 * torch.tanh(raw_score / 2.0)
        query_vector = F.normalize(self._masked_mean(text_token, text_mask, dim=0), dim=-1)
        set_summary = set_tokens.mean(dim=0)
        absent_logit = self.absent_head(torch.cat((query_vector, set_summary), dim=-1)).reshape(1)
        return {
            "match_logits": match_logits,
            "raw_score": raw_score,
            "query_vector": query_vector,
            "candidate_tokens": candidate_tokens,
            "set_tokens": set_tokens,
            "cross_attention": attention,
            "absent_logit": absent_logit,
        }

    def parameter_summary(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": int(total), "trainable_parameters": int(trainable),
            "region_dim": self.region_dim, "text_dim": self.text_dim,
            "hidden": self.hidden, "heads": self.heads, "temperature": self.temperature,
            "bounded_score": "2*tanh(raw_score/2)",
        }
