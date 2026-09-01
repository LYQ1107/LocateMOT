"""Lightweight trajectory encoder producing a raw-space fused ObjectToken.

Design evidence (docs/l1_a_reference_audit.md):
- FDTA Temporal_Adapter uses a causal temporal transformer + position/time
  embedding + missing-frame mask to aggregate detection history.
- MOTIP keeps a truncated history buffer and encodes boxes + time together.

This is a clean reimplementation: K history observations are encoded with a
2-layer causal transformer, then attention-pooled into per-modality fusion
weights. The fused vectors stay in the raw feature space (pbd 2048, region
4608, geom 5, gen) so the frozen L0-D B6 FeatureProjector can consume them as
if they were a single-frame reference token.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class _CausalTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        # x: [B,K,D]; key_padding_mask: [B,K] True=invalid
        causal = torch.triu(
            torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device),
            diagonal=1,
        )
        kp = key_padding_mask.float().unsqueeze(1) * -1e9  # [B,1,K]
        attn_mask = (causal[None] + kp).repeat_interleave(self.attn.num_heads, dim=0)
        # always allow self-attention so padded positions (all keys masked)
        # never produce NaN softmax; padding is excluded later by pooling mask
        eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
        attn_mask = attn_mask.masked_fill(eye, 0.0)
        h = self.attn(x, x, x, attn_mask=attn_mask)[0]
        x = self.norm1(x + self.dropout(h))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TrajectoryEncoder(nn.Module):
    """Aggregate K history ObjectToken observations into one fused token.

    Inputs (all [B,K,*]):
      pbd: float, 2048
      region: float, 4608 (may be zeros if unavailable)
      geom: float, 5 (normalized xyxy + area)
      gen: float, 1
      gaps: float, frame offset from current frame (>=1 for valid history)
      mask: bool, True=invalid (padding)
    Outputs: fused pbd/region/geom/gen + attention weights.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_k: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_k = max_k
        self.geom_proj = nn.Sequential(nn.Linear(5, 48), nn.GELU())
        self.gen_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.time_proj = nn.Sequential(nn.Linear(2, 32), nn.GELU())
        self.feat_proj = nn.Sequential(
            nn.Linear(48 + 16 + 32, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
        )
        self.layers = nn.ModuleList(
            [_CausalTransformerLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.pool_q = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.pool_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.valid_norm = 1e-6

    def _time_emb(self, gaps):
        return torch.stack([torch.log1p(gaps), (gaps / 100.0).clamp(max=1.0)], dim=-1)

    def forward(self, pbd, region, geom, gen, gaps, mask):
        B, K, _ = pbd.shape
        g = self.geom_proj(geom)
        s = self.gen_proj(gen.unsqueeze(-1))
        t = self.time_proj(self._time_emb(gaps))
        x = self.feat_proj(torch.cat([g, s, t], dim=-1))
        kp_mask = mask  # True=invalid
        for layer in self.layers:
            x = layer(x, kp_mask)
        scores = torch.einsum("bkd,d->bk", self.pool_proj(x), self.pool_q)
        scores = scores.masked_fill(mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)  # [B,K]
        w = w.masked_fill(mask, 0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + self.valid_norm)
        fused_pbd = torch.einsum("bk,bkd->bd", w, pbd)
        fused_region = torch.einsum("bk,bkd->bd", w, region)
        # last valid observation for geometry / gen
        last_idx = (~mask).long().cumsum(dim=1).argmax(dim=1)
        batch_idx = torch.arange(B, device=pbd.device)
        fused_geom = geom[batch_idx, last_idx]
        fused_gen = gen[batch_idx, last_idx]
        return {
            "pbd": fused_pbd,
            "region": fused_region,
            "geom": fused_geom,
            "gen": fused_gen,
            "weights": w,
        }
