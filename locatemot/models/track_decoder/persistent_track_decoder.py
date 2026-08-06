"""B4: Persistent Track Decoder with configurable query direction."""
from __future__ import annotations

import math

import torch
from torch import nn


class TrackDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )

    def forward(self, q, kv, q_mask, kv_mask):
        h = q
        h2, _ = self.self_attn(
            h, h, h,
            key_padding_mask=None if q_mask is None else ~q_mask,
        )
        h = self.norm1(h + h2)
        h2, _ = self.cross_attn(
            h, kv, kv,
            key_padding_mask=None if kv_mask is None else ~kv_mask,
        )
        h = self.norm2(h + h2)
        h = self.norm3(h + self.ffn(h))
        return h


class PersistentTrackDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        query_direction: str = "reference_query",
    ):
        super().__init__()
        assert query_direction in ("reference_query", "current_query")
        self.query_direction = query_direction
        self.layers = nn.ModuleList([
            TrackDecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.cur_key = nn.Linear(d_model, d_model)
        self.ref_key = nn.Linear(d_model, d_model)
        self.no_match_head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
        self.aux_iou_head = nn.Sequential(nn.Linear(d_model * 2, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(
        self,
        ref_tokens: torch.Tensor,
        cur_tokens: torch.Tensor,
        ref_mask: torch.Tensor,
        cur_mask: torch.Tensor,
    ) -> dict:
        B, M, D = ref_tokens.shape
        N = cur_tokens.shape[1]
        if self.query_direction == "reference_query":
            q, kv, q_mask, kv_mask = ref_tokens, cur_tokens, ref_mask, cur_mask
        else:
            q, kv, q_mask, kv_mask = cur_tokens, ref_tokens, cur_mask, ref_mask
        h = q
        for layer in self.layers:
            h = layer(h, kv, q_mask, kv_mask)
        if self.query_direction == "reference_query":
            ref_out = h
            cur_out = kv
            match = torch.bmm(ref_out, self.cur_key(cur_out).transpose(1, 2)) / math.sqrt(D)
            no_match = self.no_match_head(ref_out).squeeze(-1)
        else:
            cur_out = h
            ref_out = kv
            match = torch.bmm(cur_out, self.ref_key(ref_out).transpose(1, 2)) / math.sqrt(D)
            match = match.transpose(1, 2)  # [B,M,N]
            no_match = self.no_match_head(ref_out).squeeze(-1)
        iou_logit = None
        return {
            "match_logits": match,
            "no_match_logits": no_match,
            "ref_feats": ref_out,
            "cur_feats": cur_out,
            "iou_logit": iou_logit,
        }

    def iou_predict(self, ref_feats, cur_feats, ref_geom, cur_geom):
        # auxiliary geometry head on matched pairs [B,M,N]
        B, M, N, D = ref_feats.shape[0], ref_feats.shape[1], cur_feats.shape[1], ref_feats.shape[2]
        rf = ref_feats.unsqueeze(2).expand(B, M, N, D)
        cf = cur_feats.unsqueeze(1).expand(B, M, N, D)
        return self.aux_iou_head(torch.cat([rf, cf], dim=-1)).squeeze(-1)
