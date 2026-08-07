"""B6: relation-aware Persistent Track Decoder with strong-prior residual."""
from __future__ import annotations

import math

import torch
from torch import nn

from .features import FeatureProjector
from .persistent_track_decoder import PersistentTrackDecoder
from .relation_encoder import RelationEncoder
from .relation_features import base_affinity
from .relation_pairwise import _gap_embedding


class RelationTrackDecoderModel(nn.Module):
    """B6 modifies B4 in place:
    - adds relation embedding / scalar score / per-head attention bias
    - final affinity = BaseAffinity + alpha * tanh(bmm_logit + rel_score)
    - keeps reference self-attention and [M,N+M] NO_MATCH assignment.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_pbd_base: bool = True,
        use_region_geom: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.proj = FeatureProjector(d_model, dropout)
        self.decoder = PersistentTrackDecoder(
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            ffn_dim=ffn_dim, dropout=dropout, query_direction="reference_query",
        )
        self.rel_enc = RelationEncoder(d_model, 128, num_heads, use_region_geom, dropout)
        self.rel_score_head = nn.Linear(128, 1)
        self.no_match_head = nn.Sequential(
            nn.Linear(d_model + 2 + 3, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.w_iou = nn.Parameter(torch.tensor(0.5))
        self.w_pbd = nn.Parameter(torch.tensor(0.5)) if use_pbd_base else None
        self.alpha_logit = nn.Parameter(torch.tensor(-1.0986))
        self.beta_logit = nn.Parameter(torch.tensor(-2.9444))  # sigmoid -> ~0.05

    @property
    def alpha(self):
        return 0.5 * torch.sigmoid(self.alpha_logit)

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def forward(self, batch):
        ref = self.proj(
            batch["ref_pbd"], batch["ref_region"], batch["ref_geom"],
            batch["ref_gen"], batch["ref_cat"],
        )
        cur = self.proj(
            batch["cur_pbd"], batch["cur_region"], batch["cur_geom"],
            batch["cur_gen"], batch["cur_cat"],
        )
        B, M, D = ref.shape
        N = cur.shape[1]
        rel_emb, rel_score, per_head_bias = self.rel_enc(batch)  # B,M,N,H
        cross_bias = (self.beta * per_head_bias).reshape(B * self.decoder.layers[0].self_attn.num_heads, M, N)
        h = ref
        for layer in self.decoder.layers:
            h = layer(h, cur, batch["ref_mask"], batch["cur_mask"], cross_bias=cross_bias)
        ref_out = h
        cur_out = cur
        bmm = torch.bmm(ref_out, self.decoder.cur_key(cur_out).transpose(1, 2)) / math.sqrt(D)
        residual = bmm + self.rel_score_head(rel_emb).squeeze(-1)
        base = base_affinity(batch, self.w_iou, self.w_pbd)
        if self.residual:
            match = base + self.alpha * torch.tanh(residual)
        else:
            match = base + residual
        max_match = match.max(dim=2).values
        n_cand = torch.log1p(batch["cur_mask"].float().sum(dim=1)).unsqueeze(1).expand(B, M)
        gap_emb = _gap_embedding(batch).unsqueeze(1).expand(B, M, 2)
        no_match = self.no_match_head(torch.cat([
            ref_out, gap_emb, max_match.unsqueeze(-1),
            n_cand.unsqueeze(-1), batch["ref_gen"].float().unsqueeze(-1),
        ], dim=-1)).squeeze(-1)
        iou_logit = self.decoder.iou_predict(
            ref_out, cur_out, batch["ref_geom"], batch["cur_geom"])
        return {
            "match_logits": match,
            "no_match_logits": no_match,
            "ref_feats": ref_out,
            "cur_feats": cur_out,
            "iou_logit": iou_logit,
        }
