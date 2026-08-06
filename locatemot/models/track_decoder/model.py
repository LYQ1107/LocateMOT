"""End-to-end wrappers around FeatureProjector + B3/B4 heads."""
from __future__ import annotations

import torch
from torch import nn

from .features import FeatureProjector
from .pairwise_mlp import PairwiseMLP
from .persistent_track_decoder import PersistentTrackDecoder


class TrackDecoderModel(nn.Module):
    def __init__(self, d_model=256, num_layers=4, num_heads=8, ffn_dim=1024, dropout=0.1,
                 query_direction="reference_query"):
        super().__init__()
        self.proj = FeatureProjector(d_model, dropout)
        self.decoder = PersistentTrackDecoder(
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            ffn_dim=ffn_dim, dropout=dropout, query_direction=query_direction,
        )

    def forward(self, batch):
        ref_tokens = self.proj(
            batch["ref_pbd"], batch["ref_region"], batch["ref_geom"],
            batch["ref_gen"], batch["ref_cat"],
        )
        cur_tokens = self.proj(
            batch["cur_pbd"], batch["cur_region"], batch["cur_geom"],
            batch["cur_gen"], batch["cur_cat"],
        )
        pred = self.decoder(ref_tokens, cur_tokens, batch["ref_mask"], batch["cur_mask"])
        pred["iou_logit"] = self.decoder.iou_predict(
            pred["ref_feats"], pred["cur_feats"], batch["ref_geom"], batch["cur_geom"]
        )
        return pred


class PairwiseModel(nn.Module):
    def __init__(self, d_model=256, hidden=512, dropout=0.1):
        super().__init__()
        self.proj = FeatureProjector(d_model, dropout)
        self.mlp = PairwiseMLP(d_model=d_model, hidden=hidden, dropout=dropout)
        self.iou_head = nn.Sequential(nn.Linear(d_model * 2, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, batch):
        ref_feat = self.proj(
            batch["ref_pbd"], batch["ref_region"], batch["ref_geom"],
            batch["ref_gen"], batch["ref_cat"],
        )
        cur_feat = self.proj(
            batch["cur_pbd"], batch["cur_region"], batch["cur_geom"],
            batch["cur_gen"], batch["cur_cat"],
        )
        B, M, D = ref_feat.shape
        N = cur_feat.shape[1]
        rf = ref_feat.unsqueeze(2).expand(B, M, N, D).reshape(B * M * N, D)
        cf = cur_feat.unsqueeze(1).expand(B, M, N, D).reshape(B * M * N, D)
        rg = batch["ref_geom"].unsqueeze(2).expand(B, M, N, 5).reshape(B * M * N, 5)
        cg = batch["cur_geom"].unsqueeze(1).expand(B, M, N, 5).reshape(B * M * N, 5)
        gap = batch["gap"].unsqueeze(1).expand(B, M, N).reshape(B * M * N)
        gen = batch["cur_gen"].unsqueeze(1).expand(B, M, N).reshape(B * M * N)
        pred = self.mlp(rf, cf, rg, cg, gap, gen)
        pred["ref_feats"] = ref_feat
        pred["cur_feats"] = cur_feat
        iou_logit = self.iou_head(torch.cat([rf, cf], dim=-1)).squeeze(-1)
        pred["iou_logit"] = iou_logit.reshape(B, M, N)
        return pred
