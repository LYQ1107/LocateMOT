"""B5: relation-aware pairwise association with strong-prior residual."""
from __future__ import annotations

import torch
from torch import nn

from .features import FeatureProjector
from .relation_encoder import RelationEncoder
from .relation_features import base_affinity


def _gap_embedding(batch):
    gap = batch["gap"][:, :1].float()  # B,1 (precomputed tensors pad gap to B,32)
    return torch.stack([torch.log1p(gap.squeeze(-1)), (gap.squeeze(-1) / 100.0).clamp(max=1.0)], dim=-1)


class RelationPairwiseModel(nn.Module):
    """B5 RelationPairwise.

    match = BaseAffinity + alpha * tanh(Residual)
    Residual = MLP(ref, cur, abs(ref-cur), ref*cur, relation_embedding, relation_score,
                   base, candidate_gen, gap)
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden: int = 256,
        dropout: float = 0.1,
        use_pbd_base: bool = True,
        use_region_geom: bool = True,
        residual: bool = True,
        n_heads: int = 8,
    ):
        super().__init__()
        self.use_pbd_base = use_pbd_base
        self.residual = residual
        self.proj = FeatureProjector(d_model, dropout)
        self.rel_enc = RelationEncoder(d_model, 128, n_heads, use_region_geom, dropout)
        self.w_iou = nn.Parameter(torch.tensor(0.5))
        self.w_pbd = nn.Parameter(torch.tensor(0.5)) if use_pbd_base else None
        self.alpha_logit = nn.Parameter(torch.tensor(-1.0986))  # sigmoid -> ~0.25
        pair_in = d_model * 4 + 128 + 5  # rel_score, base, cur_gen, gap(2)
        self.pair_net = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.no_match_head = nn.Sequential(
            nn.Linear(d_model + 2 + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @property
    def alpha(self):
        return 0.5 * torch.sigmoid(self.alpha_logit)

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
        rel_emb, rel_score, _ = self.rel_enc(batch)  # B,M,N,128 ; B,M,N
        base = base_affinity(batch, self.w_iou, self.w_pbd)  # B,M,N

        rf = ref.unsqueeze(2).expand(B, M, N, D)
        cf = cur.unsqueeze(1).expand(B, M, N, D)
        gap_emb = _gap_embedding(batch).unsqueeze(1).unsqueeze(2).expand(B, M, N, 2)
        cur_gen = batch["cur_gen"].float().unsqueeze(1).unsqueeze(-1).expand(B, M, N, 1)
        x = torch.cat([
            rf, cf, (rf - cf).abs(), rf * cf,
            rel_emb,
            rel_score.unsqueeze(-1),
            base.unsqueeze(-1),
            cur_gen,
            gap_emb,
        ], dim=-1)
        residual = self.pair_net(x).squeeze(-1)  # B,M,N
        if self.residual:
            match = base + self.alpha * torch.tanh(residual)
        else:
            match = base + residual
        n_cand = torch.log1p(batch["cur_mask"].float().sum(dim=1)).unsqueeze(1).expand(B, M)
        ref_gen = batch["ref_gen"].float()
        max_match = match.max(dim=2).values
        no_match = self.no_match_head(torch.cat([
            ref, gap_emb[:, :, 0, :], max_match.unsqueeze(-1),
            n_cand.unsqueeze(-1), ref_gen.unsqueeze(-1),
        ], dim=-1)).squeeze(-1)
        return {
            "match_logits": match,
            "no_match_logits": no_match,
            "ref_feats": ref,
            "cur_feats": cur,
            "iou_logit": None,
        }
