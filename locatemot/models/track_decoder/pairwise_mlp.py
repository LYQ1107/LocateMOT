"""B3: learned pairwise MLP for two-frame association."""
from __future__ import annotations

import torch
from torch import nn


class PairwiseMLP(nn.Module):
    def __init__(self, d_model: int = 256, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        in_dim = d_model * 4 + 5 + 1 + 1  # ref, cur, abs diff, product, geom delta(5), gap, gen
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.match_head = nn.Linear(256, 1)
        self.no_match_head = nn.Linear(256 + 2, 1)

    def forward(
        self,
        ref_feat: torch.Tensor,
        cur_feat: torch.Tensor,
        ref_geom: torch.Tensor,
        cur_geom: torch.Tensor,
        gap: torch.Tensor,
        gen_score: torch.Tensor,
    ) -> dict:
        geom_delta = cur_geom - ref_geom
        x = torch.cat([
            ref_feat, cur_feat, (ref_feat - cur_feat).abs(), ref_feat * cur_feat,
            geom_delta, gap.unsqueeze(-1), gen_score.unsqueeze(-1),
        ], dim=-1)
        h = self.net(x)
        match_logit = self.match_head(h).squeeze(-1)
        nm_input = torch.cat([h, gap.unsqueeze(-1), gen_score.unsqueeze(-1)], dim=-1)
        no_match_logit = self.no_match_head(nm_input).squeeze(-1)
        return {"match_logits": match_logit, "no_match_logits": no_match_logit}
