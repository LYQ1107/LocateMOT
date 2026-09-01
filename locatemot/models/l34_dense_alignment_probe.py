"""Small word-token to dense-region alignment probe for LocateMOT L34."""
from __future__ import annotations
import torch
from torch import nn

class L34DenseAlignmentProbe(nn.Module):
    def __init__(self, dim=512, hidden=128, heads=4):
        super().__init__()
        self.text = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.region = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.coord = nn.Linear(2, hidden, bias=False)
        self.attn1 = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden); self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    @staticmethod
    def mean(x): return x.mean(1)

    def forward(self, query_tokens, region_tokens, region_coords):
        if query_tokens.ndim == 2: query_tokens = query_tokens.unsqueeze(0)
        if region_tokens.ndim == 4: region_tokens = region_tokens[0]
        if region_coords.ndim == 4: region_coords = region_coords[0]
        n = region_tokens.shape[0]
        q = self.text(torch.nan_to_num(query_tokens.float())).expand(n, -1, -1)
        r = self.region(torch.nan_to_num(region_tokens.float())) + self.coord(torch.nan_to_num(region_coords.float()))
        a, _ = self.attn1(q, r, r, need_weights=False); q = self.norm1(q + a)
        a, _ = self.attn2(q, r, r, need_weights=False); q = self.norm2(q + a)
        qmean = q.mean(1); rmean = r.mean(1)
        fused = self.head(torch.cat((qmean, rmean), -1)).squeeze(-1)
        return {"region_logits": fused, "null_logit": self.null(qmean[:1]).squeeze()}
