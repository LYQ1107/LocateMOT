"""Bidirectional spatial-token comparator for L41 identity-only diagnostics."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class L41RawRelationalIdentity(nn.Module):
    def __init__(self, patch_dim=768, relation_dim=49, hidden=96, history=8, heads=4):
        super().__init__()
        self.history = history
        self.patch = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, hidden))
        self.time_position = nn.Parameter(torch.zeros(history, hidden))
        nn.init.normal_(self.time_position, std=0.02)
        self.left_to_right = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.right_to_left = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.relation = nn.Sequential(nn.LayerNorm(relation_dim), nn.Linear(relation_dim, hidden), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _encode(self, patches, mask):
        # patches: [B, history, spatial_tokens, patch_dim]
        b, h, s, _ = patches.shape
        x = self.patch(patches.float()) + self.time_position[:h].view(1, h, 1, -1)
        x = x.reshape(b, h * s, -1)
        token_mask = mask.unsqueeze(-1).expand(b, h, s).reshape(b, h * s)
        return x, token_mask

    @staticmethod
    def _masked_mean(x, mask):
        w = mask.float().unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.0)

    def forward(self, left, right, relation, left_mask, right_mask):
        a, am = self._encode(left, left_mask); b, bm = self._encode(right, right_mask)
        a2, aw = self.left_to_right(a, b, b, key_padding_mask=~bm)
        b2, bw = self.right_to_left(b, a, a, key_padding_mask=~am)
        av = self._masked_mean(a, am); bv = self._masked_mean(b, bm)
        ac = self._masked_mean(a2, am); bc = self._masked_mean(b2, bm)
        rv = self.relation(relation.float())
        score = self.head(torch.cat((av * bv, torch.abs(av - bv), ac * bc, rv), dim=-1)).squeeze(-1)
        return {"logit": score, "attention_norm": (a2.norm(dim=-1).mean() + b2.norm(dim=-1).mean()) * 0.5,
                "left_attention": aw, "right_attention": bw}
