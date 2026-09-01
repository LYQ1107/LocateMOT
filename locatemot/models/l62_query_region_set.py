"""L62 per-level-token query-to-region/set probe.

Unlike L59, ROI tokens are kept as four level-specific token groups through
query cross-attention; only the conditioned tokens are reduced before the
candidate-set competition.
"""
from __future__ import annotations
import torch
from torch import nn

class L62QueryRegionSet(nn.Module):
    def __init__(self, image_dim=256, text_dim=256, numeric_dim=24,
                 hidden=128, levels=4, points=16, heads=4):
        super().__init__()
        if hidden % heads: raise ValueError('hidden must divide heads')
        self.levels, self.points = levels, points
        self.roi_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.roi_to_text = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.level_merge = nn.Sequential(nn.Linear(levels * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.set_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.0, batch_first=True, norm_first=True)
        self.set_norm = nn.LayerNorm(hidden)
        self.relevance = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, roi_tokens, text_memory, text_valid, numeric):
        if roi_tokens.dim() != 3 or roi_tokens.shape[1] != self.levels * self.points: raise ValueError('expected [N,levels*points,D] ROI tokens')
        if text_memory.dim() == 2: text_memory = text_memory.unsqueeze(0)
        if text_valid.dim() == 1: text_valid = text_valid.unsqueeze(0)
        n = roi_tokens.shape[0]
        roi = self.roi_proj(roi_tokens).reshape(n, self.levels, self.points, -1)
        text = self.text_proj(text_memory).expand(n, -1, -1)
        pad = (~text_valid.bool()).expand(n, -1)
        conditioned = []
        for level in range(self.levels):
            q = roi[:, level]
            a, _ = self.roi_to_text(q, text, text, key_padding_mask=pad)
            conditioned.append(q + a)
        x = torch.cat([v.mean(1) for v in conditioned], dim=-1)
        x = self.level_merge(x) + self.numeric_proj(numeric)
        x = self.set_norm(x + self.set_layer(x.unsqueeze(0)).squeeze(0))
        score = self.relevance(x).squeeze(-1)
        valid = text_valid.bool().unsqueeze(-1)
        base = self.text_proj(text_memory)
        summary = (base * valid).sum((0, 1)) / valid.sum().clamp_min(1)
        null = self.null_head(x.mean(0) + summary).reshape(())
        return {'relevance_logit': score, 'null_logit': null, 'candidate_hidden': x}
