"""Bounded temporal identity prototype over streamed raw-image embeddings."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class L40RawImageIdentity(nn.Module):
    def __init__(self, image_dim=512, numeric_dim=24, hidden=96, history=8):
        super().__init__()
        self.history = history
        self.image = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden))
        self.numeric = nn.Sequential(nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden))
        self.time = nn.Linear(1, hidden)
        self.temporal = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.prototype = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, image, numeric, mask, times):
        x = self.image(F.normalize(image.float(), dim=-1)) + self.numeric(numeric.float()) + self.time(times.float().unsqueeze(-1))
        x, _ = self.temporal(x)
        weights = self.attn(x).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        pooled = (x * weights.unsqueeze(-1)).sum(1)
        z = F.normalize(self.prototype(pooled), dim=-1)
        return {"prototype": z, "attention": weights, "prototype_norm": z.norm(dim=-1)}
