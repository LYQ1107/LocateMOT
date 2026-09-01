"""Stage L39 temporal identity prototype encoder."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class L39IdentityPrototype(nn.Module):
    """Encode an observation fragment into a bounded, metric-learning prototype."""

    def __init__(self, feature_dim=1432, hidden=96, heads=4, layers=2,
                 history=8, prototype_dim=96):
        super().__init__()
        if hidden % heads: raise ValueError("hidden must be divisible by heads")
        self.hidden = int(hidden); self.history = int(history)
        self.input = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        self.time = nn.Linear(1, hidden)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, 0.1,
                                       batch_first=True, norm_first=True,
                                       activation="gelu"), num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                    nn.Linear(hidden, prototype_dim))

    @staticmethod
    def masked_mean(x, mask):
        w = mask.float().unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.0)

    def forward(self, observations, observation_mask, observation_time):
        x = self.input(torch.nan_to_num(observations.float()))
        x = x + self.time(observation_time.float().unsqueeze(-1))
        mask = observation_mask.bool()
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = self.temporal(x, src_key_padding_mask=~mask)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = self.norm(self.masked_mean(x, mask))
        last = x[torch.arange(x.shape[0], device=x.device), mask.long().sum(1).clamp_min(1) - 1]
        # The update is bounded before normalization, so prototype scale cannot
        # become an uncontrolled identity/source shortcut.
        prototype = F.normalize(pooled + 0.1 * torch.tanh(self.update(torch.cat((pooled, last), -1))), dim=-1)
        return {"prototype": prototype, "prototype_norm": prototype.norm(dim=-1), "pooled": pooled}

    def pair_score(self, left, right):
        return (left * right).sum(-1)
