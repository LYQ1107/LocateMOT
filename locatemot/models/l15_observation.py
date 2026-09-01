"""Stage L15 query-conditioned observation scorer.

The scorer is deliberately outside the shared UIDM.  It ranks detector
proposals using a frozen crop representation, a frozen specification
embedding, and causal-frame geometry metadata.  The L11 tracker receives the
selected proposals unchanged; this module therefore cannot alter MOT/OVMOT
identity state or lifecycle behavior.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class L15ObservationHead(nn.Module):
    """Cross-modal proposal relevance head used before UIDM acquisition."""

    def __init__(self, clip_dim=512, hidden=384, dropout=0.10):
        super().__init__()
        self.clip_dim = int(clip_dim)
        # crop, text, elementwise product, absolute difference, 7 geometry
        # values, detector confidence, and explicit cosine similarity.
        self.feature_dim = 4 * self.clip_dim + 9
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.fuse = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _norm(x):
        return F.normalize(torch.nan_to_num(x.float()), dim=-1)

    def make_features(self, clip, spec, geometry, gen):
        """Build ``[N, feature_dim]`` proposal/specification features."""
        crop = self._norm(clip)
        text = self._norm(spec)
        if text.dim() == 1:
            text = text.expand(crop.shape[0], -1)
        geom = torch.nan_to_num(geometry.float())
        score = torch.nan_to_num(gen.float()).reshape(-1, 1)
        cosine = (crop * text).sum(dim=-1, keepdim=True)
        return torch.cat((crop, text, crop * text, (crop - text).abs(),
                          geom, score, cosine), dim=-1)

    def forward(self, clip, spec, geometry, gen):
        x = self.make_features(clip, spec, geometry, gen)
        return self.fuse(self.input_norm(x)).squeeze(-1)

