"""Bounded residual heads for T4/T6 motion and reactivation."""
from __future__ import annotations

import torch
from torch import nn


class _BoundedHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, alpha_init: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(0.5 * (alpha_init * 2 - 1) * 4.0))

    @property
    def alpha(self):
        return 0.5 * torch.sigmoid(self.alpha_logit)

    def forward(self, x):
        return self.alpha * torch.tanh(self.net(x).squeeze(-1))


class MotionResidualHead(_BoundedHead):
    """Adds motion-aware correction on top of frozen B6 match logits.

    Input features per (track, candidate):
      b6_match, iou(last,cand), iou(pred,cand), center_dist_norm, motion_resid_norm, gap
    """

    def __init__(self, in_dim: int = 6, hidden: int = 64, alpha_init: float = 0.25):
        super().__init__(in_dim, hidden, alpha_init)


class ReactivationResidualHead(_BoundedHead):
    """Lost-track boost combining trajectory similarity with motion.

    Input features per (track, candidate):
      b6_match, traj_cos, pbd_cos, iou(pred,cand), center_dist_norm, gap
    """

    def __init__(self, in_dim: int = 6, hidden: int = 64, alpha_init: float = 0.25):
        super().__init__(in_dim, hidden, alpha_init)
