"""Memory fusion for T5 (short-term + anchor), raw-space gated combination.

Design evidence: MeMOTR query_updater writes long_memory only for high-confidence
positive matches and updates it with an EMA; this module mirrors that behavior
with a small learned gate over anchor / EMA / trajectory-fused token.
"""
from __future__ import annotations

import torch
from torch import nn


class MemoryFusion(nn.Module):
    def __init__(self, n_modalities: int = 2, dropout: float = 0.1):
        """n_modalities: pbd (2048) and region (4608) are fused separately."""
        super().__init__()
        self.logit_a = nn.Parameter(torch.zeros(1, n_modalities, 3))  # anchor/ema/traj
        self.ema_alpha = nn.Parameter(torch.zeros(n_modalities))  # sigmoid -> ~0.5
        self.conf_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 3))
        self.dropout = nn.Dropout(dropout)

    def update_ema(self, ema, obs, alpha):
        return (1 - alpha) * ema + alpha * obs

    def forward(
        self,
        anchor_pbd, ema_pbd, traj_pbd,
        anchor_region, ema_region, traj_region,
        geom, gen, conf,
    ):
        # gate: [B,2,3] softmax over anchor/ema/traj; conf adds a per-candidate shift
        conf_bias = self.conf_mlp(conf.unsqueeze(-1)).unsqueeze(1)  # [B,1,3]
        logits = self.logit_a + conf_bias
        w = torch.softmax(logits, dim=-1)  # [B,2,3]
        fused_pbd = (
            w[:, 0, 0:1] * anchor_pbd + w[:, 0, 1:2] * ema_pbd + w[:, 0, 2:3] * traj_pbd
        )
        fused_region = (
            w[:, 1, 0:1] * anchor_region + w[:, 1, 1:2] * ema_region + w[:, 1, 2:3] * traj_region
        )
        return {
            "pbd": self.dropout(fused_pbd),
            "region": self.dropout(fused_region),
            "geom": geom,
            "gen": gen,
            "weights": w,
        }
