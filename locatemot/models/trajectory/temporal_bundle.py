"""T3-T6 trainable bundle with a single forward for training."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .memory_fusion import MemoryFusion
from .motion_predictor import MotionPredictor
from .residual_heads import MotionResidualHead, ReactivationResidualHead
from .trajectory_encoder import TrajectoryEncoder


def quantize_gap(gaps):
    reps = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    return [min(reps, key=lambda r: abs(r - g)) for g in gaps]


class TemporalBundle(nn.Module):
    """All Stage L1-A temporal modules + a small trainable no-match bias."""

    def __init__(
        self,
        trajectory_encoder=None,
        motion_predictor=None,
        memory_fusion=None,
        motion_residual_head=None,
        reactivation_head=None,
    ):
        super().__init__()
        self.trajectory_encoder = trajectory_encoder or TrajectoryEncoder()
        self.motion_predictor = motion_predictor or MotionPredictor()
        self.memory_fusion = memory_fusion or MemoryFusion()
        self.motion_residual_head = motion_residual_head or MotionResidualHead()
        self.reactivation_head = reactivation_head or ReactivationResidualHead()
        self.nm_bias = nn.Parameter(torch.tensor(0.0))

    def trajectory_refs(self, win):
        return self.trajectory_encoder(
            win["pbd"], win["region"], win["geom"], win["gen"],
            win["gaps"], win["mask"],
        )

    def memory_refs(self, anchors, emas, traj_out, geom_last, gen_last, conf):
        """anchors/emas: dicts of [B,2048]/[B,4608] raw features."""
        return self.memory_fusion(
            anchors["pbd"], emas["pbd"], traj_out["pbd"],
            anchors["region"], emas["region"], traj_out["region"],
            geom_last, gen_last, conf,
        )

    def motion_delta(self, boxes, gaps):
        return self.motion_predictor(boxes, gaps)

    def motion_residual(self, feats):
        return self.motion_residual_head(feats)

    def reactivation_residual(self, feats):
        return self.reactivation_head(feats)
