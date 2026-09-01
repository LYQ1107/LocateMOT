"""Teacher-preserving dense ranker for the isolated Stage L24 experiments.

The teacher is a frozen calibration-only linear probe over the L23 dense bank.
All learnable paths are residuals, initialized to zero, so a new experiment
starts with exactly the teacher ranking.  Pool/source/group identifiers are
intentionally absent from every score path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _flat(x: torch.Tensor, n: int) -> torch.Tensor:
    x = torch.as_tensor(x).float().reshape(n, -1)
    return torch.nan_to_num(x)


class L24ResidualDenseRanker(nn.Module):
    source_in_score = False
    uses_grouping = False
    uses_membership = False
    uses_source_acceptance = False
    uses_null_scalar = False
    uses_temporal_gru = False

    def __init__(self, teacher_weight: torch.Tensor, teacher_bias: torch.Tensor,
                 stage: str = "R1", alpha: float = 0.1, hidden: int = 128):
        super().__init__()
        if stage not in {"R1", "R2", "R3", "R4", "F1", "F2", "F3", "F4", "F5", "F6"}:
            raise ValueError(stage)
        self.stage, self.alpha, self.teacher_dim = stage, float(alpha), int(teacher_weight.numel())
        self.register_buffer("teacher_weight", teacher_weight.detach().float().reshape(1, -1))
        self.register_buffer("teacher_bias", teacher_bias.detach().float().reshape(1))
        # Canonical candidate point order is fixed by the v3 bank builder.
        coords = torch.tensor([[.5, .5], [.2, .2], [.8, .2], [.2, .8], [.8, .8]])
        freq = torch.arange(1, 9, dtype=torch.float32)[None, None, :]
        xy = coords[:, :, None]
        enc = torch.cat([torch.sin(torch.pi * freq * xy), torch.cos(torch.pi * freq * xy)], dim=1)
        self.register_buffer("point_coordinate_encoding", enc.reshape(5, -1))
        self.coord_proj = nn.Linear(self.point_coordinate_encoding.shape[1], 128, bias=False)
        self.query_proj = nn.Linear(512, 128, bias=False)
        self.roi_proj = nn.Linear(512, 128, bias=False)
        self.point_proj = nn.Linear(512, 128, bias=False)
        self.context_proj = nn.Linear(512, 128, bias=False)
        self.motion_proj = nn.Linear(16, 32, bias=False)
        # q/roi/point/context/geometry/objectness/motion is the common residual input.
        common_dim = 128 * 4 + 10 + 1 + 32
        self.common_dim = common_dim
        self.linear = nn.Linear(common_dim, 1)
        self.mlp = nn.Sequential(nn.LayerNorm(common_dim), nn.Linear(common_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.q_gate = nn.Sequential(nn.Linear(128, 128), nn.Tanh())
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.pair = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Zero residual at initialization is important: teacher is the R0 model.
        for head in (self.linear, self.mlp[-1], self.pair[-1]):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        for p in (self.teacher_weight, self.teacher_bias):
            p.requires_grad_(False)

    def teacher_features(self, query, dense_roi, dense_points, context_1p5, context_3,
                         prev_roi, geometry, neighbor, motion, lifecycle, objectness):
        n = query.shape[0]
        values = [query, dense_roi, dense_points, context_1p5, context_3, prev_roi,
                  geometry, neighbor, motion, lifecycle, objectness]
        return torch.cat([_flat(v, n) for v in values], dim=-1)

    def forward(self, query, dense_points, dense_roi, geometry, objectness,
                dense_context_1p5, dense_context_3, dense_prev_roi, motion,
                neighbor=None, lifecycle=None):
        device = next(self.parameters()).device
        query = torch.as_tensor(query, device=device).float()
        if query.ndim == 1: query = query[None, :]
        n = query.shape[0]
        dense_roi = torch.as_tensor(dense_roi, device=device).float().reshape(n, -1)
        points = torch.as_tensor(dense_points, device=device).float().reshape(n, 5, 512)
        c1 = _flat(torch.as_tensor(dense_context_1p5, device=device), n)
        c3 = _flat(torch.as_tensor(dense_context_3, device=device), n)
        prev = _flat(torch.as_tensor(dense_prev_roi, device=device), n)
        geometry = _flat(geometry, n); motion = _flat(motion, n)
        objectness = _flat(objectness, n)
        if neighbor is None: neighbor = torch.zeros(n, 10, device=device)
        if lifecycle is None: lifecycle = torch.zeros(n, 6, device=device)
        neighbor = _flat(neighbor, n); lifecycle = _flat(lifecycle, n)
        query, dense_roi, points, c1, c3, prev, geometry, motion, objectness, neighbor, lifecycle = [
            torch.nan_to_num(v) for v in (query, dense_roi, points, c1, c3, prev, geometry, motion, objectness, neighbor, lifecycle)]
        tf = self.teacher_features(query, dense_roi, points, c1, c3, prev, geometry, neighbor, motion, lifecycle, objectness)
        with torch.no_grad():
            teacher = F.linear(tf, self.teacher_weight, self.teacher_bias).squeeze(-1)
        q = F.normalize(self.query_proj(query), dim=-1)
        roi = F.normalize(self.roi_proj(dense_roi), dim=-1)
        point_tokens = self.point_proj(points) + self.coord_proj(self.point_coordinate_encoding)[None, :, :]
        point_tokens = F.normalize(point_tokens, dim=-1)
        point_pool = point_tokens.mean(dim=1)
        context = F.normalize(self.context_proj((c1 + c3) * .5), dim=-1)
        common = torch.cat((q, roi, point_pool, context, geometry, objectness, self.motion_proj(motion)), dim=-1)
        if self.stage in {"R3", "R4", "F4", "F5"}:
            gated = point_tokens * (1.0 + 0.1 * self.q_gate(q)[:, None, :])
            attended, _ = self.attn(q[:, None, :], gated, gated, need_weights=False)
            # Replace mean-pooled point tokens in the common representation;
            # keep the declared dimensionality unchanged.
            common = torch.cat((q, roi, attended[:, 0, :], context, geometry,
                                objectness, self.motion_proj(motion)), dim=-1)
        residual = common.new_zeros(n)
        if self.stage not in {"F1"}:
            if self.stage in {"R1", "F2", "F3", "F6"}:
                residual = self.linear(common).squeeze(-1)
            elif self.stage in {"R2"}:
                residual = self.mlp(common).squeeze(-1)
            elif self.stage in {"R3", "F4", "F5"}:
                residual = self.mlp(common).squeeze(-1)
            else:  # R4: add a pairwise current/history correspondence term.
                relation = torch.cat((roi, F.normalize(self.roi_proj(prev), dim=-1)), dim=-1)
                residual = self.mlp(common).squeeze(-1) + self.pair(relation).squeeze(-1)
        final = teacher + self.alpha * residual
        return {"teacher_score": teacher, "residual_score": residual, "final_score": final,
                "static_score": final, "correspondence_score": self.alpha * residual}


__all__ = ["L24ResidualDenseRanker"]
