"""Small learned motion predictor for T4.

MATR (arXiv:2509.21715) has NO VERIFIED OFFICIAL CODE; per Stage L1-A spec this
is a paper-guided clean implementation. Input: the last L observed boxes and
their frame offsets. Output: normalized delta (dcx_n, dcy_n, dlogw, dlogh)
relative to the last box. Supervised with SmoothL1 against GT deltas.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MotionPredictor(nn.Module):
    def __init__(self, window: int = 4, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.window = window
        pair_dim = 5  # dcx_n, dcy_n, dlogw, dlogh, 1/gap
        self.net = nn.Sequential(
            nn.Linear(pair_dim * (window - 1), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 4),
        )

    @staticmethod
    def _pair_feats(boxes, gaps):
        """boxes [B,L,4] xyxy, gaps [B,L] frame offsets (last=0)."""
        B, L, _ = boxes.shape
        out = []
        last = boxes[:, -1]  # [B,4]
        lw = (last[:, 2] - last[:, 0]).clamp(min=1e-3)
        lh = (last[:, 3] - last[:, 1]).clamp(min=1e-3)
        lcx = (last[:, 0] + last[:, 2]) / 2
        lcy = (last[:, 1] + last[:, 3]) / 2
        for i in range(L - 1):
            bx = boxes[:, i]
            g = gaps[:, i].clamp(min=1e-3)
            cx = (bx[:, 0] + bx[:, 2]) / 2
            cy = (bx[:, 1] + bx[:, 3]) / 2
            w = (bx[:, 2] - bx[:, 0]).clamp(min=1e-3)
            h = (bx[:, 3] - bx[:, 1]).clamp(min=1e-3)
            out.append(torch.stack([
                (cx - lcx) / lw,
                (cy - lcy) / lh,
                torch.log(w / lw),
                torch.log(h / lh),
                1.0 / g,
            ], dim=-1))
        return torch.cat(out, dim=-1) if out else torch.zeros(B, 0, device=boxes.device)

    def forward(self, boxes, gaps):
        feats = self._pair_feats(boxes, gaps)
        if feats.shape[1] < self.window - 1:
            pad = torch.zeros(feats.shape[0], (self.window - 1) - feats.shape[1], 5, device=feats.device)
            feats = torch.cat([pad, feats], dim=1)
        return self.net(feats)  # [B,4]

    def predict_box(self, boxes, gaps):
        """boxes [B,L,4] xyxy; returns predicted current box [B,4]."""
        delta = self.forward(boxes, gaps)
        last = boxes[:, -1]
        lw = (last[:, 2] - last[:, 0]).clamp(min=1e-3)
        lh = (last[:, 3] - last[:, 1]).clamp(min=1e-3)
        lcx = (last[:, 0] + last[:, 2]) / 2
        lcy = (last[:, 1] + last[:, 3]) / 2
        cx = lcx + delta[:, 0] * lw
        cy = lcy + delta[:, 1] * lh
        w = lw * torch.exp(delta[:, 2].clamp(-2, 2))
        h = lh * torch.exp(delta[:, 3].clamp(-2, 2))
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    def motion_loss(self, pred_delta, gt_boxes, last_boxes):
        """SmoothL1 on normalized deltas."""
        lw = (last_boxes[:, 2] - last_boxes[:, 0]).clamp(min=1e-3)
        lh = (last_boxes[:, 3] - last_boxes[:, 1]).clamp(min=1e-3)
        lcx = (last_boxes[:, 0] + last_boxes[:, 2]) / 2
        lcy = (last_boxes[:, 1] + last_boxes[:, 3]) / 2
        gcx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gcy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-3)
        gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-3)
        target = torch.stack([
            (gcx - lcx) / lw,
            (gcy - lcy) / lh,
            torch.log(gw / lw),
            torch.log(gh / lh),
        ], dim=-1)
        return F.smooth_l1_loss(pred_delta, target)
