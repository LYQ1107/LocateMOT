"""Lightweight explicit pairwise relation features for association repair.

Design evidence: official implementations encode pairwise geometry with small
MLPs (GRAE-3DMOT spatial_proj, TADN spatial embedding) and combine appearance
similarity with IoU directly (GMTracker, GTR). This module is a clean
reimplementation (no code copied) kept deliberately small per Stage L0-D spec.
"""
from __future__ import annotations

import torch
from torch import nn


def _iou_matrix(ref_boxes: torch.Tensor, cur_boxes: torch.Tensor, eps: float = 1e-6):
    """ref_boxes [B,M,4] xyxy, cur_boxes [B,N,4] xyxy -> [B,M,N] IoU."""
    B, M, _ = ref_boxes.shape
    N = cur_boxes.shape[1]
    rb = ref_boxes.unsqueeze(2)  # B,M,1,4
    cb = cur_boxes.unsqueeze(1)  # B,1,N,4
    ix1 = torch.maximum(rb[..., 0], cb[..., 0])
    iy1 = torch.maximum(rb[..., 1], cb[..., 1])
    ix2 = torch.minimum(rb[..., 2], cb[..., 2])
    iy2 = torch.minimum(rb[..., 3], cb[..., 3])
    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih
    area_r = torch.clamp(rb[..., 2] - rb[..., 0], min=0.0) * torch.clamp(rb[..., 3] - rb[..., 1], min=0.0)
    area_c = torch.clamp(cb[..., 2] - cb[..., 0], min=0.0) * torch.clamp(cb[..., 3] - cb[..., 1], min=0.0)
    union = area_r + area_c - inter
    return inter / (union + eps)


def _cos_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """a [B,M,D], b [B,N,D] -> [B,M,N] cosine similarity."""
    an = a / (a.norm(dim=-1, keepdim=True) + eps)
    bn = b / (b.norm(dim=-1, keepdim=True) + eps)
    return torch.bmm(an, bn.transpose(1, 2))


def build_relation_features(
    batch,
    use_region_geom: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build pairwise relation feature tensor [B,M,N,Df]."""
    ref_boxes = batch["ref_boxes"].float()
    cur_boxes = batch["cur_boxes"].float()
    B, M, _ = ref_boxes.shape
    N = cur_boxes.shape[1]

    iou = _iou_matrix(ref_boxes, cur_boxes)  # B,M,N

    ref_cx = (ref_boxes[..., 0] + ref_boxes[..., 2]) / 2
    ref_cy = (ref_boxes[..., 1] + ref_boxes[..., 3]) / 2
    cur_cx = (cur_boxes[..., 0] + cur_boxes[..., 2]) / 2
    cur_cy = (cur_boxes[..., 1] + cur_boxes[..., 3]) / 2
    dx = cur_cx.unsqueeze(1) - ref_cx.unsqueeze(2)
    dy = cur_cy.unsqueeze(1) - ref_cy.unsqueeze(2)
    cdist = torch.sqrt(dx * dx + dy * dy + eps)

    ref_w = torch.clamp(ref_boxes[..., 2] - ref_boxes[..., 0], min=eps)
    ref_h = torch.clamp(ref_boxes[..., 3] - ref_boxes[..., 1], min=eps)
    cur_w = torch.clamp(cur_boxes[..., 2] - cur_boxes[..., 0], min=eps)
    cur_h = torch.clamp(cur_boxes[..., 3] - cur_boxes[..., 1], min=eps)
    w_ratio = torch.log(cur_w.unsqueeze(1) / ref_w.unsqueeze(2) + eps)
    h_ratio = torch.log(cur_h.unsqueeze(1) / ref_h.unsqueeze(2) + eps)
    area_ratio = torch.log((cur_w * cur_h).unsqueeze(1) / (ref_w * ref_h).unsqueeze(2) + eps)

    cos_boxend = _cos_matrix(batch["ref_pbd_be"].float(), batch["cur_pbd_be"].float())
    cos_pbd = _cos_matrix(batch["ref_pbd"].float(), batch["cur_pbd"].float())

    feats = [
        iou.unsqueeze(-1),
        dx.unsqueeze(-1),
        dy.unsqueeze(-1),
        cdist.unsqueeze(-1),
        w_ratio.unsqueeze(-1),
        h_ratio.unsqueeze(-1),
        area_ratio.unsqueeze(-1),
        cos_boxend.unsqueeze(-1),
        cos_pbd.unsqueeze(-1),
    ]
    if use_region_geom:
        cos_region = _cos_matrix(batch["ref_region"].float(), batch["cur_region"].float())
        feats.append(cos_region.unsqueeze(-1))
        geom_delta = batch["cur_geom"].float().unsqueeze(1) - batch["ref_geom"].float().unsqueeze(2)
        feats.append(geom_delta)
    # generation score (candidate-side) and reference-side score
    cur_gen = batch["cur_gen"].float().unsqueeze(1).unsqueeze(-1).expand(B, M, N, 1)
    ref_gen = batch["ref_gen"].float().unsqueeze(2).unsqueeze(-1).expand(B, M, N, 1)
    feats.append(cur_gen)
    feats.append(ref_gen)
    # temporal gap encoding
    gap = batch["gap"][:, :1].float()  # B,1 (precomputed tensors pad gap to B,32)
    gap_enc = torch.stack([
        torch.log1p(gap.squeeze(-1)),
        (gap.squeeze(-1) / 100.0).clamp(max=1.0),
    ], dim=-1)  # B,2
    feats.append(gap_enc.unsqueeze(1).unsqueeze(2).expand(B, M, N, 2))
    return torch.cat(feats, dim=-1)


def base_affinity(batch, w_iou: torch.Tensor, w_pbd: torch.Tensor | None, eps: float = 1e-6):
    """BaseAffinity_ij = w_iou * f(IoU) + w_pbd * f(PBD_box_end_cos)."""
    iou = _iou_matrix(batch["ref_boxes"].float(), batch["cur_boxes"].float(), eps)
    out = w_iou * iou
    if w_pbd is not None:
        cos = _cos_matrix(batch["ref_pbd_be"].float(), batch["cur_pbd_be"].float(), eps)
        out = out + w_pbd * 0.5 * (1.0 + cos)
    return out


class RelationMLP(nn.Module):
    """D -> 128 -> 128 with LayerNorm/GELU; outputs relation embedding + scalar score."""

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(out_dim, 1)

    def forward(self, x):
        h = self.net(x)
        return h, self.score_head(h).squeeze(-1)
