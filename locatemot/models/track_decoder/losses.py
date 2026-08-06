"""Losses for B3/B4 with strict candidate_missing vs true_no_match semantics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pair_losses(pred, batch, weights, d_model: int = 256):
    """batch keys: match_targets [B,M] (-1 none), no_match_targets [B,M] (0/1),
    candidate_missing [B,M] bool, visible [B,M] bool, ref_mask [B,M], cur_mask [B,N],
    ref_geom [B,M,5], cur_geom [B,N,5], gt_iou [B,M] for matched, labels [B,M,N] for calibration.
    """
    match_logits = pred["match_logits"]  # [B,M,N] or [B] for pairwise
    no_match_logits = pred["no_match_logits"]
    B = match_logits.shape[0]
    if match_logits.dim() == 2:  # pairwise flattened: reshape handled by caller
        raise ValueError("use batched losses; pairwise loss is built in train script")

    M, N = match_logits.shape[1], match_logits.shape[2]
    tgt = batch["match_targets"]  # [B,M]
    labels = batch.get("labels")  # [B,M,N] 1 if target else 0
    valid = (tgt >= 0) & (~batch["candidate_missing"]) & batch["visible"] & batch["ref_mask"]
    assignment_loss = torch.zeros((), device=match_logits.device)
    if valid.any():
        assignment_loss = F.cross_entropy(
            match_logits[valid],
            tgt[valid].long(),
            reduction="mean",
        )

    nm_valid = (~batch["candidate_missing"]) & batch["ref_mask"]
    no_match_loss = torch.zeros((), device=match_logits.device)
    if nm_valid.any():
        no_match_loss = F.binary_cross_entropy_with_logits(
            no_match_logits[nm_valid],
            batch["no_match_targets"][nm_valid].float(),
            reduction="mean",
        )

    contrastive_loss = torch.zeros((), device=match_logits.device)
    ref_feats = pred.get("ref_feats")
    cur_feats = pred.get("cur_feats")
    if ref_feats is not None and cur_feats is not None and valid.any():
        rows, cols = valid.nonzero(as_tuple=True)
        refs = ref_feats[rows, cols]
        cur = cur_feats[rows, tgt[rows, cols].long()]
        pos_cos = F.cosine_similarity(refs, cur, dim=-1, eps=1e-8)
        # negatives: sample up to 64 other current candidates per positive
        rng_idx = torch.randint(0, B * N, (len(rows), 64), device=match_logits.device)
        neg_cur = cur_feats.view(B * N, -1)[rng_idx]
        neg_cos = F.cosine_similarity(refs.unsqueeze(1), neg_cur, dim=-1, eps=1e-8)
        contrastive_loss = (1 - pos_cos).mean() + torch.clamp(neg_cos - 0.3, min=0).mean()

    geometry_loss = torch.zeros((), device=match_logits.device)
    iou_logit = pred.get("iou_logit")
    if iou_logit is not None and "gt_iou" in batch and valid.any():
        gt_iou_map = torch.zeros_like(labels) if labels is not None else None
        rows_v, cols_v = valid.nonzero(as_tuple=True)
        if gt_iou_map is not None:
            gt_iou_map[rows_v, cols_v, tgt[rows_v, cols_v].long()] = batch["gt_iou"][rows_v, cols_v]
            geom_mask = valid.unsqueeze(-1) & (labels == 1)
            geometry_loss = F.mse_loss(
                torch.sigmoid(iou_logit[geom_mask]),
                gt_iou_map[geom_mask],
            )

    calibration_loss = torch.zeros((), device=match_logits.device)
    if labels is not None and (~batch["candidate_missing"]).any():
        mask = (~batch["candidate_missing"]).unsqueeze(-1) & batch["cur_mask"].unsqueeze(1)
        calibration_loss = F.binary_cross_entropy_with_logits(
            match_logits[mask],
            labels[mask].float(),
            reduction="mean",
        )

    total = (
        weights["assignment"] * assignment_loss
        + weights["no_match"] * no_match_loss
        + weights["contrastive"] * contrastive_loss
        + weights["geometry"] * geometry_loss
        + weights["calibration"] * calibration_loss
    )
    return {
        "loss": total,
        "assignment": assignment_loss,
        "no_match": no_match_loss,
        "contrastive": contrastive_loss,
        "geometry": geometry_loss,
        "calibration": calibration_loss,
    }
