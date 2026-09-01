"""L80-R2 train-only hard-negative and all-positive loss variant.

The detector/region/model interface is unchanged.  This module changes only
which train-time negatives are emphasized and adds an explicit mean
positive-floor term.  Same-class metadata is unavailable, so the fallback is
all-negative candidate mining with detached model-logit ranking and frozen
objectness as the tie-breaker.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from locatemot.rmot.l80_losses import balanced_bce


def l80_r2_loss(output: dict[str, torch.Tensor], labels: torch.Tensor,
                membership_mask: bool | torch.Tensor, observations: torch.Tensor,
                history_mask: torch.Tensor, category: str) -> tuple[torch.Tensor, dict[str, Any]]:
    logits = output["candidate_logits"]
    labels = labels.bool().to(device=logits.device)
    if logits.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("R2 candidate logits/labels mismatch")
    covered = bool(membership_mask) if isinstance(membership_mask, bool) else bool(membership_mask.to(logits.device).all())
    zero = logits.new_zeros(())
    pos = torch.nonzero(labels, as_tuple=False).flatten()
    neg = torch.nonzero(~labels, as_tuple=False).flatten()
    member = pair = listwise = minimum = positive_floor = zero
    hard_count = 0
    if category != "present_uncovered" and covered:
        member = balanced_bce(logits, labels)
        # Metadata is unavailable.  This is a deterministic all-negative
        # fallback: current high logits are hard, with objectness as a stable
        # query-independent tie-breaker.  It affects loss only, never rows.
        objectness = observations[:, -1].to(device=logits.device).float()
        # Stable two-pass tensor sorting avoids a host-side GPU sync: the
        # second pass ranks logits, while the first pass supplies the
        # objectness order for exact-logit ties.
        object_order = torch.argsort(objectness[neg].detach(), descending=True, stable=True)
        logit_order = torch.argsort(logits[neg][object_order].detach(), descending=True, stable=True)
        hard = neg[object_order[logit_order[:min(16, neg.numel())]]] if neg.numel() else neg
        hard_count = int(hard.numel())
        if pos.numel() and hard.numel():
            pair = F.softplus(0.75 + logits[hard].unsqueeze(0) - logits[pos].unsqueeze(1)).mean()
        if pos.numel():
            # Every positive receives a direct gradient, while the minimum
            # term specifically protects the lowest-scoring positive.
            listwise = torch.logsumexp(logits[neg], dim=0) - torch.logsumexp(logits[pos], dim=0) if neg.numel() else zero
            positive_floor = F.softplus(0.50 - logits[pos]).mean()
            minimum = F.softplus(0.75 - logits[pos].min())
    elif category == "inactive":
        # All inactive candidates are explicit negatives.  The normal BCE is
        # retained below and this bounded margin is the R2 no-match term.
        member = F.softplus(logits + 0.25).mean()

    track_target = labels.float()
    track = balanced_bce(output["track_logits"], track_target)
    if category == "present_uncovered":
        track = zero
    inactive = category == "inactive"
    null_target = torch.tensor(float(inactive), device=logits.device)
    null_loss = F.binary_cross_entropy_with_logits(output["null_logit"], null_target)
    presence_target = torch.tensor(float(category != "inactive"), device=logits.device)
    cardinality_loss = F.binary_cross_entropy_with_logits(output["cardinality_logit"], presence_target)
    valid = torch.ones_like(labels, dtype=torch.bool, device=logits.device) if category != "present_uncovered" else torch.zeros_like(labels, dtype=torch.bool, device=logits.device)
    brier = ((torch.sigmoid(logits[valid]) - labels[valid].float()) ** 2).mean() if bool(valid.any()) else zero
    continuation_target = (history_mask.to(device=logits.device).sum(dim=1) > 1).float()
    continuation = balanced_bce(output["continuation_logits"], continuation_target)
    temporal = (output["quality_logits"] - output["track_logits"]).abs().mean()
    total = (member + 1.20 * pair + 0.70 * listwise + 0.80 * minimum +
             0.60 * positive_floor + 0.30 * track + 0.20 * null_loss +
             0.15 * cardinality_loss + 0.05 * brier + 0.05 * continuation +
             0.01 * temporal)
    parts = {
        "total": float(total.detach()), "membership_bce": float(member.detach()),
        "pairwise": float(pair.detach()), "listwise": float(listwise.detach()),
        "minimum_positive": float(minimum.detach()), "positive_floor": float(positive_floor.detach()),
        "track_bce": float(track.detach()), "null": float(null_loss.detach()),
        "cardinality": float(cardinality_loss.detach()), "brier": float(brier.detach()),
        "continuation": float(continuation.detach()), "temporal": float(temporal.detach()),
        "positive_count": int(pos.numel()), "negative_count": int(neg.numel()),
        "covered_membership": bool(covered), "masked_missing_count": int(labels.numel()) if not covered else 0,
        "hard_negative_count": hard_count, "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "hard_negative_selection": "detached current membership logit, frozen objectness, native row-index tie-break",
        "category": str(category),
    }
    return total, parts


__all__ = ["l80_r2_loss"]
