"""Unit-local L77 losses; no cross-expression/frame negatives."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_loss(output: dict[str, torch.Tensor], data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["match_logits"]
    target = data["membership_target"].to(logits.device).float()
    coverage = data["coverage_mask"].to(logits.device).bool()
    positive = (target > 0.5) & coverage
    negative = (target <= 0.5) & coverage
    zero = logits.sum() * 0.0

    if bool(positive.any()) and bool(negative.any()):
        pos_count = positive.sum().to(logits.dtype)
        neg_count = negative.sum().to(logits.dtype)
        weights = torch.where(positive, neg_count / pos_count.clamp_min(1.0), torch.ones_like(target))
        bce = F.binary_cross_entropy_with_logits(logits[coverage], target[coverage], weight=weights[coverage])
        pairwise = F.softplus(0.25 - logits[positive, None] + logits[negative][None, :]).mean()
    elif bool(positive.any()):
        bce = F.binary_cross_entropy_with_logits(logits[positive], target[positive])
        pairwise = zero
    elif bool(negative.any()):
        bce = F.binary_cross_entropy_with_logits(logits[negative], target[negative])
        pairwise = zero
    else:
        bce = zero
        pairwise = zero

    if bool(positive.any()):
        if bool(negative.any()):
            negative_logsumexp = torch.logsumexp(logits[negative], dim=0)
            listwise = F.softplus(negative_logsumexp - logits[positive] + 0.25).mean()
            minimum_positive = F.softplus(logits[negative].max() - logits[positive].min() + 0.25)
        else:
            listwise = -F.logsigmoid(logits[positive]).mean()
            minimum_positive = -F.logsigmoid(logits[positive].min())
    else:
        listwise = zero
        minimum_positive = zero

    inactive = bool(float(data["null_target"].reshape(-1)[0]) > 0.5)
    inactive_no_match = F.softplus(logits[coverage] + 0.25).mean() if inactive and bool(coverage.any()) else zero
    absent_target = data["null_target"].to(logits.device).reshape(1)
    absent_loss = F.binary_cross_entropy_with_logits(output["absent_logit"].reshape(1), absent_target)
    brier = (torch.sigmoid(logits[coverage]) - target[coverage]).square().mean() if bool(coverage.any()) else zero
    regularization = 0.0 * logits.square().mean()
    total = (bce + 0.50 * pairwise + 0.50 * listwise + 0.25 * minimum_positive
             + 0.50 * inactive_no_match + 0.20 * absent_loss + 0.05 * brier + regularization)
    parts: dict[str, float] = {
        "balanced_bce": float(bce.detach()), "pairwise_all_negative_fallback": float(pairwise.detach()),
        "all_positive_listwise": float(listwise.detach()), "minimum_positive": float(minimum_positive.detach()),
        "inactive_no_match": float(inactive_no_match.detach()), "absent_loss": float(absent_loss.detach()),
        "brier": float(brier.detach()), "regularization": float(regularization.detach()),
        "total": float(total.detach()), "positive_count": float(positive.sum()),
        "negative_count": float(negative.sum()), "masked_missing_count": float((~coverage).sum()),
        "coverage_mask": float(coverage.any()), "inactive_unit": float(inactive),
    }
    return total, parts
