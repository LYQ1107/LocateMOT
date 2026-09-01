"""Unit-local losses for the isolated L71 correspondence head."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_loss(output: dict[str, torch.Tensor], data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["correspondence_logits"]
    target = data["membership_target"]
    coverage = data["coverage_mask"].bool()
    positive = (target > 0.5) & coverage
    negative = (target <= 0.5) & coverage
    zero = logits.sum() * 0.0

    if bool(positive.any()) and bool(negative.any()):
        pairwise = F.softplus(0.2 - logits[positive, None] + logits[negative][None, :]).mean()
    else:
        pairwise = zero
    if bool(positive.any()):
        listwise = torch.logsumexp(logits[coverage], dim=0) - torch.logsumexp(logits[positive], dim=0)
        if bool(negative.any()):
            listwise = listwise + F.softplus(logits[negative].max() - logits[positive].min() + 0.2)
    else:
        listwise = zero

    # A target with no referent is truly inactive.  A present-uncovered unit
    # has coverage_mask=False and therefore cannot manufacture negatives.
    inactive = bool(float(data["null_target"].reshape(-1)[0]) > 0.5)
    no_match = F.softplus(logits[coverage] + 0.2).mean() if inactive and bool(coverage.any()) else zero
    regularizer = 0.001 * logits.square().mean()
    total = pairwise + 0.5 * listwise + 0.5 * no_match + regularizer
    parts: dict[str, float] = {
        "pairwise": float(pairwise.detach()),
        "all_positive_listwise": float(listwise.detach()),
        "inactive_no_match": float(no_match.detach()),
        "logit_l2": float(regularizer.detach()),
        "total": float(total.detach()),
        "positive_count": float(positive.sum()),
        "negative_count": float(negative.sum()),
        "masked_missing_count": float((~coverage).sum()),
        "coverage_mask": float(coverage.any()),
        "inactive_unit": float(inactive),
    }
    return total, parts
