"""Fit-only, group-local losses for the L82 rank probe."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _smooth_min(value: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    return -temperature * torch.logsumexp(-value / temperature, dim=0)


def _smooth_max(value: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    return temperature * torch.logsumexp(value / temperature, dim=0)


def l82_rank_loss(
    interaction: torch.Tensor,
    labels: torch.Tensor,
    membership_mask: torch.Tensor,
    categories: list[str],
    margin: float = 0.50,
    query_swap_margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute only within-frame losses and retain all positive rows.

    ``present_uncovered`` units carry an all-false membership vector but are
    masked out of membership correspondence losses.  Inactive units are
    explicit no-match examples.  The query-swap term compares only different
    expressions on the same frame and never crosses a frame/video boundary.
    """
    if interaction.ndim != 2 or labels.shape != interaction.shape:
        raise ValueError("L82 loss shape mismatch")
    q_count, n_count = interaction.shape
    labels = labels.bool()
    membership_mask = membership_mask.bool().reshape(-1)
    if membership_mask.numel() != q_count:
        raise ValueError("membership mask must have one entry per query")
    if len(categories) != q_count:
        raise ValueError("category/query mismatch")

    zero = interaction.sum() * 0.0
    bce_terms: list[torch.Tensor] = []
    pair_terms: list[torch.Tensor] = []
    minimum_terms: list[torch.Tensor] = []
    floor_terms: list[torch.Tensor] = []
    brier_terms: list[torch.Tensor] = []
    inactive_terms: list[torch.Tensor] = []
    swap_terms: list[torch.Tensor] = []
    positive_count = negative_count = hard_pair_count = masked_missing_count = 0
    minimum_positive_count = 0

    for q in range(q_count):
        if categories[q] == "present_uncovered":
            masked_missing_count += n_count
            continue
        if categories[q] == "inactive":
            inactive_terms.append(F.softplus(interaction[q] + margin).mean())
            negative_count += n_count
            brier_terms.append(torch.sigmoid(interaction[q]).square().mean())
            continue
        if not bool(membership_mask[q]):
            masked_missing_count += n_count
            continue
        pos = interaction[q][labels[q]]
        neg = interaction[q][~labels[q]]
        positive_count += int(pos.numel())
        negative_count += int(neg.numel())
        if pos.numel() == 0:
            continue
        values = interaction[q]
        y = labels[q].to(dtype=values.dtype)
        if neg.numel():
            positive_weight = max(1.0, float(neg.numel()) / max(1, pos.numel()))
            per_row = F.binary_cross_entropy_with_logits(values, y, reduction="none")
            weights = torch.where(labels[q], per_row.new_tensor(positive_weight), per_row.new_tensor(1.0))
            bce_terms.append((per_row * weights).sum() / weights.sum().clamp_min(1.0))
            pair = F.softplus(float(margin) - pos[:, None] + neg[None, :])
            pair_terms.append(pair.mean())
            hard_pair_count += int(pos.numel() * neg.numel())
            smooth_low = _smooth_min(pos)
            smooth_high = _smooth_max(neg)
            minimum_terms.append(F.softplus(float(margin) + smooth_high - smooth_low))
        else:
            bce_terms.append(F.binary_cross_entropy_with_logits(values, y))
        # Every positive participates in this floor term, including the lowest.
        floor_terms.append(F.softplus(-pos).mean())
        brier_terms.append((torch.sigmoid(values) - y).square().mean())
        minimum_positive_count += 1

    # Query-axis supervision is restricted to exact same-frame label flips.
    for left in range(q_count):
        if categories[left] == "present_uncovered" or not bool(membership_mask[left]):
            continue
        for right in range(left + 1, q_count):
            if categories[right] == "present_uncovered" or not bool(membership_mask[right]):
                continue
            flip = labels[left] != labels[right]
            if not bool(flip.any()):
                continue
            left_values = interaction[left][flip]
            right_values = interaction[right][flip]
            left_positive = labels[left][flip]
            signed = torch.where(left_positive, left_values - right_values, right_values - left_values)
            swap_terms.append(F.softplus(float(query_swap_margin) - signed).mean())

    def average(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    bce = average(bce_terms)
    pairwise = average(pair_terms)
    minimum = average(minimum_terms)
    floor = average(floor_terms)
    brier = average(brier_terms)
    inactive = average(inactive_terms)
    query_swap = average(swap_terms)
    total = bce + 0.50 * pairwise + 0.50 * minimum + 0.25 * floor + 0.20 * query_swap + 0.20 * inactive + 0.05 * brier
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite L82 loss")
    return total, {
        "total": total.detach(),
        "membership_bce": bce.detach(),
        "pairwise": pairwise.detach(),
        "minimum_positive": minimum.detach(),
        "positive_floor": floor.detach(),
        "query_swap": query_swap.detach(),
        "inactive_no_match": inactive.detach(),
        "brier": brier.detach(),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "hard_pair_count": hard_pair_count,
        "minimum_positive_count": minimum_positive_count,
        "masked_missing_count": masked_missing_count,
        "same_class_hard_negative_metadata": "unavailable",
        "hard_negative_fallback": "all current-frame negatives; no verified class metadata",
        "finite": True,
    }


__all__ = ["l82_rank_loss"]
