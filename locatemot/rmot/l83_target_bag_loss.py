"""Faithful duplicate-aware target-bag loss for L83."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from locatemot.rmot.l83_target_bags import bag_values, build_target_bag_layout, positive_target_bags


def _mean_or_zero(values: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else zero


def l83_target_bag_loss(
    interaction: torch.Tensor,
    membership_mask: torch.Tensor,
    categories: list[str],
    target_ids: list[list[str]],
    candidate_gt: list[list[str | None]],
    bag_margin: float = 0.50,
    query_margin: float = 0.25,
    inactive_margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute group-local losses using max-pooled unique target bags.

    ``present_uncovered`` masks all correspondence terms.  Inactive units use
    an absolute negative anchor, while covered units use balanced positive and
    negative bag classification plus candidate-axis ranking.  Query-axis
    comparisons use only exact real-label flips within this frame group.
    """
    if interaction.ndim != 2:
        raise ValueError("interaction must be [Q,N]")
    q_count, n_count = interaction.shape
    if len(categories) != q_count or len(target_ids) != q_count or len(candidate_gt) != q_count:
        raise ValueError("query metadata length mismatch")
    mask = membership_mask.bool().reshape(-1)
    if mask.numel() != q_count:
        raise ValueError("membership_mask must have one value per query")
    for values in candidate_gt:
        if len(values) != n_count:
            raise ValueError("candidate_gt row count mismatch")
    if not bool(torch.isfinite(interaction.float()).all()):
        raise FloatingPointError("nonfinite L83 interaction")

    zero = interaction.sum() * 0.0
    bag_cls_terms: list[torch.Tensor] = []
    candidate_axis_terms: list[torch.Tensor] = []
    query_axis_terms: list[torch.Tensor] = []
    inactive_terms: list[torch.Tensor] = []
    positive_bags = negative_bags = background_bags = 0
    query_flip_count = masked_missing_count = inactive_count = 0
    positive_target_count = 0
    minimum_positive_values: list[torch.Tensor] = []

    layouts = [build_target_bag_layout(values) for values in candidate_gt]
    bag_cache: list[tuple[list[tuple[str, str | int]], torch.Tensor, torch.Tensor]] = []
    for q in range(q_count):
        keys, scores, positive_mask = bag_values(interaction[q], layouts[q], target_ids[q])
        bag_cache.append((keys, scores, positive_mask))
        referred_bags = positive_target_bags(layouts[q], target_ids[q])
        if categories[q] == "present_uncovered" or (target_ids[q] and not referred_bags):
            masked_missing_count += n_count
            continue
        if categories[q] == "inactive" or not target_ids[q]:
            inactive_count += 1
            if scores.numel():
                inactive_terms.append(F.softplus(float(inactive_margin) + scores.max()))
                negative_bags += int(scores.numel())
            continue
        pos = scores[positive_mask]
        neg = scores[~positive_mask]
        positive_bags += int(pos.numel())
        negative_bags += int(neg.numel())
        background_bags += sum(1 for key, _ in keys if key[0] == "background")
        positive_target_count += len(referred_bags)
        if pos.numel():
            bag_cls_terms.append(F.softplus(-pos).mean())
            weakest = pos.min()
            minimum_positive_values.append(weakest)
            if neg.numel():
                bag_cls_terms.append(F.softplus(neg).mean())
                candidate_axis_terms.append(F.softplus(float(bag_margin) + neg.max() - weakest))
        elif scores.numel():
            bag_cls_terms.append(F.softplus(scores).mean())

    # Exact query-axis flips: a target bag is positive for one real expression
    # and negative for another expression in this same frame group.
    for left in range(q_count):
        if categories[left] in {"inactive", "present_uncovered"} or not bool(mask[left]):
            continue
        for right in range(left + 1, q_count):
            if categories[right] in {"inactive", "present_uncovered"} or not bool(mask[right]):
                continue
            left_targets = set(str(x) for x in target_ids[left])
            right_targets = set(str(x) for x in target_ids[right])
            differing = (left_targets ^ right_targets).intersection(layouts[left].unique_target_ids)
            for target in sorted(differing):
                if target not in layouts[right].target_to_rows:
                    continue
                left_is_pos = target in left_targets
                left_score = bag_cache[left][1][bag_cache[left][0].index(("target", target))]
                right_score = bag_cache[right][1][bag_cache[right][0].index(("target", target))]
                signed = left_score - right_score if left_is_pos else right_score - left_score
                query_axis_terms.append(F.softplus(float(query_margin) - signed))
                query_flip_count += 1

    bag_cls = _mean_or_zero(bag_cls_terms, zero)
    candidate_axis = _mean_or_zero(candidate_axis_terms, zero)
    query_axis = _mean_or_zero(query_axis_terms, zero)
    inactive = _mean_or_zero(inactive_terms, zero)
    total = bag_cls + candidate_axis + 0.50 * query_axis + 0.25 * inactive
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite L83 target-bag loss")
    return total, {
        "total": total.detach(), "bag_classification": bag_cls.detach(),
        "candidate_axis": candidate_axis.detach(), "query_axis": query_axis.detach(),
        "inactive_no_match": inactive.detach(),
        "positive_bag_count": positive_bags, "negative_bag_count": negative_bags,
        "background_bag_count": background_bags, "positive_target_count": positive_target_count,
        "query_flip_count": query_flip_count, "inactive_count": inactive_count,
        "masked_missing_count": masked_missing_count,
        "minimum_positive_count": len(minimum_positive_values),
        "same_class_hard_negative_metadata": "unavailable",
        "hard_negative_fallback": "all non-referred target bags and background singleton bags",
        "bag_score": "max within each unique candidate_gt target bag",
        "finite": True,
    }


__all__ = ["l83_target_bag_loss"]
