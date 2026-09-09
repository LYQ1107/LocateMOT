"""Registered R0 target-bag and coverage-presence losses."""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.nn import functional as F

from locatemot.rmot.l83_target_bags import bag_values, build_target_bag_layout


def _zero(value: torch.Tensor) -> torch.Tensor:
    return value.sum() * 0.0


def r0_membership_set_loss(
    membership_logits: torch.Tensor,
    categories: Iterable[str],
    target_ids: Iterable[Iterable[object]],
    candidate_gt: Iterable[Iterable[object | None]],
    *,
    weakest_tau: float = 0.20,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the fixed R0 covered/inactive target-bag objective.

    A row-level positive term is retained in addition to target-bag terms so
    every positive observation receives gradient, while the bag terms keep
    same-GT fragments jointly positive.
    """
    if membership_logits.ndim != 2:
        raise ValueError("membership_logits must be [Q,N]")
    categories = [str(value) for value in categories]
    target_ids = [tuple(str(x) for x in values) for values in target_ids]
    candidate_gt = [[None if value is None else str(value) for value in values] for values in candidate_gt]
    q_count, n_count = membership_logits.shape
    if len(categories) != q_count or len(target_ids) != q_count or len(candidate_gt) != q_count:
        raise ValueError("R0 loss query count mismatch")
    total = _zero(membership_logits)
    info: dict[str, Any] = {"queries": q_count, "positive_count": 0, "negative_count": 0,
                            "masked_present_uncovered": 0, "hard_negative_count": 0,
                            "finite": True, "components": []}
    tau = float(weakest_tau)
    if tau <= 0:
        raise ValueError("weakest_tau must be positive")
    for q in range(q_count):
        if len(candidate_gt[q]) != n_count:
            raise ValueError("candidate_gt/logit length mismatch")
        values = membership_logits[q]
        category = categories[q]
        if category == "present_uncovered":
            info["masked_present_uncovered"] += 1
            info["components"].append({"category": category, "membership_masked": True})
            continue
        labels = torch.tensor(
            [value is not None and value in set(target_ids[q]) for value in candidate_gt[q]],
            device=values.device, dtype=torch.bool,
        )
        if category == "inactive":
            component = F.softplus(values).mean()
            total = total + component
            info["negative_count"] += int(labels.numel())
            info["components"].append({"category": category, "positive_count": 0,
                                       "negative_count": int(labels.numel()), "inactive_loss": float(component.detach())})
            continue
        if not bool(labels.any()):
            raise ValueError(f"covered active unit has no positive candidate at query {q}")
        pos_rows = values[labels]
        neg_rows = values[~labels]
        layout = build_target_bag_layout(candidate_gt[q])
        _bag_keys, bag_scores, bag_positive = bag_values(values, layout, target_ids[q])
        positive_bags = bag_scores[bag_positive]
        negative_bags = bag_scores[~bag_positive]
        positive_cls = F.softplus(-pos_rows).mean()
        negative_cls = F.softplus(neg_rows).mean() if neg_rows.numel() else _zero(values)
        softmin_pos = -tau * torch.logsumexp(-positive_bags / tau, dim=0)
        positive_floor = F.softplus(-softmin_pos)
        hard_negative = negative_bags.max() if negative_bags.numel() else _zero(values)
        set_margin = F.softplus(0.50 + hard_negative - softmin_pos) if negative_bags.numel() else _zero(values)
        component = positive_cls + negative_cls + positive_floor + set_margin
        total = total + component
        info["positive_count"] += int(pos_rows.numel())
        info["negative_count"] += int(neg_rows.numel())
        info["hard_negative_count"] += int(negative_bags.numel())
        info["components"].append({
            "category": category, "positive_count": int(pos_rows.numel()),
            "negative_count": int(neg_rows.numel()), "positive_bag_count": int(positive_bags.numel()),
            "negative_bag_count": int(negative_bags.numel()),
            "minimum_positive_logit": float(positive_bags.min().detach()),
            "hard_negative_logit": float(hard_negative.detach()),
            "positive_cls": float(positive_cls.detach()), "negative_cls": float(negative_cls.detach()),
            "positive_floor": float(positive_floor.detach()), "set_margin": float(set_margin.detach()),
        })
    total = total / max(1, q_count)
    if not bool(torch.isfinite(total.float()).all()):
        raise FloatingPointError("nonfinite R0 membership loss")
    info["loss"] = float(total.detach())
    return total, info


def r0_coverage_presence_loss(
    coverage_presence_logits: torch.Tensor,
    categories: Iterable[str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Balanced query-level coverage-presence BCE for the four strata."""
    values = coverage_presence_logits.reshape(-1)
    categories = [str(value) for value in categories]
    if len(categories) != values.numel():
        raise ValueError("coverage/category count mismatch")
    target = torch.tensor([category in {"positive", "multi_positive"} for category in categories],
                          dtype=values.dtype, device=values.device)
    pos = target > 0.5
    neg = ~pos
    losses = []
    if bool(pos.any()):
        losses.append(F.binary_cross_entropy_with_logits(values[pos], target[pos]))
    if bool(neg.any()):
        losses.append(F.binary_cross_entropy_with_logits(values[neg], target[neg]))
    loss = torch.stack(losses).mean() if losses else _zero(values)
    if not bool(torch.isfinite(loss.float()).all()):
        raise FloatingPointError("nonfinite R0 coverage loss")
    return loss, {"loss": float(loss.detach()), "positive_queries": int(pos.sum()),
                  "negative_queries": int(neg.sum()), "finite": True}


def r0_total_loss(
    membership_logits: torch.Tensor,
    coverage_presence_logits: torch.Tensor,
    categories: Iterable[str],
    target_ids: Iterable[Iterable[object]],
    candidate_gt: Iterable[Iterable[object | None]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    membership, membership_info = r0_membership_set_loss(
        membership_logits, categories, target_ids, candidate_gt)
    coverage, coverage_info = r0_coverage_presence_loss(coverage_presence_logits, categories)
    total = membership + 0.50 * coverage
    if not bool(torch.isfinite(total.float()).all()):
        raise FloatingPointError("nonfinite R0 total loss")
    return total, {"membership": membership_info, "coverage": coverage_info, "loss": float(total.detach()), "finite": True}


__all__ = ["r0_membership_set_loss", "r0_coverage_presence_loss", "r0_total_loss"]
