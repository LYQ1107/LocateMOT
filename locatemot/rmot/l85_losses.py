"""Group-local, multi-positive losses for the L85 factorized RMOT model."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _balanced_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    parts = []
    pos = targets > 0.5
    neg = ~pos
    if bool(pos.any()):
        parts.append(F.binary_cross_entropy_with_logits(logits[pos], targets[pos]))
    if bool(neg.any()):
        parts.append(F.binary_cross_entropy_with_logits(logits[neg], targets[neg]))
    return torch.stack(parts).mean() if parts else logits.new_zeros(())


def _rank_terms(logits: torch.Tensor, labels: torch.Tensor, objectness: torch.Tensor,
                margin: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return pairwise, listwise, minimum-positive terms for one candidate bag."""
    labels = labels.bool()
    pos = torch.nonzero(labels, as_tuple=False).flatten()
    neg = torch.nonzero(~labels, as_tuple=False).flatten()
    zero = logits.sum() * 0.0
    if not pos.numel() or not neg.numel():
        minimum = F.softplus(float(margin) - logits[pos].min()) if pos.numel() else zero
        return zero, zero, minimum, int(pos.numel()), int(neg.numel())
    order = torch.argsort(objectness[neg].float(), descending=True, stable=True)
    hard = neg[order[: min(16, int(neg.numel()))]]
    pair = F.softplus(float(margin) + logits[hard].unsqueeze(0) - logits[pos].unsqueeze(1)).mean()
    listwise = torch.logsumexp(logits[neg], dim=0) - torch.logsumexp(logits[pos], dim=0)
    minimum = F.softplus(float(margin) - logits[pos].min())
    return pair, listwise, minimum, int(pos.numel()), int(neg.numel())


def l85_loss(
    output: dict[str, torch.Tensor],
    labels: list[torch.Tensor],
    membership_masks: list[bool | torch.Tensor],
    categories: list[str],
    observations: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    temporal_enabled: bool,
    semantic_margin: float = 0.50,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the registered L85 loss for one complete frame/query group.

    Labels are expression-level candidate membership labels attached only after
    the label-free cache item has been built. ``present_uncovered`` masks all
    candidate membership/ranking terms instead of manufacturing negatives.
    """
    scores = output["membership"]
    static = output["r_static"]
    r_total = output["r_total"]
    presence = output["presence"]
    null_logit = output["null_logit"]
    if scores.ndim != 2 or len(labels) != scores.shape[0] or len(categories) != scores.shape[0]:
        raise ValueError("L85 group/query shape mismatch")
    n = scores.shape[1]
    if observations.shape != (n, 1432) or history_mask.shape[0] != n:
        raise ValueError("L85 observation shape mismatch")
    if not bool(torch.isfinite(scores.float()).all() and torch.isfinite(static.float()).all() and
                torch.isfinite(r_total.float()).all() and torch.isfinite(presence.float()).all() and
                torch.isfinite(null_logit.float()).all()):
        raise FloatingPointError("nonfinite L85 loss input")

    zero = scores.sum() * 0.0
    rank_total_terms: list[torch.Tensor] = []
    rank_static_terms: list[torch.Tensor] = []
    membership_terms: list[torch.Tensor] = []
    presence_terms: list[torch.Tensor] = []
    null_terms: list[torch.Tensor] = []
    temporal_terms: list[torch.Tensor] = []
    positive_count = negative_count = hard_count = masked_missing = inactive_count = 0
    minimum_positive_values: list[torch.Tensor] = []
    categories_seen: dict[str, int] = {}
    objectness = observations[:, -1].float()

    for q, (label, category) in enumerate(zip(labels, categories)):
        label = label.to(device=scores.device).bool().reshape(-1)
        if label.numel() != n:
            raise ValueError(f"label length mismatch at query {q}: {label.numel()} != {n}")
        categories_seen[str(category)] = categories_seen.get(str(category), 0) + 1
        pos = torch.nonzero(label, as_tuple=False).flatten()
        neg = torch.nonzero(~label, as_tuple=False).flatten()
        positive_count += int(pos.numel()); negative_count += int(neg.numel())
        covered = bool(membership_masks[q]) if isinstance(membership_masks[q], bool) else bool(membership_masks[q].all())
        if category == "present_uncovered" or not covered:
            masked_missing += n
        else:
            pair, listwise, minimum, _, _ = _rank_terms(r_total[q], label, objectness, semantic_margin)
            s_pair, s_listwise, s_minimum, _, _ = _rank_terms(static[q], label, objectness, semantic_margin)
            rank_total_terms.extend((pair, listwise, minimum))
            rank_static_terms.extend((s_pair, s_listwise, s_minimum))
            membership_terms.append(_balanced_bce(scores[q], label))
            if pos.numel():
                minimum_positive_values.append(r_total[q, pos].min())
            hard_count += min(16, int(neg.numel()))
        present_target = torch.tensor(float(category != "inactive"), device=scores.device)
        presence_terms.append(F.binary_cross_entropy_with_logits(presence[q], present_target))
        if category == "inactive":
            inactive_count += 1
            # Explicit no-match pressure over all rows, not a NULL filter.
            null_terms.append(F.binary_cross_entropy_with_logits(null_logit[q], torch.ones_like(null_logit[q])))
            null_terms.append(F.softplus(float(semantic_margin) + scores[q]).mean())
        elif category == "present_uncovered":
            # It is present, but the candidate bank has no positive row.
            null_terms.append(F.binary_cross_entropy_with_logits(null_logit[q], torch.zeros_like(null_logit[q])))
        else:
            null_terms.append(F.binary_cross_entropy_with_logits(null_logit[q], torch.zeros_like(null_logit[q])))
            if pos.numel():
                null_terms.append(F.softplus(float(semantic_margin) + null_logit[q] - r_total[q, pos].min()))

    rank_total = torch.stack(rank_total_terms).mean() if rank_total_terms else zero
    rank_static = torch.stack(rank_static_terms).mean() if rank_static_terms else zero
    membership = torch.stack(membership_terms).mean() if membership_terms else zero
    presence_loss = torch.stack(presence_terms).mean() if presence_terms else zero
    null_loss = torch.stack(null_terms).mean() if null_terms else zero
    if temporal_enabled:
        history_target = (history_mask.to(device=scores.device).sum(dim=1) > 1).float()
        gate = output["temporal_gate"]
        temporal_terms.append(F.binary_cross_entropy_with_logits(gate, history_target.unsqueeze(0).expand_as(gate)))
        temporal_terms.append(output["temporal_correction"].float().pow(2).mean() * 0.05)
    temporal = torch.stack(temporal_terms).mean() if temporal_terms else zero
    total = rank_total + 0.30 * rank_static + membership + 0.50 * presence_loss + 0.50 * null_loss + 0.10 * temporal
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite L85 total loss")
    parts = {
        "total": float(total.detach()), "semantic_rank_r_total": float(rank_total.detach()),
        "semantic_rank_r_static": float(rank_static.detach()), "membership_s": float(membership.detach()),
        "presence_b": float(presence_loss.detach()), "null_rank": float(null_loss.detach()),
        "temporal": float(temporal.detach()), "positive_count": positive_count,
        "negative_count": negative_count, "hard_negative_count": hard_count,
        "masked_missing_count": masked_missing, "inactive_count": inactive_count,
        "minimum_positive_mean": float(torch.stack(minimum_positive_values).mean().detach()) if minimum_positive_values else None,
        "categories": categories_seen, "temporal_enabled": bool(temporal_enabled),
        "same_class_hard_negative_metadata": "unavailable; all-negative objectness fallback",
        "finite": True,
    }
    return total, parts


__all__ = ["l85_loss"]
