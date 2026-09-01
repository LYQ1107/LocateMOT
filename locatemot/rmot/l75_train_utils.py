"""Small deterministic sampling/loss helpers for the L75 sidecar."""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def sample_candidate_indices(record: dict[str, Any], bank: Any,
                             max_negatives: int = 16) -> list[int]:
    """Keep every positive and a fixed query-independent negative subset.

    The negative ordering uses only frozen objectness and native row order.  It
    is a cost control for the VLM training path; the full candidate count and
    all row keys remain in the record and evaluation never calls this helper.
    """
    n = int(record["candidate_count"])
    if n != len(record["row_offsets"]):
        raise AssertionError("candidate count drift before sampling")
    positives = [int(value) for value in record.get("positive_indices", [])]
    if any(value < 0 or value >= n for value in positives):
        raise AssertionError("positive index outside current candidate set")
    positive_set = set(positives)
    negatives = [index for index in range(n) if index not in positive_set]
    objectness = bank.tensors["objectness"]
    rows = [int(row) for row in record["row_offsets"]]
    ranked = sorted(
        negatives,
        key=lambda index: (-float(objectness[rows[index]]), index),
    )
    # Positives are intentionally placed first so all multi-positive rows can
    # participate in one minimum/listwise term before negative chunks are
    # streamed.  This is an internal compute order; the immutable bank row
    # order and complete candidate record are unchanged.
    selected = sorted(positive_set) + ranked[:int(max_negatives)]
    if any(value < 0 or value >= n for value in selected):
        raise AssertionError("sampled candidate index contract failed")
    return selected


def l75_loss(match_logits: torch.Tensor, absent_logits: torch.Tensor,
             labels: torch.Tensor, category: str, coverage_mask: bool,
             marker: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Expression-level, within-frame loss with explicit multi-positive terms."""
    logits = match_logits.float().reshape(-1)
    labels = labels.float().reshape(-1).to(device=logits.device)
    absent = absent_logits.float().reshape(-1)
    if logits.numel() != labels.numel():
        raise AssertionError("loss logit/label length mismatch")
    if not torch.isfinite(logits).all() or not torch.isfinite(absent).all():
        raise AssertionError("nonfinite L75 logits")
    pos = labels > 0.5
    neg = ~pos
    zero = logits.new_zeros(())
    parts: dict[str, torch.Tensor] = {
        "balanced_bce": zero, "pairwise": zero, "min_positive": zero,
        "listwise": zero, "inactive_negative": zero, "absent": zero,
        "regularization": zero,
    }
    if coverage_mask:
        if bool(pos.any()) and bool(neg.any()):
            pos_loss = F.binary_cross_entropy_with_logits(logits[pos], torch.ones_like(logits[pos]))
            neg_loss = F.binary_cross_entropy_with_logits(logits[neg], torch.zeros_like(logits[neg]))
            parts["balanced_bce"] = 0.5 * (pos_loss + neg_loss)
        elif bool(pos.any()):
            parts["balanced_bce"] = F.binary_cross_entropy_with_logits(logits[pos], torch.ones_like(logits[pos]))
        elif bool(neg.any()):
            parts["balanced_bce"] = F.binary_cross_entropy_with_logits(logits[neg], torch.zeros_like(logits[neg]))
        if bool(pos.any()) and bool(neg.any()):
            margin = 0.20
            parts["pairwise"] = F.softplus(
                margin - logits[pos].unsqueeze(1) + logits[neg].unsqueeze(0)
            ).mean()
        if bool(pos.any()):
            # All positives occur in the denominator; BCE and this term give
            # every positive a direct gradient, including the lowest one.
            parts["min_positive"] = F.softplus(0.20 - logits[pos].min())
            all_scores = torch.cat([logits[pos], logits[neg]])
            parts["listwise"] = -torch.logsumexp(logits[pos], dim=0) + torch.logsumexp(all_scores, dim=0)
    elif category == "inactive":
        # Inactive is the only complete no-target frame.  This is not a NULL
        # post-filter and is applied only to the sampled rows in the fit cost
        # control; all rows remain part of the unit audit.
        parts["inactive_negative"] = F.softplus(logits + 0.50).mean()
    # The absent head is diagnostic at evaluation, but its supervised loss is
    # explicit and frame-level: inactive=1, any expression target=0.  A
    # present-uncovered unit is not converted into membership negatives.
    absent_target = torch.full_like(absent, 1.0 if category == "inactive" else 0.0)
    parts["absent"] = F.binary_cross_entropy_with_logits(absent, absent_target)
    if marker is not None:
        parts["regularization"] = 1e-5 * marker.float().pow(2).mean()
    total = (
        parts["balanced_bce"] + parts["pairwise"] + 0.50 * parts["min_positive"] +
        0.25 * parts["listwise"] + parts["inactive_negative"] +
        0.25 * parts["absent"] + parts["regularization"]
    )
    if not torch.isfinite(total):
        raise AssertionError("nonfinite L75 loss")
    return total, {name: float(value.detach().cpu()) for name, value in parts.items()}


def gradient_row_summary(logits: torch.Tensor, labels: torch.Tensor,
                         coverage_mask: bool) -> dict[str, Any]:
    grad = logits.grad
    labels = labels.detach().reshape(-1)
    if grad is None:
        return {"available": False, "positive_nonzero": 0, "negative_nonzero": 0}
    g = grad.detach().float().reshape(-1).abs()
    pos = labels > 0.5
    neg = ~pos
    return {
        "available": True,
        "coverage_mask": bool(coverage_mask),
        "positive_count": int(pos.sum()),
        "negative_count": int(neg.sum()),
        "positive_nonzero": int((g[pos] > 1e-12).sum()) if bool(pos.any()) else 0,
        "negative_nonzero": int((g[neg] > 1e-12).sum()) if bool(neg.any()) else 0,
        "max_abs": float(g.max()) if g.numel() else 0.0,
        "all_finite": bool(torch.isfinite(g).all()),
        "masked_uncovered_has_membership_gradient": bool((not coverage_mask) and bool(g.abs().sum() > 0)),
    }
