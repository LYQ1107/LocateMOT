"""Expression-level, multi-positive L80 losses."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def balanced_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    targets = targets.to(dtype=logits.dtype)
    pieces = []
    positive = targets > 0.5
    negative = ~positive
    if bool(positive.any()):
        pieces.append(F.binary_cross_entropy_with_logits(logits[positive], targets[positive]))
    if bool(negative.any()):
        pieces.append(F.binary_cross_entropy_with_logits(logits[negative], targets[negative]))
    return torch.stack(pieces).mean() if pieces else logits.new_zeros(())


def l80_loss(output: dict[str, torch.Tensor], labels: torch.Tensor,
             membership_mask: bool | torch.Tensor, observations: torch.Tensor,
             history_mask: torch.Tensor, category: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute one-unit losses; ``present_uncovered`` masks membership terms."""
    logits = output["candidate_logits"]
    labels = labels.bool().to(device=logits.device)
    if logits.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("candidate logits/labels mismatch")
    if isinstance(membership_mask, bool):
        covered = bool(membership_mask)
    else:
        covered = bool(membership_mask.to(device=logits.device).all())
    zero = logits.new_zeros(())
    pos = torch.nonzero(labels, as_tuple=False).flatten()
    neg = torch.nonzero(~labels, as_tuple=False).flatten()
    if category == "present_uncovered" or not covered:
        member = zero
        pair = zero
        listwise = zero
        minimum = zero
    else:
        member = balanced_bce(logits, labels)
        objectness = observations[:, -1].to(device=logits.device).float()
        hard = neg[torch.argsort(objectness[neg], descending=True, stable=True)[:min(16, neg.numel())]] if neg.numel() else neg
        pair = F.softplus(0.50 + logits[hard].unsqueeze(0) - logits[pos].unsqueeze(1)).mean() if pos.numel() and hard.numel() else zero
        listwise = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[pos], dim=0) if pos.numel() else zero
        minimum = F.softplus(0.50 - logits[pos].min()) if pos.numel() else zero
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
    # Continuation is an identity-quality auxiliary: it uses only whether a
    # current observation has a causal history entry, never future/GT IDs.
    continuation_target = (history_mask.to(device=logits.device).sum(dim=1) > 1).float()
    continuation = balanced_bce(output["continuation_logits"], continuation_target)
    # A small temporal regularizer keeps the sequence-quality auxiliary finite
    # without imposing a future or cross-expression comparison.
    temporal = (output["quality_logits"] - output["track_logits"]).abs().mean()
    total = (member + 0.80 * pair + 0.50 * listwise + 0.50 * minimum +
             0.30 * track + 0.20 * null_loss + 0.15 * cardinality_loss +
             0.05 * brier + 0.05 * continuation + 0.01 * temporal)
    parts = {
        "total": float(total.detach()), "membership_bce": float(member.detach()),
        "pairwise": float(pair.detach()), "listwise": float(listwise.detach()),
        "minimum_positive": float(minimum.detach()), "track_bce": float(track.detach()),
        "null": float(null_loss.detach()), "cardinality": float(cardinality_loss.detach()),
        "brier": float(brier.detach()), "continuation": float(continuation.detach()),
        "temporal": float(temporal.detach()), "positive_count": int(pos.numel()),
        "negative_count": int(neg.numel()), "covered_membership": bool(covered),
        "masked_missing_count": int(labels.numel()) if not covered else 0,
        "hard_negative_count": int(min(16, neg.numel())) if covered else 0,
        "category": str(category),
    }
    return total, parts


__all__ = ["balanced_bce", "l80_loss"]
