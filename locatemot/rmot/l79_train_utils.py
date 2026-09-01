"""L79 loss and deterministic fit sampling helpers."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F


LOSS_WEIGHTS = {
    "frame_membership": 1.00,
    "track_set": 1.00,
    "same_frame_hard_pair": 1.00,
    "all_positive_min_margin": 0.75,
    "null_inactive": 0.75,
    "fragment_continuation": 0.50,
    "temporal_consistency": 0.50,
    "observation_quality": 0.25,
    "teacher_identity_stability": 0.15,
    "source_cross_fragment_consistency": 0.10,
}


def _zero(value: torch.Tensor) -> torch.Tensor:
    return value.sum() * 0.0


def _balanced_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected_logits = logits[mask]
    selected_labels = labels[mask].float()
    if selected_logits.numel() == 0:
        return _zero(logits)
    positive = selected_labels > 0.5
    negative = ~positive
    parts = []
    if bool(positive.any()):
        parts.append(F.binary_cross_entropy_with_logits(selected_logits[positive], selected_labels[positive]))
    if bool(negative.any()):
        parts.append(F.binary_cross_entropy_with_logits(selected_logits[negative], selected_labels[negative]))
    return torch.stack(parts).mean()


def _pairwise(pos: torch.Tensor, neg: torch.Tensor, margin: float = 0.20) -> torch.Tensor:
    if pos.numel() == 0 or neg.numel() == 0:
        return _zero(pos if pos.numel() else neg)
    return F.softplus(margin - pos[:, None] + neg[None, :]).mean()


def compute_l79_loss(outputs: dict[str, torch.Tensor], labels: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute all registered losses within one `(video,query,frame)` unit."""
    frame_logits = outputs["frame_membership_logits"]
    track_logits = outputs["track_relevance_logits"]
    quality_logits = outputs["observation_quality_logits"]
    continuation_logits = outputs["continuation_logits"]
    null_logit = outputs["null_logit"]
    y = labels["labels"].to(device=frame_logits.device, dtype=torch.bool)
    mask = labels["membership_mask"].to(device=frame_logits.device, dtype=torch.bool)
    if y.shape != frame_logits.shape or not bool(mask.shape == y.shape):
        raise AssertionError("L79 label/logit shape mismatch")
    pos = frame_logits[y & mask]
    neg = frame_logits[(~y) & mask]
    track_pos = track_logits[y & mask]
    track_neg = track_logits[(~y) & mask]
    frame_membership = _balanced_bce(frame_logits, y, mask)
    track_set = _balanced_bce(track_logits, y, mask)
    hard_pair = _pairwise(pos, neg) + _pairwise(track_pos, track_neg)
    # This term touches every positive row; it is deliberately not max-positive.
    min_margin = F.softplus(0.50 - pos).mean() if pos.numel() else _zero(frame_logits)
    min_track_margin = F.softplus(0.50 - track_pos).mean() if track_pos.numel() else _zero(track_logits)
    if labels["category"] == "inactive":
        null_inactive = F.binary_cross_entropy_with_logits(null_logit, torch.ones_like(null_logit))
        null_inactive = null_inactive + F.softplus(frame_logits + 0.20).mean()
    elif labels["candidate_present"]:
        null_inactive = F.binary_cross_entropy_with_logits(null_logit, torch.zeros_like(null_logit))
    else:
        # Present-uncovered is not an inactive negative: no membership/NULL loss.
        null_inactive = _zero(frame_logits)
    quality_target = labels["history_mask_last"].to(device=quality_logits.device, dtype=quality_logits.dtype)
    observation_quality = F.binary_cross_entropy_with_logits(quality_logits, quality_target)
    continuation = F.binary_cross_entropy_with_logits(continuation_logits, y.to(continuation_logits.dtype)) if bool(mask.any()) else _zero(continuation_logits)
    temporal = F.smooth_l1_loss(torch.sigmoid(frame_logits), torch.sigmoid(track_logits).detach())
    current = F.normalize(outputs["current_vector"].float(), dim=-1)
    track_vector = F.normalize(outputs["track_vector"].float(), dim=-1)
    identity_stability = (1.0 - (current * track_vector).sum(dim=-1)).mean()
    cross_fragment = (1.0 - (current * track_vector.detach()).sum(dim=-1)).mean()
    terms = {
        "frame_membership": frame_membership,
        "track_set": track_set,
        "same_frame_hard_pair": hard_pair,
        "all_positive_min_margin": min_margin + min_track_margin,
        "null_inactive": null_inactive,
        "fragment_continuation": continuation,
        "temporal_consistency": temporal,
        "observation_quality": observation_quality,
        "teacher_identity_stability": identity_stability,
        "source_cross_fragment_consistency": cross_fragment,
    }
    total = _zero(frame_logits)
    for name, value in terms.items():
        total = total + LOSS_WEIGHTS[name] * value
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("nonfinite L79 loss")
    details = {
        "loss_terms": {name: float(value.detach().cpu()) for name, value in terms.items()},
        "weighted_total": float(total.detach().cpu()),
        "positive_count": int(pos.numel()),
        "negative_count": int(neg.numel()),
        "masked_missing_count": int((~mask).sum().item()),
        "minimum_positive_logit": float(pos.detach().min().cpu()) if pos.numel() else None,
        "mean_positive_logit": float(pos.detach().mean().cpu()) if pos.numel() else None,
        "mean_negative_logit": float(neg.detach().mean().cpu()) if neg.numel() else None,
        "same_class_hard_negative_metadata": "unavailable",
        "hard_negative_fallback": "all current-frame negatives",
    }
    return total, details


def deterministic_fit_order(units: list[dict[str, Any]], seed: int = 20260829) -> list[dict[str, Any]]:
    required = {"positive", "multi_positive", "inactive", "present_uncovered"}
    domains = {str(x["dataset"]) for x in units}
    if domains != {"refer_kitti_v1", "refer_kitti_v2"}:
        raise AssertionError(f"L79 fit domains drifted: {domains}")
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        groups[(str(unit["dataset"]), str(unit["category"]))].append(unit)
    for values in groups.values():
        values.sort(key=lambda x: (str(x["video"]), int(x["frame_id"]), int(x["query_id"]), str(x["unit_key"])))
        if values:
            shift = rng.randrange(len(values))
            values[:] = values[shift:] + values[:shift]
    result = sorted(units, key=lambda x: (str(x["video"]), str(x["dataset"]), str(x["category"]), int(x["frame_id"]), int(x["query_id"]), str(x["unit_key"])))
    if not required.issubset({str(x["category"]) for x in result}):
        raise AssertionError("L79 fit order misses a required category")
    if len(result) != len(units) or len({str(x["unit_key"]) for x in result}) != len(units):
        raise AssertionError("L79 fit order changed unit cardinality")
    return result


def smoke_stratified_order(units: list[dict[str, Any]], seed: int = 20260829, count: int = 100) -> list[dict[str, Any]]:
    order = deterministic_fit_order(units, seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in order:
        groups[(str(unit["dataset"]), str(unit["category"]))].append(unit)
    keys = sorted(groups)
    result = []
    index = 0
    while len(result) < count:
        key = keys[index % len(keys)]
        result.append(groups[key][(index // len(keys)) % len(groups[key])])
        index += 1
    return result


__all__ = ["LOSS_WEIGHTS", "compute_l79_loss", "deterministic_fit_order", "smoke_stratified_order"]
