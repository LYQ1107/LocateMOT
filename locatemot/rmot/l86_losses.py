"""Faithful L86 target-bag, presence/NULL and temporal identity losses."""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F

from locatemot.rmot.l83_target_bags import bag_values, build_target_bag_layout, positive_target_bags
from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss


def _zero(value: torch.Tensor) -> torch.Tensor:
    return value.sum() * 0.0


def l86_membership_loss(
    energy: torch.Tensor,
    categories: list[str],
    target_ids: list[list[str]],
    candidate_gt: list[list[str | None]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Classify unique target bags; never penalize a missing present target."""
    if energy.ndim != 2 or len(categories) != energy.shape[0]:
        raise ValueError("L86 membership shape mismatch")
    terms: list[torch.Tensor] = []
    pos_bags = neg_bags = masked = 0
    for q in range(energy.shape[0]):
        layout = build_target_bag_layout(candidate_gt[q])
        _, scores, positive = bag_values(energy[q], layout, target_ids[q])
        category = str(categories[q])
        if category == "present_uncovered":
            masked += int(energy.shape[1])
            continue
        if category == "inactive" or not target_ids[q]:
            if scores.numel():
                terms.append(F.softplus(scores).mean())
                neg_bags += int(scores.numel())
            continue
        pos = scores[positive]
        neg = scores[~positive]
        if pos.numel():
            terms.append(F.softplus(-pos).mean())
            pos_bags += int(pos.numel())
        if neg.numel():
            terms.append(F.softplus(neg).mean())
            neg_bags += int(neg.numel())
    result = torch.stack(terms).mean() if terms else _zero(energy)
    return result, {
        "membership": float(result.detach()),
        "positive_target_bags": pos_bags,
        "negative_target_bags": neg_bags,
        "masked_missing_count": masked,
        "finite": bool(torch.isfinite(result)),
    }


def l86_presence_loss(presence: torch.Tensor, categories: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    if presence.ndim != 1 or len(categories) != presence.numel():
        raise ValueError("L86 presence shape mismatch")
    target = torch.tensor([float(category != "inactive") for category in categories], device=presence.device)
    positive = target > 0.5
    terms = []
    if bool(positive.any()):
        terms.append(F.binary_cross_entropy_with_logits(presence[positive], target[positive]))
    if bool((~positive).any()):
        terms.append(F.binary_cross_entropy_with_logits(presence[~positive], target[~positive]))
    result = torch.stack(terms).mean() if terms else _zero(presence)
    return result, {"presence": float(result.detach()), "positive_frames": int(positive.sum()), "inactive_frames": int((~positive).sum())}


def l86_null_competing_loss(
    energy: torch.Tensor,
    null_logit: torch.Tensor,
    categories: list[str],
    target_ids: list[list[str]],
    candidate_gt: list[list[str | None]],
    margin: float = 0.50,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Make NULL compete with target-bag evidence independently of presence."""
    terms: list[torch.Tensor] = []
    inactive = active = masked = 0
    for q in range(energy.shape[0]):
        layout = build_target_bag_layout(candidate_gt[q])
        _, scores, positive = bag_values(energy[q], layout, target_ids[q])
        category = str(categories[q])
        if category == "present_uncovered":
            masked += 1
            continue
        if category == "inactive" or not target_ids[q]:
            if scores.numel():
                terms.append(F.softplus(float(margin) + scores.max() - null_logit[q]))
            inactive += 1
            continue
        pos = scores[positive]
        if pos.numel():
            terms.append(F.softplus(float(margin) + null_logit[q] - pos.min()))
            active += 1
    result = torch.stack(terms).mean() if terms else _zero(energy)
    return result, {"null": float(result.detach()), "inactive": inactive, "active": active, "masked_missing": masked}


def target_bag_embedding(
    state: torch.Tensor,
    scores: torch.Tensor,
    candidate_gt: list[str | None],
    target: str,
) -> torch.Tensor | None:
    rows = [index for index, value in enumerate(candidate_gt) if value is not None and str(value) == str(target)]
    if not rows:
        return None
    row_index = torch.tensor(rows, dtype=torch.long, device=state.device)
    weights = torch.softmax(scores[row_index] / 0.10, dim=0)
    vector = (weights.unsqueeze(-1) * state[row_index]).sum(dim=0)
    return F.normalize(vector, dim=0)


def l86_temporal_identity_loss(
    current_output: dict[str, torch.Tensor],
    current_labels: list[dict[str, Any]],
    previous: list[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]],
    margin: float = 0.20,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use real same-query/target pairs only; target IDs never enter tensors."""
    terms: list[torch.Tensor] = []
    positive_pairs = negative_pairs = 0
    current_qids = [int(item["query_id"]) for item in current_labels]
    for previous_output, previous_labels in previous:
        previous_by_qid = {int(item["query_id"]): (index, item) for index, item in enumerate(previous_labels)}
        for q, current_item in enumerate(current_labels):
            qid = int(current_item["query_id"])
            if qid not in previous_by_qid:
                continue
            prev_q, prev_item = previous_by_qid[qid]
            current_targets = {str(value) for value in current_item.get("target_ids", [])}
            previous_targets = {str(value) for value in prev_item.get("target_ids", [])}
            shared = sorted(current_targets.intersection(previous_targets))
            for target in shared:
                current_embedding = target_bag_embedding(
                    current_output["temporal_state"][q], current_output["r_total"][q], current_item["candidate_gt"], target
                )
                previous_embedding = target_bag_embedding(
                    previous_output["temporal_state"][prev_q], previous_output["r_total"][prev_q], prev_item["candidate_gt"], target
                )
                if current_embedding is None or previous_embedding is None:
                    continue
                other: list[torch.Tensor] = []
                for other_target in sorted(previous_targets - {target}):
                    value = target_bag_embedding(
                        previous_output["temporal_state"][prev_q], previous_output["r_total"][prev_q], prev_item["candidate_gt"], other_target
                    )
                    if value is not None:
                        other.append(value)
                if not other:
                    continue
                pos_sim = F.cosine_similarity(current_embedding, previous_embedding, dim=0)
                neg_sim = torch.stack([F.cosine_similarity(current_embedding, value, dim=0) for value in other]).max()
                terms.append(F.softplus(float(margin) + neg_sim - pos_sim))
                positive_pairs += 1
                negative_pairs += len(other)
    result = torch.stack(terms).mean() if terms else _zero(current_output["r_total"])
    return result, {"temporal_identity": float(result.detach()), "positive_pairs": positive_pairs, "negative_pairs": negative_pairs, "finite": bool(torch.isfinite(result))}


def l86_loss(
    current_output: dict[str, torch.Tensor],
    current_labels: list[dict[str, Any]],
    observations: torch.Tensor,
    temporal_pairs: list[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]],
    *,
    temporal_enabled: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    categories = [str(item["category"]) for item in current_labels]
    target_ids = [list(item["target_ids"]) for item in current_labels]
    candidate_gt = [list(item["candidate_gt"]) for item in current_labels]
    membership_mask = torch.tensor([bool(item["coverage_mask"]) for item in current_labels], device=observations.device)
    interaction = current_output["r_total"]
    static_loss, static_info = l83_target_bag_loss(interaction, membership_mask, categories, target_ids, candidate_gt)
    static_loss_value, static_info_value = l83_target_bag_loss(current_output["r_static"], membership_mask, categories, target_ids, candidate_gt)
    membership, membership_info = l86_membership_loss(current_output["candidate_energy"], categories, target_ids, candidate_gt)
    presence, presence_info = l86_presence_loss(current_output["presence_logit"], categories)
    null_loss, null_info = l86_null_competing_loss(current_output["candidate_energy"], current_output["null_logit"], categories, target_ids, candidate_gt)
    if temporal_enabled:
        temporal, temporal_info = l86_temporal_identity_loss(current_output, current_labels, temporal_pairs)
    else:
        temporal = _zero(interaction)
        temporal_info = {"temporal_identity": 0.0, "positive_pairs": 0, "negative_pairs": 0}
    delta = current_output["temporal_delta"].float().pow(2).mean() * 0.01
    total = static_loss + 0.30 * static_loss_value + membership + 0.50 * presence + 0.50 * null_loss + 0.10 * temporal + delta
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite L86 total loss")
    info = {
        "total": float(total.detach()),
        "semantic_total": float(static_loss.detach()),
        "semantic_static": float(static_loss_value.detach()),
        "membership": float(membership.detach()),
        "presence": float(presence.detach()),
        "null": float(null_loss.detach()),
        "temporal_id": float(temporal.detach()),
        "delta_reg": float(delta.detach()),
        "positive_target_bags": int(membership_info["positive_target_bags"]),
        "negative_target_bags": int(membership_info["negative_target_bags"]),
        "positive_pairs": int(temporal_info["positive_pairs"]),
        "negative_pairs": int(temporal_info["negative_pairs"]),
        "inactive_count": int(sum(category == "inactive" for category in categories)),
        "present_uncovered_count": int(sum(category == "present_uncovered" for category in categories)),
        "positive_count": int(sum(item["positive_count"] for item in current_labels)),
        "masked_missing_count": int(membership_info["masked_missing_count"]),
        "same_class_hard_negative_metadata": "unavailable; unique target-bag all-negative fallback",
        "temporal_enabled": bool(temporal_enabled),
        "finite": True,
    }
    return total, info


__all__ = [
    "l86_loss", "l86_membership_loss", "l86_null_competing_loss", "l86_presence_loss",
    "l86_temporal_identity_loss", "target_bag_embedding",
]
