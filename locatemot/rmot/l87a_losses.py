"""L87-A loss: L86 objective with corrected real temporal negatives.

The only training-science change from L86 is temporal negative construction.
For a shared same-query target, negatives are all non-referred target bags
available in either the current or previous frame, not only other referred
targets.  Candidate rows are never removed and target IDs are supervision-only.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss
from locatemot.rmot.l86_losses import (
    _zero,
    l86_membership_loss,
    l86_null_competing_loss,
    l86_presence_loss,
    target_bag_embedding,
)


def l87a_temporal_identity_loss(
    current_output: dict[str, torch.Tensor],
    current_labels: list[dict[str, Any]],
    previous: list[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]],
    margin: float = 0.20,
) -> tuple[torch.Tensor, dict[str, Any]]:
    terms: list[torch.Tensor] = []
    positive_pairs = negative_pairs = 0
    anchors = anchors_with_earlier_positive = anchors_with_negative = 0
    negative_target_bags = single_pairs = multi_pairs = 0
    for previous_output, previous_labels in previous:
        previous_by_qid = {int(item["query_id"]): (index, item)
                           for index, item in enumerate(previous_labels)}
        for q, current_item in enumerate(current_labels):
            qid = int(current_item["query_id"])
            if qid not in previous_by_qid:
                continue
            anchors += 1
            prev_q, previous_item = previous_by_qid[qid]
            referred = {str(value) for value in current_item.get("target_ids", [])}
            previous_referred = {str(value) for value in previous_item.get("target_ids", [])}
            shared = sorted(referred.intersection(previous_referred))
            previous_available = {str(value) for value in previous_item.get("candidate_gt", []) if value is not None}
            current_available = {str(value) for value in current_item.get("candidate_gt", []) if value is not None}
            negative_targets = sorted((previous_available | current_available) - referred)
            for target in shared:
                current_embedding = target_bag_embedding(
                    current_output["temporal_state"][q], current_output["r_total"][q],
                    current_item["candidate_gt"], target)
                previous_embedding = target_bag_embedding(
                    previous_output["temporal_state"][prev_q], previous_output["r_total"][prev_q],
                    previous_item["candidate_gt"], target)
                if current_embedding is None or previous_embedding is None:
                    continue
                anchors_with_earlier_positive += 1
                negatives: list[torch.Tensor] = []
                for negative_target in negative_targets:
                    values: list[torch.Tensor] = []
                    current_value = target_bag_embedding(
                        current_output["temporal_state"][q], current_output["r_total"][q],
                        current_item["candidate_gt"], negative_target)
                    previous_value = target_bag_embedding(
                        previous_output["temporal_state"][prev_q], previous_output["r_total"][prev_q],
                        previous_item["candidate_gt"], negative_target)
                    if current_value is not None:
                        values.append(current_value)
                    if previous_value is not None:
                        values.append(previous_value)
                    if values:
                        negatives.append(torch.stack(values).mean(dim=0))
                if not negatives:
                    continue
                anchors_with_negative += 1
                negative_target_bags += len(negatives)
                pos_sim = F.cosine_similarity(current_embedding, previous_embedding, dim=0)
                neg_sim = torch.stack([
                    F.cosine_similarity(current_embedding, value, dim=0) for value in negatives
                ]).max()
                terms.append(F.softplus(float(margin) + neg_sim - pos_sim))
                positive_pairs += 1
                negative_pairs += len(negatives)
                if len(shared) == 1:
                    single_pairs += 1
                else:
                    multi_pairs += 1
    result = torch.stack(terms).mean() if terms else _zero(current_output["r_total"])
    return result, {
        "temporal_identity": float(result.detach()), "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs, "temporal_anchors": anchors,
        "anchors_with_earlier_positive": anchors_with_earlier_positive,
        "anchors_with_negative": anchors_with_negative, "negative_target_bags": negative_target_bags,
        "single_target_temporal_pairs": single_pairs, "multi_target_temporal_pairs": multi_pairs,
        "negative_contract": "(previous_available | current_available) - referred_targets",
        "margin": float(margin), "finite": bool(torch.isfinite(result)),
    }


def l87a_loss(
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
    semantic, semantic_info = l83_target_bag_loss(interaction, membership_mask, categories, target_ids, candidate_gt)
    static, static_info = l83_target_bag_loss(current_output["r_static"], membership_mask, categories, target_ids, candidate_gt)
    membership, membership_info = l86_membership_loss(current_output["candidate_energy"], categories, target_ids, candidate_gt)
    presence, presence_info = l86_presence_loss(current_output["presence_logit"], categories)
    null_loss, null_info = l86_null_competing_loss(
        current_output["candidate_energy"], current_output["null_logit"], categories, target_ids, candidate_gt)
    if temporal_enabled:
        temporal, temporal_info = l87a_temporal_identity_loss(current_output, current_labels, temporal_pairs)
    else:
        temporal = _zero(interaction)
        temporal_info = {"temporal_identity": 0.0, "positive_pairs": 0, "negative_pairs": 0,
                         "temporal_anchors": 0, "anchors_with_earlier_positive": 0,
                         "anchors_with_negative": 0, "negative_target_bags": 0,
                         "single_target_temporal_pairs": 0, "multi_target_temporal_pairs": 0}
    delta = current_output["temporal_delta"].float().pow(2).mean() * 0.01
    total = semantic + 0.30 * static + membership + 0.50 * presence + 0.50 * null_loss + 0.10 * temporal + delta
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite L87-A total loss")
    info = {
        "total": float(total.detach()), "semantic_total": float(semantic.detach()),
        "semantic_static": float(static.detach()), "membership": float(membership.detach()),
        "presence": float(presence.detach()), "null": float(null_loss.detach()),
        "temporal_id": float(temporal.detach()), "delta_reg": float(delta.detach()),
        "positive_target_bags": int(membership_info["positive_target_bags"]),
        "negative_target_bags": int(membership_info["negative_target_bags"]),
        "positive_pairs": int(temporal_info["positive_pairs"]),
        "negative_pairs": int(temporal_info["negative_pairs"]),
        "temporal_anchors": int(temporal_info.get("temporal_anchors", 0)),
        "anchors_with_earlier_positive": int(temporal_info.get("anchors_with_earlier_positive", 0)),
        "anchors_with_negative": int(temporal_info.get("anchors_with_negative", 0)),
        "negative_target_bags_temporal": int(temporal_info.get("negative_target_bags", 0)),
        "single_target_temporal_pairs": int(temporal_info.get("single_target_temporal_pairs", 0)),
        "multi_target_temporal_pairs": int(temporal_info.get("multi_target_temporal_pairs", 0)),
        "inactive_count": int(sum(category == "inactive" for category in categories)),
        "present_uncovered_count": int(sum(category == "present_uncovered" for category in categories)),
        "positive_count": int(sum(item["positive_count"] for item in current_labels)),
        "masked_missing_count": int(membership_info["masked_missing_count"]),
        "same_class_hard_negative_metadata": "unavailable; unique target-bag all-negative fallback",
        "temporal_negative_metadata": "real candidate_gt target bags only; no synthetic objectness negative",
        "temporal_enabled": bool(temporal_enabled), "finite": True,
    }
    return total, info


__all__ = ["l87a_loss", "l87a_temporal_identity_loss"]
