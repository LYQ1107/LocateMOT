"""Corrected unique-target-bag metrics for L83."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from locatemot.rmot.l83_target_bags import bag_values, build_target_bag_layout


def roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    """Rank-based ROC-AUC with average ranks for ties."""
    if len(y_true) != len(y_score) or not y_true or len(set(y_true)) < 2:
        return None
    order = sorted(range(len(y_score)), key=lambda i: (float(y_score[i]), i))
    ranks = [0.0] * len(order)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and float(y_score[order[end]]) == float(y_score[order[pos]]):
            end += 1
        rank = (pos + 1 + end) / 2.0
        for index in range(pos, end):
            ranks[order[index]] = rank
        pos = end
    positive = sum(int(value) for value in y_true)
    negative = len(y_true) - positive
    rank_sum = sum(ranks[index] for index, value in enumerate(y_true) if value)
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def _average_precision(scores: list[float], labels: list[bool]) -> float | None:
    if not scores or not any(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    seen = total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            seen += 1.0
            total += seen / rank
    return float(total / seen)


def _target_bag_index(keys: list[tuple[str, str | int]], target: str) -> int | None:
    try:
        return keys.index(("target", target))
    except ValueError:
        return None


def group_metrics(data: Any, scores: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute row diagnostics and primary unique-bag metrics for one frame group."""
    values = scores.detach().float().cpu()
    if values.ndim != 2 or tuple(values.shape) != tuple(data.labels.shape):
        raise ValueError(f"score shape drift for {data.group_key}: {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"nonfinite score for {data.group_key}")
    labels = data.labels.bool().cpu().numpy()
    row_hard_total = row_hard_bad = 0
    bag_hard_total = bag_hard_bad = 0
    bag_query_total = bag_hit1 = 0
    bag_recall5_numerator = bag_recall5_denominator = 0
    multi_total = multi_exact = 0
    row_aps: list[float] = []
    bag_aps: list[float] = []
    row_margins: list[float] = []
    bag_margins: list[float] = []
    swap_y: list[int] = []
    swap_scores: list[float] = []
    swap_pair_correct = swap_pair_total = 0
    inactive_total = inactive_false = inactive_empty = 0
    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for q in range(values.shape[0]):
        category = str(data.categories[q])
        category_stats[category]["queries"] += 1
        row_score = values[q]
        row_labels = labels[q]
        if category == "inactive":
            inactive_total += 1
            accepted = bool((row_score >= 0.0).any())
            inactive_false += int(accepted)
            inactive_empty += int(not accepted)
            category_stats[category]["false_accept"] += int(accepted)
            continue
        if category == "present_uncovered" or not bool(data.membership_mask[q]):
            category_stats[category]["masked"] += 1
            continue
        target_ids = tuple(str(x) for x in data.target_ids[q])
        layout = build_target_bag_layout(data.candidate_gt[q])
        keys, bag_score, positive_mask = bag_values(row_score, layout, target_ids)
        positive_bags = [index for index, positive in enumerate(positive_mask.tolist()) if positive]
        negative_bags = [index for index, positive in enumerate(positive_mask.tolist()) if not positive]
        if not positive_bags:
            category_stats["present_uncovered"]["masked"] += 1
            continue
        category_stats[category]["covered"] += 1
        if row_labels.any():
            row_order = np.argsort(-row_score.numpy(), kind="stable")
            row_ap = _average_precision(row_score.tolist(), row_labels.tolist())
            if row_ap is not None:
                row_aps.append(row_ap)
            if (~row_labels).any():
                row_min = float(row_score[row_labels].min())
                row_max = float(row_score[~row_labels].max())
                row_margins.append(row_min - row_max)
                row_hard_total += 1
                row_hard_bad += int(row_max >= row_min)
        if negative_bags:
            positive_min = float(bag_score[positive_mask].min())
            negative_max = float(bag_score[~positive_mask].max())
            bag_margins.append(positive_min - negative_max)
            bag_hard_total += 1
            bag_hard_bad += int(negative_max >= positive_min)
        bag_order = sorted(range(len(keys)), key=lambda index: (-float(bag_score[index]), index))
        bag_query_total += 1
        bag_hit1 += int(positive_mask[bag_order[0]])
        positive_count = int(positive_mask.sum())
        bag_recall5_numerator += sum(int(positive_mask[index]) for index in bag_order[:5])
        bag_recall5_denominator += positive_count
        bag_scores = bag_score.tolist()
        bag_aps_value = _average_precision(bag_scores, positive_mask.tolist())
        if bag_aps_value is not None:
            bag_aps.append(bag_aps_value)
        if category == "multi_positive" or len(target_ids) > 1:
            multi_total += 1
            required = set(target_ids)
            top_t = bag_order[:len(required)]
            seen_targets = {keys[index][1] for index in top_t if keys[index][0] == "target"}
            multi_exact += int(required.issubset(seen_targets))

    # Exact cross-expression real-label flips within this frame group.
    for left in range(values.shape[0]):
        if data.categories[left] in {"inactive", "present_uncovered"} or not bool(data.membership_mask[left]):
            continue
        for right in range(left + 1, values.shape[0]):
            if data.categories[right] in {"inactive", "present_uncovered"} or not bool(data.membership_mask[right]):
                continue
            left_targets = set(str(x) for x in data.target_ids[left])
            right_targets = set(str(x) for x in data.target_ids[right])
            layout = build_target_bag_layout(data.candidate_gt[left])
            for target in sorted((left_targets ^ right_targets).intersection(layout.unique_target_ids)):
                li = _target_bag_index(bag_values(values[left], layout, left_targets)[0], target)
                ri = _target_bag_index(bag_values(values[right], layout, right_targets)[0], target)
                if li is None or ri is None:
                    continue
                left_score = float(bag_values(values[left], layout, left_targets)[1][li])
                right_score = float(bag_values(values[right], layout, right_targets)[1][ri])
                if target in left_targets:
                    positive_score, negative_score = left_score, right_score
                else:
                    positive_score, negative_score = right_score, left_score
                delta = positive_score - negative_score
                swap_pair_total += 1
                swap_pair_correct += int(delta > 0.0)
                swap_y.extend([1, 0])
                swap_scores.extend([positive_score, negative_score])

    per_group = {
        "format": "locatemot-l83-target-bag-group-metrics-v1",
        "group_key": str(data.group_key), "dataset": str(data.dataset), "video": str(data.video),
        "frame_id": int(data.frame_id), "query_count": len(data.query_unit_keys),
        "candidate_count": int(data.candidate_count),
        "row_hard_violation": (row_hard_bad / row_hard_total) if row_hard_total else None,
        "row_hard_total": row_hard_total, "row_hard_bad": row_hard_bad,
        "target_bag_hard_violation": (bag_hard_bad / bag_hard_total) if bag_hard_total else None,
        "target_bag_hard_total": bag_hard_total, "target_bag_hard_bad": bag_hard_bad,
        "target_bag_hit_at1": bag_hit1, "target_bag_query_total": bag_query_total,
        "target_bag_recall_at5_numerator": bag_recall5_numerator,
        "target_bag_recall_at5_denominator": bag_recall5_denominator,
        "multi_target_exact_topT": (multi_exact / multi_total) if multi_total else None,
        "multi_target_total": multi_total, "multi_target_exact": multi_exact,
        "query_swap_pair_accuracy": (swap_pair_correct / swap_pair_total) if swap_pair_total else None,
        "query_swap_pair_correct": swap_pair_correct, "query_swap_pair_total": swap_pair_total,
        "query_swap_roc_auc": roc_auc(swap_y, swap_scores),
        "row_ap_macro": float(np.mean(row_aps)) if row_aps else None,
        "target_bag_ap_macro": float(np.mean(bag_aps)) if bag_aps else None,
        "row_margin_mean": float(np.mean(row_margins)) if row_margins else None,
        "target_bag_margin_mean": float(np.mean(bag_margins)) if bag_margins else None,
        "score_mean": float(values.mean()), "score_std": float(values.std(unbiased=False)),
        "inactive_false_acceptance": (inactive_false / inactive_total) if inactive_total else None,
        "inactive_false": inactive_false, "inactive_total": inactive_total,
        "inactive_empty_count": inactive_empty,
        "category_stats": {key: dict(value) for key, value in category_stats.items()},
        "query_ids": [int(x) for x in data.query_ids], "categories": list(data.categories),
        "positive_counts": [int(x) for x in data.labels.sum(dim=1).tolist()],
        "row_keys_digest": list(data.row_keys_digest), "row_offsets": list(data.row_offsets),
        "candidate_indices": list(data.candidate_indices), "pool_ids": list(data.pool_ids),
        "candidate_deletion": False, "candidate_truncation": False,
        "candidate_count_complete": True, "finite": True,
    }
    return per_group, {"swap_y": swap_y, "swap_scores": swap_scores, "row_margins": row_margins, "bag_margins": bag_margins}


def aggregate_group_metrics(records: list[dict[str, Any]], auxiliaries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    def ratio(num: str, den: str) -> float | None:
        denominator = sum(int(row.get(den, 0)) for row in records)
        return float(sum(int(row.get(num, 0)) for row in records) / denominator) if denominator else None

    def mean(field: str) -> float | None:
        values = [float(row[field]) for row in records if row.get(field) is not None and math.isfinite(float(row[field]))]
        return float(np.mean(values)) if values else None

    y_true: list[int] = []
    y_score: list[float] = []
    if auxiliaries:
        for item in auxiliaries:
            y_true.extend(int(x) for x in item.get("swap_y", []))
            y_score.extend(float(x) for x in item.get("swap_scores", []))
    result = {
        "format": "locatemot-l83-target-bag-aggregate-metrics-v1", "group_count": len(records),
        "row_hard_violation": ratio("row_hard_bad", "row_hard_total"),
        "target_bag_hard_violation": ratio("target_bag_hard_bad", "target_bag_hard_total"),
        "target_bag_hit_at1": ratio("target_bag_hit_at1", "target_bag_query_total"),
        "target_bag_recall_at5": ratio("target_bag_recall_at5_numerator", "target_bag_recall_at5_denominator"),
        "multi_target_exact_topT": ratio("multi_target_exact", "multi_target_total"),
        "query_swap_pair_accuracy": ratio("query_swap_pair_correct", "query_swap_pair_total"),
        "query_swap_roc_auc": roc_auc(y_true, y_score),
        "row_ap_macro": mean("row_ap_macro"), "target_bag_ap_macro": mean("target_bag_ap_macro"),
        "row_margin_mean": mean("row_margin_mean"), "target_bag_margin_mean": mean("target_bag_margin_mean"),
        "score_mean": mean("score_mean"), "score_std": mean("score_std"),
        "inactive_false_acceptance": ratio("inactive_false", "inactive_total"),
        "inactive_total": sum(int(row.get("inactive_total", 0)) for row in records),
        "inactive_false": sum(int(row.get("inactive_false", 0)) for row in records),
        "target_bag_hard_total": sum(int(row.get("target_bag_hard_total", 0)) for row in records),
        "target_bag_query_total": sum(int(row.get("target_bag_query_total", 0)) for row in records),
        "target_bag_recall_at5_denominator": sum(int(row.get("target_bag_recall_at5_denominator", 0)) for row in records),
        "multi_target_total": sum(int(row.get("multi_target_total", 0)) for row in records),
        "query_swap_pair_total": sum(int(row.get("query_swap_pair_total", 0)) for row in records),
        "candidate_deletion": any(bool(row.get("candidate_deletion", True)) for row in records),
        "candidate_truncation": any(bool(row.get("candidate_truncation", True)) for row in records),
        "finite": all(bool(row.get("finite", False)) for row in records),
    }
    return result


def breakdowns(records: list[dict[str, Any]], auxiliaries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("dataset", "video"):
        result[field] = {}
        for value in sorted({str(row[field]) for row in records}):
            indexes = [index for index, row in enumerate(records) if str(row[field]) == value]
            result[field][value] = aggregate_group_metrics(
                [records[index] for index in indexes],
                [auxiliaries[index] for index in indexes] if auxiliaries else None,
            )
    result["group_count"] = len(records)
    return result


__all__ = ["aggregate_group_metrics", "breakdowns", "group_metrics", "roc_auc"]
