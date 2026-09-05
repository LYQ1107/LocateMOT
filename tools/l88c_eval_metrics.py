#!/usr/bin/env python3
"""L88C metrics with the registered candidate-vs-NULL emission contract.

This file is intentionally separate from ``l88_eval_metrics.py``.  L88C
replays existing scores without training and changes only the deployment
equation: NULL competes with each candidate energy, never with query-level
presence.  Target-bag and legacy-row metrics remain separate.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from l88_eval_metrics import GRID, NULL_MARGIN_GRID, summary, target_bags


def corrected_emission_mask(
    candidate_energy: np.ndarray | Iterable[float],
    presence_logit: float,
    null_logit: float,
    candidate_threshold: float,
    presence_threshold: float,
    null_margin: float,
) -> np.ndarray:
    """Return the registered row-level L88C emission mask.

    ``null_logit`` is a query/frame constant and is compared to every
    candidate energy.  Presence is an independent query/frame gate.
    """
    values = np.asarray(candidate_energy, dtype=np.float64)
    if values.ndim != 1:
        raise AssertionError(f"candidate energy must be one-dimensional: {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(
            np.asarray([presence_logit, null_logit, candidate_threshold,
                        presence_threshold, null_margin], dtype=np.float64)).all():
        raise FloatingPointError("nonfinite L88C emission input")
    return ((values >= float(candidate_threshold))
            & ((values - float(null_logit)) >= float(null_margin))
            & (float(presence_logit) >= float(presence_threshold)))


def _bag_row_indices(candidate_gt: list[str | None]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    background: list[int] = []
    for index, value in enumerate(candidate_gt):
        if value is None:
            background.append(index)
        else:
            groups.setdefault(str(value), []).append(index)
    return [groups[target] for target in sorted(groups)] + [[index] for index in background]


def _metric(records: list[dict[str, Any]], candidate_threshold: float,
            presence_threshold: float, null_margin: float, *, stratify: bool) -> dict[str, Any]:
    row_tp = row_fp = row_fn = row_selected = row_positive = 0
    row_top1 = row_top5 = row_target_units = row_empty = 0
    row_hard_total = row_hard_bad = 0
    row_margins: list[float] = []
    row_best: list[float] = []
    row_average: list[float] = []
    multi_recall: list[float] = []
    multi_exact: list[float] = []

    bag_tp = bag_fp = bag_fn = bag_selected = bag_positive = 0
    bag_hit1 = bag_hit5 = bag_query_total = 0
    bag_hard_total = bag_hard_bad = 0
    bag_margins: list[float] = []
    distinct_hit = distinct_total = 0
    distinct_multi_exact: list[float] = []
    inactive_units = inactive_accept = inactive_fp_rows = 0
    present_uncovered_units = candidate_present_units = 0
    empty = 0
    score_values: list[float] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in records:
        scores = np.asarray(row["score"], dtype=np.float64)
        labels = np.asarray(row["labels"], dtype=bool)
        candidate_gt = [None if value is None else str(value) for value in row["candidate_gt"]]
        if scores.ndim != 1 or scores.shape != labels.shape or len(candidate_gt) != scores.size:
            raise AssertionError(f"L88C metric row shape drift: {row['unit_key']}")
        if len(row["row_keys"]) != scores.size or not np.isfinite(scores).all():
            raise AssertionError(f"L88C metric key/finite drift: {row['unit_key']}")
        score_values.extend(scores.tolist())
        selected = corrected_emission_mask(
            scores, float(row["presence_logit"]), float(row["null_logit"]),
            candidate_threshold, presence_threshold, null_margin)
        row_tp += int((selected & labels).sum())
        row_fp += int((selected & ~labels).sum())
        row_fn += int((~selected & labels).sum())
        row_selected += int(selected.sum())
        row_positive += int(labels.sum())
        row_empty += int(not selected.any())
        if labels.any():
            order = np.argsort(-scores, kind="stable")
            row_target_units += 1
            row_top1 += int(bool(labels[order[:1]].any()))
            row_top5 += int(bool(labels[order[:5]].any()))
        pos = np.flatnonzero(labels)
        neg = np.flatnonzero(~labels)
        if pos.size and neg.size:
            minimum = float(scores[pos].min())
            maximum = float(scores[neg].max())
            row_margins.append(minimum - maximum)
            row_best.append(float(scores[pos].max() - maximum))
            row_average.append(float(scores[pos].mean() - maximum))
            row_hard_total += 1
            row_hard_bad += int(maximum >= minimum)
        if pos.size > 1:
            multi_recall.append(float(selected[pos].sum() / pos.size))
            multi_exact.append(float(selected[pos].all()))

        target_ids = [str(value) for value in row["target_ids"]]
        bags, bag_positive_mask, _ = target_bags(scores, candidate_gt, target_ids)
        bag_indices = _bag_row_indices(candidate_gt)
        if len(bag_indices) != len(bags):
            raise AssertionError(f"L88C bag order drift: {row['unit_key']}")
        # Derive bag emission from the actual row mask.  This is equivalent to
        # max-energy thresholding because NULL is a query-level constant, but
        # prevents the replay from silently diverging from final inference.
        bag_selected_mask = np.asarray([bool(selected[indexes].any()) for indexes in bag_indices], dtype=bool)
        bag_tp += int((bag_selected_mask & bag_positive_mask).sum())
        bag_fp += int((bag_selected_mask & ~bag_positive_mask).sum())
        bag_fn += int((~bag_selected_mask & bag_positive_mask).sum())
        bag_selected += int(bag_selected_mask.sum())
        bag_positive += int(bag_positive_mask.sum())
        if bag_positive_mask.any():
            bag_order = np.argsort(-bags, kind="stable")
            bag_query_total += 1
            bag_hit1 += int(bool(bag_positive_mask[bag_order[:1]].any()))
            bag_hit5 += int(bool(bag_positive_mask[bag_order[:5]].any()))
            distinct_total += int(bag_positive_mask.sum())
            distinct_hit += int((bag_selected_mask & bag_positive_mask).sum())
            if bag_positive_mask.sum() > 1:
                distinct_multi_exact.append(float((bag_selected_mask & bag_positive_mask)[bag_positive_mask].all()))
            bag_neg = ~bag_positive_mask
            if bag_neg.any():
                minimum = float(bags[bag_positive_mask].min())
                maximum = float(bags[bag_neg].max())
                bag_margins.append(minimum - maximum)
                bag_hard_total += 1
                bag_hard_bad += int(maximum >= minimum)

        category = str(row.get("category", "unknown"))
        by_dataset[str(row["dataset"])].append(row)
        by_category[category].append(row)
        if category == "inactive":
            inactive_units += 1
            inactive_accept += int(bool(selected.any()))
            inactive_fp_rows += int((selected & ~labels).sum())
        elif category == "present_uncovered":
            present_uncovered_units += 1
        else:
            candidate_present_units += 1

    def ratio(num: int, den: int) -> float:
        return float(num / max(1, den))

    result: dict[str, Any] = {
        "units": len(records),
        "candidate_rows": int(sum(len(row["score"]) for row in records)),
        "candidate_rows_retained": True, "candidate_deletion": False,
        "candidate_truncation": False, "finite_scores": True,
        "candidate_threshold": float(candidate_threshold),
        "presence_threshold": float(presence_threshold), "null_margin": float(null_margin),
        "target_bag_tp": bag_tp, "target_bag_false": bag_fp, "target_bag_fn": bag_fn,
        "target_bag_selected": bag_selected, "target_bag_positive": bag_positive,
        "target_bag_precision": ratio(bag_tp, bag_selected),
        "target_bag_recall": ratio(bag_tp, bag_tp + bag_fn),
        "target_bag_f1": float(2.0 * bag_tp / max(1.0, 2.0 * bag_tp + bag_fp + bag_fn)),
        "target_bag_hard_violation": ratio(bag_hard_bad, bag_hard_total),
        "target_bag_hard_total": bag_hard_total, "target_bag_hard_bad": bag_hard_bad,
        "target_bag_hit1": ratio(bag_hit1, bag_query_total),
        "target_bag_hit5": ratio(bag_hit5, bag_query_total),
        "target_bag_query_total": bag_query_total,
        "distinct_target_recall": ratio(distinct_hit, distinct_total),
        "distinct_target_hit": distinct_hit, "distinct_target_total": distinct_total,
        "distinct_multi_target_exact": float(np.mean(distinct_multi_exact)) if distinct_multi_exact else None,
        "distinct_multi_target_units": len(distinct_multi_exact),
        "target_bag_margin": summary(bag_margins),
        "legacy_true_positive_rows": row_tp, "legacy_false_positive_rows": row_fp,
        "legacy_false_negative_rows": row_fn, "legacy_selected_rows": row_selected,
        "legacy_positive_rows": row_positive,
        "legacy_candidate_precision": ratio(row_tp, row_selected),
        "legacy_candidate_recall": ratio(row_tp, row_tp + row_fn),
        "legacy_fp_per_frame": ratio(row_fp, len(records)),
        "legacy_predictions_per_positive": ratio(row_selected, row_positive),
        "legacy_top1": ratio(row_top1, row_target_units),
        "legacy_top5": ratio(row_top5, row_target_units),
        "legacy_row_hard_violation": ratio(row_hard_bad, row_hard_total),
        "legacy_row_hard_total": row_hard_total, "legacy_row_hard_bad": row_hard_bad,
        "legacy_row_strict_margin": summary(row_margins),
        "legacy_row_best_margin": summary(row_best),
        "legacy_row_average_margin": summary(row_average),
        "legacy_row_multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "legacy_row_multi_target_exact": float(np.mean(multi_exact)) if multi_exact else None,
        "multi_positive_units": len(multi_recall), "empty_rate": ratio(empty, len(records)),
        "inactive_units": inactive_units,
        "inactive_false_acceptance": ratio(inactive_accept, inactive_units),
        "inactive_false_positive_rows": inactive_fp_rows,
        "present_uncovered_units": present_uncovered_units,
        "candidate_present_units": candidate_present_units,
        "score_distribution": summary(score_values),
    }
    # The original helper names this count row_empty; preserve the actual
    # value explicitly for compatibility with the historical report.
    result["empty_rate"] = ratio(row_empty, len(records))
    if stratify:
        result["per_dataset"] = {
            key: _metric(values, candidate_threshold, presence_threshold, null_margin, stratify=False)
            for key, values in sorted(by_dataset.items())
        }
        result["per_category"] = {
            key: _metric(values, candidate_threshold, presence_threshold, null_margin, stratify=False)
            for key, values in sorted(by_category.items())
        }
    return result


def metric(records: list[dict[str, Any]], candidate_threshold: float,
           presence_threshold: float = -1.0, null_margin: float = 0.0) -> dict[str, Any]:
    return _metric(records, candidate_threshold, presence_threshold, null_margin, stratify=True)


def fit_rule_set(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for presence in GRID:
        for null_margin in NULL_MARGIN_GRID:
            for candidate in GRID:
                measured = metric(records, candidate, presence, null_margin)
                measured["rule"] = "grid_candidate_energy_null"
                candidates.append(measured)

    def rule_b_key(value: dict[str, Any]) -> tuple[Any, ...]:
        return (float(value["target_bag_f1"]), -float(value["inactive_false_acceptance"]),
                float(value["distinct_target_recall"]),
                float(value["distinct_multi_target_exact"] or 0.0),
                -float(value["target_bag_false"]), -float(value["candidate_threshold"]),
                -float(value["presence_threshold"]), -float(value["null_margin"]))

    def rule_r_key(value: dict[str, Any]) -> tuple[Any, ...]:
        if float(value["target_bag_precision"]) < 0.08:
            return (-1, -1.0, -float(value["inactive_false_acceptance"]),
                    -float(value["target_bag_false"]))
        return (1, float(value["distinct_target_recall"]), float(value["target_bag_precision"]),
                -float(value["inactive_false_acceptance"]), -float(value["target_bag_false"]),
                -float(value["candidate_threshold"]), -float(value["presence_threshold"]),
                -float(value["null_margin"]))

    def rule_p_key(value: dict[str, Any]) -> tuple[Any, ...]:
        if float(value["distinct_target_recall"]) < 0.60:
            return (-1, -1.0, -float(value["inactive_false_acceptance"]),
                    -float(value["target_bag_false"]))
        return (1, float(value["target_bag_precision"]), float(value["distinct_target_recall"]),
                float(value["distinct_multi_target_exact"] or 0.0),
                -float(value["inactive_false_acceptance"]), -float(value["target_bag_false"]),
                -float(value["candidate_threshold"]), -float(value["presence_threshold"]),
                -float(value["null_margin"]))

    chosen = {"B": max(candidates, key=rule_b_key),
              "R": max(candidates, key=rule_r_key),
              "P": max(candidates, key=rule_p_key)}
    return {
        name: {
            "rule": name,
            "candidate_threshold": float(value["candidate_threshold"]),
            "presence_threshold": float(value["presence_threshold"]),
            "null_margin": float(value["null_margin"]),
            "metrics": value,
            "tie_rule": (
                "B: higher target-bag F1, lower inactive false acceptance, higher distinct recall, "
                "higher multi-target exact, fewer false bags, then lower grid thresholds" if name == "B" else
                "R: precision>=0.08, higher distinct recall, then higher precision/lower inactive/fewer false bags" if name == "R" else
                "P: distinct recall>=0.60, higher precision, then higher recall/multi-target exact/lower inactive/fewer false bags"
            ),
        }
        for name, value in chosen.items()
    }


__all__ = ["GRID", "NULL_MARGIN_GRID", "corrected_emission_mask", "fit_rule_set",
           "metric", "summary", "target_bags"]
