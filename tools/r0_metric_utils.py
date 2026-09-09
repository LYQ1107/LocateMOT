#!/usr/bin/env python3
"""Small streaming metrics for the fixed R0 zero-logit dev rule.

This is deliberately separate from model code.  It consumes complete score
records and never changes the candidate set.  The deployment rule registered
for R0 is ``presence_logit >= 0`` and ``membership_logit >= 0``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if float(denominator) else 0.0


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


class ZeroLogitMetrics:
    """Streaming row and target-bag metrics for one scope."""

    def __init__(self, *, stratify: bool = True) -> None:
        self.stratify = bool(stratify)
        self.rows = 0
        self.candidate_rows = 0
        self.row_tp = self.row_fp = self.row_fn = self.row_selected = self.row_positive = 0
        self.row_target_units = self.row_top1 = self.row_top5 = self.row_empty = 0
        self.row_hard_total = self.row_hard_bad = 0
        self.row_margins: list[float] = []
        self.row_best: list[float] = []
        self.row_average: list[float] = []
        self.multi_recall: list[float] = []
        self.multi_exact: list[float] = []
        self.bag_tp = self.bag_fp = self.bag_fn = self.bag_selected = self.bag_positive = 0
        self.bag_hit1 = self.bag_hit5 = self.bag_query_total = 0
        self.bag_hard_total = self.bag_hard_bad = 0
        self.bag_margins: list[float] = []
        self.distinct_hit = self.distinct_total = 0
        self.distinct_multi_exact: list[float] = []
        self.inactive_units = self.inactive_accept = self.inactive_fp_rows = 0
        self.present_uncovered_units = self.candidate_present_units = 0
        self.score_count = 0
        self.score_sum = self.score_sq = 0.0
        self.score_min = math.inf
        self.score_max = -math.inf
        self.by_dataset: dict[str, ZeroLogitMetrics] = {} if stratify else {}
        self.by_category: dict[str, ZeroLogitMetrics] = {} if stratify else {}

    def add(self, row: dict[str, Any]) -> None:
        scores = [float(value) for value in row["score"]]
        labels = [bool(value) for value in row["labels"]]
        candidate_gt = [None if value is None else str(value) for value in row["candidate_gt"]]
        if len(scores) != len(labels) or len(scores) != len(candidate_gt):
            raise AssertionError(f"R0 metric row length drift: {row.get('unit_key')}")
        if not all(math.isfinite(value) for value in scores):
            raise FloatingPointError(f"R0 nonfinite score: {row.get('unit_key')}")
        if len(row.get("row_keys", [])) != len(scores) or int(row["candidate_count"]) != len(scores):
            raise AssertionError(f"R0 metric candidate/key drift: {row.get('unit_key')}")
        presence = float(row["presence_logit"])
        if not math.isfinite(presence):
            raise FloatingPointError(f"R0 nonfinite presence: {row.get('unit_key')}")
        unit_gate = presence >= 0.0
        selected = [(value >= 0.0 and unit_gate) for value in scores]
        self.rows += 1
        self.candidate_rows += len(scores)
        self.row_tp += sum(int(a and b) for a, b in zip(selected, labels))
        self.row_fp += sum(int(a and not b) for a, b in zip(selected, labels))
        self.row_fn += sum(int((not a) and b) for a, b in zip(selected, labels))
        self.row_selected += sum(selected)
        self.row_positive += sum(labels)
        self.row_empty += int(not any(selected))
        positive_indices = [index for index, value in enumerate(labels) if value]
        negative_indices = [index for index, value in enumerate(labels) if not value]
        if positive_indices:
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            self.row_target_units += 1
            self.row_top1 += int(any(labels[index] for index in order[:1]))
            self.row_top5 += int(any(labels[index] for index in order[:5]))
        if positive_indices and negative_indices:
            minimum = min(scores[index] for index in positive_indices)
            maximum = max(scores[index] for index in negative_indices)
            self.row_margins.append(minimum - maximum)
            self.row_best.append(max(scores[index] for index in positive_indices) - maximum)
            self.row_average.append(sum(scores[index] for index in positive_indices) / len(positive_indices) - maximum)
            self.row_hard_total += 1
            self.row_hard_bad += int(maximum >= minimum)
        if len(positive_indices) > 1:
            hit = sum(selected[index] for index in positive_indices) / len(positive_indices)
            self.multi_recall.append(float(hit))
            self.multi_exact.append(float(all(selected[index] for index in positive_indices)))

        target_ids = {str(value) for value in row.get("target_ids", [])}
        groups: dict[str, list[int]] = defaultdict(list)
        backgrounds: list[int] = []
        for index, value in enumerate(candidate_gt):
            if value is None:
                backgrounds.append(index)
            else:
                groups[value].append(index)
        bags: list[tuple[float, bool, bool]] = []
        for target in sorted(groups):
            indexes = groups[target]
            value = max(scores[index] for index in indexes)
            bags.append((value, target in target_ids, bool(unit_gate and value >= 0.0)))
        for index in backgrounds:
            bags.append((scores[index], False, bool(unit_gate and scores[index] >= 0.0)))
        positive_bags = [bag for bag in bags if bag[1]]
        selected_bags = [bag for bag in bags if bag[2]]
        self.bag_positive += len(positive_bags)
        self.bag_selected += len(selected_bags)
        self.bag_tp += sum(int(value[1] and value[2]) for value in bags)
        self.bag_fp += sum(int((not value[1]) and value[2]) for value in bags)
        self.bag_fn += sum(int(value[1] and (not value[2])) for value in bags)
        if positive_bags:
            order = sorted(range(len(bags)), key=lambda index: (-bags[index][0], index))
            self.bag_query_total += 1
            self.bag_hit1 += int(any(bags[index][1] for index in order[:1]))
            self.bag_hit5 += int(any(bags[index][1] for index in order[:5]))
        if len(positive_bags) and any(not value[1] for value in bags):
            minimum = min(value[0] for value in positive_bags)
            maximum = max(value[0] for value in bags if not value[1])
            self.bag_margins.append(minimum - maximum)
            self.bag_hard_total += 1
            self.bag_hard_bad += int(maximum >= minimum)
        self.distinct_total += len(positive_bags)
        self.distinct_hit += sum(int(value[1] and value[2]) for value in bags)
        if len(positive_bags) > 1:
            self.distinct_multi_exact.append(float(all(value[2] for value in positive_bags)))

        category = str(row.get("category", "unknown"))
        if category == "inactive":
            self.inactive_units += 1
            self.inactive_accept += int(any(selected))
            self.inactive_fp_rows += sum(selected)
        if category == "present_uncovered":
            self.present_uncovered_units += 1
        if bool(row.get("candidate_present", False)):
            self.candidate_present_units += 1
        for value in scores:
            self.score_count += 1
            self.score_sum += value
            self.score_sq += value * value
            self.score_min = min(self.score_min, value)
            self.score_max = max(self.score_max, value)

        if self.stratify:
            dataset = str(row.get("dataset", "unknown"))
            self.by_dataset.setdefault(dataset, ZeroLogitMetrics(stratify=False)).add(row)
            self.by_category.setdefault(category, ZeroLogitMetrics(stratify=False)).add(row)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "min": None, "p50": None, "max": None}
        ordered = sorted(values)
        return {"count": len(values), "mean": float(sum(values) / len(values)),
                "min": float(ordered[0]), "p50": float(ordered[(len(ordered) - 1) // 2]),
                "max": float(ordered[-1])}

    def finish(self) -> dict[str, Any]:
        variance = self.score_sq / self.score_count - (self.score_sum / self.score_count) ** 2 if self.score_count else 0.0
        result: dict[str, Any] = {
            "units": self.rows, "candidate_rows": self.candidate_rows,
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "finite_scores": True, "rule": {"membership_threshold": 0.0, "presence_threshold": 0.0,
                                               "null_logit": 0.0, "null_margin": 0.0,
                                               "description": "membership_logit >= 0 and coverage_presence_logit >= 0"},
            "legacy_candidate_precision": _ratio(self.row_tp, self.row_selected),
            "legacy_candidate_recall": _ratio(self.row_tp, self.row_tp + self.row_fn),
            "legacy_fp_per_frame": _ratio(self.row_fp, self.rows),
            "legacy_predictions_per_positive": _ratio(self.row_selected, self.row_positive),
            "legacy_top1": _ratio(self.row_top1, self.row_target_units),
            "legacy_top5": _ratio(self.row_top5, self.row_target_units),
            "legacy_row_hard_violation": _ratio(self.row_hard_bad, self.row_hard_total),
            "legacy_row_hard_total": self.row_hard_total, "legacy_row_hard_bad": self.row_hard_bad,
            "legacy_row_strict_margin": self._summary(self.row_margins),
            "legacy_row_best_margin": self._summary(self.row_best),
            "legacy_row_average_margin": self._summary(self.row_average),
            "legacy_row_multi_positive_recall": _mean(self.multi_recall),
            "legacy_row_multi_target_exact": _mean(self.multi_exact),
            "multi_positive_units": len(self.multi_recall),
            "empty_rate": _ratio(self.row_empty, self.rows),
            "target_bag_precision": _ratio(self.bag_tp, self.bag_selected),
            "target_bag_recall": _ratio(self.bag_tp, self.bag_tp + self.bag_fn),
            "target_bag_f1": _ratio(2 * self.bag_tp, 2 * self.bag_tp + self.bag_fp + self.bag_fn),
            "target_bag_hard_violation": _ratio(self.bag_hard_bad, self.bag_hard_total),
            "target_bag_hard_total": self.bag_hard_total, "target_bag_hard_bad": self.bag_hard_bad,
            "target_bag_margin": self._summary(self.bag_margins),
            "target_bag_hit1": _ratio(self.bag_hit1, self.bag_query_total),
            "target_bag_hit5": _ratio(self.bag_hit5, self.bag_query_total),
            "target_bag_query_total": self.bag_query_total,
            "distinct_target_recall": _ratio(self.distinct_hit, self.distinct_total),
            "distinct_target_hit": self.distinct_hit, "distinct_target_total": self.distinct_total,
            "distinct_multi_target_exact": _mean(self.distinct_multi_exact),
            "distinct_multi_target_units": len(self.distinct_multi_exact),
            "inactive_units": self.inactive_units,
            "inactive_false_acceptance": _ratio(self.inactive_accept, self.inactive_units),
            "inactive_false_positive_rows": self.inactive_fp_rows,
            "present_uncovered_units": self.present_uncovered_units,
            "candidate_present_units": self.candidate_present_units,
            "score_distribution": {"count": self.score_count,
                                    "mean": self.score_sum / self.score_count if self.score_count else None,
                                    "std": math.sqrt(max(0.0, variance)) if self.score_count else None,
                                    "min": None if not self.score_count else self.score_min,
                                    "max": None if not self.score_count else self.score_max},
        }
        if self.stratify:
            result["per_dataset"] = {key: value.finish() for key, value in sorted(self.by_dataset.items())}
            result["per_category"] = {key: value.finish() for key, value in sorted(self.by_category.items())}
        return result


def metric_from_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    accumulator = ZeroLogitMetrics()
    for row in records:
        accumulator.add(row)
    return accumulator.finish()


__all__ = ["ZeroLogitMetrics", "metric_from_records"]
