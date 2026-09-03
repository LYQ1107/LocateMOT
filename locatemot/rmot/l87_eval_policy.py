"""Byte-identical L87 target-bag selection and candidate-vs-NULL policy.

This module is intentionally shared by the isolated A/B worktrees.  It is
the only place defining the corrected deployment mask and the corrected
unique-target-bag metrics.  Candidate rows remain intact; only emission
metrics use the frozen rule.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


GRID = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
NULL_GRID = (0.0, 0.25, 0.5, 0.75)


def emission_mask(
    candidate_energy: Iterable[float],
    presence_logit: float,
    null_logit: float,
    candidate_threshold: float,
    presence_threshold: float,
    null_margin: float,
) -> np.ndarray:
    """Correct candidate-vs-NULL emission rule, with no row deletion."""
    energy = np.asarray(list(candidate_energy), dtype=np.float64)
    if not np.isfinite(energy).all() or not np.isfinite([presence_logit, null_logit]).all():
        raise FloatingPointError("nonfinite L87 emission inputs")
    return (
        (float(presence_logit) >= float(presence_threshold))
        & (energy >= float(candidate_threshold))
        & ((energy - float(null_logit)) >= float(null_margin))
    )


def target_bags(
    score: Iterable[float], candidate_gt: Iterable[object | None], target_ids: Iterable[object]
) -> tuple[list[dict[str, Any]], set[str]]:
    values = np.asarray(list(score), dtype=np.float64)
    gt = [None if value is None else str(value) for value in candidate_gt]
    if values.ndim != 1 or len(gt) != values.size:
        raise ValueError("L87 score/candidate_gt length mismatch")
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite L87 score")
    referred = {str(value) for value in target_ids}
    rows: dict[str, list[int]] = {}
    background: list[int] = []
    for index, value in enumerate(gt):
        if value is None:
            background.append(index)
        else:
            rows.setdefault(value, []).append(index)
    result: list[dict[str, Any]] = []
    for target in sorted(rows):
        indexes = rows[target]
        result.append({"kind": "target", "id": target, "rows": indexes,
                       "score": float(values[indexes].max()), "referred": target in referred})
    for index in background:
        result.append({"kind": "background", "id": str(index), "rows": [index],
                       "score": float(values[index]), "referred": False})
    return result, referred


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "mean": float(array.mean()),
            "p50": float(np.quantile(array, 0.5)), "p90": float(np.quantile(array, 0.9))}


def metric(
    records: list[dict[str, Any]],
    candidate_threshold: float,
    presence_threshold: float,
    null_margin: float,
) -> dict[str, Any]:
    """Compute legacy row diagnostics and corrected unique target-bag metrics."""
    if not records:
        raise ValueError("empty L87 records")
    selected_rows = positive_rows = true_positive_rows = false_positive_rows = 0
    false_negative_rows = 0
    candidate_scores: list[float] = []
    inactive_units = inactive_accept = inactive_fp = 0
    present_uncovered_units = 0
    empty = 0
    target_total = target_selected = 0
    target_predicted = target_false = 0
    hard_values: list[bool] = []
    hard_margins: list[float] = []
    bag_hit1 = bag_hit5 = bag_units = 0
    multi_total = multi_selected = multi_exact = 0
    legacy_top1 = legacy_top5 = legacy_top_units = 0
    legacy_strict: list[float] = []
    legacy_best: list[float] = []
    legacy_average: list[float] = []
    for row in records:
        score = np.asarray(row["score"], dtype=np.float64)
        labels = np.asarray(row.get("labels", [False] * len(score)), dtype=bool)
        gt = list(row.get("candidate_gt", [None] * len(score)))
        targets = [str(value) for value in row.get("target_ids", [])]
        if score.size != int(row["candidate_count"]) or labels.size != score.size or len(gt) != score.size:
            raise AssertionError(f"L87 metric row length drift: {row.get('unit_key')}")
        if not np.isfinite(score).all():
            raise FloatingPointError(f"nonfinite L87 metric row: {row.get('unit_key')}")
        selected = emission_mask(score, float(row["presence_logit"]), float(row["null_logit"]),
                                 candidate_threshold, presence_threshold, null_margin)
        selected_rows += int(selected.sum()); positive_rows += int(labels.sum())
        true_positive_rows += int((selected & labels).sum())
        false_positive_rows += int((selected & ~labels).sum())
        false_negative_rows += int((~selected & labels).sum())
        candidate_scores.extend(score.tolist())
        if not bool(selected.any()):
            empty += 1
        category = str(row.get("category", "unknown"))
        if category == "inactive":
            inactive_units += 1; inactive_accept += int(bool(selected.any())); inactive_fp += int((selected & ~labels).sum())
        if category == "present_uncovered":
            present_uncovered_units += 1
        bags, referred = target_bags(score, gt, targets)
        by_id = {str(item["id"]): item for item in bags if item["kind"] == "target"}
        if targets and category != "present_uncovered":
            target_total += len(referred)
            selected_for_target: set[str] = set()
            for target in sorted(referred):
                item = by_id.get(target)
                if item is not None:
                    target_total += 0
                    if bool(selected[item["rows"]].any()):
                        selected_for_target.add(target)
            target_selected += len(selected_for_target)
            target_false += sum(int(bool(selected[item["rows"]].any())) for item in bags
                                if not item["referred"] and item["kind"] == "target")
            target_false += sum(int(bool(selected[item["rows"]].any())) for item in bags
                                if item["kind"] == "background")
            positives = [float(by_id[target]["score"]) for target in sorted(referred) if target in by_id]
            negatives = [float(item["score"]) for item in bags if not item["referred"]]
            if positives:
                bag_units += 1
                ordered = sorted(bags, key=lambda item: (-float(item["score"]), item["kind"], str(item["id"])))
                bag_hit1 += int(bool(ordered) and bool(ordered[0]["referred"]))
                bag_hit5 += int(any(bool(item["referred"]) for item in ordered[:5]))
                if negatives:
                    margin = float(min(positives) - max(negatives))
                    hard_margins.append(margin); hard_values.append(bool(margin < 0.0))
            if len(referred) > 1:
                multi_total += len(referred)
                multi_selected += len(selected_for_target)
                multi_exact += int(all(target in selected_for_target for target in referred))
        order = np.argsort(-score, kind="stable")
        if labels.any():
            legacy_top_units += 1
            legacy_top1 += int(bool(labels[order[:1]].any())); legacy_top5 += int(bool(labels[order[:5]].any()))
            pos = score[labels]; neg = score[~labels]
            if neg.size:
                legacy_strict.append(float(pos.min() - neg.max())); legacy_best.append(float(pos.max() - neg.max()))
                legacy_average.append(float(pos.mean() - neg.max()))
    target_recall = target_selected / max(1, target_total)
    target_precision = target_selected / max(1, target_selected + target_false)
    target_f1 = 2.0 * target_precision * target_recall / max(1e-12, target_precision + target_recall)
    legacy_precision = true_positive_rows / max(1, selected_rows)
    legacy_recall = true_positive_rows / max(1, true_positive_rows + false_negative_rows)
    return {
        "units": len(records), "candidate_rows": int(sum(int(row["candidate_count"]) for row in records)),
        "positive_rows": positive_rows, "selected_rows": selected_rows,
        "true_positive_rows": true_positive_rows, "false_positive_rows": false_positive_rows,
        "false_negative_rows": false_negative_rows, "candidate_precision": float(legacy_precision),
        "candidate_recall": float(legacy_recall), "fp_per_frame": float(false_positive_rows / max(1, len(records))),
        "predictions_per_positive": float(selected_rows / max(1, positive_rows)),
        "top1": float(legacy_top1 / max(1, legacy_top_units)), "top5": float(legacy_top5 / max(1, legacy_top_units)),
        "strict_margin": _distribution(legacy_strict), "best_margin": _distribution(legacy_best),
        "average_margin": _distribution(legacy_average),
        "hard_violation": float(np.mean(hard_values)) if hard_values else 1.0,
        "target_bag_hard_violation": float(np.mean(hard_values)) if hard_values else 1.0,
        "target_bag_strict_margin": _distribution(hard_margins), "target_bag_hit1": float(bag_hit1 / max(1, bag_units)),
        "target_bag_hit5": float(bag_hit5 / max(1, bag_units)), "target_bag_units": bag_units,
        "distinct_target_recall": float(target_recall), "target_bag_precision": float(target_precision),
        "target_bag_f1": float(target_f1), "target_bag_total": int(target_total),
        "target_bag_selected": int(target_selected), "target_bag_false": int(target_false),
        "multi_positive_recall": float(multi_selected / max(1, multi_total)) if multi_total else None,
        "minimum_positive_coverage": float(multi_selected / max(1, multi_total)) if multi_total else None,
        "multi_target_exact": float(multi_exact / max(1, sum(1 for row in records if len(row.get("target_ids", [])) > 1 and row.get("category") != "present_uncovered"))) if any(len(row.get("target_ids", [])) > 1 and row.get("category") != "present_uncovered" for row in records) else None,
        "empty_rate": float(empty / max(1, len(records))), "inactive_units": inactive_units,
        "inactive_false_acceptance": float(inactive_accept / max(1, inactive_units)),
        "inactive_false_positive_rows": inactive_fp, "present_uncovered_units": present_uncovered_units,
        "score_distribution": _distribution(candidate_scores), "candidate_threshold": float(candidate_threshold),
        "presence_threshold": float(presence_threshold), "null_margin": float(null_margin),
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
    }


def fit_rules(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates = [metric(records, candidate, presence, margin)
                  for candidate in GRID for presence in GRID for margin in NULL_GRID]
    def with_rule(name: str, value: dict[str, Any]) -> dict[str, Any]:
        return {"rule": name, "candidate_threshold": value["candidate_threshold"],
                "presence_threshold": value["presence_threshold"], "null_margin": value["null_margin"],
                "metrics": value}
    b = max(candidates, key=lambda x: (float(x["target_bag_f1"]), -float(x["inactive_false_acceptance"]),
                                       float(x["distinct_target_recall"]), float(x["multi_target_exact"] or 0.0),
                                       -float(x["target_bag_false"]), float(x["candidate_threshold"])))
    eligible_r = [x for x in candidates if float(x["target_bag_precision"]) >= 0.08]
    r = max(eligible_r or candidates, key=lambda x: (float(x["distinct_target_recall"]),
                                                     -float(x["inactive_false_acceptance"]),
                                                     float(x["multi_target_exact"] or 0.0)))
    eligible_p = [x for x in candidates if float(x["distinct_target_recall"]) >= 0.60]
    p = max(eligible_p or candidates, key=lambda x: (float(x["target_bag_precision"]),
                                                     float(x["multi_target_exact"] or 0.0),
                                                     -float(x["inactive_false_acceptance"])))
    return {"B": with_rule("B", b), "R": with_rule("R", r), "P": with_rule("P", p)}


def checkpoint_selection_key(value: dict[str, Any], epoch: int) -> list[float | int]:
    """Exact L87 six-field checkpoint tuple."""
    return [float(value["target_bag_hard_violation"]), -float(value["target_bag_hit1"]),
            -float(value["multi_target_exact"] or 0.0), float(value["inactive_false_acceptance"]),
            -float(value["distinct_target_recall"]), int(epoch)]


def contract_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "unique_unit_keys": len({str(row["unit_key"]) for row in records}),
        "candidate_rows_retained": all(bool(row.get("candidate_rows_retained", True)) for row in records),
        "candidate_deletion": any(bool(row.get("candidate_deletion", False)) for row in records),
        "candidate_truncation": any(bool(row.get("candidate_truncation", False)) for row in records),
        "finite_scores": all(np.isfinite(np.asarray(row["score"], dtype=np.float64)).all() for row in records),
    }


__all__ = ["GRID", "NULL_GRID", "emission_mask", "target_bags", "metric", "fit_rules",
           "checkpoint_selection_key", "contract_summary"]
