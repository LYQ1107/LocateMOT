#!/usr/bin/env python3
"""Evaluate Yes/No probabilities from a prepared CPU prediction JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _binary_target(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "1", "positive"}:
            return 1
        if normalized in {"no", "false", "0", "negative"}:
            return 0
        raise ValueError(f"unsupported binary target: {value!r}")
    return int(bool(value))


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    metadata = []
    for index, row in enumerate(records):
        if "p_yes" not in row:
            continue
        target = row.get("target_yes", row.get("label"))
        if target is None:
            continue
        p_yes = float(row["p_yes"])
        if not 0.0 <= p_yes <= 1.0:
            raise ValueError(f"row {index} p_yes outside [0,1]")
        pairs.append((p_yes, _binary_target(target)))
        metadata.append(row)
    if not pairs:
        return {"status": "NO_LABELED_PREDICTIONS", "count": 0, "brier": None, "accuracy": None}
    brier = sum((p - target) ** 2 for p, target in pairs) / len(pairs)
    predictions = [int(p >= 0.5) for p, _ in pairs]
    labels = [target for _, target in pairs]
    tp = sum(pred == 1 and target == 1 for pred, target in zip(predictions, labels))
    tn = sum(pred == 0 and target == 0 for pred, target in zip(predictions, labels))
    fp = sum(pred == 1 and target == 0 for pred, target in zip(predictions, labels))
    fn = sum(pred == 0 and target == 1 for pred, target in zip(predictions, labels))
    accuracy = (tp + tn) / len(pairs)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    negative_precision = tn / max(1, tn + fn)
    negative_recall = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + negative_recall)
    positives = [p for p, target in pairs if target == 1]
    negatives = [p for p, target in pairs if target == 0]
    direction = None
    if positives and negatives:
        direction = sum(pos > neg for pos in positives for neg in negatives) / (len(positives) * len(negatives))
    # Rank-based ROC/PR areas avoid requiring sklearn in the CPU preparation
    # environment.  Ties use the standard half-credit convention for ROC.
    positives = [p for p, target in pairs if target == 1]
    negatives = [p for p, target in pairs if target == 0]
    roc_auc = None
    if positives and negatives:
        roc_auc = sum(1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positives for neg in negatives) / (len(positives) * len(negatives))
    ranked = sorted(pairs, key=lambda item: item[0], reverse=True)
    total_positive = sum(target for _, target in ranked)
    pr_area = 0.0
    seen_positive = 0
    previous_recall = 0.0
    for index, (_, target) in enumerate(ranked, start=1):
        seen_positive += target
        current_recall = seen_positive / max(1, total_positive)
        current_precision = seen_positive / index
        pr_area += (current_recall - previous_recall) * current_precision
        previous_recall = current_recall
    # Ten-bin ECE is reported for calibration diagnostics, not used as a
    # training signal.
    calibration_bins = []
    ece = 0.0
    for bin_index in range(10):
        lo, hi = bin_index / 10.0, (bin_index + 1) / 10.0
        selected = [(p, target) for p, target in pairs if (lo <= p < hi) or (bin_index == 9 and p <= hi)]
        if selected:
            mean_p = sum(p for p, _ in selected) / len(selected)
            mean_y = sum(target for _, target in selected) / len(selected)
            ece += len(selected) / len(pairs) * abs(mean_p - mean_y)
            calibration_bins.append({"bin": bin_index, "count": len(selected), "mean_p": mean_p, "mean_y": mean_y})
    groups = {}
    for group_key in ("video", "hard_negative_category", "occlusion_bucket", "identity_gap_bucket"):
        grouped = {}
        for row, pair in zip(metadata, pairs):
            value = row.get(group_key)
            if value is not None:
                grouped.setdefault(str(value), []).append(pair)
        if grouped:
            groups[group_key] = {name: evaluate([{"p_yes": p, "target_yes": y} for p, y in values]) for name, values in grouped.items()}
    return {
        "status": "YESNO_EVALUATION_COMPLETE_CPU_ONLY",
        "count": len(pairs),
        "brier": brier,
        "accuracy": accuracy,
        "positive_precision": precision,
        "positive_recall": recall,
        "negative_precision": negative_precision,
        "negative_recall": negative_recall,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "roc_auc": roc_auc,
        "pr_auc": pr_area if positives else None,
        "ece": ece,
        "calibration_bins": calibration_bins,
        "groups": groups,
        "pairwise_probability_direction": direction,
        "scientific_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    report = evaluate(rows)
    report["predictions"] = str(args.predictions.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
