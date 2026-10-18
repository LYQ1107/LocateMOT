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


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for index, row in enumerate(records):
        if "p_yes" not in row:
            continue
        target = row.get("target_yes", row.get("label"))
        if target is None:
            continue
        p_yes = float(row["p_yes"])
        if not 0.0 <= p_yes <= 1.0:
            raise ValueError(f"row {index} p_yes outside [0,1]")
        pairs.append((p_yes, int(bool(target))))
    if not pairs:
        return {"status": "NO_LABELED_PREDICTIONS", "count": 0, "brier": None, "accuracy": None}
    brier = sum((p - target) ** 2 for p, target in pairs) / len(pairs)
    accuracy = sum((p >= 0.5) == bool(target) for p, target in pairs) / len(pairs)
    positives = [p for p, target in pairs if target == 1]
    negatives = [p for p, target in pairs if target == 0]
    direction = None
    if positives and negatives:
        direction = sum(pos > neg for pos in positives for neg in negatives) / (len(positives) * len(negatives))
    return {
        "status": "YESNO_EVALUATION_COMPLETE_CPU_ONLY",
        "count": len(pairs),
        "brier": brier,
        "accuracy": accuracy,
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
