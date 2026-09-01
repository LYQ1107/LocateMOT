"""Summarize per-query TrackEval metrics for the formal L16 RMOT runs.

The official TrackEval COMBINED row is the primary pooled result.  This tool
adds query-macro means, percentile bootstrap intervals over complete query
units, and overlapping language-family summaries for uncertainty and
generalization analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import FAMILY_NAMES, expression_family_vector


METRICS = (
    "HOTA___AUC", "DetA___AUC", "AssA___AUC", "DetRe___AUC",
    "DetPr___AUC", "AssRe___AUC", "AssPr___AUC", "LocA___AUC", "IDF1",
)


def family_names(seq: str) -> list[str]:
    expression = seq.split("+", 1)[1] if "+" in seq else seq
    flags = expression_family_vector(expression).tolist()
    return [name for name, flag in zip(FAMILY_NAMES, flags) if flag > 0.5]


def bootstrap(values: np.ndarray, rng: np.random.Generator,
              n_boot: int) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[indices].mean(axis=1)
    return mean, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    csv_path = Path(args.eval_root) / "uidm16" / "pedestrian_detailed.csv"
    with csv_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("seq") != "COMBINED"]
    if not rows:
        raise RuntimeError(f"no TrackEval query rows in {csv_path}")

    values = {}
    for metric in METRICS:
        values[metric] = np.asarray([float(row[metric]) for row in rows], np.float64)

    rng = np.random.default_rng(args.seed)
    macro = {metric: float(np.mean(value) * 100.0)
             for metric, value in values.items()}
    ci = {}
    for metric, value in values.items():
        mean, low, high = bootstrap(value, rng, args.n_boot)
        ci[metric] = {
            "mean_percent": mean * 100.0,
            "low_percent": low * 100.0,
            "high_percent": high * 100.0,
        }

    family_rows = {}
    for family in FAMILY_NAMES:
        selected = [row for row in rows if family in family_names(row["seq"])]
        family_rows[family] = {
            "queries": len(selected),
            "metrics_percent": {
                metric: float(np.mean([float(row[metric]) for row in selected]) * 100.0)
                if selected else None
                for metric in METRICS
            },
        }

    payload = {
        "dataset": args.dataset,
        "source_csv": str(csv_path),
        "query_count": len(rows),
        "metrics": list(METRICS),
        "query_macro_percent": macro,
        "bootstrap_95_percent": ci,
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed,
                       "unit": "complete query", "statistic": "query macro mean"},
        "overlapping_family_summary_percent": family_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "dataset": args.dataset,
        "queries": len(rows),
        "query_macro_percent": macro,
        "bootstrap_95_percent": ci,
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
