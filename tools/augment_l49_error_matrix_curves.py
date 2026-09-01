#!/usr/bin/env python3
"""Add saved training and validation curves to the frozen L49 error matrix.

The fit/validation/test matrix is already complete.  This small CPU-only
augmentation does not read labels or scores again and cannot change the
selected checkpoint, thresholds, or test results.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
MATRIX = ROOT / "outputs/l49/val/error_matrix.json"
TRACE = ROOT / "outputs/l49/train/joint_long5000/loss_trace.json"
VALIDATION = ROOT / "outputs/l49/val/validation_metrics.json"


LOSS_KEYS = (
    "total", "membership", "pairwise", "listwise_all_positive",
    "min_positive", "continuation", "null", "sequence_consistency",
    "calibration_brier", "inactive", "teacher_distillation",
    "gradient_norm",
)
METRIC_KEYS = (
    "top1_frame_recall", "top5_frame_recall", "recall", "precision",
    "false_positive_candidates_per_frame", "hard_violation_rate",
    "strict_min_positive_margin", "best_positive_margin",
    "average_positive_margin", "multi_positive_recall", "empty_output_rate",
    "null_frame_false_acceptance", "predictions_per_positive",
)


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def training_curve(trace, step):
    rows = trace[:step]
    window = rows[-min(100, len(rows)):]
    result = {key: mean([row.get(key) for row in window]) for key in LOSS_KEYS}
    result["step"] = int(step)
    result["window_steps"] = len(window)
    result["trace_rows"] = len(rows)
    result["all_finite"] = all(
        value is not None and value == value and abs(float(value)) != float("inf")
        for row in window for key, value in row.items()
        if isinstance(value, (int, float))
    )
    return result


def metric_slice(record):
    result = {}
    for key in METRIC_KEYS:
        value = record.get(key)
        if isinstance(value, dict):
            value = value.get("mean")
        result[key] = value
    return result


def validation_curve(results, step):
    record = results[str(step)]
    domains = {}
    for domain in ("refer_kitti_v1", "refer_kitti_v2"):
        item = record["per_domain"][domain]
        domains[domain] = {
            "threshold": item["l49_final"]["threshold"],
            "baseline": metric_slice(item["baseline"]),
            "l49_final": metric_slice(item["l49_final"]),
            "delta": item["delta"],
        }
    return {
        "step": int(step),
        "validation_macro_f1": record["validation_macro_f1"],
        "domains": domains,
        "validation_loss": None,
        "validation_loss_note": (
            "The frozen validation evaluator is inference-only and did not write "
            "a differentiable loss trace; candidate margins and hard-violation "
            "curves are recorded here instead."
        ),
    }


def atomic_write(path: Path, payload):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    matrix = json.loads(MATRIX.read_text())
    trace = json.loads(TRACE.read_text())
    validation = json.loads(VALIDATION.read_text())
    steps = [100, 250, 500, 1000, 2500, 5000]
    missing = [step for step in steps
               if str(step) not in validation["checkpoint_results"]]
    if missing:
        raise KeyError(f"validation metrics missing steps: {missing}")
    if len(trace) < max(steps):
        raise ValueError(f"loss trace has {len(trace)} rows, expected {max(steps)}")
    matrix["training_validation_curves"] = {
        "source": {
            "loss_trace": str(TRACE.resolve()),
            "validation_metrics": str(VALIDATION.resolve()),
            "official_test_labels_used": False,
            "selected_checkpoint_or_threshold_changed": False,
        },
        "training_loss_windows": [training_curve(trace, step) for step in steps],
        "validation_checkpoint_curves": [
            validation_curve(validation["checkpoint_results"], step)
            for step in steps
        ],
    }
    matrix["provenance"]["training_validation_curves_augmented"] = True
    atomic_write(MATRIX, matrix)
    print(json.dumps({
        "matrix": str(MATRIX.resolve()),
        "status": matrix["matrix_status"],
        "steps": steps,
        "official_test_labels_used": False,
        "selected_checkpoint_or_threshold_changed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
