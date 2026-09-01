#!/usr/bin/env python3
"""Finalize the L47 B1 gate without changing the evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric(result: dict, key: str) -> float:
    return float(result[key])


def evaluate_split(result_bundle: dict, split: str) -> dict:
    teacher = result_bundle[split]["teacher"]
    student = result_bundle[split]["score"]
    top1_delta = _metric(student, "top1_frame_recall") - _metric(
        teacher, "top1_frame_recall"
    )
    recall_delta = _metric(student, "recall") - _metric(teacher, "recall")
    hard_drop = _metric(teacher, "contract_hard_violation_rate") - _metric(
        student, "contract_hard_violation_rate"
    )
    precision_delta = _metric(student, "precision") - _metric(teacher, "precision")
    fp_delta = _metric(
        student, "false_positive_candidates_per_frame"
    ) - _metric(teacher, "false_positive_candidates_per_frame")
    multi_positive_delta = _metric(student, "multi_positive_recall") - _metric(
        teacher, "multi_positive_recall"
    )
    empty_delta = _metric(student, "empty_output_rate") - _metric(
        teacher, "empty_output_rate"
    )
    null_delta = _metric(student, "null_frame_false_acceptance") - _metric(
        teacher, "null_frame_false_acceptance"
    )
    flip_rate = _metric(
        student["rank_flip"], "teacher_correct_flip_rate"
    )
    checks = {
        "top1_within_tolerance": abs(top1_delta) <= 0.02,
        "recall_within_tolerance": abs(recall_delta) <= 0.03,
        "hard_violation_drop_at_least_0.05": hard_drop >= 0.05,
        "precision_not_down_more_than_0.01": precision_delta >= -0.01,
        "fp_per_frame_not_up_more_than_0.10": fp_delta <= 0.10,
        "teacher_correct_flip_rate_at_most_0.01": flip_rate <= 0.01,
        "multi_positive_recall_not_down_more_than_0.03": multi_positive_delta
        >= -0.03,
        "empty_rate_not_increased_more_than_0.03": empty_delta <= 0.03,
    }
    return {
        "teacher": {
            "top1": _metric(teacher, "top1_frame_recall"),
            "recall": _metric(teacher, "recall"),
            "hard_violation": _metric(teacher, "contract_hard_violation_rate"),
            "precision": _metric(teacher, "precision"),
            "fp_per_frame": _metric(teacher, "false_positive_candidates_per_frame"),
            "multi_positive_recall": _metric(teacher, "multi_positive_recall"),
            "empty_rate": _metric(teacher, "empty_output_rate"),
            "null_false_acceptance": _metric(teacher, "null_frame_false_acceptance"),
        },
        "anchored_residual": {
            "top1": _metric(student, "top1_frame_recall"),
            "recall": _metric(student, "recall"),
            "hard_violation": _metric(student, "contract_hard_violation_rate"),
            "precision": _metric(student, "precision"),
            "fp_per_frame": _metric(student, "false_positive_candidates_per_frame"),
            "multi_positive_recall": _metric(student, "multi_positive_recall"),
            "empty_rate": _metric(student, "empty_output_rate"),
            "null_false_acceptance": _metric(student, "null_frame_false_acceptance"),
        },
        "deltas_student_minus_teacher": {
            "top1": top1_delta,
            "recall": recall_delta,
            "hard_violation_drop": hard_drop,
            "precision": precision_delta,
            "fp_per_frame": fp_delta,
            "multi_positive_recall": multi_positive_delta,
            "empty_rate": empty_delta,
            "null_false_acceptance": null_delta,
        },
        "rank_flip": student["rank_flip"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    payload = json.loads(source.read_text())
    gate = {
        "protocol": "L47-B1 teacher-anchored residual gate v1",
        "reference_variant": "L29 frozen teacher",
        "student_variant": "anchored residual score",
        "screening_labels_used_for_selection": False,
        "full_96_screening_authorized": False,
        "trackeval_authorized": False,
        "validation": evaluate_split(payload["results"], "validation"),
        "screening_100": evaluate_split(payload["results"], "screening_100"),
    }
    gate["passed"] = gate["validation"]["passed"] and gate["screening_100"]["passed"]
    gate["failed_requirements"] = {
        split: [name for name, ok in details["checks"].items() if not ok]
        for split, details in gate.items()
        if split in {"validation", "screening_100"}
    }
    output = {
        "format": "locatemot-l47-candidate-gate-v1",
        "stage": "L47-B1",
        "source_evaluation": str(source.resolve()),
        "source_checkpoint": payload["checkpoint"],
        "source_checkpoint_sha256": payload["checkpoint_sha256"],
        "manifest": payload["manifest"],
        "fit_calibration": payload["fit_calibration"],
        "validation_units": payload["validation_units"],
        "screening_units": payload["screening_units"],
        "screening_query_count": payload["screening_query_count"],
        "screening_label_provenance": payload["screening_label_provenance"],
        "gate": gate,
        "decision": "b1_passed" if gate["passed"] else "b1_failed_stage_stop",
        "next_authorized_action": (
            "none; write failure decomposition and propose one minimal next hypothesis"
            if not gate["passed"]
            else "full 96-query screening may proceed under frozen calibration"
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "decision": output["decision"]}))


if __name__ == "__main__":
    main()
