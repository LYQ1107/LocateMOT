#!/usr/bin/env python3
"""Freeze the corrected L88C strategy after internal dev TrackEval only."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from l88_eval_common import MANIFEST, MANIFEST_SHA, THREAD, sha256, write_json


WORK_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
RULES = ("B", "R", "P")


def _mean(values: list[float]) -> float:
    if not values:
        raise AssertionError("empty macro aggregation")
    return sum(float(value) for value in values) / len(values)


def _epoch_from_source(source: str) -> int:
    for part in Path(source).parts:
        if part.startswith("candidate_epoch"):
            return int(part[len("candidate_epoch"):])
    raise AssertionError(f"missing candidate epoch in TrackEval source: {source}")


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88C selection output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    try:
        shortlist_path = args.shortlist.resolve()
        matrix_path = args.trackeval_matrix.resolve()
        shortlist = json.loads(shortlist_path.read_text())
        matrix = json.loads(matrix_path.read_text())
        if shortlist.get("status") != "complete" or matrix.get("status") != "complete":
            raise AssertionError("selection inputs are incomplete")
        if matrix.get("scope_key") != "dev" or not matrix.get("full_video"):
            raise AssertionError("selection TrackEval must be complete internal dev full video")
        if matrix.get("screening_gt_used") or matrix.get("official_test_labels_read"):
            raise AssertionError("forbidden labels in dev TrackEval matrix")
        shortlist_rows = list(shortlist.get("shortlist", []))
        if not 1 <= len(shortlist_rows) <= 5:
            raise AssertionError(f"shortlist count drift: {len(shortlist_rows)}")
        by_epoch = {
            int(row["checkpoint_info"]["epoch"]): row for row in shortlist_rows
        }
        if len(by_epoch) != len(shortlist_rows):
            raise AssertionError("duplicate shortlist epochs")
        indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
        for row in matrix.get("results", []):
            epoch = _epoch_from_source(str(row["source"]))
            key = (epoch, str(row["rule"]), str(row["dataset"]))
            if key in indexed:
                raise AssertionError(f"duplicate TrackEval result: {key}")
            indexed[key] = row
        expected = {(epoch, rule, dataset) for epoch in by_epoch for rule in RULES for dataset in DATASETS}
        if set(indexed) != expected:
            raise AssertionError(
                f"TrackEval result contract drift; missing={sorted(expected - set(indexed))}, "
                f"extra={sorted(set(indexed) - expected)}"
            )

        candidates: list[dict[str, Any]] = []
        for epoch in sorted(by_epoch):
            candidate = by_epoch[epoch]
            for rule in RULES:
                fit = candidate.get("rule_fits", {}).get(rule)
                if not isinstance(fit, dict):
                    raise AssertionError(f"missing corrected rule fit {epoch}/{rule}")
                per_dataset = {dataset: indexed[(epoch, rule, dataset)] for dataset in DATASETS}
                macro: dict[str, float] = {}
                for name in ("HOTA___AUC", "DetA___AUC", "AssA___AUC"):
                    values = [float(per_dataset[dataset]["metrics_raw"][name]) for dataset in DATASETS]
                    if not all(value == value and abs(value) != float("inf") for value in values):
                        raise AssertionError(f"nonfinite TrackEval metric {epoch}/{rule}/{name}")
                    macro[name] = _mean(values)
                fit_metrics = fit.get("metrics", {})
                distinct_recall = float(fit_metrics.get("distinct_target_recall", 0.0))
                inactive = float(fit_metrics.get("inactive_false_acceptance", 1.0))
                if not (0 <= distinct_recall <= 1 and 0 <= inactive <= 1):
                    raise AssertionError(f"invalid fit metric {epoch}/{rule}")
                selection_tuple = (
                    -macro["HOTA___AUC"], -macro["DetA___AUC"], -macro["AssA___AUC"],
                    -distinct_recall, inactive, epoch,
                )
                candidates.append({
                    "epoch": epoch, "rule": rule,
                    "candidate_name": candidate.get("candidate_name"),
                    "checkpoint_info": candidate["checkpoint_info"], "rule_fit": fit,
                    "trackeval_by_dataset": per_dataset, "trackeval_macro": macro,
                    "distinct_target_recall": distinct_recall,
                    "inactive_false_acceptance": inactive,
                    "selection_tuple": list(selection_tuple),
                })
        candidates.sort(key=lambda row: tuple(row["selection_tuple"]))
        best_by_epoch = [
            min((row for row in candidates if row["epoch"] == epoch),
                key=lambda value: tuple(value["selection_tuple"]))
            for epoch in sorted(by_epoch)
        ]
        final = min(best_by_epoch, key=lambda row: tuple(row["selection_tuple"]))
        payload = {
            "format": "locatemot-l88c-dev-checkpoint-selection-v1", "status": "complete",
            "evidence_type": "corrected candidate-vs-NULL internal dev TrackEval selection only",
            "selection_rule": ["higher_dev_hota_macro", "higher_dev_deta_macro",
                               "higher_dev_assa_macro", "higher_distinct_target_recall",
                               "lower_inactive_false_acceptance", "earlier_epoch"],
            "macro_definition": "unweighted arithmetic mean of V1/V2 dev TrackEval AUC values",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "shortlist": str(shortlist_path), "shortlist_sha256": sha256(shortlist_path),
            "trackeval_matrix": str(matrix_path), "trackeval_matrix_sha256": sha256(matrix_path),
            "candidate_rule_records": candidates, "best_rule_per_epoch": best_by_epoch,
            "final_selection": final,
            "selection_frozen_before_fixed_validation": True,
            "fixed_validation_read": False, "zero_training": True,
            "corrected_candidate_vs_null": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
            "no_hota_or_trackeval": False, "candidate_deletion": False,
            "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "failure_root_cause": None,
            "next_action": "run corrected fixed 16-calibration/24-validation semantic replay",
        }
        write_json(out / "checkpoint_selection.json", payload)
        write_json(out / "selection.json", {
            "format": payload["format"], "status": "complete",
            "final_selection": final, "best_rule_per_epoch": best_by_epoch,
            "selection_frozen_before_fixed_validation": True,
            "fixed_validation_read": False, "zero_training": True,
            "corrected_candidate_vs_null": True, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": True, "no_hota_or_trackeval": False,
        })
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-dev-selection-status-v1", "status": "complete",
            "shortlist_count": len(shortlist_rows), "rule_count": len(candidates),
            "final_epoch": int(final["epoch"]), "final_rule": str(final["rule"]),
            "fixed_validation_read": False, "zero_training": True,
            "corrected_candidate_vs_null": True, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": True, "no_hota_or_trackeval": False,
        })
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88C checkpoint selection — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-dev-selection-status-v1", "status": "incomplete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "zero_training": True, "corrected_candidate_vs_null": True,
            "failure_root_cause": "first traceback in INCOMPLETE.md",
            "fixed_validation_read": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--trackeval-matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
