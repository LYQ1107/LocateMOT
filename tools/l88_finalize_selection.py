#!/usr/bin/env python3
"""Freeze the L88 checkpoint/rule using only internal fit/dev TrackEval.

The selection is deliberately separate from TrackEval execution and from the
fixed 16-calibration/24-validation evaluation.  It consumes only the
pre-registered shortlist/rule fits and completed internal dev TrackEval rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any


WORK_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l19/protocol/kitti_fast_eval_manifest.json").resolve()
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
RULES = ("B", "R", "P")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def mean(values: list[float]) -> float:
    if not values:
        raise AssertionError("empty metric aggregation")
    return sum(float(value) for value in values) / len(values)


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 selection output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    try:
        shortlist_path = args.shortlist.resolve()
        matrix_path = args.trackeval_matrix.resolve()
        shortlist = json.loads(shortlist_path.read_text())
        matrix = json.loads(matrix_path.read_text())
        if shortlist.get("status") != "complete" or matrix.get("status") != "complete":
            raise AssertionError("selection inputs are not complete")
        if matrix.get("screening_gt_used") or matrix.get("official_test_labels_read"):
            raise AssertionError("forbidden labels present in TrackEval matrix")
        shortlist_rows = shortlist.get("shortlist", [])
        if len(shortlist_rows) != 5:
            raise AssertionError(f"shortlist count drift: {len(shortlist_rows)}")
        by_epoch: dict[int, dict[str, Any]] = {}
        for row in shortlist_rows:
            epoch = int(row["checkpoint_info"]["epoch"])
            if epoch in by_epoch:
                raise AssertionError(f"duplicate shortlist epoch: {epoch}")
            by_epoch[epoch] = row
        results = matrix.get("results", [])
        indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
        for row in results:
            source = Path(str(row["source"])).resolve()
            parts = source.parts
            try:
                epoch = int(next(value[len("candidate_epoch"):]
                                 for value in parts if value.startswith("candidate_epoch")))
            except StopIteration as exc:
                raise AssertionError(f"TrackEval source has no candidate epoch: {source}") from exc
            key = (epoch, str(row["rule"]), str(row["dataset"]))
            if key in indexed:
                raise AssertionError(f"duplicate TrackEval result: {key}")
            indexed[key] = row
        expected = {(epoch, rule, dataset) for epoch in by_epoch for rule in RULES for dataset in DATASETS}
        if set(indexed) != expected:
            missing = sorted(expected - set(indexed)); extra = sorted(set(indexed) - expected)
            raise AssertionError(f"TrackEval result contract drift; missing={missing}, extra={extra}")

        candidate_rules: list[dict[str, Any]] = []
        for epoch in sorted(by_epoch):
            candidate = by_epoch[epoch]
            fit_rules = candidate.get("rule_fits", {})
            for rule in RULES:
                fit = fit_rules.get(rule)
                if not isinstance(fit, dict):
                    raise AssertionError(f"missing rule fit {epoch}/{rule}")
                per_dataset: dict[str, Any] = {}
                for dataset in DATASETS:
                    row = indexed[(epoch, rule, dataset)]
                    metrics = row.get("metrics_raw", {})
                    for name in ("HOTA___AUC", "DetA___AUC", "AssA___AUC"):
                        value = float(metrics[name])
                        if not (value == value) or abs(value) == float("inf"):
                            raise AssertionError(f"nonfinite TrackEval metric {epoch}/{rule}/{dataset}/{name}")
                    per_dataset[dataset] = row
                macro = {
                    name: mean([float(per_dataset[dataset]["metrics_raw"][name]) for dataset in DATASETS])
                    for name in ("HOTA___AUC", "DetA___AUC", "AssA___AUC")
                }
                dev_metrics = fit.get("metrics", {})
                distinct_recall = float(dev_metrics.get("distinct_target_recall", 0.0))
                inactive_fa = float(dev_metrics.get("inactive_false_acceptance", 1.0))
                if not (0.0 <= distinct_recall <= 1.0 and 0.0 <= inactive_fa <= 1.0):
                    raise AssertionError(f"invalid dev rule metrics {epoch}/{rule}")
                candidate_rules.append({
                    "epoch": epoch, "rule": rule, "checkpoint_info": candidate["checkpoint_info"],
                    "rule_fit": fit, "trackeval_by_dataset": per_dataset,
                    "trackeval_macro": macro,
                    "distinct_target_recall": distinct_recall,
                    "inactive_false_acceptance": inactive_fa,
                    "selection_tuple": [
                        -macro["HOTA___AUC"], -macro["DetA___AUC"], -macro["AssA___AUC"],
                        -distinct_recall, inactive_fa, epoch,
                    ],
                })
        # The exact tuple is lexicographic: higher HOTA, then DetA, AssA,
        # distinct-target recall; lower inactive acceptance; earlier epoch.
        candidate_rules.sort(key=lambda row: tuple(row["selection_tuple"]))
        best_by_epoch: list[dict[str, Any]] = []
        for epoch in sorted(by_epoch):
            options = [row for row in candidate_rules if row["epoch"] == epoch]
            best = min(options, key=lambda row: tuple(row["selection_tuple"]))
            best_by_epoch.append(best)
        final = min(best_by_epoch, key=lambda row: tuple(row["selection_tuple"]))
        payload = {
            "format": "locatemot-l88-dev-checkpoint-selection-v1", "status": "complete",
            "evidence_type": "internal fit/dev full-video TrackEval selection only",
            "selection_rule": ["higher_dev_hota_macro", "higher_dev_deta_macro",
                                "higher_dev_assa_macro", "higher_distinct_target_recall",
                                "lower_inactive_false_acceptance", "earlier_epoch"],
            "macro_definition": "unweighted arithmetic mean of complete V1 and V2 dev TrackEval AUC values",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "shortlist": str(shortlist_path), "shortlist_sha256": sha256(shortlist_path),
            "trackeval_matrix": str(matrix_path), "trackeval_matrix_sha256": sha256(matrix_path),
            "candidate_rule_records": candidate_rules, "best_rule_per_epoch": best_by_epoch,
            "final_selection": final, "selection_frozen_before_fixed_validation": True,
            "fixed_validation_read": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": True, "no_hota_or_trackeval": False,
            "candidate_deletion": False, "candidate_truncation": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "failure_root_cause": None,
            "next_action": "run fixed 16-calibration/24-validation semantic evaluation with frozen selection",
        }
        write_json(out / "checkpoint_selection.json", payload)
        write_json(out / "selection.json", {"format": payload["format"], "status": "complete",
                                             "final_selection": final, "best_rule_per_epoch": best_by_epoch,
                                             "selection_frozen_before_fixed_validation": True,
                                             "screening_gt_used": False, "official_test_labels_read": False,
                                             "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
                                             "no_hota_or_trackeval": False})
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": payload["format"], "status": "complete",
                                          "shortlist_count": len(by_epoch), "rule_count": len(candidate_rules),
                                          "final_epoch": int(final["epoch"]), "final_rule": str(final["rule"]),
                                          "fixed_validation_read": False, "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": True, "no_hota_or_trackeval": False})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 checkpoint selection — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l88-dev-checkpoint-selection-v1",
                                          "status": "incomplete", "command": command, "cwd": str(WORK_ROOT),
                                          "luna_thread": THREAD, "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "fixed_validation_read": False, "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": False, "no_hota_or_trackeval": True})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--trackeval-matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
