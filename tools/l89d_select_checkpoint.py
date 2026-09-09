#!/usr/bin/env python3
"""Freeze the registered L89D dev checkpoint/rule from true-timeline metrics."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from l89d_fullvideo_common import (
    MANIFEST_SHA,
    RULES,
    SEED,
    THREAD,
    WORK_ROOT,
    command_line,
    manifest_assertion,
    sha256_file,
    standard_flags,
    write_json,
)


FORMAT = "locatemot-l89d-checkpoint-selection-v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89D selection output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89D cwd: {Path.cwd()}")
        manifest_assertion()
        source_path = args.source_selection.resolve()
        source = _load(source_path)
        if source.get("status") != "complete" or str(source.get("protocol_repair_stage")) != "L89C":
            raise AssertionError("L89C source selection is incomplete or wrong stage")
        shortlist = source.get("shortlist")
        if not isinstance(shortlist, list) or not shortlist:
            raise AssertionError("L89C source shortlist missing")
        by_path: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(shortlist):
            candidate = dict(item)
            info = dict(candidate.get("checkpoint_info") or {})
            checkpoint = Path(str(info["path"])).resolve()
            if not checkpoint.is_file() or sha256_file(checkpoint) != str(info["sha256"]):
                raise AssertionError(f"shortlist checkpoint SHA drift: {checkpoint}")
            rules = candidate.get("rule_fits")
            if not isinstance(rules, dict) or set(RULES) - set(rules):
                raise AssertionError(f"shortlist rules incomplete at index {index}")
            for rule in RULES:
                for name in ("candidate_threshold", "presence_threshold", "null_margin"):
                    if name not in rules[rule] or not isinstance(rules[rule][name], (int, float)):
                        raise AssertionError(f"missing threshold {rule}/{name}")
            key = str(checkpoint)
            if key in by_path:
                raise AssertionError("duplicate shortlist checkpoint")
            by_path[key] = candidate

        matrix_path = args.dev_trackeval.resolve()
        matrix = _load(matrix_path)
        if matrix.get("status") != "complete" or matrix.get("format") != "locatemot-l89d-trackeval-matrix-v1":
            raise AssertionError("L89D dev TrackEval matrix incomplete")
        if matrix.get("scope") != "dev" or not matrix.get("timeline_contract_passed"):
            raise AssertionError("dev matrix lacks timeline proof")
        results = matrix.get("results")
        if not isinstance(results, list) or len(results) != len(shortlist) * 3:
            raise AssertionError(f"dev matrix result count drift: {len(results) if isinstance(results, list) else None}")
        seen: set[tuple[str, str]] = set()
        legal: list[dict[str, Any]] = []
        for result in results:
            path = str(Path(str(result["checkpoint_info"]["path"])).resolve())
            rule = str(result.get("rule"))
            if path not in by_path or rule not in RULES:
                raise AssertionError(f"dev matrix item not in source shortlist: {path}/{rule}")
            pair = (path, rule)
            if pair in seen:
                raise AssertionError(f"duplicate dev matrix item: {pair}")
            seen.add(pair)
            legal.append(result)
        if len(seen) != len(shortlist) * 3:
            raise AssertionError("not every shortlist checkpoint has B/R/P matrix result")

        def selection_key(result: dict[str, Any]) -> tuple[float, float, float, float, float, float, int]:
            inactive = float(result.get("inactive_false_acceptance", 1.0))
            rule_order = {"B": 2, "R": 1, "P": 0}
            return (
                float(result["hota"]), float(result["deta"]), float(result["assa"]),
                float(result.get("distinct_target_recall", -1.0)), -inactive,
                -float(result["checkpoint_info"]["epoch"]), rule_order[str(result["rule"])],
            )

        chosen = max(legal, key=selection_key)
        chosen_path = str(Path(str(chosen["checkpoint_info"]["path"])).resolve())
        source_candidate = by_path[chosen_path]
        rule = str(chosen["rule"])
        rule_object = dict(source_candidate["rule_fits"][rule])
        # This is an assertion, not a refit: every threshold is copied from
        # the immutable L89C source selection.
        if any(float(rule_object[name]) != float(source_candidate["rule_fits"][rule][name]) for name in ("candidate_threshold", "presence_threshold", "null_margin")):
            raise AssertionError("threshold copy mismatch")
        payload = {
            "format": FORMAT, "status": "complete", "stage": "L89D true-full-video dev selection",
            "evaluation_contract": "candidate_energy_vs_null", "protocol_repair_stage": "L89D",
            "zero_training": True, "checkpoint_weights_changed": False, "new_checkpoint_created": False,
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "source_selection": str(source_path), "source_selection_sha256": sha256_file(source_path),
            "dev_trackeval_source": str(matrix_path), "dev_trackeval_source_sha256": sha256_file(matrix_path),
            "shortlist": shortlist, "shortlist_count": len(shortlist),
            "trackeval_candidates": results,
            "selection_key": "(HOTA, DetA, AssA, distinct_target_recall, -inactive_false_acceptance, -epoch, deterministic_rule_order)",
            "rule_order": {"B": 2, "R": 1, "P": 0},
            "final_selection": {
                "checkpoint_info": dict(chosen["checkpoint_info"]), "rule": rule,
                "rule_fit": dict(rule_object.get("metrics", rule_object)),
                "rule_object": rule_object, "selection_metrics": chosen,
            },
            "selection_frozen_before_fixed_validation": True,
            "fixed_calibration_read": False, "fixed_validation_read": False,
            "true_fullvideo_timeline": True, "timeline_contract_passed": True,
            "thresholds_refit": False, "candidate_deletion": False, "candidate_truncation": False,
            "manifest_sha256": MANIFEST_SHA,
            **standard_flags(hota_trackeval_run=True), "failure_root_cause": None,
            "next_action": "run unchanged fixed semantic evaluator, then L89D internal true full-video replay",
        }
        write_json(out / "checkpoint_selection.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {
            "format": FORMAT, "status": "complete", "protocol_repair_stage": "L89D",
            "selected_epoch": int(chosen["checkpoint_info"]["epoch"]), "rule": rule,
            "shortlist_count": len(shortlist), "timeline_contract_passed": True,
            "thresholds_refit": False, "zero_training": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
        })
        return 0
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text("# L89D checkpoint selection — INCOMPLETE\n\n" + traceback.format_exc(), encoding="utf-8")
        write_json(out / "status.json", {"format": FORMAT, "status": "incomplete", "command": command,
                                          "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                          "next_action": "repair first selection-contract error in a new output",
                                          **standard_flags(hota_trackeval_run=False)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--dev-trackeval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
