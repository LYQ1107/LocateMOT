#!/usr/bin/env python3
"""Freeze the registered L89E checkpoint/rule from phase-consistent dev TrackEval."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
for value in (WORK_ROOT, WORK_ROOT / "tools", ROOT, ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from l89d_fullvideo_common import (  # noqa: E402
    MANIFEST_SHA,
    RULES,
    SEED,
    THREAD,
    WORK_ROOT as COMMON_WORK_ROOT,
    command_line,
    manifest_assertion,
    sha256_file,
    standard_flags,
    write_json,
)
from l89_select_checkpoint import read_scores  # noqa: E402


FORMAT = "locatemot-l89e-checkpoint-selection-v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89E selection output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    try:
        if Path.cwd().resolve() != COMMON_WORK_ROOT:
            raise RuntimeError(f"wrong L89E worktree cwd: {Path.cwd()}")
        manifest_assertion()
        source_path = args.source_selection.resolve()
        source = _load(source_path)
        if source.get("format") != "locatemot-l89e-phase-consistent-shortlist-v1" or source.get("status") != "complete":
            raise AssertionError("L89E shortlist is incomplete or wrong format")
        if not bool(source.get("phase_consistent_temporal")) or not bool(source.get("thresholds_refit_on_fit_dev_only")):
            raise AssertionError("L89E shortlist phase/refit contract missing")
        shortlist = source.get("shortlist")
        if not isinstance(shortlist, list) or not shortlist or len(shortlist) > 5:
            raise AssertionError("invalid L89E shortlist")
        by_path: dict[str, dict[str, Any]] = {}
        for item in shortlist:
            info = dict(item.get("checkpoint_info") or {})
            path = str(Path(str(info["path"])).resolve())
            if not Path(path).is_file() or sha256_file(Path(path)) != str(info["sha256"]):
                raise AssertionError(f"shortlist checkpoint SHA drift: {path}")
            if path in by_path:
                raise AssertionError("duplicate shortlist checkpoint")
            if set(RULES) - set(item.get("rule_fits") or {}):
                raise AssertionError(f"shortlist rules incomplete: {path}")
            by_path[path] = dict(item)
        matrix_path = args.dev_trackeval.resolve()
        matrix = _load(matrix_path)
        if matrix.get("format") != "locatemot-l89e-trackeval-matrix-v1" or matrix.get("status") != "complete":
            raise AssertionError("L89E dev TrackEval matrix incomplete")
        if matrix.get("scope") != "dev" or not bool(matrix.get("timeline_contract_passed")) or not bool(matrix.get("phase_consistency_passed")):
            raise AssertionError("L89E dev matrix lacks timeline/phase proof")
        results = matrix.get("results")
        if not isinstance(results, list) or len(results) != len(shortlist) * 3:
            raise AssertionError("L89E dev matrix result count drift")
        seen: set[tuple[str, str]] = set()
        legal: list[dict[str, Any]] = []
        for result in results:
            path = str(Path(str(result["checkpoint_info"]["path"])).resolve())
            rule = str(result.get("rule"))
            if path not in by_path or rule not in RULES:
                raise AssertionError(f"matrix item not in shortlist: {path}/{rule}")
            if not bool(result.get("phase_consistent_temporal")):
                raise AssertionError("matrix item lacks phase consistency")
            pair = (path, rule)
            if pair in seen:
                raise AssertionError(f"duplicate matrix item: {pair}")
            seen.add(pair)
            legal.append(result)
        if len(seen) != len(shortlist) * 3:
            raise AssertionError("not every shortlist checkpoint has B/R/P result")

        def key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float, int]:
            inactive = float(item.get("inactive_false_acceptance", 1.0))
            # Exact registered tie order: B > R > P.
            rule_order = {"B": 2, "R": 1, "P": 0}
            return (
                float(item["hota"]), float(item["deta"]), float(item["assa"]),
                float(item.get("distinct_target_recall", -1.0)), -inactive,
                -float(item["checkpoint_info"]["epoch"]), rule_order[str(item["rule"])],
            )

        chosen = max(legal, key=key)
        chosen_path = str(Path(str(chosen["checkpoint_info"]["path"])).resolve())
        source_candidate = by_path[chosen_path]
        rule = str(chosen["rule"])
        rule_object = dict(source_candidate["rule_fits"][rule])
        policy = dict(chosen.get("phase_policy") or {})
        if not policy or not bool(chosen.get("phase_consistent_temporal")):
            raise AssertionError("chosen phase policy missing")
        payload = {
            "format": FORMAT,
            "status": "complete",
            "stage": "L89E phase-consistent true-full-video dev selection",
            "evaluation_contract": "candidate_energy_vs_null",
            "protocol_repair_stage": "L89E",
            "zero_training": True,
            "checkpoint_weights_changed": False,
            "new_checkpoint_created": False,
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "seed": SEED,
            "source_selection": str(source_path),
            "source_selection_sha256": sha256_file(source_path),
            "dev_trackeval_source": str(matrix_path),
            "dev_trackeval_source_sha256": sha256_file(matrix_path),
            "shortlist": shortlist,
            "shortlist_count": len(shortlist),
            "trackeval_candidates": results,
            "selection_key": "(HOTA, DetA, AssA, distinct_target_recall, -inactive_false_acceptance, -epoch, deterministic_rule_order)",
            "rule_order": {"B": 2, "R": 1, "P": 0},
            "final_selection": {
                "checkpoint_info": dict(chosen["checkpoint_info"]),
                "rule": rule,
                "rule_fit": dict(rule_object.get("metrics", rule_object)),
                "rule_object": rule_object,
                "selection_metrics": chosen,
                "phase_policy": policy,
            },
            "selection_frozen_before_fixed_validation": True,
            "fixed_calibration_read": False,
            "fixed_validation_read": False,
            "true_fullvideo_timeline": True,
            "timeline_contract_passed": True,
            "phase_consistent_temporal": True,
            "thresholds_refit_after_fixed": False,
            "thresholds_refit_on_fit_dev_only": True,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "manifest_sha256": MANIFEST_SHA,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            **standard_flags(hota_trackeval_run=True),
            "no_hota_or_trackeval": False,
            "failure_root_cause": None,
            "next_action": "run fixed 16-calibration/24-validation semantic replay with the frozen phase policy",
        }
        write_json(out / "checkpoint_selection.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-l89e-selection-provenance-v1"})
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "complete",
            "selected_epoch": int(chosen["checkpoint_info"]["epoch"]),
            "selected_phase": policy["phase"],
            "selected_rule": rule,
            "phase_consistent_temporal": True,
            "selection_frozen_before_fixed_validation": True,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "zero_training": True,
            "hota_trackeval_run": True,
        })
        return 0
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text("# L89E checkpoint selection — INCOMPLETE\n\n" + traceback.format_exc(), encoding="utf-8")
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "incomplete",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair first L89E selection contract error and use a new output",
            **standard_flags(hota_trackeval_run=False),
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--dev-trackeval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
