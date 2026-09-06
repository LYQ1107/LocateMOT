#!/usr/bin/env python3
"""TrackEval wrapper for L88C corrected full-video outputs.

The local TrackEval invocation is the already audited implementation.  The
wrapper makes the provenance of this corrected, zero-training replay explicit
without modifying the historical L88 TrackEval helper or its outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from l88_eval_common import MANIFEST, MANIFEST_SHA, THREAD, sha256, write_json
import l88_trackeval_matrix as legacy


WORK_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_INTERNAL_SEQUENCES = {
    "refer_kitti_v1": 86,
    "refer_kitti_v2": 537,
}


def _load_frozen_selection(selection_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = selection_path.resolve()
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "complete":
        raise AssertionError("frozen L88C selection is incomplete")
    if not bool(selection.get("selection_frozen_before_fixed_validation")):
        raise AssertionError("L88C selection was not frozen before fixed validation")
    final = selection.get("final_selection")
    if not isinstance(final, dict):
        raise AssertionError("L88C final_selection missing")
    checkpoint_info = dict(final["checkpoint_info"])
    if int(final["epoch"]) != 30 or int(checkpoint_info["epoch"]) != 30:
        raise AssertionError("frozen L88C epoch drift")
    if str(final["rule"]) != "B":
        raise AssertionError(f"frozen L88C rule drift: {final['rule']}")
    rule_fit = dict(final["rule_fit"])
    expected = {
        "candidate_threshold": 1.0,
        "presence_threshold": -1.0,
        "null_margin": 0.0,
    }
    for key, expected_value in expected.items():
        actual = float(rule_fit[key])
        if actual != expected_value:
            raise AssertionError(f"frozen L88C threshold drift {key}: {actual} != {expected_value}")
    checkpoint_path = Path(str(checkpoint_info["path"])).resolve()
    if sha256(checkpoint_path) != str(checkpoint_info["sha256"]):
        raise AssertionError(f"frozen L88C checkpoint SHA drift: {checkpoint_path}")
    return selection, {
        "selection_path": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "checkpoint_info": checkpoint_info,
        "rule": "B",
        "rule_fit": rule_fit,
    }


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88C TrackEval output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        roots = [path.resolve() for path in args.inference_roots]
        if not roots:
            raise AssertionError("no inference roots")
        selection_info = None
        if args.expected_scope == "internal":
            if args.selection is None:
                raise ValueError("--selection is required for internal final TrackEval")
            _, selection_info = _load_frozen_selection(args.selection)

        results: list[dict[str, Any]] = []
        for root in roots:
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete":
                raise AssertionError(f"inference root incomplete: {root}")
            if not bool(summary.get("full_video")):
                raise AssertionError(f"not full-video: {root}")
            scope_key = str(summary.get("scope_key") or "")
            if scope_key != args.expected_scope:
                raise AssertionError(f"scope drift: {scope_key} != {args.expected_scope}")
            if summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
                raise AssertionError(f"forbidden labels in source: {root}")

            if args.expected_scope == "internal":
                if not bool(summary.get("frozen_selection_mode")):
                    raise AssertionError("internal final inference is not frozen-selection mode")
                if list(summary.get("selected_rule_names", [])) != ["B"]:
                    raise AssertionError(
                        f"internal final rules drift: {summary.get('selected_rule_names')}"
                    )
                if int(summary.get("frozen_checkpoint_epoch")) != 30:
                    raise AssertionError("internal final frozen checkpoint epoch drift")
                if not bool(summary.get("selection_frozen_before_fixed_validation")):
                    raise AssertionError("internal selection freeze marker missing")
                if str(summary.get("strategy_source")) != str(selection_info["selection_path"]):
                    raise AssertionError("internal strategy source drift")

            candidates = list(summary.get("candidates", []))
            if args.expected_scope == "internal" and len(candidates) != 1:
                raise AssertionError(f"internal final candidate count drift: {len(candidates)}")
            for candidate in candidates:
                candidate_rules = candidate.get("rules")
                if not isinstance(candidate_rules, dict) or not candidate_rules:
                    raise AssertionError("candidate rules missing")
                rule_names = sorted(candidate_rules.keys())
                if args.expected_scope == "internal" and rule_names != ["B"]:
                    raise AssertionError(f"internal final must contain only Rule B: {rule_names}")
                if args.expected_scope == "internal":
                    checkpoint = candidate.get("checkpoint", {})
                    if int(checkpoint.get("epoch")) != 30:
                        raise AssertionError("internal candidate checkpoint epoch drift")
                    if str(checkpoint.get("sha256")) != str(selection_info["checkpoint_info"]["sha256"]):
                        raise AssertionError("internal candidate checkpoint SHA drift")
                for rule_name in rule_names:
                    for dataset in legacy.DATASETS:
                        source = root / f"candidate_epoch{int(candidate['checkpoint']['epoch']):03d}" / rule_name / dataset
                        destination = out / root.name / f"candidate_epoch{int(candidate['checkpoint']['epoch']):03d}" / rule_name / dataset
                        result = legacy.run_dataset(source, destination, dataset, rule_name)
                        if args.expected_scope == "internal":
                            expected = EXPECTED_INTERNAL_SEQUENCES[dataset]
                            if int(result["sequence_count"]) != expected:
                                raise AssertionError(
                                    f"internal sequence-count drift {dataset}: "
                                    f"{result['sequence_count']} != {expected}"
                                )
                        results.append(result)

        expected_count = 2 if args.expected_scope == "internal" else None
        if expected_count is not None and len(results) != expected_count:
            raise AssertionError(f"internal result count drift: {len(results)} != {expected_count}")
        payload = {
            "format": "locatemot-l88c-internal-final-trackeval-v1" if args.expected_scope == "internal" else "locatemot-l88c-trackeval-matrix-v1",
            "status": "complete",
            "scope_key": args.expected_scope,
            "evidence_type": (
                "full-video internal V1/V2 validation TrackEval for frozen L88C epoch30 Rule B corrected strategy"
                if args.expected_scope == "internal" else
                "full-video corrected candidate-vs-NULL TrackEval matrix"
            ),
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "inference_roots": [str(path) for path in roots],
            "source_summary_sha256": {str(path): sha256(path / "summary.json") for path in roots},
            "results": results, "manifest_sha256": MANIFEST_SHA,
            "zero_training": True, "corrected_candidate_vs_null": True,
            "selection_source": selection_info["selection_path"] if selection_info else None,
            "selection_source_sha256": selection_info["selection_sha256"] if selection_info else None,
            "frozen_epoch": 30 if selection_info else None,
            "frozen_rule": "B" if selection_info else None,
            "candidate_threshold": 1.0 if selection_info else None,
            "presence_threshold": -1.0 if selection_info else None,
            "null_margin": 0.0 if selection_info else None,
            "fit_dev_labels_only": False,
            "internal_validation_labels_only": args.expected_scope == "internal",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
            "no_hota_or_trackeval": False, "candidate_deletion": False,
            "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            "wall_seconds": time.perf_counter() - started,
            "failure_root_cause": None,
            "next_action": "write final internal validation report" if args.expected_scope == "internal" else "stop",
        }
        write_json(out / "trackeval_matrix.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {
            "format": payload["format"], "status": "complete", "scope_key": args.expected_scope,
            "result_count": len(results), "manifest_sha256": MANIFEST_SHA,
            "zero_training": True, "corrected_candidate_vs_null": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
            "no_hota_or_trackeval": False,
        })
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88C TrackEval matrix — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-trackeval-status-v1", "status": "incomplete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "zero_training": True, "corrected_candidate_vs_null": True,
            "failure_root_cause": "first traceback in INCOMPLETE.md",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": False,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--selection", type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
