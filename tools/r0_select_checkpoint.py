#!/usr/bin/env python3
"""Freeze one R0 checkpoint using only legal dev TrackEval evidence."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r0_common import check_manifest, sha256_file, write_json  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909


def choose(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise AssertionError("empty legal R0 dev TrackEval result")

    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        metrics = item.get("trackeval", {}).get("metrics_raw", {})
        dev = item.get("dev_metrics") or {}
        inactive = float(dev.get("inactive_false_acceptance", 1.0))
        epoch = int(item["checkpoint_info"]["epoch"])
        return (float(metrics.get("HOTA___AUC", -1.0)),
                float(metrics.get("DetA___AUC", -1.0)),
                float(metrics.get("AssA___AUC", -1.0)),
                float(dev.get("distinct_target_recall", -1.0)),
                -inactive, -epoch)

    return max(results, key=key)


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 selection output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    base = {"format": "locatemot-r0-checkpoint-selection-v1", "status": "incomplete",
            "command": command, "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD,
            "seed": SEED, "dataset": args.dataset, "trackeval_source": str(args.trackeval.resolve()),
            "manifest_sha256": check_manifest(), "selection_data": "legal dev TrackEval only",
            "fixed_calibration_read": False, "fixed_validation_read": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
            "candidate_deletion": False, "candidate_truncation": False,
            "failure_root_cause": None, "next_action": "run fixed post-selection diagnostics"}
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 selection cwd: {Path.cwd()}")
        matrix = json.loads(args.trackeval.resolve().read_text(encoding="utf-8"))
        if matrix.get("status") != "complete":
            raise AssertionError("R0 dev TrackEval matrix incomplete")
        results = [value for value in matrix.get("results", []) if str(value.get("checkpoint_info", {}).get("dataset")) in {args.dataset, ""}]
        if not results:
            # The checkpoint info is allowed to omit the dataset in legacy
            # package wrappers; the inference scope remains the guard.
            results = list(matrix.get("results", []))
        chosen = choose(results)
        selection = {
            **base, "status": "complete", "selected": {
                "checkpoint_info": chosen["checkpoint_info"], "rule": chosen["rule"],
                "dev_metrics": chosen.get("dev_metrics"), "trackeval": chosen.get("trackeval"),
                "selection_tuple": [chosen.get("hota", -1.0), chosen.get("deta", -1.0),
                                     chosen.get("assa", -1.0),
                                     (chosen.get("dev_metrics") or {}).get("distinct_target_recall", -1.0),
                                     -(chosen.get("dev_metrics") or {}).get("inactive_false_acceptance", 1.0),
                                     -int(chosen["checkpoint_info"]["epoch"])],
            }, "candidate_results": results, "candidate_count": len(results),
            "selection_frozen_before_fixed_validation": True,
            "trackeval_source_sha256": sha256_file(args.trackeval.resolve()),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "selection.json", selection); write_json(out / "provenance.json", selection); write_json(out / "status.json", selection)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 checkpoint selection — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "hota_trackeval_run": False, "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2"), required=True)
    parser.add_argument("--trackeval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
