#!/usr/bin/env python3
"""Build the registered L89E maximum-five shortlist from fit/dev rows."""
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
    SEED,
    THREAD,
    WORK_ROOT as COMMON_WORK_ROOT,
    command_line,
    manifest_assertion,
    sha256_file,
    standard_flags,
    write_json,
)
from l89_select_checkpoint import build_shortlist, read_scores  # noqa: E402


FORMAT = "locatemot-l89e-phase-consistent-shortlist-v1"
EXPECTED_EPOCHS = set(range(2, 41, 2))


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89E shortlist output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    try:
        if Path.cwd().resolve() != COMMON_WORK_ROOT:
            raise RuntimeError(f"wrong L89E worktree cwd: {Path.cwd()}")
        manifest_assertion()
        grouped = read_scores(args.scores.resolve())
        epochs = {int(rows[0]["checkpoint"]["epoch"]) for rows in grouped.values() if rows}
        if epochs != EXPECTED_EPOCHS:
            raise AssertionError(f"merged epoch set drift: {sorted(epochs)}")
        shortlist = build_shortlist(grouped)
        if len(shortlist) < 1 or len(shortlist) > 5:
            raise AssertionError(f"registered shortlist size drift: {len(shortlist)}")
        shortlist_epochs = [int(item["checkpoint_info"]["epoch"]) for item in shortlist]
        if not {8, 20, 40}.issubset(set(shortlist_epochs)):
            raise AssertionError(f"fixed shortlist epochs missing: {shortlist_epochs}")
        for item in shortlist:
            if not item.get("rule_fits") or set(("B", "R", "P")) - set(item["rule_fits"]):
                raise AssertionError("shortlist corrected B/R/P fits missing")
        payload = {
            "format": FORMAT,
            "status": "complete",
            "stage": "L89E fit/dev phase-consistent shortlist; fixed validation pending",
            "evaluation_contract": "candidate_energy_vs_null",
            "protocol_repair_stage": "L89E",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "seed": SEED,
            "source_scores": str(args.scores.resolve()),
            "source_scores_sha256": sha256_file(args.scores.resolve()),
            "shortlist": shortlist,
            "shortlist_count": len(shortlist),
            "shortlist_epochs": shortlist_epochs,
            "shortlist_rule": {
                "fixed_epochs": [8, 20, 40],
                "best": ["target_bag_f1", "distinct_target_recall_with_precision_0.08"],
                "max": 5,
                "rules_refit_on_fit_dev_only": True,
            },
            "phase_consistent_temporal": True,
            "stage_s_rescored": True,
            "tj_sparse_scores_reused": True,
            "thresholds_refit_on_fit_dev_only": True,
            "fixed_calibration_read": False,
            "fixed_validation_read": False,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "zero_training": True,
            "new_checkpoint_created": False,
            "checkpoint_weights_changed": False,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            **standard_flags(hota_trackeval_run=False),
            "failure_root_cause": None,
            "next_action": "run phase-consistent true-full-video dev matrix",
        }
        write_json(out / "shortlist.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-l89e-shortlist-provenance-v1"})
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "complete",
            "shortlist_count": len(shortlist),
            "shortlist_epochs": shortlist_epochs,
            "phase_consistent_temporal": True,
            "fixed_validation_read": False,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "zero_training": True,
        })
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89E shortlist — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "incomplete",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair the first shortlist contract error in a new output",
            **standard_flags(hota_trackeval_run=False),
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
