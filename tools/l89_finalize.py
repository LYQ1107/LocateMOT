#!/usr/bin/env python3
"""Freeze the registered L89 dev selection for the internal-only pass.

This is a small provenance adapter: it does not train, rescore, read fixed
validation labels, or alter a checkpoint.  The full-video runner consumes the
registered shortlist shape, so this tool exposes only the already selected
checkpoint while retaining its frozen B/R/P rule objects for diagnostic
matrix output.  The selected rule remains explicit in ``final_selection``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 final-selection output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89 cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        selection_path = args.selection.resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "complete" or not selection.get("selection_frozen_before_fixed_validation"):
            raise AssertionError("L89 selection is not complete/frozen")
        final = selection.get("final_selection") or {}
        checkpoint_info = dict(final.get("checkpoint_info") or {})
        rule_name = str(final.get("rule") or "")
        if not checkpoint_info or rule_name not in {"B", "R", "P"}:
            raise AssertionError("L89 final selection lacks checkpoint or registered rule")
        checkpoint_path = Path(str(checkpoint_info["path"])).resolve()
        if sha256_file(checkpoint_path) != str(checkpoint_info["sha256"]):
            raise AssertionError("selected checkpoint SHA drift")
        source_item = None
        for item in selection.get("shortlist", []):
            if str(Path(str(item.get("checkpoint_info", {}).get("path", ""))).resolve()) == str(checkpoint_path):
                source_item = item
                break
        if source_item is None:
            raise AssertionError("selected checkpoint absent from frozen shortlist")
        rule_fits = source_item.get("rule_fits") or {}
        if set(rule_fits) != {"B", "R", "P"}:
            raise AssertionError("frozen shortlist does not contain all registered rules")
        selected_rule_object = final.get("rule_object") or rule_fits[rule_name]
        candidate = {
            "shortlist_index": 0,
            "reason": "frozen_final_dev_trackeval_selection",
            "checkpoint_info": checkpoint_info,
            "rule_fits": rule_fits,
            "selected_rule": rule_name,
            "selected_rule_object": selected_rule_object,
        }
        payload = {
            "format": "locatemot-l89-final-selection-v1",
            "status": "complete",
            "stage": "frozen internal-only selection adapter; no fixed validation labels read",
            "command": command,
            "cwd": str(WORK_ROOT),
            "luna_thread": THREAD,
            "seed": SEED,
            "source_selection": str(selection_path),
            "source_selection_sha256": sha256_file(selection_path),
            "checkpoint_info": checkpoint_info,
            "selected_rule": rule_name,
            "selected_rule_object": selected_rule_object,
            "shortlist": [candidate],
            "shortlist_count": 1,
            "selection_frozen_before_fixed_validation": True,
            "fixed_calibration_read": False,
            "fixed_validation_read": False,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
            "no_screening_or_official_test": True,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "failure_root_cause": None,
            "next_action": "run frozen internal V1/V2 inference and TrackEval",
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "final_selection.json", payload)
        write_json(out / "status.json", {
            "format": "locatemot-l89-final-selection-v1",
            "status": "complete",
            "selected_epoch": int(checkpoint_info["epoch"]),
            "selected_rule": rule_name,
            "selection_frozen_before_fixed_validation": True,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
        })
        write_json(out / "provenance.json", payload | {"format": "locatemot-l89-final-selection-provenance-v1"})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89 final selection — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {
            "format": "locatemot-l89-final-selection-v1",
            "status": "incomplete",
            "command": command,
            "cwd": str(WORK_ROOT),
            "luna_thread": THREAD,
            "failure_root_cause": "first traceback in INCOMPLETE.md",
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
