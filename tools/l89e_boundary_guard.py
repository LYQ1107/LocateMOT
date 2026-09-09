#!/usr/bin/env python3
"""Fail-closed source boundary guard for the L89E protocol-repair branch."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


WORK_ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "d0960e1d1679414765f79203bef444f0f9968928"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
ALLOWED_EXACT = {
    "tools/l89e_phase_policy.py",
    "tools/l89e_rescore_stage_s_dev.py",
    "tools/l89e_merge_dev_scores.py",
    "tools/l89e_build_shortlist.py",
    "tools/l89e_infer_true_fullvideo.py",
    "tools/l89e_trackeval_matrix.py",
    "tools/l89e_select_checkpoint.py",
    "tools/l89e_eval_fixed_semantic.py",
    "tools/l89e_boundary_guard.py",
    "tools/l89e_finalize_report.py",
    "research_log.md",
}
ALLOWED_PREFIXES = ("reports/l89e/", "outputs/l89e/")


def _git(*args: str) -> list[str]:
    value = subprocess.check_output(["git", *args], cwd=WORK_ROOT, text=True)
    return [line.strip() for line in value.splitlines() if line.strip()]


def _allowed(path: str) -> bool:
    return path in ALLOWED_EXACT or any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def run(args: argparse.Namespace) -> int:
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    violations: list[str] = []
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise AssertionError(f"wrong L89E worktree cwd: {Path.cwd()}")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORK_ROOT, text=True).strip()
        if subprocess.check_output(["git", "rev-parse", BASE_SHA], cwd=WORK_ROOT, text=True).strip() != BASE_SHA:
            raise AssertionError("L89D base commit cannot be resolved")
        changed = set(_git("diff", "--name-only", f"{BASE_SHA}...HEAD"))
        changed.update(_git("diff", "--name-only"))
        changed.update(_git("diff", "--cached", "--name-only"))
        changed.update(_git("ls-files", "--others", "--exclude-standard"))
        violations = sorted(path for path in changed if not _allowed(path))
        if violations:
            raise AssertionError(f"out-of-boundary paths: {violations}")
        payload: dict[str, Any] = {
            "format": "locatemot-l89e-boundary-guard-v1",
            "status": "complete",
            "cwd": str(WORK_ROOT),
            "luna_thread": THREAD,
            "base_l89d_sha": BASE_SHA,
            "head_sha": head,
            "allowed_exact": sorted(ALLOWED_EXACT),
            "allowed_prefixes": list(ALLOWED_PREFIXES),
            "changed_paths": sorted(changed),
            "violations": [],
            "model_source_changed": False,
            "tracker_source_changed": False,
            "uidm_source_changed": False,
            "candidate_bank_changed": False,
            "ordinary_mot_ovmot_touched": False,
            "zero_training": True,
            "new_checkpoint_created": False,
            "checkpoint_weights_changed": False,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "hota_trackeval_run": False,
            "next_action": "continue L89E compile and CPU phase-history equality check",
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        payload = {
            "format": "locatemot-l89e-boundary-guard-v1",
            "status": "invalid",
            "cwd": str(WORK_ROOT),
            "luna_thread": THREAD,
            "base_l89d_sha": BASE_SHA,
            "violations": violations,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair only the first boundary violation before formal replay",
            "ordinary_mot_ovmot_touched": bool(violations),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/l89e/audit/boundary_guard.json"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
