#!/usr/bin/env python3
"""Verify R1 frozen-source identities before cache/training/evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from locatemot.paths import R1_ANCHOR, R1_RULE

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.r1_aligned_track_conditioning import ANCHOR_SHA256, load_l89e_rule, sha256_file  # noqa: E402
from tools.r1_common import MANIFEST, MANIFEST_SHA, THREAD, check_manifest, file_meta, standard_flags, write_json  # noqa: E402

ANCHOR = R1_ANCHOR
SELECTION = R1_RULE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 boundary output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_sha = check_manifest()
    anchor_sha = sha256_file(ANCHOR)
    if anchor_sha != ANCHOR_SHA256:
        raise AssertionError(f"R1 anchor hash drift: {anchor_sha}")
    rule = load_l89e_rule(SELECTION)
    if (rule["candidate_threshold"], rule["presence_threshold"], rule["null_margin"]) != (1.0, 0.5, 0.0):
        raise AssertionError(f"R1 Rule-B drift: {rule}")
    payload = {
        "format": "locatemot-r1-boundary-guard-v1", "status": "complete",
        "command": " ".join([str(sys.executable), *sys.argv]), "cwd": str(Path.cwd().resolve()),
        "thread": THREAD, "manifest": file_meta(MANIFEST), "manifest_sha256": manifest_sha,
        "anchor": {"path": str(ANCHOR), "sha256": anchor_sha, "expected_sha256": ANCHOR_SHA256},
        "rule": rule, "selection": file_meta(SELECTION), "production_files_modified": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "next_action": "build or load only label-free R1 aligned cache",
        "failure_root_cause": None, **standard_flags(training_run=False),
    }
    write_json(out / "contract.json", payload)
    write_json(out / "status.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
