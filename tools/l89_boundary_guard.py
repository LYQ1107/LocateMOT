#!/usr/bin/env python3
"""Fail-closed boundary audit for the isolated L89 worktree.

The guard is intentionally small and read-only apart from its compact JSON
record.  It audits tracked, staged, and untracked paths because the L89 work
is developed before its first commit.  The base commit is the frozen
pre-L89 revision requested by the stage protocol.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89").resolve()
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
BASE_SHA = "ece9fc5549a9210424eb127fb10430c3fed7b7ba"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def command(args: list[str]) -> list[str]:
    result = subprocess.run(
        args, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def allowed(path: str) -> bool:
    value = path.replace("\\", "/")
    return (
        value == "research_log.md"
        or value.startswith("locatemot/models/l89_")
        or value.startswith("locatemot/rmot/l89_")
        or value.startswith("tools/l89_")
        or value.startswith("reports/l89/")
        or value.startswith("outputs/l89/")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l89/audit/boundary_guard.json")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong L89 worktree cwd: {Path.cwd()}")
    tracked = command(["git", "diff", "--name-only", f"{BASE_SHA}...HEAD"])
    unstaged = command(["git", "diff", "--name-only"])
    staged = command(["git", "diff", "--cached", "--name-only"])
    untracked = command(["git", "ls-files", "--others", "--exclude-standard"])
    all_paths = sorted(set(tracked + unstaged + staged + untracked))
    forbidden = [path for path in all_paths if not allowed(path)]
    manifest_sha = sha256_file(MANIFEST)
    payload: dict[str, Any] = {
        "format": "locatemot-l89-boundary-guard-v1",
        "status": "complete" if not forbidden and manifest_sha == MANIFEST_SHA else "invalid",
        "command": " ".join(["python", *(__import__("sys").argv)]),
        "cwd": str(ROOT), "asset_root": str(ASSET_ROOT), "luna_thread": THREAD, "base_commit": BASE_SHA,
        "tracked_diff_paths": tracked, "unstaged_paths": unstaged,
        "staged_paths": staged, "untracked_paths": untracked,
        "audited_paths": all_paths, "forbidden_paths": forbidden,
        "allowed_policy": [
            "research_log.md", "locatemot/models/l89_*", "locatemot/rmot/l89_*",
            "tools/l89_*", "reports/l89/**", "outputs/l89/**",
        ],
        "manifest": {"path": str(MANIFEST), "sha256": manifest_sha, "expected_sha256": MANIFEST_SHA,
                     "unchanged": manifest_sha == MANIFEST_SHA},
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False,
        "hota_trackeval_run": False,
        "failure_root_cause": None if not forbidden and manifest_sha == MANIFEST_SHA else "boundary or manifest check failed",
        "next_action": "continue L89 only after guard is complete" if not forbidden and manifest_sha == MANIFEST_SHA else "stop and repair first boundary violation",
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "complete":
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
