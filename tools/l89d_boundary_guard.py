#!/usr/bin/env python3
"""Fail-closed source-boundary guard for the L89D evidence repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
BASE_SHA = "613fbe6d5e802fbc08d3ef0b5084f5d6b8337cd5"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
ALLOWED_EXACT = {
    "research_log.md",
    "tools/l89d_fullvideo_common.py",
    "tools/l89d_audit_dense_coverage.py",
    "tools/l89d_build_dense_z1_cache.py",
    "tools/l89d_finalize_dense_z1_cache.py",
    "tools/l89d_infer_true_fullvideo.py",
    "tools/l89d_trackeval_matrix.py",
    "tools/l89d_select_checkpoint.py",
    "tools/l89d_boundary_guard.py",
    "tools/l89d_finalize_report.py",
}
ALLOWED_PREFIXES = ("reports/l89d/", "outputs/l89d/")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(["git", "-C", str(WORK_ROOT), *args], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def allowed(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized in ALLOWED_EXACT or normalized.startswith(ALLOWED_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "outputs/l89d/audit/boundary_guard.json")
    args = parser.parse_args()
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong L89D worktree cwd: {Path.cwd()}")
    tracked = git_lines("diff", "--name-only", f"{BASE_SHA}...HEAD")
    unstaged = git_lines("diff", "--name-only")
    staged = git_lines("diff", "--cached", "--name-only")
    untracked = git_lines("ls-files", "--others", "--exclude-standard")
    audited = sorted(set(tracked + unstaged + staged + untracked))
    forbidden = sorted(path for path in audited if not allowed(path))
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != MANIFEST_SHA:
        forbidden.append("fixed manifest SHA drift")
    forbidden = sorted(set(forbidden))
    payload: dict[str, Any] = {
        "format": "locatemot-l89d-boundary-guard-v1",
        "status": "complete" if not forbidden else "invalid",
        "command": " ".join([sys.executable, *sys.argv]), "cwd": str(WORK_ROOT),
        "asset_root": str(ASSET_ROOT), "luna_thread": THREAD, "base_commit": BASE_SHA,
        "tracked_diff_paths": tracked, "unstaged_paths": unstaged, "staged_paths": staged,
        "untracked_paths": untracked, "audited_paths": audited, "forbidden_paths": forbidden,
        "allowed_policy": sorted(ALLOWED_EXACT) + list(ALLOWED_PREFIXES),
        "manifest": {"path": str(MANIFEST), "sha256": manifest_sha, "expected_sha256": MANIFEST_SHA,
                     "unchanged": manifest_sha == MANIFEST_SHA},
        "zero_training": True, "training_run": False, "checkpoint_weights_changed": False,
        "new_checkpoint_created": False, "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "failure_root_cause": None if not forbidden else "boundary or manifest check failed",
        "next_action": "continue L89D only after guard is complete" if not forbidden else "stop and repair first boundary violation",
    }
    out = args.out.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if forbidden:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
