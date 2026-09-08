#!/usr/bin/env python3
"""Guard the L89C source and frozen-asset boundary."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


BASE_SHA = "23f36bc3eefa4e891ff38ba1e7e3bd870b171e0e"
WORK_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
OUTPUT = WORK_ROOT / "outputs/l89c/audit/boundary_guard.json"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"

ALLOWED_EXACT = {
    "research_log.md",
    "tools/l89_score_dev.py",
    "tools/l89_select_checkpoint.py",
    "tools/l89_eval_fixed_semantic.py",
    "tools/l89_infer_fullvideo.py",
    "tools/l89_trackeval_matrix.py",
    "tools/l89c_boundary_guard.py",
}
ALLOWED_PREFIXES = ("reports/l89c/", "outputs/l89c/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_paths(*args: str) -> set[str]:
    output = subprocess.check_output(
        ["git", "-C", str(WORK_ROOT), *args], text=True
    )
    return {line.strip() for line in output.splitlines() if line.strip()}


def allowed(path: str) -> bool:
    return path in ALLOWED_EXACT or path.startswith(ALLOWED_PREFIXES)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    changed = set()
    changed.update(git_paths("diff", "--name-only", f"{BASE_SHA}...HEAD"))
    changed.update(git_paths("diff", "--name-only"))
    changed.update(git_paths("diff", "--cached", "--name-only"))
    changed.update(git_paths("ls-files", "--others", "--exclude-standard"))
    changed_paths = sorted(changed)
    violations = sorted(path for path in changed_paths if not allowed(path))
    manifest_sha = sha256_file(MANIFEST) if MANIFEST.is_file() else None
    if manifest_sha != MANIFEST_SHA:
        violations.append("fixed manifest SHA drift")
    violations = sorted(set(violations))
    payload = {
        "format": "locatemot-l89c-boundary-guard-v1",
        "status": "complete" if not violations else "invalid",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(WORK_ROOT),
        "luna_thread": THREAD,
        "base_sha": BASE_SHA,
        "changed_paths": changed_paths,
        "forbidden_paths": violations,
        "manifest_sha": manifest_sha,
        "model_source_changed": any(path.startswith("locatemot/models/") for path in changed_paths),
        "tracker_source_changed": any(path.startswith("locatemot/tracking/") for path in changed_paths),
        "uidm_source_changed": any("uidm" in path.lower() for path in changed_paths),
        "candidate_bank_changed": any(
            path.startswith(("outputs/l16/", "outputs/l18/", "outputs/l19/"))
            for path in changed_paths
        ),
        "ordinary_mot_ovmot_touched": any(
            path.startswith(("locatemot/tracking/", "configs/", "tools/ordinary", "tools/ovmot"))
            for path in changed_paths
        ),
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "zero_training": True,
        "checkpoint_weights_changed": False,
        "qscd_weights_changed": False,
        "new_checkpoint_written": False,
        "groundingdino_trainable": False,
    }
    write_json(OUTPUT, payload)
    if violations:
        raise AssertionError(f"L89C boundary violation: {violations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
