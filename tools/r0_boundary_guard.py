#!/usr/bin/env python3
"""Guard the R0 sidecar boundary against accidental legacy edits.

The guard is intentionally small: it checks the worktree relative to the
registered L89E base and the current index/untracked set.  R0 is a sidecar,
so no existing production source or historical artifact may be part of its
change set.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_SHA = "7fa201e4967f71a42ba669e90f330a6e3979e1cd"
ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EXACT = {
    "locatemot/models/r0_track_grounding.py",
    "locatemot/rmot/r0_visual_tokens.py",
    "locatemot/rmot/r0_dense_data.py",
    "locatemot/rmot/r0_geometry.py",
    "locatemot/rmot/r0_losses.py",
    "locatemot/rmot/r0_safe_target_source.py",
}
ALLOWED_PREFIXES = ("tools/r0_", "reports/r0/", "outputs/r0/")


def _git(*args: str) -> list[str]:
    value = subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT
    )
    return [line.strip() for line in value.splitlines() if line.strip()]


def allowed(path: str) -> bool:
    return path == "research_log.md" or path in ALLOWED_EXACT or any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def main() -> int:
    if not (ROOT / ".git").exists() and not (ROOT / ".git").is_file():
        print(f"R0_BOUNDARY_FAIL: not a git worktree: {ROOT}")
        return 1
    changed: set[str] = set()
    commands = [
        ("base_to_head", ("diff", "--name-only", f"{BASE_SHA}...HEAD")),
        ("worktree", ("diff", "--name-only")),
        ("cached", ("diff", "--cached", "--name-only")),
        ("untracked", ("ls-files", "--others", "--exclude-standard")),
    ]
    failures: list[tuple[str, str]] = []
    for label, args in commands:
        try:
            paths = _git(*args)
        except subprocess.CalledProcessError as exc:
            print(f"R0_BOUNDARY_FAIL: {label}: {exc.output.strip()}")
            return 1
        for path in paths:
            changed.add(path)
            if not allowed(path):
                failures.append((label, path))
    print(f"base_sha={BASE_SHA}")
    print(f"worktree={ROOT}")
    print(f"changed_path_count={len(changed)}")
    if failures:
        print("R0_BOUNDARY_FAIL")
        for label, path in failures:
            print(f"{label}: {path}")
        return 1
    print("R0_BOUNDARY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
