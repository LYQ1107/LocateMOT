#!/usr/bin/env python3
"""Build the fixed R1 fit request schedule without constructing features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r1_common import (  # noqa: E402
    SEED,
    THREAD,
    check_manifest,
    load_indexes,
    provenance_inputs,
    build_request_manifest,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--tiles-per-epoch", type=int, default=3000)
    parser.add_argument("--max-queries", type=int, default=8)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 request output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"R1 request builder must run from {WORK_ROOT}, got {Path.cwd()}")
    if int(args.epochs) != 6 or int(args.tiles_per_epoch) != 3000 or int(args.max_queries) != 8:
        raise ValueError("R1 request schedule is fixed at 6 x 3000 with Q<=8")
    manifest_sha = check_manifest()
    indexes = load_indexes("fit")
    payload = build_request_manifest(indexes, epochs=args.epochs, tiles_per_epoch=args.tiles_per_epoch,
                                     seed=SEED, max_queries=args.max_queries)
    payload["command"] = " ".join([str(sys.executable), *sys.argv])
    payload["cwd"] = str(Path.cwd().resolve())
    payload["thread"] = THREAD
    payload["manifest_sha256"] = manifest_sha
    payload["inputs"] = provenance_inputs()
    payload["outputs"] = {"request_manifest": str(out / "request_manifest.json")}
    payload["failure_root_cause"] = None
    payload["next_action"] = "audit request schedule, then build only label-free aligned states"
    write_json(out / "request_manifest.json", payload)
    write_json(out / "provenance.json", {
        "format": "locatemot-r1-request-provenance-v1", "status": "complete", "command": payload["command"],
        "cwd": payload["cwd"], "thread": THREAD, "seed": SEED, "manifest_sha256": manifest_sha,
        "inputs": payload["inputs"], "outputs": payload["outputs"], "labels_in_request": False,
        "failure_root_cause": None, "next_action": payload["next_action"],
    })
    write_json(out / "status.json", {
        "format": "locatemot-r1-request-status-v1", "status": "complete", "command": payload["command"],
        "inputs": payload["inputs"], "outputs": payload["outputs"], "failure_root_cause": None,
        "next_action": payload["next_action"], "thread": THREAD, "manifest_sha256": manifest_sha,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
