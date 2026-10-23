#!/usr/bin/env python3
"""Collect the two user-requested parallel controls and their retries.

This report is separate from the main G2 collection because these controls are
cross-model comparisons: G1/DanceTrack and G200/AnimalTrack plus the queued
G200 OVMOT recovery. Failed or partial attempts are retained as status evidence
but never promoted to metric entries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
DEFAULT_OWNERS = (
    RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_shared_gpu5_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_shared_gpu5_retry.queue_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_shared_gpu5_v2/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_animaltrack_shared_gpu7_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_ovmot_dedicated_gpu7_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_ovmot_dedicated_gpu7_retry_v2.queue_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_ovmot_dedicated_gpu7_v2/owner_state.json",
)


def read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256(path: Path | None, cache: dict[str, str | None]) -> str | None:
    if path is None or not path.is_file():
        return None
    key = str(path)
    if key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    cache[key] = digest.hexdigest()
    return cache[key]


def artifact(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_dir():
        path = path / "model.safetensors"
    return path


def collect(path: Path, hashes: dict[str, str | None]) -> dict[str, Any]:
    state = read(path)
    if state is None:
        return {"owner_state": str(path), "status": "MISSING", "accepted": {}}
    completed = state.get("completed") or {}
    if isinstance(completed, dict) and "frames" in completed:
        completed = {state.get("current_scope", state.get("scope", state.get("label", "completed"))): completed}
    accepted: dict[str, Any] = {}
    for name, item in completed.items() if isinstance(completed, dict) else ():
        if not isinstance(item, dict):
            continue
        metrics = item.get("official_metrics")
        result = item.get("result")
        frames = item.get("frames")
        if not isinstance(metrics, dict) or not isinstance(result, dict) or not isinstance(frames, int):
            continue
        model = item.get("model") or state.get("model")
        manifest = item.get("manifest") or state.get("manifest")
        accepted[str(name)] = {
            "task": item.get("task") or state.get("task"),
            "model": model,
            "model_sha256": sha256(artifact(model), hashes),
            "manifest": manifest,
            "manifest_sha256": sha256(Path(manifest) if manifest else None, hashes),
            "frames": frames,
            "official_metrics": metrics,
            "result_status": result.get("status"),
        }
    return {
        "owner_state": str(path),
        "status": state.get("status"),
        "current_stage": state.get("current_stage"),
        "physical_gpu": state.get("physical_gpu"),
        "model": state.get("model"),
        "manifest": state.get("manifest"),
        "failed": state.get("failed"),
        "accepted": accepted,
        "claim_boundary": "Only accepted entries have complete official receipts; failed/partial entries remain status evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--owner", type=Path, action="append", default=[])
    parser.add_argument("--require-terminal", action="store_true")
    args = parser.parse_args()
    owners = args.owner or list(DEFAULT_OWNERS)
    hashes: dict[str, str | None] = {}
    records = [collect(path, hashes) for path in owners]
    report = {
        "schema": "groundjev.extra_shared_controls.receipt_collection.v1",
        "project_root": "/data2/user/LocateMOT",
        "out": str(args.out),
        "owners": records,
        "accepted_scope_count": sum(len(record["accepted"]) for record in records),
        "terminal_owner_count": sum(str(record.get("status", "")).endswith(("COMPLETE", "FAILED")) for record in records),
        "owner_count": len(records),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("out", "accepted_scope_count", "terminal_owner_count", "owner_count")}, ensure_ascii=False))
    if args.require_terminal and report["terminal_owner_count"] != len(records):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
