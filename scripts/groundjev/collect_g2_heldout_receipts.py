#!/usr/bin/env python3
"""Collect complete G2/G0 held-out receipts without promoting partial runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
DEFAULT_OWNERS = (
    RUNTIME_ROOT / "outputs/groundjev/heldout/g2_clean_independent_gpu3_v1/owner_state.json",
    RUNTIME_ROOT / "outputs/groundjev/heldout/g2_clean_independent_gpu5_v1/owner_state.json",
    RUNTIME_ROOT / "outputs/groundjev/heldout/g2_clean_independent_gpu6_v1/owner_state.json",
    RUNTIME_ROOT / "outputs/groundjev/heldout/g2_domain_dancetrack_gpu5_v1/owner_state.json",
    RUNTIME_ROOT / "outputs/groundjev/heldout/g2_domain_animaltrack_gpu6_v1/owner_state.json",
    RUNTIME_ROOT / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video63_v1/owner_state.json",
    RUNTIME_ROOT / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video63_repaired_v1/owner_state.json",
)
G2_BUNDLE = RUNTIME_ROOT / "models/GroundAnything-DLM-G2-CLEAN-V2/model.safetensors"
G2_MANIFEST = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "g2_clean_train_manifest.json"
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifact(path: str | None) -> Path | None:
    """Resolve a model bundle directory to its immutable weight file."""
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "model.safetensors"
    return candidate


def collect(path: Path) -> dict[str, Any]:
    state = read_json(path)
    if state is None:
        return {"owner_state": str(path), "status": "MISSING", "accepted": {}}
    accepted: dict[str, Any] = {}
    default_model = state.get("model") or state.get("g2_bundle")
    default_manifest = state.get("manifest")
    completed_root = state.get("completed") or {}
    # Older owners store one completed block under a scope key; the generic
    # parallel runner stores that same block directly at the owner level.
    # Normalize both shapes before applying the receipt acceptance contract.
    if isinstance(completed_root, dict) and (
        "frames" in completed_root
        and ("result" in completed_root or "official_metrics" in completed_root)
    ):
        completed_items = {
            state.get("current_scope", state.get("scope", state.get("label", "completed"))): completed_root
        }
    else:
        completed_items = completed_root
    for name, item in completed_items.items():
        if name == "g2_dependency" or not isinstance(item, dict):
            continue
        metrics = item.get("official_metrics")
        result = item.get("result")
        frames = item.get("frames")
        if isinstance(metrics, dict) and isinstance(result, dict) and isinstance(frames, int):
            model_path = item.get("model") or default_model
            manifest_path = item.get("manifest") or default_manifest
            accepted[name] = {
                "task": item.get("task") or state.get("task"),
                "model": model_path,
                "model_sha256": sha256(model_artifact(model_path)),
                "manifest": manifest_path,
                "manifest_sha256": sha256(Path(manifest_path)) if manifest_path else None,
                "frames": frames,
                "official_metrics": metrics,
                "result_status": result.get("status"),
            }
    model_file = model_artifact(default_model)
    return {
        "owner_state": str(path),
        "status": state.get("status"),
        "current_scope": state.get("current_scope", state.get("scope")),
        "physical_gpu": state.get("physical_gpu"),
        "failed": state.get("failed"),
        "model": default_model,
        "model_sha256": sha256(model_file) if model_file else None,
        "manifest": default_manifest,
        "manifest_sha256": sha256(Path(default_manifest)) if default_manifest else None,
        "accepted": accepted,
        "claim_boundary": "Only accepted entries have complete official receipts; partial or failed entries are status evidence only.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--owner", type=Path, action="append", default=[])
    args = parser.parse_args()
    owners = args.owner or list(DEFAULT_OWNERS)
    report = {
        "schema": "groundjev.g2.heldout.receipt_collection.v1",
        "project_root": "/data2/user/LocateMOT",
        "g2_bundle": str(G2_BUNDLE),
        "g2_bundle_sha256": sha256(G2_BUNDLE),
        "g2_train_manifest": str(G2_MANIFEST),
        "g2_train_manifest_sha256": sha256(G2_MANIFEST),
        "owners": [collect(path) for path in owners],
    }
    report["accepted_scope_count"] = sum(len(owner["accepted"]) for owner in report["owners"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "accepted_scope_count": report["accepted_scope_count"],
                      "owner_count": len(report["owners"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
