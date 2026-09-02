#!/usr/bin/env python3
"""Complete compact, derived L81 fit artifact views without rerunning fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def complete(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    loss = json.loads((directory / "loss_trace.json").read_text())
    sampling = json.loads((directory / "sampling_trace.json").read_text())
    if len(loss) != len(sampling):
        raise AssertionError(f"trace length mismatch: {directory}")
    unit_path = directory / "unit_sequence.jsonl"
    log_path = directory / "train_log.jsonl"
    if unit_path.exists() or log_path.exists():
        raise FileExistsError(f"derived fit artifact already exists: {directory}")
    with unit_path.open("w") as handle:
        for row in sampling:
            handle.write(json.dumps({
                "format": "locatemot-l81-unit-sequence-v1", "step": row["step"],
                "unit_key": row["unit_key"], "dataset": row["dataset"], "video": row["video"],
                "frame_id": row["frame_id"], "category": row["category"],
                "candidate_count": row["candidate_count"], "positive_count": row["positive_count"],
                "candidate_key_digest": row["candidate_key_digest"],
                "candidate_deletion": row["candidate_deletion"],
                "candidate_truncation": row["candidate_truncation"],
            }, ensure_ascii=False) + "\n")
    with log_path.open("w") as handle:
        for row in loss:
            handle.write(json.dumps({"format": "locatemot-l81-train-log-v1", **row}, ensure_ascii=False) + "\n")
    milestones = []
    for checkpoint in sorted(directory.glob("checkpoint_l81_step*.pt")):
        package = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_state = package.get("model_state_dict", {})
        norm = 0.0
        for tensor in model_state.values():
            if torch.is_tensor(tensor):
                norm += float(tensor.float().pow(2).sum())
        milestones.append({
            "step": int(package.get("step", 0)), "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size,
            "parameter_norm": norm ** 0.5, "strict_reload_required": True,
        })
        del package
    audit = {
        "format": "locatemot-l81-fit-artifact-audit-v1", "status": "complete",
        "directory": str(directory.resolve()), "trace_steps": len(loss),
        "unit_sequence": str(unit_path.resolve()), "train_log": str(log_path.resolve()),
        "milestones": milestones, "derived_without_training": True,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
    }
    write_json(directory / "milestones.json", audit)
    write_json(directory / "artifact_audit.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", action="append", type=Path, required=True)
    args = parser.parse_args()
    results = [complete(path.resolve()) for path in args.directory]
    print(json.dumps({"status": "complete", "artifacts": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
