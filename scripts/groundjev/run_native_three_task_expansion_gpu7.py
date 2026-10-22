#!/usr/bin/env python3
"""Run a recoverable single-GPU native GroundJEV three-task expansion.

This owner is an additional real MOT/RMOT/OVMOT readout run.  It uses the
protected final G2 Clean adapter, one frozen visual backbone, the audited
native source manifest, and an isolated output directory on physical GPU7.
It does not touch the existing G0/G2 held-out owners or the 240-step pilot.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
PYTHON = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"
RUNNER = RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"
TRAINER = PROJECT_ROOT / "tools/groundjev/coupled/train_native_three_task.py"
MANIFEST = RUNTIME_ROOT / "outputs/groundjev/training_data/real_three_task_joint_native_v1/train.sources.json"
CHECKPOINT = RUNTIME_ROOT / "models/GroundAnything-DLM"
ADAPTER = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final/adapter_projector_final.pt")
SOURCE_ROOT = RUNTIME_ROOT / "external/GroundAnything"
OUTPUT = Path("/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_expansion_gpu7_v2")
OWNER = RUNTIME_ROOT / "outputs/groundjev/training/groundjev_native_three_task_expansion_gpu7_v2/owner_state.json"
LOG = RUNTIME_ROOT / "outputs/groundjev/training/groundjev_native_three_task_expansion_gpu7_v2/TRAINING.log"
MANIFEST_SHA = "cc632115aa2b87161dc7b8413d08c3c471e1225179559a76509e7366190764c1"
MODEL_SHA = "3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2"
ROWS_PER_TASK = 72
STEPS = 1200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save(state: dict) -> None:
    OWNER.parent.mkdir(parents=True, exist_ok=True)
    temporary = OWNER.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(OWNER)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed", "optimizer_updates")}, ensure_ascii=False), flush=True)


def main() -> int:
    old = read_json(OWNER)
    if old and str(old.get("status", "")).endswith(("COMPLETE", "FAILED")):
        raise FileExistsError(f"terminal expansion owner already exists: {OWNER}")
    if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
        raise FileExistsError(f"expansion owner already live: {old['owner_pid']}")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite expansion output: {OUTPUT}")
    if not MANIFEST.is_file() or sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("native manifest missing or SHA mismatch")
    for path in (CHECKPOINT, ADAPTER, SOURCE_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    command = [
        str(PYTHON), str(RUNNER), "--gpus", "7", "--", str(PYTHON), str(TRAINER),
        "--manifest", str(MANIFEST), "--checkpoint", str(CHECKPOINT),
        "--adapter", str(ADAPTER), "--runtime-root", str(RUNTIME_ROOT),
        "--source-root", str(SOURCE_ROOT), "--model-sha256", MODEL_SHA,
        "--output", str(OUTPUT), "--device", "cuda:0",
        "--rows-per-task", str(ROWS_PER_TASK), "--steps", str(STEPS),
        "--max-native-boxes", "16", "--native-forward-chunk-size", "64",
        "--seed", "20261022",
    ]
    state = {
        "status": "GPU7_NATIVE_THREE_TASK_EXPANSION_RUNNING",
        "current_stage": "LAUNCH",
        "owner_pid": os.getpid(),
        "physical_gpu": 7,
        "manifest": str(MANIFEST),
        "manifest_sha256": MANIFEST_SHA,
        "checkpoint": str(CHECKPOINT),
        "adapter": str(ADAPTER),
        "adapter_step": 18227,
        "model_sha256": MODEL_SHA,
        "output": str(OUTPUT),
        "rows_per_task": ROWS_PER_TASK,
        "steps_requested": STEPS,
        "g2_frozen": True,
        "single_backbone": True,
        "old_pilot_preserved": "/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_v1/receipt.json",
        "command": command,
        "optimizer_updates": 0,
        "gpu_launched": False,
        "failed": None,
    }
    save(state)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("w", encoding="utf-8") as stream:
            state.update(current_stage="TRAINING", gpu_launched=True)
            save(state)
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        receipt = read_json(OUTPUT / "receipt.json")
        if result.returncode != 0 or not receipt or receipt.get("status") != "GROUNDJEV_NATIVE_THREE_TASK_COUPLED_READOUT_COMPLETE":
            state.update(status="GPU7_NATIVE_THREE_TASK_EXPANSION_FAILED", current_stage="FAILED", failed={"returncode": result.returncode, "receipt": receipt})
            save(state)
            return result.returncode or 1
        counts = receipt.get("optimizer_updates_by_task") or {}
        if any(int(counts.get(task, 0)) <= 0 for task in ("mot", "rmot", "ovmot")):
            raise RuntimeError(f"missing per-task optimizer updates: {counts}")
        state.update(status="GPU7_NATIVE_THREE_TASK_EXPANSION_COMPLETE", current_stage="COMPLETE", optimizer_updates=int(receipt.get("optimizer_steps_requested", 0)), receipt=str(OUTPUT / "receipt.json"), task_counts=receipt.get("task_counts"), optimizer_updates_by_task=counts, peak_memory_allocated_bytes=receipt.get("peak_memory_allocated_bytes"))
        save(state)
        return 0
    except Exception as error:
        state.update(status="GPU7_NATIVE_THREE_TASK_EXPANSION_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
