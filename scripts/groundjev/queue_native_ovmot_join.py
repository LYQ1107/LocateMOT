#!/usr/bin/env python3
"""Wait for the complete native TAO stream and build the real joint source.

This owner is CPU-only.  It never opens a model service and never launches an
optimizer.  Once the strict eight-shard merge receipt exists, it performs the
offline TAO TRAIN geometry/category join and then rebuilds the three-task
manifest with native OVMOT rows.  Partial outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
MERGED_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_merged_v1"
JOIN_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_join_v1"
JOINT_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/real_three_task_joint_native_v1"
# These preparation entrypoints are repository code.  The runtime tree holds
# the data, OneJev environment and long-lived outputs, but does not mirror all
# source scripts.  Resolve the joiners from the checked-out LocateMOT project
# so the owner cannot fail only because a runtime copy was never staged.
JOIN_SCRIPT = PROJECT_ROOT / "tools/groundjev/prepare_native_ovmot_train_join.py"
JOINT_SCRIPT = PROJECT_ROOT / "tools/groundjev/prepare_real_three_task_joint_manifest.py"
PYTHON = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed", "completed")}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--queue-root", type=Path, default=RUNTIME_ROOT / "outputs/groundjev/training_data/native_ovmot_join_queue_v1")
    args = parser.parse_args()
    owner = args.queue_root / "owner_state.json"
    old = read_json(owner)
    if old and str(old.get("status", "")).endswith(("_COMPLETE", "_FAILED")):
        raise FileExistsError(f"terminal join owner already exists: {owner}")
    if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
        raise FileExistsError(f"join owner already live: {old['owner_pid']}")
    state = {
        "status": "WAITING_COMPLETE_NATIVE_TAO_STREAM",
        "current_stage": "WAITING_COMPLETE_NATIVE_TAO_STREAM",
        "owner_pid": os.getpid(),
        "merged_root": str(MERGED_ROOT),
        "join_root": str(JOIN_ROOT),
        "joint_root": str(JOINT_ROOT),
        "gpu_launched": False,
        "optimizer_updates": 0,
        "native_ovmot_training_rows": False,
        "completed": {},
        "failed": None,
    }
    save(owner, state)
    try:
        while True:
            receipt = read_json(MERGED_ROOT / "receipt.json")
            if receipt is None:
                state.update(status="WAITING_COMPLETE_NATIVE_TAO_STREAM", current_stage="WAITING_COMPLETE_NATIVE_TAO_STREAM")
                save(owner, state)
                time.sleep(args.poll_seconds)
                continue
            if receipt.get("status") != "COMPLETE_FULL_NATIVE_TAO_TRAIN_CANDIDATE_STREAM":
                state.update(status="NATIVE_TAO_STREAM_FAILED", current_stage="FAILED", failed=receipt)
                save(owner, state)
                return 2
            break
        if JOIN_ROOT.exists() or JOINT_ROOT.exists():
            raise FileExistsError("refusing to overwrite a previous native join or joint manifest")
        state.update(status="JOINING_TAO_TRAIN_ANNOTATIONS_OFFLINE", current_stage="JOINING_TAO_TRAIN_ANNOTATIONS_OFFLINE")
        save(owner, state)
        JOIN_ROOT.parent.mkdir(parents=True, exist_ok=True)
        join_log = args.queue_root / "NATIVE_OVMOT_JOIN.log"
        command = [str(PYTHON), str(JOIN_SCRIPT), "--candidates", str(MERGED_ROOT / "frame_candidates.jsonl"), "--out", str(JOIN_ROOT)]
        with join_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"native OVMOT join exited {result.returncode}; see {join_log}")
        join_manifest = read_json(JOIN_ROOT / "manifest.json")
        if join_manifest is None or join_manifest.get("native_ovmot_training_rows") is not True or join_manifest.get("ovmot_training_mode") != "native":
            raise RuntimeError("native OVMOT join did not produce a full native training manifest")
        state["completed"]["native_ovmot_join"] = join_manifest
        save(owner, state)
        state.update(status="BUILDING_NATIVE_THREE_TASK_MANIFEST", current_stage="BUILDING_NATIVE_THREE_TASK_MANIFEST")
        save(owner, state)
        joint_log = args.queue_root / "NATIVE_THREE_TASK_MANIFEST.log"
        command = [str(PYTHON), str(JOINT_SCRIPT), "--out", str(JOINT_ROOT), "--ovmot-native-manifest", str(JOIN_ROOT / "manifest.json")]
        with joint_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"native three-task manifest exited {result.returncode}; see {joint_log}")
        joint_manifest = read_json(JOINT_ROOT / "train.sources.json")
        joint_receipt = read_json(JOINT_ROOT / "manifest_receipt.json")
        if joint_manifest is None or joint_receipt is None or joint_manifest.get("native_ovmot_training_rows") is not True or joint_manifest.get("ovmot_training_mode") != "native":
            raise RuntimeError("native three-task manifest did not pass the native OVMOT gate")
        state["completed"]["native_three_task_manifest"] = {
            "manifest": str(JOINT_ROOT / "train.sources.json"),
            "manifest_receipt": str(JOINT_ROOT / "manifest_receipt.json"),
            "manifest_sha256": joint_receipt.get("manifest_sha256"),
            "rows": joint_manifest.get("rows"),
            "rows_per_task": joint_manifest.get("rows_per_task"),
        }
        state.update(status="NATIVE_OVMOT_JOIN_AND_THREE_TASK_MANIFEST_COMPLETE", current_stage="COMPLETE",
                     native_ovmot_training_rows=True, gpu_launched=False, optimizer_updates=0)
        save(owner, state)
        return 0
    except Exception as error:
        state.update(status="NATIVE_OVMOT_JOIN_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner, state)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
