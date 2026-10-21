#!/usr/bin/env python3
"""Queue the real three-task GroundJEV readout behind the protected G2 run.

This owner replaces the paused independent OneJev SFT owner.  It waits for
the native TAO join manifest and the exact G2 Clean terminal receipt, then
uses one G2 model plus one shared typed readout on GPU4.  GPU5 remains reserved
for the registered G1 comparison and GPU7 for the independent held-out gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
G2_ROOT = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final")
NATIVE_MANIFEST = RUNTIME_ROOT / "outputs/groundjev/training_data/real_three_task_joint_native_v1/train.sources.json"
NATIVE_RECEIPT = RUNTIME_ROOT / "outputs/groundjev/training_data/real_three_task_joint_native_v1/manifest_receipt.json"
BUNDLE_STATE = RUNTIME_ROOT / "outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1/owner_state.json"
HELDOUT_STATE = RUNTIME_ROOT / "outputs/groundjev/heldout/g2_clean_independent_gpu7_v1/owner_state.json"
QUEUE_ROOT = RUNTIME_ROOT / "outputs/groundjev/training/groundjev_native_three_task_v1_queue"
TRAIN_ROOT = Path("/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_v1")
PYTHON = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"
RESOURCE_RUNNER = RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"
TRAINER = PROJECT_ROOT / "tools/groundjev/coupled/train_native_three_task.py"
CHECKPOINT = RUNTIME_ROOT / "models/GroundAnything-DLM"
SOURCE_ROOT = RUNTIME_ROOT / "external/GroundAnything"
GPU = 4
MIN_FREE_MIB = 30_000
G2_STEPS = 18_227
G2_MANIFEST_SHA = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
MODEL_SHA = "3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(json.dumps({key: state.get(key) for key in ("status", "current_stage", "optimizer_updates", "failed")}, ensure_ascii=False), flush=True)


def g2_terminal() -> tuple[str, dict | None]:
    receipt = read_json(G2_ROOT / "final_receipt.json")
    if receipt is None:
        contract = read_json(G2_ROOT / "trainer_contract.json")
        if contract is not None and contract.get("status") not in {"G2_FORMAL_TRAINING_RUNNING"}:
            return "failed", contract
        return "waiting", None
    valid = (
        receipt.get("status") == "G2_EXACT_COVERAGE_TRAINING_COMPLETE"
        and int(receipt.get("steps", -1)) == G2_STEPS
        and receipt.get("manifest_sha256") == G2_MANIFEST_SHA
        and (G2_ROOT / "adapter_projector_final.pt").is_file()
    )
    return ("complete", receipt) if valid else ("failed", receipt)


def native_manifest() -> tuple[str, dict | None]:
    manifest = read_json(NATIVE_MANIFEST)
    receipt = read_json(NATIVE_RECEIPT)
    if manifest is None or receipt is None:
        return "waiting", None
    if manifest.get("native_ovmot_training_rows") is not True or manifest.get("ovmot_training_mode") != "native":
        return "failed", {"manifest": manifest, "receipt": receipt, "reason": "native OVMOT gate is false"}
    if manifest.get("heldout_exclusion", {}).get("MOT20-05") is None:
        return "failed", {"manifest": manifest, "reason": "held-out exclusion audit is missing"}
    if receipt.get("manifest_sha256") != sha256(NATIVE_MANIFEST):
        return "failed", {"manifest": manifest, "receipt": receipt, "reason": "manifest SHA mismatch"}
    if int(manifest.get("rows", 0)) <= 0:
        return "failed", {"manifest": manifest, "receipt": receipt, "reason": "empty native manifest"}
    return "complete", {"manifest": manifest, "receipt": receipt}


def dependency(path: Path, accepted: set[str]) -> tuple[str, dict | None]:
    value = read_json(path)
    if value is None:
        return "waiting", None
    status = str(value.get("status", ""))
    if status in accepted:
        return "complete", value
    if status.endswith("FAILED") or status.endswith("DEPENDENCY_FAILED"):
        return "failed", value
    return "waiting", value


def gpu_ready(gpu: int) -> tuple[bool, dict]:
    apps = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    free = int(subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0])
    pids = [line.strip() for line in apps.splitlines() if line.strip()]
    # The research task limits our allocation to one physical GPU, but does
    # not require killing unrelated jobs.  A co-resident launch is safe when
    # the measured free memory is comfortably above the readout pilot's
    # requirement; keep the foreign PID evidence in the owner receipt.
    return (free >= MIN_FREE_MIB), {
        "compute_pids": pids,
        "free_memory_MiB": free,
        "minimum_free_memory_MiB": MIN_FREE_MIB,
        "co_resident_foreign_apps": bool(pids),
        "resource_policy": "one physical GPU; co-resident allowed when free memory threshold passes; foreign jobs untouched",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, default=QUEUE_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--rows-per-task", type=int, default=24)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--max-native-boxes", type=int, default=16)
    parser.add_argument("--native-forward-chunk-size", type=int, default=64,
                        help="restart the frozen native G2 provider after this many unique inputs")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="physical GPU for the one-card pilot; G2 formal training is never touched")
    parser.add_argument("--training-root", type=Path, default=TRAIN_ROOT,
                        help="separate checkpoint/receipt directory for this coupled run")
    args = parser.parse_args()
    owner = args.queue_root / "owner_state.json"
    previous = read_json(owner)
    if previous and previous.get("owner_pid") and Path(f"/proc/{previous['owner_pid']}").exists():
        raise FileExistsError(f"GroundJEV three-task owner already live: {previous['owner_pid']}")
    if previous and str(previous.get("status", "")).endswith(("_COMPLETE", "_FAILED")):
        raise FileExistsError(f"terminal GroundJEV queue already exists: {args.queue_root}")
    args.queue_root.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "WAITING_G2_NATIVE_MANIFEST",
        "current_stage": "WAITING_G2_NATIVE_MANIFEST",
        "owner_pid": os.getpid(),
        "g2_root": str(G2_ROOT),
        "g2_manifest_sha256": G2_MANIFEST_SHA,
        "native_manifest": str(NATIVE_MANIFEST),
        "physical_gpu": args.gpu,
        "max_native_boxes": args.max_native_boxes,
        "native_forward_chunk_size": args.native_forward_chunk_size,
        "training_root": str(args.training_root),
        "checkpoint": str(CHECKPOINT),
        "source_root": str(SOURCE_ROOT),
        "optimizer_updates": 0,
        "gpu_launched": False,
        "failed": None,
        "completed": {},
        "old_independent_onejev_queue_paused": True,
        "resumed_wait_owner": bool(previous),
    }
    save(owner, state)
    try:
        while True:
            g2, g2e = g2_terminal()
            native, native_e = native_manifest()
            state["dependency_snapshot"] = {"g2": g2, "native_manifest": native}
            if g2 == "failed" or native == "failed":
                state.update(status="GROUNDJEV_NATIVE_THREE_TASK_DEPENDENCY_FAILED", current_stage="FAILED",
                             failed={"g2": g2e, "native_manifest": native_e})
                save(owner, state)
                return 2
            if g2 == "complete" and native == "complete":
                break
            state.update(status="WAITING_G2_NATIVE_MANIFEST", current_stage="WAITING_G2_NATIVE_MANIFEST")
            save(owner, state)
            time.sleep(args.poll_seconds)
        state["completed"]["g2"] = g2e
        state["completed"]["native_manifest"] = native_e
        state["manifest_sha256"] = native_e["receipt"]["manifest_sha256"]
        while True:
            ready, probe = gpu_ready(args.gpu)
            state["resource_probe"] = probe
            if ready:
                break
            state.update(status=f"WAITING_GPU{args.gpu}_RESOURCE", current_stage=f"WAITING_GPU{args.gpu}_RESOURCE")
            save(owner, state)
            time.sleep(args.poll_seconds)
        output = args.training_root
        command = [
            str(PYTHON), str(RESOURCE_RUNNER), "--gpus", str(args.gpu), "--", str(PYTHON), str(TRAINER),
            "--manifest", str(NATIVE_MANIFEST), "--checkpoint", str(CHECKPOINT),
            "--adapter", str(G2_ROOT / "adapter_projector_final.pt"),
            "--runtime-root", str(RUNTIME_ROOT), "--source-root", str(SOURCE_ROOT),
            "--model-sha256", MODEL_SHA, "--output", str(output), "--device", "cuda:0",
            "--rows-per-task", str(args.rows_per_task), "--steps", str(args.steps),
            "--max-native-boxes", str(args.max_native_boxes),
            "--native-forward-chunk-size", str(args.native_forward_chunk_size),
        ]
        state.update(status="GROUNDJEV_NATIVE_THREE_TASK_TRAINING_RUNNING",
                     current_stage="GROUNDJEV_NATIVE_THREE_TASK_TRAINING_RUNNING",
                     command=command, gpu_launched=True)
        save(owner, state)
        log = args.queue_root / "TRAINING.log"
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=RUNTIME_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        receipt = read_json(output / "receipt.json")
        if result.returncode != 0 or receipt is None or receipt.get("status") != "GROUNDJEV_NATIVE_THREE_TASK_COUPLED_READOUT_COMPLETE":
            state.update(status="GROUNDJEV_NATIVE_THREE_TASK_TRAINING_FAILED", current_stage="FAILED",
                         failed={"returncode": result.returncode, "receipt": receipt})
            save(owner, state)
            return result.returncode or 3
        state["completed"]["training"] = receipt
        state.update(status="GROUNDJEV_NATIVE_THREE_TASK_TRAINING_COMPLETE", current_stage="COMPLETE",
                     optimizer_updates=sum(receipt.get("optimizer_updates_by_task", {}).values()), full_goal_complete=False)
        save(owner, state)
        return 0
    except Exception as error:
        state.update(status="GROUNDJEV_NATIVE_THREE_TASK_TRAINING_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
