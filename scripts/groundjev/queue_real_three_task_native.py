#!/usr/bin/env python3
"""Launch the real native MOT/RMOT/OVMOT OneJev run after all gates.

This owner is distinct from the paused teacher-control queue.  It waits for
the CPU native-TAO join owner, the terminal G2/G1/bundle/held-out receipts and
an idle four-GPU window, then writes a run-specific config and launches one
distributed source-manifest SFT.  The source row count and SHA are read from
the native manifest at runtime; no old 131,990-row teacher manifest is reused.
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
NATIVE_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/real_three_task_joint_native_v1"
MANIFEST = NATIVE_ROOT / "train.sources.json"
MANIFEST_RECEIPT = NATIVE_ROOT / "manifest_receipt.json"
BASE_CONFIG = PROJECT_ROOT / "configs/groundjev/real_three_task_joint_v2_compact_runtime.json"
QUEUE_ROOT = RUNTIME_ROOT / "outputs/groundjev/training/real_three_task_joint_native_v1_queue"
TRAIN_ROOT = Path("/data3/user/LocateMOT-checkpoints/onejev_real_three_task_joint_native_v1")
CONFIG_OUT = QUEUE_ROOT / "native_runtime.json"
G2_ROOT = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final")
G1_STATE = RUNTIME_ROOT / "outputs/groundjev/ground_adaptation/native_tracking200_gpu5_g1_v1/queue_owner_state.json"
BUNDLE_STATE = RUNTIME_ROOT / "outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1/owner_state.json"
HELDOUT_STATE = RUNTIME_ROOT / "outputs/groundjev/heldout/g2_clean_independent_gpu7_v1/owner_state.json"
PYTHON = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"
RUNNER = RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"
SFT = RUNTIME_ROOT / "tools/groundjev/run_source_manifest_sft.py"
GPUS = (4, 5, 6, 7)
G2_MANIFEST_SHA = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
G2_STEPS = 18227


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed", "optimizer_updates")}, ensure_ascii=False), flush=True)


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


def g2_status() -> tuple[str, dict | None]:
    value = read_json(G2_ROOT / "final_receipt.json")
    if value is None:
        return "waiting", None
    valid = value.get("status") == "G2_EXACT_COVERAGE_TRAINING_COMPLETE" and int(value.get("steps", -1)) == G2_STEPS and value.get("manifest_sha256") == G2_MANIFEST_SHA
    return ("complete", value) if valid else ("failed", value)


def native_status() -> tuple[str, dict | None]:
    manifest = read_json(MANIFEST)
    receipt = read_json(MANIFEST_RECEIPT)
    if manifest is None or receipt is None:
        return "waiting", None
    if manifest.get("native_ovmot_training_rows") is not True or manifest.get("ovmot_training_mode") != "native":
        return "failed", {"reason": "native OVMOT gate is false", "manifest": manifest}
    if int(manifest.get("rows", -1)) <= 0 or int(receipt.get("optimizer_updates", -1)) != 0 or receipt.get("gpu_launched") is not False:
        return "failed", {"reason": "native manifest receipt contract mismatch", "receipt": receipt}
    if receipt.get("manifest_sha256") != sha256(MANIFEST):
        return "failed", {"reason": "native manifest SHA mismatch", "receipt": receipt}
    return "complete", {"manifest": manifest, "receipt": receipt}


def devices_ready() -> tuple[bool, dict]:
    probes = {}
    ready = True
    for gpu in GPUS:
        apps = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        free = int(subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[0])
        pids = [line.strip() for line in apps.splitlines() if line.strip()]
        probes[str(gpu)] = {"compute_apps": pids, "free_memory_MiB": free}
        ready = ready and not pids and free >= 32768
    return ready, probes


def build_config(rows: int) -> dict:
    config = json.loads(BASE_CONFIG.read_text())
    config.update({
        "train": str(MANIFEST),
        "name": "onejev_real_three_task_joint_native_v1",
        "runs_dir": str(TRAIN_ROOT.parent),
        "max_steps": 0,
        "epochs": 1,
    })
    config["native_manifest_rows"] = rows
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--queue-root", type=Path, default=QUEUE_ROOT)
    args = parser.parse_args()
    owner = args.queue_root / "owner_state.json"
    old = read_json(owner)
    if old and str(old.get("status", "")).endswith(("_COMPLETE", "_FAILED")):
        raise FileExistsError(f"terminal native joint owner already exists: {owner}")
    if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
        raise FileExistsError(f"native joint owner already live: {old['owner_pid']}")
    state = {
        "status": "WAITING_NATIVE_OVMOT_MANIFEST",
        "current_stage": "WAITING_NATIVE_OVMOT_MANIFEST",
        "owner_pid": os.getpid(),
        "manifest": str(MANIFEST),
        "native_manifest_sha256": None,
        "physical_gpus": list(GPUS),
        "training_root": str(TRAIN_ROOT),
        "config": str(CONFIG_OUT),
        "optimizer_updates": 0,
        "gpu_launched": False,
        "old_teacher_queue_preserved": True,
        "completed": {},
        "failed": None,
    }
    save(owner, state)
    try:
        while True:
            native, ne = native_status()
            if native == "failed":
                state.update(status="NATIVE_JOINT_MANIFEST_FAILED", current_stage="FAILED", failed=ne)
                save(owner, state)
                return 2
            if native != "complete":
                state.update(status="WAITING_NATIVE_OVMOT_MANIFEST", current_stage="WAITING_NATIVE_OVMOT_MANIFEST")
                save(owner, state)
                time.sleep(args.poll_seconds)
                continue
            break
        manifest = ne["manifest"]
        rows = int(manifest["rows"])
        config = build_config(rows)
        if CONFIG_OUT.exists() and json.loads(CONFIG_OUT.read_text()) != config:
            raise FileExistsError(f"refusing to overwrite a different config: {CONFIG_OUT}")
        args.queue_root.mkdir(parents=True, exist_ok=True)
        CONFIG_OUT.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        state["native_manifest_sha256"] = sha256(MANIFEST)
        state["rows"] = rows
        state["rows_per_task"] = manifest.get("rows_per_task")
        state["completed"]["native_manifest"] = ne["receipt"]
        state.update(status="WAITING_G2_G1_BUNDLE_HELDOUT_AND_GPUS", current_stage="WAITING_G2_G1_BUNDLE_HELDOUT_AND_GPUS")
        save(owner, state)
        while True:
            g2, ge = g2_status()
            g1, g1e = dependency(G1_STATE, {"G1_GPU5_ADAPTATION_AND_CONTROLS_COMPLETE"})
            bundle, be = dependency(BUNDLE_STATE, {"G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED", "G2_CLEAN_BUNDLE_EXPORT_COMPLETE"})
            heldout, he = dependency(HELDOUT_STATE, {"G2_CLEAN_FULL_INDEPENDENT_HELDOUT_COMPLETE"})
            snapshot = {"g2": g2, "g1": g1, "bundle": bundle, "heldout": heldout}
            state["dependency_snapshot"] = snapshot
            if any(item == "failed" for item in (g2, g1, bundle, heldout)):
                state.update(status="NATIVE_JOINT_DEPENDENCY_FAILED", current_stage="FAILED", failed={"snapshot": snapshot, "evidence": {"g2": ge, "g1": g1e, "bundle": be, "heldout": he}})
                save(owner, state)
                return 3
            if all(item == "complete" for item in (g2, g1, bundle, heldout)):
                ready, probes = devices_ready()
                state["resource_probe"] = probes
                if ready:
                    state["completed"].update({"g2": ge, "g1": g1e, "bundle": be, "heldout": he})
                    break
            save(owner, state)
            time.sleep(args.poll_seconds)
        command = [str(RUNNER), "--gpus", *map(str, GPUS), "--", str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4", str(SFT), "--source-training-backend", "fla", "--config", str(CONFIG_OUT), "--paired-index-shuffle", "--distributed-sft", "--deterministic-execution", "--seed-global-rngs"]
        state.update(status="REAL_NATIVE_THREE_TASK_TRAINING_RUNNING", current_stage="REAL_NATIVE_THREE_TASK_TRAINING_RUNNING", command=command, gpu_launched=True)
        save(owner, state)
        log_path = args.queue_root / "TRAINING.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=RUNTIME_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        records = []
        log_jsonl = TRAIN_ROOT / "log.jsonl"
        if log_jsonl.is_file():
            for line in log_jsonl.read_text(errors="ignore").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        done = next((item for item in reversed(records) if item.get("event") == "done"), None)
        state["terminal_record"] = done
        if result.returncode != 0 or done is None or int(done.get("examples", -1)) != rows:
            state.update(status="REAL_NATIVE_THREE_TASK_TRAINING_FAILED", current_stage="FAILED", failed={"returncode": result.returncode, "expected_rows": rows, "terminal_record": done})
            save(owner, state)
            return result.returncode or 4
        state.update(status="REAL_NATIVE_THREE_TASK_TRAINING_COMPLETE", current_stage="COMPLETE", optimizer_updates=int(done.get("steps", 0)), full_goal_complete=False)
        save(owner, state)
        return 0
    except Exception as error:
        state.update(status="REAL_NATIVE_THREE_TASK_TRAINING_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner, state)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
