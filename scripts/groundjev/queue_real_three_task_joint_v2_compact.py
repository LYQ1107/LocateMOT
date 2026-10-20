#!/usr/bin/env python3
"""Queue a corrected MOT/RMOT/OVMOT OneJev comparison after protected gates.

This is a new run and never resumes or overwrites the paused historical
four-GPU source course.  It uses the audited 131,990-row manifest, but will
not launch an optimizer while the manifest contains only OVMOT Base
teacher-control rows.  The owner waits for native OVMOT training rows in
addition to the terminal G2/G1/bundle/held-out gates and an exclusive
four-GPU window before launching one registered distributed epoch.
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
G2_ROOT = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final")
G1_STATE = RUNTIME_ROOT / "outputs/groundjev/ground_adaptation/native_tracking200_gpu5_g1_v1/queue_owner_state.json"
BUNDLE_STATE = RUNTIME_ROOT / "outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1/owner_state.json"
HELDOUT_STATE = RUNTIME_ROOT / "outputs/groundjev/heldout/g2_clean_independent_gpu7_v1/owner_state.json"
MANIFEST = RUNTIME_ROOT / "outputs/groundjev/training_data/real_three_task_joint_v2_compact/train.sources.json"
MANIFEST_RECEIPT = MANIFEST.parent / "manifest_receipt.json"
CONFIG = PROJECT_ROOT / "configs/groundjev/real_three_task_joint_v2_compact_runtime.json"
QUEUE_ROOT = RUNTIME_ROOT / "outputs/groundjev/training/real_three_task_joint_v2_compact_queue"
TRAIN_ROOT = Path("/data3/user/LocateMOT-checkpoints/onejev_real_three_task_joint_v2_compact")
GPUS = (4, 5, 6, 7)
G2_STEPS = 18227
MANIFEST_SHA = "cbdafce9e5a587c00b053ee58ea6435db4221d38036ee7574fe8a3f0f854e54c"
ROWS = 131990
TRAINABLE_ROWS = 131990
OVERLENGTH_ROWS = 0


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed", "last_exit_code")}, ensure_ascii=False), flush=True)


def terminal_state(path: Path, complete_status: str | set[str], failed_suffix: str | set[str]) -> tuple[str, dict | None]:
    state = read_json(path)
    if state is None:
        return "waiting", None
    status = str(state.get("status", ""))
    complete_statuses = {complete_status} if isinstance(complete_status, str) else complete_status
    failed_statuses = {failed_suffix} if isinstance(failed_suffix, str) else failed_suffix
    if status in complete_statuses:
        return "complete", state
    if status in failed_statuses or status.endswith("_FAILED") or status.endswith("DEPENDENCY_FAILED"):
        return "failed", state
    return "waiting", state


def g2_status() -> tuple[str, dict | None]:
    receipt = read_json(G2_ROOT / "final_receipt.json")
    if receipt is not None:
        if (receipt.get("status") == "G2_EXACT_COVERAGE_TRAINING_COMPLETE"
                and int(receipt.get("steps", -1)) >= G2_STEPS
                and receipt.get("manifest_sha256") == "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"):
            return "complete", receipt
        return "failed", receipt
    contract = read_json(G2_ROOT / "trainer_contract.json")
    if contract is not None and contract.get("status") != "G2_FORMAL_TRAINING_RUNNING":
        return "failed", contract
    return "waiting", None


def manifest_status() -> tuple[str, dict | None]:
    receipt = read_json(MANIFEST_RECEIPT)
    manifest = read_json(MANIFEST)
    if receipt is None or manifest is None:
        return "failed", {"reason": "missing_corrected_manifest"}
    native_rows = bool(manifest.get("native_ovmot_training_rows"))
    provenance = manifest.get("candidate_provenance") or {}
    native_rows = native_rows or bool(provenance.get("native_ovmot_training_rows"))
    if (receipt.get("status") != "REAL_THREE_TASK_COMPACT_SOURCE_MANIFEST_PREPARED_NOT_TRAINED"
            or receipt.get("manifest_sha256") != MANIFEST_SHA
            or int(receipt.get("rows", -1)) != ROWS
            or int(receipt.get("optimizer_updates", -1)) != 0
            or receipt.get("gpu_launched") is not False):
        return "failed", {"reason": "manifest_receipt_contract_mismatch", "receipt": receipt}
    if (not native_rows) or manifest.get("ovmot_training_mode") != "native":
        return "waiting_native_ovmot", {
            "reason": "native_ovmot_rows_required_before_optimizer_launch",
            "ovmot_training_mode": manifest.get("ovmot_training_mode"),
            "native_ovmot_rows_pending": manifest.get("native_ovmot_rows_pending"),
            "candidate_provenance_native_ovmot_training_rows": provenance.get("native_ovmot_training_rows"),
        }
    return "complete", receipt


def compute_pids(gpu: int) -> list[int]:
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=False, capture_output=True, text=True)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def free_memory_mib() -> dict[int, int]:
    result = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, GPUS)), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
    if len(values) != len(GPUS):
        raise RuntimeError(f"unexpected free-memory response: {result.stdout!r}")
    return dict(zip(GPUS, values))


def devices_ready() -> tuple[bool, dict]:
    pids = {gpu: compute_pids(gpu) for gpu in GPUS}
    free = free_memory_mib()
    ready = all(not pids[gpu] and free[gpu] >= 32768 for gpu in GPUS)
    return ready, {"compute_pids": pids, "free_memory_MiB": free}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, default=QUEUE_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    owner = args.queue_root / "owner_state.json"
    old = read_json(owner)
    if old and old.get("status", "").endswith(("_COMPLETE", "_FAILED")):
        raise FileExistsError(f"terminal queue already exists: {owner}")
    if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
        raise FileExistsError(f"queue owner is already live: {old['owner_pid']}")
    state = {
        "status": "WAITING_G2_G1_BUNDLE_HELDOUT_AND_EXCLUSIVE_GPUS",
        "current_stage": "WAITING_G2_G1_BUNDLE_HELDOUT_AND_EXCLUSIVE_GPUS",
        "owner_pid": os.getpid(), "project_root": str(PROJECT_ROOT), "runtime_root": str(RUNTIME_ROOT),
        "g2_root": str(G2_ROOT), "g1_state": str(G1_STATE), "bundle_state": str(BUNDLE_STATE),
        "heldout_state": str(HELDOUT_STATE), "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA,
        "physical_gpus": list(GPUS), "training_root": str(TRAIN_ROOT), "config": str(CONFIG),
        "rows": ROWS, "ovmot_training_mode": "base_teacher_control", "native_ovmot_rows_pending": True,
        "planned_trainable_rows": TRAINABLE_ROWS, "overlength_rows_excluded": OVERLENGTH_ROWS,
        "max_len": 32768, "full_source_exposure": True, "optimizer_updates": 0,
        "completed": {}, "failed": None, "old_queue_restarted": False,
    }
    save(owner, state)
    try:
        while True:
            m, me = manifest_status()
            if m == "waiting_native_ovmot":
                state.update(
                    status="WAITING_NATIVE_OVMOT_ROWS",
                    current_stage="WAITING_NATIVE_OVMOT_ROWS",
                    native_ovmot_rows_pending=True,
                    manifest_gate=me,
                )
                save(owner, state)
                time.sleep(args.poll_seconds)
                continue
            if m == "failed":
                state.update(status="REAL_THREE_TASK_JOINT_MANIFEST_FAILED", current_stage="FAILED", failed=me)
                save(owner, state); return 2
            g2, ge = g2_status()
            if g2 == "failed":
                state.update(status="REAL_THREE_TASK_JOINT_G2_DEPENDENCY_FAILED", current_stage="G2_DEPENDENCY_FAILED", failed={"dependency": "G2", "evidence": ge})
                save(owner, state); return 2
            g1, g1e = terminal_state(G1_STATE, "G1_GPU5_ADAPTATION_AND_CONTROLS_COMPLETE", "G1_GPU5_QUEUE_FAILED")
            bundle, be = terminal_state(
                BUNDLE_STATE,
                {"G2_CLEAN_BUNDLE_EXPORT_COMPLETE", "G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED"},
                "G2_CLEAN_BUNDLE_EXPORT_FAILED",
            )
            heldout, he = terminal_state(HELDOUT_STATE, "G2_CLEAN_FULL_INDEPENDENT_HELDOUT_COMPLETE", "G2_CLEAN_FULL_INDEPENDENT_HELDOUT_FAILED")
            if g1 == "failed" or bundle == "failed":
                state.update(status="REAL_THREE_TASK_JOINT_DEPENDENCY_FAILED", current_stage="DEPENDENCY_FAILED",
                             failed={"g1": g1e, "bundle": be})
                save(owner); return 2
            state["gate_snapshot"] = {"g2": g2, "g1": g1, "bundle": bundle, "heldout": heldout}
            if g2 == "complete" and g1 == "complete" and bundle == "complete" and heldout in ("complete", "failed"):
                try:
                    ready, evidence = devices_ready()
                except Exception as error:
                    ready, evidence = False, {"error": repr(error)}
                state["resource_probe"] = evidence
                if ready:
                    state["completed"].update({"g2": ge, "g1": g1e, "bundle": be, "heldout": he})
                    break
            save(owner, state)
            time.sleep(args.poll_seconds)
        command = [str(RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"), "--gpus", *map(str, GPUS), "--",
                   str(RUNTIME_ROOT / ".venv-onejev-py310/bin/python"), "-m", "torch.distributed.run",
                   "--standalone", "--nproc-per-node", "4", str(RUNTIME_ROOT / "tools/groundjev/run_source_manifest_sft.py"),
                   "--source-training-backend", "fla", "--config", str(CONFIG), "--paired-index-shuffle",
                   "--distributed-sft", "--deterministic-execution", "--seed-global-rngs"]
        state.update(status="REAL_THREE_TASK_JOINT_TRAINING_RUNNING", current_stage="REAL_THREE_TASK_JOINT_TRAINING_RUNNING", command=command)
        save(owner, state)
        args.queue_root.mkdir(parents=True, exist_ok=True)
        with (args.queue_root / "TRAINING.log").open("w") as log:
            result = subprocess.run(command, cwd=RUNTIME_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        state["last_exit_code"] = result.returncode
        records = []
        log_jsonl = TRAIN_ROOT / "log.jsonl"
        if log_jsonl.is_file():
            for line in log_jsonl.read_text(errors="ignore").splitlines():
                try: records.append(json.loads(line))
                except json.JSONDecodeError: pass
        done = next((row for row in reversed(records) if row.get("event") == "done"), None)
        state["terminal_record"] = done
        if result.returncode != 0 or not done or int(done.get("examples", -1)) != TRAINABLE_ROWS:
            state.update(status="REAL_THREE_TASK_JOINT_TRAINING_FAILED", current_stage="FAILED",
                         failed={"reason": "nonzero_or_incomplete_terminal_receipt", "expected_trainable_rows": TRAINABLE_ROWS,
                                 "source_rows": ROWS, "overlength_rows_excluded": OVERLENGTH_ROWS})
            save(owner, state); return result.returncode or 3
        state.update(status="REAL_THREE_TASK_JOINT_TRAINING_COMPLETE_FULL_SOURCE_EXPOSURE", current_stage="COMPLETE",
                     completed={"training": done}, optimizer_updates=int(done.get("steps", 0)), full_goal_complete=False,
                     source_exposure="131990/131990 rows; compact_competition_v1 applied at manifest load")
        save(owner, state); return 0
    except Exception as error:
        state.update(status="REAL_THREE_TASK_JOINT_TRAINING_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
