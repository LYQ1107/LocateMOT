#!/usr/bin/env python3
"""Resume the real MOT/RMOT/OVMOT OneJev run after protected dependencies.

The historical four-GPU source run reached non-durable step 3525 and has no
terminal receipt.  Its last complete FP32/Adam/RNG state is step 3000.  This
owner deliberately waits for the protected G2 course, the queued GPU5 G1
adaptation, and a clean four-device allowlist before resuming from that state.
It never overwrites the historical run or starts alongside another compute
process on the selected devices.
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
G2_ROOT = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/"
    "G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final"
)
G1_STATE = RUNTIME_ROOT / "outputs/groundjev/ground_adaptation/" \
    "native_tracking200_gpu5_g1_v1/queue_owner_state.json"
HELDOUT_STATE = RUNTIME_ROOT / "outputs/groundjev/heldout/" \
    "g2_clean_independent_gpu7_v1/owner_state.json"
QUEUE_ROOT = RUNTIME_ROOT / "outputs/groundjev/training/" \
    "full_joint_source_epoch1_resume_queue_v2"
TRAIN_ROOT = Path(
    "/data3/user/LocateMOT-checkpoints/onejev_full_joint_source_epoch1_v1"
)
TRAIN_CONFIG = RUNTIME_ROOT / "outputs/groundjev/training/" \
    "full_joint_source_epoch1_owner_v1/training_config.json"
RESUME_STATE = TRAIN_ROOT / "latest_training_state.pt"
GPUS = (4, 5, 6, 7)
G2_STEPS = 18227
MANIFEST_SHA = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
HISTORICAL_DURABLE_STEP = 3000


def atomic_save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(json.dumps({k: state.get(k) for k in (
        "status", "current_stage", "resume_step", "last_exit_code", "failed"
    )}, ensure_ascii=False), flush=True)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def g2_status() -> tuple[str, dict | None]:
    receipt = read_json(G2_ROOT / "final_receipt.json")
    if receipt is not None:
        if (receipt.get("status") == "G2_EXACT_COVERAGE_TRAINING_COMPLETE"
                and int(receipt.get("steps", -1)) >= G2_STEPS
                and receipt.get("manifest_sha256") == MANIFEST_SHA):
            return "complete", receipt
        return "failed", receipt
    contract = read_json(G2_ROOT / "trainer_contract.json")
    if contract is not None and contract.get("status") not in (
            "G2_FORMAL_TRAINING_RUNNING",):
        return "failed", contract
    return "waiting", None


def g1_status() -> tuple[str, dict | None]:
    state = read_json(G1_STATE)
    if state is None:
        return "waiting", None
    status = state.get("status")
    if status == "G1_GPU5_ADAPTATION_AND_CONTROLS_COMPLETE":
        return "complete", state
    if status == "G1_GPU5_QUEUE_FAILED" or status == "G1_GPU5_NOT_STARTED_G2_DEPENDENCY_FAILED":
        return "failed", state
    return "waiting", state


def heldout_status() -> tuple[str, dict | None]:
    """Serialize the four-GPU resume behind the independent G2 held-out gate.

    The held-out owner uses GPU7 and must get a chance to run before the
    four-rank OneJev job claims GPUs 4--7.  A terminal held-out failure is
    retained as evidence but does not deadlock the main training objective.
    """
    state = read_json(HELDOUT_STATE)
    if state is None:
        return "waiting", None
    status = str(state.get("status", ""))
    if status == "G2_CLEAN_FULL_INDEPENDENT_HELDOUT_COMPLETE":
        return "complete", state
    if status == "G2_CLEAN_FULL_INDEPENDENT_HELDOUT_FAILED" or status.endswith("DEPENDENCY_FAILED"):
        return "failed", state
    return "waiting", state


def compute_pids(gpu: int) -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def free_memory_mib() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "-i", ",".join(map(str, GPUS)),
         "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    values = [int(x.strip()) for x in result.stdout.splitlines() if x.strip()]
    if len(values) != len(GPUS):
        raise RuntimeError(f"unexpected memory query: {result.stdout!r}")
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
    parser.add_argument(
        "--resume-paused",
        action="store_true",
        help="Explicitly clear the GroundJEV redirection pause marker and re-register this historical comparison.",
    )
    args = parser.parse_args()
    owner = args.queue_root / "owner_state.json"
    pause_marker = args.queue_root / "PAUSED_FOR_GROUNDJEV_SINGLE_BACKBONE"
    if pause_marker.exists() and not args.resume_paused:
        paused = read_json(owner) or {}
        print(json.dumps({
            "status": paused.get("status", "PAUSED_FOR_GROUNDJEV_SINGLE_BACKBONE"),
            "queue_root": str(args.queue_root),
            "resume_step": paused.get("resume_step", HISTORICAL_DURABLE_STEP),
            "message": "historical independent OneJev queue is paused; pass --resume-paused only for an explicit comparison run",
        }, ensure_ascii=False), flush=True)
        return 0
    if args.resume_paused:
        pause_marker.unlink(missing_ok=True)
    if args.queue_root.exists():
        old = read_json(owner)
        if old and old.get("status") in {
            "ONEJEV_FULL_SOURCE_RESUME_COMPLETE",
            "ONEJEV_FULL_SOURCE_RESUME_FAILED",
        }:
            raise FileExistsError(f"terminal queue already exists: {owner}")
        if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
            raise FileExistsError(f"queue owner is already live: {old['owner_pid']}")
    args.queue_root.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "WAITING_G2_G1_AND_EXCLUSIVE_GPU_SET",
        "current_stage": "WAITING_G2_G1_AND_EXCLUSIVE_GPU_SET",
        "owner_pid": os.getpid(),
        "project_root": str(PROJECT_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "g2_root": str(G2_ROOT),
        "g1_state": str(G1_STATE),
        "heldout_state": str(HELDOUT_STATE),
        "physical_gpus": list(GPUS),
        "historical_training_root": str(TRAIN_ROOT),
        "resume_state": str(RESUME_STATE),
        "historical_non_durable_last_observed_step": 3525,
        "resume_step": HISTORICAL_DURABLE_STEP,
        "expected_total_steps": 28690,
        "completed": {},
        "failed": None,
        "other_active_jobs_untouched": True,
        "feature_cache": False,
    }
    atomic_save(owner, state)
    try:
        while True:
            g2, g2_evidence = g2_status()
            if g2 == "failed":
                state.update(status="ONEJEV_FULL_SOURCE_RESUME_G2_DEPENDENCY_FAILED",
                             current_stage="G2_DEPENDENCY_FAILED",
                             failed={"dependency": "G2", "evidence": g2_evidence})
                atomic_save(owner, state)
                return 2
            g1, g1_evidence = g1_status()
            if g1 == "failed":
                state.update(status="ONEJEV_FULL_SOURCE_RESUME_G1_DEPENDENCY_FAILED",
                             current_stage="G1_DEPENDENCY_FAILED",
                             failed={"dependency": "G1", "evidence": g1_evidence})
                atomic_save(owner, state)
                return 2
            if g2 == "complete" and g1 == "complete":
                state["completed"]["g2"] = g2_evidence
                state["completed"]["g1"] = {
                    "status": g1_evidence.get("status"),
                    "current_stage": g1_evidence.get("current_stage"),
                }
                heldout, heldout_evidence = heldout_status()
                if heldout == "waiting":
                    state["heldout_gate"] = "WAITING_INDEPENDENT_G2_HELDOUT_TERMINAL"
                    state["heldout_gate_evidence"] = heldout_evidence
                    atomic_save(owner, state)
                    time.sleep(args.poll_seconds)
                    continue
                state["completed"]["heldout_gate"] = {
                    "status": heldout,
                    "evidence": heldout_evidence,
                }
                try:
                    ready, evidence = devices_ready()
                except Exception as error:
                    ready, evidence = False, {"error": repr(error)}
                state["resource_probe"] = evidence
                state["resource_gate"] = {
                    "ready": ready,
                    "rule": "no compute PIDs and at least 32768 MiB free on every requested GPU",
                    "shared_memory_mode": False,
                }
                atomic_save(owner, state)
                if ready:
                    break
            time.sleep(args.poll_seconds)

        state.update(status="STARTING_ONEJEV_FULL_SOURCE_RESUME",
                     current_stage="STARTING_ONEJEV_FULL_SOURCE_RESUME")
        atomic_save(owner, state)
        if not RESUME_STATE.is_file():
            raise FileNotFoundError(f"missing durable resume state: {RESUME_STATE}")
        latest = read_json(TRAIN_ROOT / "latest_training_state.json")
        if latest is None or int(latest.get("step", -1)) != HISTORICAL_DURABLE_STEP:
            raise RuntimeError(f"resume state is not the registered step {HISTORICAL_DURABLE_STEP}: {latest}")
        command = [
            "python3", str(RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"),
            "--gpus", *map(str, GPUS), "--",
            str(RUNTIME_ROOT / ".venv-onejev-py310/bin/python"),
            "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4",
            str(RUNTIME_ROOT / "tools/groundjev/run_source_manifest_sft.py"),
            "--source-training-backend", "fla",
            "--config", str(TRAIN_CONFIG),
            "--paired-index-shuffle", "--distributed-sft",
            "--deterministic-execution", "--seed-global-rngs",
            "--resume-state", str(RESUME_STATE),
        ]
        state["command"] = command
        state["current_stage"] = "ONEJEV_FULL_SOURCE_RESUMING"
        state["status"] = "ONEJEV_FULL_SOURCE_RESUMING"
        atomic_save(owner, state)
        with (args.queue_root / "TRAINING_RESUME.log").open("w") as log:
            result = subprocess.run(command, cwd=RUNTIME_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        state["last_exit_code"] = result.returncode
        log_jsonl = TRAIN_ROOT / "log.jsonl"
        records = []
        if log_jsonl.is_file():
            for line in log_jsonl.read_text(errors="ignore").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        done = next((r for r in reversed(records) if r.get("event") == "done"), None)
        state["last_observed_step"] = max(
            (int(r.get("step", -1)) for r in records if "step" in r), default=-1
        )
        state["terminal_record"] = done
        if result.returncode != 0:
            state.update(status="ONEJEV_FULL_SOURCE_RESUME_FAILED",
                         current_stage="FAILED",
                         failed={"reason": "trainer_nonzero_exit", "exit_code": result.returncode})
            atomic_save(owner, state)
            return result.returncode or 1
        if not done or int(done.get("steps", -1)) != 28690:
            state.update(status="ONEJEV_FULL_SOURCE_RESUME_FAILED",
                         current_stage="FAILED",
                         failed={"reason": "missing_or_incomplete_terminal_receipt",
                                 "expected_steps": 28690, "terminal_record": done})
            atomic_save(owner, state)
            return 3
        state.update(status="ONEJEV_FULL_SOURCE_RESUME_COMPLETE",
                     current_stage="COMPLETE",
                     completed={"training": done})
        atomic_save(owner, state)
        return 0
    except Exception as error:
        state.update(status="ONEJEV_FULL_SOURCE_RESUME_FAILED",
                     current_stage="FAILED", failed={"error": repr(error)})
        atomic_save(owner, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
