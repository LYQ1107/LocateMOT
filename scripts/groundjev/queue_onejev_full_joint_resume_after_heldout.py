#!/usr/bin/env python3
"""Resume the real MOT/RMOT/OVMOT OneJev run after held-out evaluation.

The historical four-GPU run has a durable step-3000 FP32/Adam/RNG state but
was stopped before a terminal receipt.  This queue waits for the current
held-out owners and an exclusive four-GPU window, then resumes into a new
checkpoint name so the historical directory remains immutable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/data2/user/LocateMOT")
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
PYTHON = RUNTIME / ".venv-onejev-py310/bin/python"
RUNNER = RUNTIME / "tools/groundjev/run_with_resources.py"
TRAINER = RUNTIME / "tools/groundjev/run_source_manifest_sft.py"
SOURCE_CONFIG = RUNTIME / "outputs/groundjev/training/full_joint_source_epoch1_owner_v1/training_config.json"
HISTORICAL_ROOT = Path("/data3/user/LocateMOT-checkpoints/onejev_full_joint_source_epoch1_v1")
RESUME_STATE = HISTORICAL_ROOT / "latest_training_state.pt"
RESUME_META = HISTORICAL_ROOT / "latest_training_state.json"
FINALIZER = RUNTIME / "outputs/groundjev/heldout/groundjev_heldout_g200_finalization_receipt.json"
QUEUE_ROOT = RUNTIME / "outputs/groundjev/training/full_joint_source_epoch1_resume_after_heldout_v1"
TRAIN_NAME = "onejev_full_joint_source_epoch1_after_heldout_v1"
TRAIN_ROOT = Path("/data3/user/LocateMOT-checkpoints") / TRAIN_NAME
GPUS = (3, 5, 6, 7)
EXPECTED_RESUME_STEP = 3000
EXPECTED_TOTAL_STEPS = 28690

OWNER_DIRS = (
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu3_v1",
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu5_v1",
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu6_v1",
    RUNTIME / "outputs/groundjev/heldout/g2_domain_dancetrack_gpu5_v1",
    RUNTIME / "outputs/groundjev/heldout/g2_domain_animaltrack_gpu6_v1",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu5_v1",
    RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_gpu3_v1",
    RUNTIME / "outputs/groundjev/heldout/g1_animaltrack_gpu6_v1",
    RUNTIME / "outputs/groundjev/heldout/g200_dancetrack_gpu3_v2",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_rmot_single_gpu7_v1",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_rmot_multi_gpu7_v1",
)


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save(state: dict) -> None:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_ROOT / "owner_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(QUEUE_ROOT / "owner_state.json")
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "resume_step", "last_exit_code", "failed")}, ensure_ascii=False), flush=True)


def terminal(path: Path) -> bool:
    state = read(path / "owner_state.json")
    return bool(state) and str(state.get("status", "")).endswith(("COMPLETE", "FAILED"))


def compute_pids(gpu: int) -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    return [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]


def free_memory() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "-i", ",".join(map(str, GPUS)), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    values = [int(x.strip()) for x in result.stdout.splitlines() if x.strip()]
    if len(values) != len(GPUS):
        raise RuntimeError(f"unexpected memory query: {result.stdout!r}")
    return dict(zip(GPUS, values))


def build_config() -> Path:
    source = read(SOURCE_CONFIG)
    if source is None:
        raise FileNotFoundError(SOURCE_CONFIG)
    source["name"] = TRAIN_NAME
    source["runs_dir"] = str(TRAIN_ROOT.parent)
    path = QUEUE_ROOT / "training_config.json"
    path.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    owner_path = QUEUE_ROOT / "owner_state.json"
    old = read(owner_path)
    if old and old.get("status") in {"ONEJEV_FULL_SOURCE_RESUME_COMPLETE", "ONEJEV_FULL_SOURCE_RESUME_FAILED"}:
        raise FileExistsError(f"terminal queue already exists: {owner_path}")
    state = old or {
        "status": "WAITING_HELDOUT_TERMINAL_AND_EXCLUSIVE_GPU_SET",
        "current_stage": "WAITING_HELDOUT_TERMINAL_AND_EXCLUSIVE_GPU_SET",
        "owner_pid": os.getpid(),
        "project_root": str(ROOT),
        "runtime_root": str(RUNTIME),
        "heldout_finalizer": str(FINALIZER),
        "physical_gpus": list(GPUS),
        "historical_training_root": str(HISTORICAL_ROOT),
        "resume_state": str(RESUME_STATE),
        "resume_step": EXPECTED_RESUME_STEP,
        "expected_total_steps": EXPECTED_TOTAL_STEPS,
        "new_training_root": str(TRAIN_ROOT),
        "other_active_jobs_untouched": True,
        "failed": None,
    }
    state["owner_pid"] = os.getpid()
    save(state)
    try:
        while True:
            finalizer = read(FINALIZER)
            all_terminal = all(terminal(path) for path in OWNER_DIRS)
            if finalizer is not None and all_terminal:
                state["heldout_gate"] = {"status": "COMPLETE", "accepted_scope_count": finalizer.get("accepted_scope_count")}
                pids = {gpu: compute_pids(gpu) for gpu in GPUS}
                free = free_memory()
                state["resource_probe"] = {"compute_pids": pids, "free_memory_MiB": free}
                if all(not pids[gpu] and free[gpu] >= 32768 for gpu in GPUS):
                    break
            save(state)
            time.sleep(60)

        if not RESUME_STATE.is_file() or not RESUME_META.is_file():
            raise FileNotFoundError("registered step-3000 resume state is missing")
        meta = read(RESUME_META)
        if not meta or int(meta.get("step", -1)) != EXPECTED_RESUME_STEP:
            raise RuntimeError(f"resume metadata is not step {EXPECTED_RESUME_STEP}: {meta}")
        config = build_config()
        state.update(status="STARTING_ONEJEV_FULL_SOURCE_RESUME", current_stage="STARTING_ONEJEV_FULL_SOURCE_RESUME", config=str(config))
        command = [
            "python3", str(RUNNER), "--gpus", *map(str, GPUS), "--",
            str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4",
            str(TRAINER), "--source-training-backend", "fla", "--config", str(config),
            "--paired-index-shuffle", "--distributed-sft", "--deterministic-execution", "--seed-global-rngs",
            "--resume-state", str(RESUME_STATE),
        ]
        state.update(status="ONEJEV_FULL_SOURCE_RESUMING", current_stage="ONEJEV_FULL_SOURCE_RESUMING", command=command)
        save(state)
        with (QUEUE_ROOT / "TRAINING_RESUME.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=RUNTIME, stdout=log, stderr=subprocess.STDOUT, check=False)
        state["last_exit_code"] = result.returncode
        log_jsonl = TRAIN_ROOT / "log.jsonl"
        records = []
        if log_jsonl.is_file():
            for line in log_jsonl.read_text(errors="ignore").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        state["last_observed_step"] = max((int(r.get("step", -1)) for r in records if "step" in r), default=-1)
        done = next((r for r in reversed(records) if r.get("event") == "done"), None)
        state["terminal_record"] = done
        if result.returncode != 0 or not done or int(done.get("steps", -1)) != EXPECTED_TOTAL_STEPS:
            state.update(status="ONEJEV_FULL_SOURCE_RESUME_FAILED", current_stage="FAILED", failed={"exit_code": result.returncode, "terminal_record": done})
            save(state)
            return result.returncode or 1
        state.update(status="ONEJEV_FULL_SOURCE_RESUME_COMPLETE", current_stage="COMPLETE", completed={"training": done})
        save(state)
        return 0
    except Exception as error:
        state.update(status="ONEJEV_FULL_SOURCE_RESUME_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
