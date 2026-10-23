#!/usr/bin/env python3
"""Queue a clean G200 TAO OVMOT retry after GPU7 controls drain.

The prior dedicated attempt and the pre-start queue are preserved as failed
receipts. This queue waits for the repaired G200 RMOT retry as well as the
G200 MOT20 owner, then starts one dedicated G200 service on a fresh port and
resumes the preserved causal OVMOT prefix. It never terminates foreign
processes or overwrites a failed run.
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
RUNNER = ROOT / "scripts/groundjev/run_g2_parallel_scope.py"
GPU = 7
PORT = 8243
WAIT_OWNERS = (
    RUNTIME / "outputs/groundjev/heldout/g200_mot20_05_gpu7_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_rmot_shared_gpu7_v3/owner_state.json",
)
MANIFEST = RUNTIME / "outputs/groundjev/multitask_pilot_data/ovmot_tao_base_novel63_full.json"
MODEL = RUNTIME / "models/GroundAnything-DLM-TrackProj200"
RESUME_FROM = RUNTIME / "outputs/groundjev/heldout/g200_ovmot_shared_gpu7_v2/G200_GPU7_OVMOT/tao_video63_base_novel40_shared"
OUT = RUNTIME / "outputs/groundjev/heldout/g200_ovmot_dedicated_gpu7_v2"
QUEUE_STATE = RUNTIME / "outputs/groundjev/heldout/g200_ovmot_dedicated_gpu7_retry_v3.queue_state.json"
MIN_FREE_MIB = 32768
POLL_SECONDS = 60
MAX_WAIT_SECONDS = 24 * 60 * 60


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save(state: dict) -> None:
    QUEUE_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(QUEUE_STATE)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed", "returncode")}, ensure_ascii=False), flush=True)


def terminal(path: Path) -> bool:
    value = read(path)
    return bool(value) and str(value.get("status", "")).endswith(("COMPLETE", "FAILED"))


def gpu_probe() -> tuple[int, list[str]]:
    free = int(subprocess.run(
        ["nvidia-smi", "-i", str(GPU), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0])
    apps = subprocess.run(
        ["nvidia-smi", "-i", str(GPU), "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return free, [line.strip() for line in apps if line.strip()]


def main() -> int:
    state = {
        "status": "WAITING_G200_GPU7_CONTROLS",
        "current_stage": "WAITING_G200_GPU7_CONTROLS",
        "queue_pid": os.getpid(),
        "physical_gpu": GPU,
        "port": PORT,
        "wait_owners": [str(path) for path in WAIT_OWNERS],
        "manifest": str(MANIFEST),
        "model": str(MODEL),
        "resume_from": str(RESUME_FROM),
        "out": str(OUT),
        "preserved_failed_attempt": str(RUNTIME / "outputs/groundjev/heldout/g200_ovmot_dedicated_gpu7_v1/failure_receipt.json"),
        "no_foreign_process_termination": True,
        "failed": None,
    }
    save(state)
    started = time.monotonic()
    try:
        if OUT.exists() and any(OUT.iterdir()):
            raise FileExistsError(f"refusing non-empty retry output: {OUT}")
        if not (RESUME_FROM / "frame_decisions.jsonl").is_file():
            raise FileNotFoundError(f"missing preserved causal prefix: {RESUME_FROM / 'frame_decisions.jsonl'}")
        while not all(terminal(path) for path in WAIT_OWNERS):
            if time.monotonic() - started > MAX_WAIT_SECONDS:
                raise TimeoutError("G200 GPU7 controls did not terminate within 24 hours")
            state["dependency_status"] = {str(path): (read(path) or {}).get("status", "MISSING") for path in WAIT_OWNERS}
            save(state)
            time.sleep(POLL_SECONDS)
        state["dependency_status"] = {str(path): (read(path) or {}).get("status", "MISSING") for path in WAIT_OWNERS}
        while True:
            free, apps = gpu_probe()
            state["resource_probe"] = {"free_memory_MiB": free, "compute_apps": apps, "minimum_free_memory_MiB": MIN_FREE_MIB}
            if free >= MIN_FREE_MIB:
                break
            if time.monotonic() - started > MAX_WAIT_SECONDS:
                raise TimeoutError("GPU7 did not reach the free-memory floor within 24 hours")
            state.update(status="WAITING_G200_GPU7_MEMORY", current_stage="RESOURCE_GATE")
            save(state)
            time.sleep(POLL_SECONDS)
        command = [
            str(PYTHON), str(RUNNER), "--gpu", str(GPU), "--port", str(PORT), "--task", "ovmot",
            "--manifest", str(MANIFEST), "--out", str(OUT), "--label", "tao_video63_base_novel40_dedicated_retry",
            "--model", str(MODEL), "--model-label", "G200_GPU7_OVMOT_DEDICATED_RETRY",
            "--ovmot-chunk-size", "192", "--grounding-parser", "official_scoring_repaired",
            "--grounding-request-timeout", "900",
            "--resume-from", str(RESUME_FROM),
        ]
        state.update(status="STARTING_G200_OVMOT_DEDICATED_RETRY", current_stage="STARTING", command=command)
        save(state)
        result = subprocess.run(command, cwd=ROOT, check=False)
        state["returncode"] = result.returncode
        state.update(
            status="G200_OVMOT_DEDICATED_RETRY_COMPLETE" if result.returncode == 0 else "G200_OVMOT_DEDICATED_RETRY_FAILED",
            current_stage="COMPLETE" if result.returncode == 0 else "FAILED",
        )
        if result.returncode:
            state["failed"] = {"returncode": result.returncode}
        save(state)
        return result.returncode
    except Exception as error:
        state.update(status="G200_OVMOT_DEDICATED_RETRY_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
