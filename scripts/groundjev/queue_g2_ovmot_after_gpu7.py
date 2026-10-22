#!/usr/bin/env python3
"""Start a faster independent TAO OVMOT scope when GPU7 has room.

The active GPU5 OVMOT owner is never touched.  This queue waits internally
for a safe co-resident window after either GPU7 RMOT owner exits, then invokes
the same one-scope runner with a larger vocabulary chunk.  A failed fast
attempt remains a separate receipt; it cannot invalidate the 96-class run.
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
PORT = 8238
OUT = RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_fast_v1"
QUEUE_STATE = RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_fast_v1.queue_state.json"
MANIFEST = RUNTIME / "outputs/groundjev/multitask_pilot_data/ovmot_tao_independent105_full.json"
MODEL = RUNTIME / "models/GroundAnything-DLM-G2-CLEAN-V2"
MIN_FREE_MIB = 32768


def save(state: dict) -> None:
    QUEUE_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(QUEUE_STATE)
    print(json.dumps({k: state.get(k) for k in ("status", "free_memory_MiB", "returncode", "failed")}, ensure_ascii=False), flush=True)


def free_memory() -> tuple[int, list[str]]:
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
        "status": "WAITING_GPU7_FAST_OVMOT_MEMORY",
        "owner_pid": os.getpid(),
        "physical_gpu": GPU,
        "port": PORT,
        "manifest": str(MANIFEST),
        "model": str(MODEL),
        "output": str(OUT),
        "ovmot_chunk_size": 192,
        "min_free_memory_MiB": MIN_FREE_MIB,
        "failed": None,
    }
    save(state)
    try:
        while True:
            free, apps = free_memory()
            state.update(free_memory_MiB=free, compute_apps=apps)
            if free >= MIN_FREE_MIB:
                break
            save(state)
            time.sleep(30)
        command = [
            str(PYTHON), str(RUNNER), "--gpu", str(GPU), "--port", str(PORT),
            "--task", "ovmot", "--manifest", str(MANIFEST), "--out", str(OUT),
            "--label", "tao_video105_independent40", "--model", str(MODEL),
            "--model-label", "G2_CLEAN_V2", "--ovmot-chunk-size", "192",
        ]
        state.update(status="STARTING_GPU7_FAST_OVMOT", command=command)
        save(state)
        result = subprocess.run(command, cwd=ROOT, check=False)
        state["returncode"] = result.returncode
        state["status"] = "GPU7_FAST_OVMOT_COMPLETE" if result.returncode == 0 else "GPU7_FAST_OVMOT_FAILED"
        if result.returncode:
            state["failed"] = {"returncode": result.returncode}
        save(state)
        return result.returncode
    except Exception as error:
        state.update(status="GPU7_FAST_OVMOT_FAILED", failed={"error": repr(error)})
        save(state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
