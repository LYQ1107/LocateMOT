#!/usr/bin/env python3
"""Single durable gate for the registered held-out finalizers and OneJev.

The evaluation owners are independent long-running processes. This
orchestrator is the one blocking waiter for the current batch: it waits for
all registered owners (including the shared-service additions), prints a
consolidated result receipt, runs the three receipt finalizers, and only then
hands control to the real native MOT/RMOT/OVMOT OneJev queue. The historical
teacher-control resume queue is preserved and is never invoked here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from queue_onejev_full_joint_resume_after_heldout import OWNER_DIRS


ROOT = Path("/data2/user/LocateMOT")
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
PYTHON = RUNTIME / ".venv-onejev-py310/bin/python"
QUEUE_ROOT = RUNTIME / "outputs/groundjev/heldout"
STATE_PATH = QUEUE_ROOT / "heldout_finalizers_onejev_orchestrator_20261023.json"
FINALIZERS = (
    ROOT / "scripts/groundjev/queue_groundjev_heldout_finalize_multidomain.py",
    ROOT / "scripts/groundjev/queue_groundjev_heldout_finalize_with_g200.py",
    ROOT / "scripts/groundjev/queue_groundjev_heldout_finalize_with_g200_fast_ovmot.py",
)
RESULT_PRINTER = ROOT / "scripts/groundjev/print_heldout_consolidated_result.py"
RESULT_JSON = QUEUE_ROOT / "groundjev_heldout_consolidated_result_20261023.json"
RESULT_MD = QUEUE_ROOT / "groundjev_heldout_consolidated_result_20261023.md"
ONEJEV_QUEUE = ROOT / "scripts/groundjev/queue_real_three_task_native.py"
ONEJEV_QUEUE_ROOT = RUNTIME / "outputs/groundjev/training/real_three_task_joint_native_after_heldout_v2_queue"
ONEJEV_TRAIN_ROOT = Path("/data3/user/LocateMOT-checkpoints/onejev_real_three_task_joint_native_after_heldout_v2")
ONEJEV_MANIFEST = RUNTIME / "outputs/groundjev/training_data/real_three_task_joint_native_v1/train.sources.json"
EXTRA_WAIT_STATE_FILES = (
    QUEUE_ROOT / "g200_animaltrack_shared_gpu7_v1/owner_state.json",
    QUEUE_ROOT / "g1_dancetrack_shared_gpu5_retry.queue_state.json",
    QUEUE_ROOT / "g200_ovmot_dedicated_gpu7_retry.queue_state.json",
    QUEUE_ROOT / "extra_shared_controls_collection.queue_state.json",
)


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def terminal(state: dict | None) -> bool:
    return bool(state) and str(state.get("status", "")).endswith(
        ("COMPLETE", "FAILED", "STOPPED", "CANCELLED")
    )


def save(state: dict) -> None:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temp.replace(STATE_PATH)


def main() -> int:
    state = {
        "status": "WAITING_ALL_REGISTERED_HELDOUT_OWNERS_AND_EXTRA_CONTROLS",
        "current_stage": "WAITING_ALL_REGISTERED_HELDOUT_OWNERS_AND_EXTRA_CONTROLS",
        "owner_pid": os.getpid(),
        "owner_dirs": [str(path) for path in OWNER_DIRS],
        "extra_wait_state_files": [str(path) for path in EXTRA_WAIT_STATE_FILES],
        "finalizers": [str(path) for path in FINALIZERS],
        "result_printer": str(RESULT_PRINTER),
        "result_json": str(RESULT_JSON),
        "result_markdown": str(RESULT_MD),
        "onejev_queue": str(ONEJEV_QUEUE),
        "failed": None,
    }
    save(state)
    while True:
        states = {str(path): read(path / "owner_state.json") for path in OWNER_DIRS}
        states.update({str(path): read(path) for path in EXTRA_WAIT_STATE_FILES})
        if all(terminal(item) for item in states.values()):
            state["heldout_states"] = {
                path: (item or {}).get("status", "MISSING") for path, item in states.items()
            }
            break
        time.sleep(60)

    state.update(status="PRINTING_HELDOUT_CONSOLIDATED_RESULT", current_stage="CONSOLIDATED_RESULT")
    save(state)
    result_command = [
        str(PYTHON), str(RESULT_PRINTER),
        "--out", str(RESULT_JSON),
        "--markdown-out", str(RESULT_MD),
    ]
    result = subprocess.run(result_command, cwd=ROOT, check=False)
    state["consolidated_result_exit_code"] = result.returncode
    if result.returncode:
        state.update(status="HELDOUT_CONSOLIDATED_RESULT_FAILED", current_stage="FAILED")
        state["failed"] = {"result_printer_exit_code": result.returncode}
        save(state)
        return result.returncode

    state.update(status="RUNNING_HELDOUT_FINALIZERS", current_stage="FINALIZERS")
    save(state)
    finalizer_results = []
    for index, script in enumerate(FINALIZERS, start=1):
        log_path = QUEUE_ROOT / f"orchestrator_finalizer_{index}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(PYTHON), str(script)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        finalizer_results.append({"script": str(script), "exit_code": result.returncode, "log": str(log_path)})
        state["finalizer_results"] = finalizer_results
        save(state)

    state.update(status="RUNNING_ONEJEV_REAL_NATIVE_QUEUE", current_stage="ONEJEV_QUEUE")
    save(state)
    log_path = QUEUE_ROOT / "orchestrator_onejev_queue.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [
                str(PYTHON), str(ONEJEV_QUEUE),
                "--queue-root", str(ONEJEV_QUEUE_ROOT),
                "--training-root", str(ONEJEV_TRAIN_ROOT),
                "--native-manifest", str(ONEJEV_MANIFEST),
                "--gpus", "5", "7",
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    state["onejev_exit_code"] = result.returncode
    state["onejev_log"] = str(log_path)
    if result.returncode:
        state.update(status="HELDOUT_FINALIZERS_COMPLETE_REAL_NATIVE_ONEJEV_FAILED", current_stage="FAILED")
        state["failed"] = {"onejev_exit_code": result.returncode}
        save(state)
        return result.returncode
    state.update(status="HELDOUT_FINALIZERS_AND_REAL_NATIVE_ONEJEV_COMPLETE", current_stage="COMPLETE")
    save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
