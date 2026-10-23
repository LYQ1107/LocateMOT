#!/usr/bin/env python3
"""Durably collect extra shared-control receipts after all retry queues finish."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/data2/user/LocateMOT")
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
PYTHON = RUNTIME / ".venv-onejev-py310/bin/python"
COLLECTOR = ROOT / "scripts/groundjev/collect_extra_shared_control_receipts.py"
WAIT_STATES = (
    RUNTIME / "outputs/groundjev/heldout/g200_animaltrack_shared_gpu7_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_shared_gpu5_retry.queue_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_ovmot_dedicated_gpu7_retry_v3.queue_state.json",
)
OUT = RUNTIME / "outputs/groundjev/heldout/extra_shared_controls_final_receipt.json"
QUEUE_STATE = RUNTIME / "outputs/groundjev/heldout/extra_shared_controls_collection.queue_state.json"
POLL_SECONDS = 60
MAX_WAIT_SECONDS = 36 * 60 * 60


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def terminal(path: Path) -> bool:
    value = read(path)
    return bool(value) and str(value.get("status", "")).endswith(("COMPLETE", "FAILED"))


def save(state: dict) -> None:
    QUEUE_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(QUEUE_STATE)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed", "returncode")}, ensure_ascii=False), flush=True)


def main() -> int:
    state = {
        "status": "WAITING_EXTRA_SHARED_CONTROLS",
        "current_stage": "WAITING_EXTRA_SHARED_CONTROLS",
        "queue_pid": os.getpid(),
        "wait_states": [str(path) for path in WAIT_STATES],
        "collector": str(COLLECTOR),
        "out": str(OUT),
        "failed": None,
    }
    save(state)
    started = time.monotonic()
    try:
        while not all(terminal(path) for path in WAIT_STATES):
            if time.monotonic() - started > MAX_WAIT_SECONDS:
                raise TimeoutError("extra shared-control queues did not terminate within 36 hours")
            state["dependency_status"] = {str(path): (read(path) or {}).get("status", "MISSING") for path in WAIT_STATES}
            save(state)
            time.sleep(POLL_SECONDS)
        state["dependency_status"] = {str(path): (read(path) or {}).get("status", "MISSING") for path in WAIT_STATES}
        command = [str(PYTHON), str(COLLECTOR), "--out", str(OUT)]
        state.update(status="COLLECTING_EXTRA_SHARED_CONTROLS", current_stage="COLLECTING", command=command)
        save(state)
        result = subprocess.run(command, cwd=ROOT, check=False)
        state["returncode"] = result.returncode
        state.update(
            status="EXTRA_SHARED_CONTROLS_COLLECTION_COMPLETE" if result.returncode == 0 else "EXTRA_SHARED_CONTROLS_COLLECTION_FAILED",
            current_stage="COMPLETE" if result.returncode == 0 else "FAILED",
        )
        if result.returncode:
            state["failed"] = {"returncode": result.returncode}
        save(state)
        return result.returncode
    except Exception as error:
        state.update(status="EXTRA_SHARED_CONTROLS_COLLECTION_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
