#!/usr/bin/env python3
"""Retry the G1 DanceTrack control after the shared GPU5 queue drains.

The first attempts are preserved as failed receipts: GPU5/8240 returned HTTP
503 while three existing clients were using the same service, and the first
retry output root was already non-empty. This queue does not start another
server. It waits until the repaired G1 RMOT retry and OVMOT scope have terminal
receipts, then retries the identical independent scope under a fresh output
root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

ROOT = Path("/data2/user/LocateMOT")
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
PYTHON = RUNTIME / ".venv-onejev-py310/bin/python"
RUNNER = ROOT / "scripts/groundjev/run_shared_service_scope.py"
GPU = 5
SERVICE_URL = "http://127.0.0.1:8240"
SERVICE_OWNER = RUNTIME / "outputs/groundjev/heldout/g1_mot20_05_gpu5_resume_v1/owner_state.json"
WAIT_OWNERS = (
    RUNTIME / "outputs/groundjev/heldout/g1_rmot_shared_gpu5_v4_retry.queue_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_ovmot_shared_gpu5_v1/owner_state.json",
)
MANIFEST = RUNTIME / "outputs/groundjev/heldout_inputs/g2_domain_v1/dancetrack0004_val1203.json"
MODEL = RUNTIME / "models/GroundAnything-DLM-TrackProjG1-GPU5-V2"
OUT = RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_shared_gpu5_v3"
QUEUE_STATE = RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_shared_gpu5_retry.queue_state.json"
MIN_WAIT_SECONDS = 60
MAX_WAIT_SECONDS = 12 * 60 * 60


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


def service_ready() -> bool:
    try:
        with urlopen(SERVICE_URL.rstrip("/") + "/v1/models", timeout=10) as response:
            payload = json.load(response)
        return any(item.get("id") == "groundinganything" for item in payload.get("data", []))
    except (OSError, ValueError):
        return False


def main() -> int:
    state = {
        "status": "WAITING_G1_SHARED_CAPACITY",
        "current_stage": "WAITING_G1_SHARED_CAPACITY",
        "queue_pid": os.getpid(),
        "physical_gpu": GPU,
        "service_url": SERVICE_URL,
        "service_owner": str(SERVICE_OWNER),
        "wait_owners": [str(path) for path in WAIT_OWNERS],
        "manifest": str(MANIFEST),
        "model": str(MODEL),
        "out": str(OUT),
        "preserved_failed_attempt": str(RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_shared_gpu5_v1/failure_receipt.json"),
        "no_extra_scheduler": True,
        "failed": None,
    }
    save(state)
    started = time.monotonic()
    try:
        if OUT.exists() and any(OUT.iterdir()):
            raise FileExistsError(f"refusing non-empty retry output: {OUT}")
        while not all(terminal(path) for path in WAIT_OWNERS):
            if time.monotonic() - started > MAX_WAIT_SECONDS:
                raise TimeoutError("shared G1 clients did not reach terminal receipts within 12 hours")
            state["dependency_status"] = {str(path): (read(path) or {}).get("status", "MISSING") for path in WAIT_OWNERS}
            save(state)
            time.sleep(MIN_WAIT_SECONDS)
        state["dependency_status"] = {str(path): (read(path) or {}).get("status", "MISSING") for path in WAIT_OWNERS}
        state.update(status="WAITING_G1_SERVICE_HEALTH", current_stage="SERVICE_CHECK")
        save(state)
        while not service_ready():
            if time.monotonic() - started > MAX_WAIT_SECONDS:
                raise TimeoutError("G1 shared service did not become ready within 12 hours")
            time.sleep(MIN_WAIT_SECONDS)
        command = [
            str(PYTHON), str(RUNNER), "--gpu", str(GPU), "--service-url", SERVICE_URL,
            "--service-owner", str(SERVICE_OWNER), "--task", "mot", "--manifest", str(MANIFEST),
            "--out", str(OUT), "--model", str(MODEL), "--model-label", "G1_GPU5_DANCE_SHARED",
            "--label", "dancetrack0004_g1_shared_retry", "--grounding-parser", "official_scoring",
        ]
        state.update(status="RETRYING_G1_DANCETRACK_SHARED", current_stage="INFERENCE", command=command)
        save(state)
        result = subprocess.run(command, cwd=RUNTIME, check=False)
        state["returncode"] = result.returncode
        state.update(
            status="G1_DANCETRACK_SHARED_RETRY_COMPLETE" if result.returncode == 0 else "G1_DANCETRACK_SHARED_RETRY_FAILED",
            current_stage="COMPLETE" if result.returncode == 0 else "FAILED",
        )
        if result.returncode:
            state["failed"] = {"returncode": result.returncode}
        save(state)
        return result.returncode
    except Exception as error:
        state.update(status="G1_DANCETRACK_SHARED_RETRY_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
