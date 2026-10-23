#!/usr/bin/env python3
"""Resume the failed G1 shared-service RMOT scope without a new scheduler.

The original v2 owner is preserved as a failed receipt: its frame-660 request
hit the registered HTTP timeout.  This retry restores the exact completed
prefix into a fresh output root, uses the same GPU5 service and parser/scorer,
and records a terminal receipt independently.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


ROOT = Path("/data2/user/LocateMOT")
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
PYTHON = RUNTIME / ".venv-onejev-py310/bin/python"
RUN_VIDEO = RUNTIME / "tools/groundjev/run_video.py"
SCORER = RUNTIME / "tools/groundjev/evaluate_multitask.py"
GPU = 5
SERVICE_URL = "http://127.0.0.1:8240"
SERVICE_OWNER = RUNTIME / "outputs/groundjev/heldout/g1_mot20_05_gpu5_resume_v1/owner_state.json"
WAIT_OWNERS = (
    SERVICE_OWNER,
    RUNTIME / "outputs/groundjev/heldout/g1_ovmot_shared_gpu5_v1/owner_state.json",
)
MANIFEST = RUNTIME / "outputs/groundjev/multitask_pilot_data/rmot_single_empty_full.json"
MODEL = RUNTIME / "models/GroundAnything-DLM-TrackProjG1-GPU5-V2"
RESUME_FROM = RUNTIME / "outputs/groundjev/heldout/g1_rmot_shared_gpu5_v2/G1_GPU5_RMOT/rmot_single0007_shared"
OUT = RUNTIME / "outputs/groundjev/heldout/g1_rmot_shared_gpu5_v3"
QUEUE_STATE = RUNTIME / "outputs/groundjev/heldout/g1_rmot_shared_gpu5_retry.queue_state.json"
POLL_SECONDS = 30
MAX_WAIT_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 900


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "status": "WAITING_G1_RMOT_RETRY",
        "current_stage": "WAITING_G1_RMOT_RETRY",
        "queue_pid": os.getpid(),
        "physical_gpu": GPU,
        "service_url": SERVICE_URL,
        "service_owner": str(SERVICE_OWNER),
        "wait_owners": [str(path) for path in WAIT_OWNERS],
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "model": str(MODEL),
        "model_sha256": sha256(MODEL / "model.safetensors"),
        "resume_from": str(RESUME_FROM),
        "out": str(OUT),
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "preserved_failed_attempt": str(RUNTIME / "outputs/groundjev/heldout/g1_rmot_shared_gpu5_v2/failure_receipt.json"),
        "no_extra_scheduler": True,
        "failed": None,
    }
    save(state)
    started = time.monotonic()
    try:
        if OUT.exists() and any(OUT.iterdir()):
            raise FileExistsError(f"refusing non-empty retry output: {OUT}")
        if not (RESUME_FROM / "frame_decisions.jsonl").is_file():
            raise FileNotFoundError(RESUME_FROM / "frame_decisions.jsonl")
        while not all(terminal(path) for path in WAIT_OWNERS):
            if time.monotonic() - started > MAX_WAIT_SECONDS:
                raise TimeoutError("G1 shared-service dependencies did not terminate within 24 hours")
            state["dependency_status"] = {str(path): (read(path) or {}).get("status", "MISSING") for path in WAIT_OWNERS}
            save(state)
            time.sleep(POLL_SECONDS)
        state["dependency_status"] = {str(path): (read(path) or {}).get("status", "MISSING") for path in WAIT_OWNERS}
        while not service_ready():
            if time.monotonic() - started > MAX_WAIT_SECONDS:
                raise TimeoutError("G1 shared service did not become ready within 24 hours")
            state.update(status="WAITING_G1_RMOT_SERVICE_HEALTH", current_stage="SERVICE_CHECK")
            save(state)
            time.sleep(POLL_SECONDS)

        env = os.environ.copy()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
        output = OUT / "G1_GPU5_RMOT_RETRY" / "rmot_single0007_shared_retry"
        command = [
            str(PYTHON), str(RUN_VIDEO), "--task", "rmot", "--video", str(MANIFEST),
            "--backend", "simple", "--grounding-url", SERVICE_URL.rstrip("/") + "/v1",
            "--grounding-parser", "official_scoring", "--grounding-decoder-policy", "tracking_dense",
            "--perception-failure-policy", "empty_and_record", "--grounding-request-timeout",
            str(REQUEST_TIMEOUT_SECONDS), "--resume-from", str(RESUME_FROM), "--out", str(output),
        ]
        state.update(status="RUNNING_G1_RMOT_RETRY", current_stage="INFERENCE", command=command)
        save(state)
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "retry_inference.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=RUNTIME, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"inference exited with {result.returncode}; see {OUT / 'retry_inference.log'}")

        scorer_command = [str(PYTHON), str(SCORER), "--video", str(MANIFEST), "--run", str(output)]
        state.update(status="SCORING_G1_RMOT_RETRY", current_stage="SCORING", scorer_command=scorer_command)
        save(state)
        with (OUT / "retry_eval.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(scorer_command, cwd=RUNTIME, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"official scorer exited with {result.returncode}; see {OUT / 'retry_eval.log'}")
        result_json = read(output / "result.json")
        metrics = read(output / "official_metrics.json")
        if result_json is None or metrics is None:
            raise RuntimeError("missing result.json or official_metrics.json")
        observed = [json.loads(line)["frame"] for line in (output / "frame_decisions.jsonl").read_text().splitlines() if line.strip()]
        expected = (read(MANIFEST) or {}).get("frame_numbers", [])
        if observed != expected:
            raise RuntimeError(f"frame contract drift: observed {len(observed)} vs expected {len(expected)}")
        state["completed"] = {
            "frames": len(observed), "result": result_json, "official_metrics": metrics,
            "inference_command": command, "scorer_command": scorer_command,
        }
        state.update(status="G1_RMOT_SHARED_RETRY_COMPLETE", current_stage="COMPLETE")
        save(state)
        (OUT / "receipt.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 0
    except Exception as error:
        state.update(status="G1_RMOT_SHARED_RETRY_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(state)
        (OUT / "failure_receipt.json").parent.mkdir(parents=True, exist_ok=True)
        (OUT / "failure_receipt.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
