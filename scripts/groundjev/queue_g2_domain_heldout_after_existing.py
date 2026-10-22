#!/usr/bin/env python3
"""Run one additional legal G2 Clean domain scope after a current owner ends.

The two instances are launched independently: GPU5 receives DanceTrack
validation and GPU6 receives AnimalTrack test.  Each instance waits internally
for its existing G2 held-out owner to reach a terminal state, then performs a
fresh service/inference/scoring run with separate receipts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen


RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
VENV = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"
SERVE_PY = RUNTIME_ROOT / "tools/groundjev/serve_groundanything.py"
RUN_VIDEO = RUNTIME_ROOT / "tools/groundjev/run_video.py"
SCORER = RUNTIME_ROOT / "tools/groundjev/evaluate_video.py"
G2_BUNDLE = RUNTIME_ROOT / "models/GroundAnything-DLM-G2-CLEAN-V2"
CURRENT_ROOT = RUNTIME_ROOT / "outputs/groundjev/heldout"
MANIFEST_ROOT = RUNTIME_ROOT / "outputs/groundjev/heldout_inputs/g2_domain_v1"

ASSIGNMENTS = {
    5: {
        "name": "dancetrack0004_val1203",
        "manifest": MANIFEST_ROOT / "dancetrack0004_val1203.json",
        "port": 8230,
        "out": CURRENT_ROOT / "g2_domain_dancetrack_gpu5_v1",
    },
    6: {
        "name": "animaltrack_chicken1_test306",
        "manifest": MANIFEST_ROOT / "animaltrack_chicken1_test306.json",
        "port": 8231,
        "out": CURRENT_ROOT / "g2_domain_animaltrack_gpu6_v1",
    },
}

CURRENT_OWNERS = {
    5: CURRENT_ROOT / "g2_clean_independent_gpu5_v1/owner_state.json",
    6: CURRENT_ROOT / "g2_clean_independent_gpu6_v1/owner_state.json",
}


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
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed", "completed")}, ensure_ascii=False), flush=True)


def ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            payload = json.load(response)
        return any(item.get("id") == "groundinganything" for item in payload.get("data", []))
    except (OSError, ValueError):
        return False


def stop_group(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass


def gpu_ready(gpu: int) -> tuple[bool, dict]:
    apps = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    memory = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    free = int(memory.splitlines()[0].strip())
    pids = [line.strip() for line in apps.splitlines() if line.strip()]
    return free >= 32768, {"free_memory_MiB": free, "compute_apps": pids}


def run_logged(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=tuple(ASSIGNMENTS), required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    assignment = ASSIGNMENTS[args.gpu]
    out = Path(assignment["out"])
    owner_path = out / "owner_state.json"
    if out.exists() and any(out.iterdir()):
        old = read_json(owner_path)
        if old and str(old.get("status", "")).endswith(("COMPLETE", "FAILED")):
            raise FileExistsError(f"terminal domain owner already exists: {owner_path}")
        raise FileExistsError(f"refusing to overwrite domain owner output: {out}")
    manifest = read_json(Path(assignment["manifest"]))
    if manifest is None or manifest.get("training_on_this_scope") is not False:
        raise RuntimeError(f"invalid independent manifest: {assignment['manifest']}")
    if not (G2_BUNDLE / "model.safetensors").is_file():
        raise FileNotFoundError(G2_BUNDLE / "model.safetensors")
    state = {
        "status": f"WAITING_EXISTING_GPU{args.gpu}_HELDOUT_TERMINAL",
        "current_stage": f"WAITING_EXISTING_GPU{args.gpu}_HELDOUT_TERMINAL",
        "owner_pid": os.getpid(),
        "physical_gpu": args.gpu,
        "scope": assignment["name"],
        "manifest": str(assignment["manifest"]),
        "g2_bundle": str(G2_BUNDLE),
        "current_owner": str(CURRENT_OWNERS[args.gpu]),
        "completed": {},
        "failed": None,
        "same_input_parser_decoder_tracker_scorer": True,
    }
    save(owner_path, state)
    service = None
    try:
        while True:
            current = read_json(CURRENT_OWNERS[args.gpu])
            current_status = str(current.get("status", "")) if current else "MISSING"
            if current_status.endswith(("COMPLETE", "FAILED")):
                state["completed"]["dependency_owner"] = {
                    "status": current_status,
                    "state_path": str(CURRENT_OWNERS[args.gpu]),
                }
                break
            time.sleep(args.poll_seconds)
        while True:
            try:
                free, resource = gpu_ready(args.gpu)
            except Exception as error:
                free, resource = False, {"error": repr(error)}
            state["resource_probe"] = resource
            if free:
                break
            time.sleep(args.poll_seconds)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
        out.mkdir(parents=True, exist_ok=True)
        state.update(status=f"WAITING_G2_CLEAN_V2_SERVICE_GPU{args.gpu}", current_stage="LAUNCH_SERVICE")
        save(owner_path, state)
        launch = [
            "python3", str(RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"),
            "--gpus", str(args.gpu), "--", str(RUNTIME_ROOT / "external/GroundAnything/.venv-serve/bin/python"),
            str(SERVE_PY), "--model", str(G2_BUNDLE), "--port", str(assignment["port"]),
            "--mem-fraction-static", ".35", "--max-queued-requests", "2",
            "--out", str(out / "G2_CLEAN_V2_service"),
        ]
        service_log = (out / "G2_CLEAN_V2_service.log").open("w", encoding="utf-8")
        service = subprocess.Popen(launch, cwd=RUNTIME_ROOT, env=env, stdout=service_log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        state["service_pid"] = service.pid
        deadline = time.monotonic() + 900
        while not ready(int(assignment["port"])):
            if service.poll() is not None:
                raise RuntimeError(f"service exited with {service.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("service readiness timeout")
            time.sleep(2)
        output = out / "G2_CLEAN_V2" / assignment["name"]
        command = [
            str(VENV), str(RUN_VIDEO), "--task", "mot", "--video", str(assignment["manifest"]),
            "--backend", "simple", "--grounding-url", f"http://127.0.0.1:{assignment['port']}/v1",
            "--grounding-parser", "official_scoring", "--grounding-decoder-policy", "tracking_dense",
            "--perception-failure-policy", "empty_and_record", "--out", str(output),
        ]
        state.update(status=f"RUNNING_G2_CLEAN_V2_{assignment['name']}", current_stage="INFERENCE", current_scope=assignment["name"])
        save(owner_path, state)
        run_logged(command, RUNTIME_ROOT, env, out / "G2_CLEAN_V2_inference.log")
        scorer_command = [str(VENV), str(SCORER), "--video", str(assignment["manifest"]), "--run", str(output)]
        run_logged(scorer_command, RUNTIME_ROOT, env, out / "G2_CLEAN_V2_eval.log")
        result = read_json(output / "result.json")
        metrics = read_json(output / "official_metrics.json")
        if result is None or metrics is None:
            raise RuntimeError("missing result or official metrics")
        expected = manifest["frame_numbers"]
        observed = [json.loads(line)["frame"] for line in (output / "frame_decisions.jsonl").read_text().splitlines()]
        if observed != expected:
            raise RuntimeError(f"frame contract drift: {len(observed)} vs {len(expected)}")
        state["completed"][assignment["name"]] = {
            "frames": len(observed), "result": result, "official_metrics": metrics,
            "inference_command": command, "scorer_command": scorer_command,
        }
        state.update(status="G2_DOMAIN_HELDOUT_COMPLETE", current_stage="COMPLETE", heldout_claim=True)
        save(owner_path, state)
        return 0
    except Exception as error:
        state.update(status="G2_DOMAIN_HELDOUT_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner_path, state)
        return 1
    finally:
        stop_group(service)


if __name__ == "__main__":
    raise SystemExit(main())
