#!/usr/bin/env python3
"""Run one independent G2 Clean scope beside an existing held-out owner.

This is intentionally a one-scope owner so a small, independent validation
can start immediately without waiting for the longer MOT20/RMOT queue.  It
uses the same serving, parser, decoder, tracker and official scorer as the
main held-out queue and writes a standalone receipt.
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
MOT_SCORER = RUNTIME_ROOT / "tools/groundjev/evaluate_video.py"
MULTITASK_SCORER = RUNTIME_ROOT / "tools/groundjev/evaluate_multitask.py"
G2_BUNDLE = RUNTIME_ROOT / "models/GroundAnything-DLM-G2-CLEAN-V2"


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


def service_ready(port: int) -> bool:
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


def run_logged(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task", choices=("mot", "rmot", "ovmot"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", type=Path, default=G2_BUNDLE)
    parser.add_argument("--model-label", default="G2_CLEAN_V2")
    args = parser.parse_args()

    owner_path = args.out / "owner_state.json"
    if args.out.exists() and any(args.out.iterdir()):
        old = read_json(owner_path)
        if old and str(old.get("status", "")).endswith("COMPLETE"):
            raise FileExistsError(f"terminal owner already exists: {owner_path}")
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    manifest = read_json(args.manifest)
    if manifest is None:
        raise FileNotFoundError(args.manifest)
    if manifest.get("training_on_this_scope") is not False:
        raise RuntimeError(f"scope is not independent: {args.manifest}")
    if not (args.model / "model.safetensors").is_file():
        raise FileNotFoundError(args.model / "model.safetensors")

    state = {
        "status": f"G2_PARALLEL_SCOPE_GPU{args.gpu}_STARTING",
        "current_stage": "RESOURCE_GATE",
        "owner_pid": os.getpid(),
        "physical_gpu": args.gpu,
        "port": args.port,
        "task": args.task,
        "label": args.label,
        "manifest": str(args.manifest),
        "manifest_sha256": None,
        "model": str(args.model),
        "model_label": args.model_label,
        "same_input_parser_decoder_tracker_scorer": True,
        "completed": {},
        "failed": None,
    }
    save(owner_path, state)
    service = None
    try:
        free_text = subprocess.run(
            ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        free = int(free_text.splitlines()[0])
        state["resource_probe"] = {"free_memory_MiB": free, "minimum_free_memory_MiB": 32768}
        if free < 32768:
            raise RuntimeError(f"insufficient free memory on GPU{args.gpu}: {free} MiB")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
        args.out.mkdir(parents=True, exist_ok=True)
        service_log = (args.out / "G2_CLEAN_V2_service.log").open("w", encoding="utf-8")
        launch = [
            "python3", str(RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"),
            "--gpus", str(args.gpu), "--", str(RUNTIME_ROOT / "external/GroundAnything/.venv-serve/bin/python"),
            str(SERVE_PY), "--model", str(args.model), "--port", str(args.port),
            "--mem-fraction-static", ".30", "--max-queued-requests", "2",
            "--out", str(args.out / "G2_CLEAN_V2_service"),
        ]
        service = subprocess.Popen(launch, cwd=RUNTIME_ROOT, env=env, stdout=service_log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        state["service_pid"] = service.pid
        state["service_command"] = launch
        state.update(status="G2_PARALLEL_SCOPE_SERVICE_STARTING", current_stage="SERVICE_STARTING")
        save(owner_path, state)
        deadline = time.monotonic() + 900
        while not service_ready(args.port):
            if service.poll() is not None:
                raise RuntimeError(f"service exited with {service.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("service readiness timeout")
            time.sleep(2)

        output = args.out / args.model_label / args.label
        command = [
            str(VENV), str(RUN_VIDEO), "--task", args.task, "--video", str(args.manifest),
            "--backend", "simple", "--grounding-url", f"http://127.0.0.1:{args.port}/v1",
            "--grounding-parser", "official_scoring", "--grounding-decoder-policy", "tracking_dense",
            "--perception-failure-policy", "empty_and_record", "--out", str(output),
        ]
        if args.task == "ovmot":
            command += ["--grounding-vocabulary-chunk-size", "96", "--category-policy", "native_label"]
        state.update(status=f"RUNNING_{args.model_label}_{args.label}", current_stage="INFERENCE", current_scope=args.label)
        save(owner_path, state)
        run_logged(command, RUNTIME_ROOT, env, args.out / "G2_CLEAN_V2_inference.log")
        scorer = MOT_SCORER if args.task == "mot" else MULTITASK_SCORER
        scorer_command = [str(VENV), str(scorer), "--video", str(args.manifest), "--run", str(output)]
        run_logged(scorer_command, RUNTIME_ROOT, env, args.out / "G2_CLEAN_V2_eval.log")
        result = read_json(output / "result.json")
        metrics = read_json(output / "official_metrics.json")
        if result is None or metrics is None:
            raise RuntimeError("missing result.json or official_metrics.json")
        expected = manifest.get("frame_numbers", [])
        frame_path = output / "frame_decisions.jsonl"
        observed = [json.loads(line)["frame"] for line in frame_path.read_text().splitlines()]
        if observed != expected:
            raise RuntimeError(f"frame contract drift: observed {len(observed)} vs expected {len(expected)}")
        state["completed"] = {
            "frames": len(observed), "result": result, "official_metrics": metrics,
            "inference_command": command, "scorer_command": scorer_command,
        }
        state.update(status="G2_PARALLEL_SCOPE_COMPLETE", current_stage="COMPLETE", heldout_claim=True)
        state["receipt"] = {"status": state["status"], "task": args.task, "label": args.label,
                             "frames": len(observed), "official_metrics": metrics}
        save(owner_path, state)
        (args.out / "receipt.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 0
    except Exception as error:
        state.update(status="G2_PARALLEL_SCOPE_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner_path, state)
        (args.out / "failure_receipt.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 1
    finally:
        stop_group(service)


if __name__ == "__main__":
    raise SystemExit(main())
