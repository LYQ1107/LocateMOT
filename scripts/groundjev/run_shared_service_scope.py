#!/usr/bin/env python3
"""Run an independent held-out scope through an already-running service.

This launcher deliberately does not start or stop SGLang.  It is used when a
GPU already owns one validated GroundAnything service and we want a second
task to run without creating another scheduler.  The output is isolated and
gets the same parser, decoder and official scorer contract as the service's
primary owner.
"""

from __future__ import annotations

import argparse
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
MOT_SCORER = RUNTIME / "tools/groundjev/evaluate_video.py"
MULTITASK_SCORER = RUNTIME / "tools/groundjev/evaluate_multitask.py"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed", "completed")}, ensure_ascii=False), flush=True)


def service_ready(url: str) -> dict:
    with urlopen(url.rstrip("/") + "/v1/models", timeout=10) as response:
        payload = json.load(response)
    if not any(item.get("id") == "groundinganything" for item in payload.get("data", [])):
        raise RuntimeError(f"service does not expose groundinganything: {payload}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--service-owner", type=Path, required=True)
    parser.add_argument("--task", choices=("mot", "rmot", "ovmot"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--grounding-parser",
        choices=("official_scoring", "official_scoring_repaired"),
        default=None,
    )
    parser.add_argument("--ovmot-chunk-size", type=int, default=192)
    args = parser.parse_args()

    owner_path = args.out / "owner_state.json"
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    manifest = read_json(args.manifest)
    if manifest is None:
        raise FileNotFoundError(args.manifest)
    if manifest.get("training_on_this_scope") is not False:
        raise RuntimeError(f"scope is not independent: {args.manifest}")
    if not (args.model / "model.safetensors").is_file():
        raise FileNotFoundError(args.model / "model.safetensors")
    if not args.service_owner.is_file():
        raise FileNotFoundError(f"primary service owner is missing: {args.service_owner}")

    parser_policy = args.grounding_parser or (
        "official_scoring_repaired" if args.task == "ovmot" else "official_scoring"
    )
    state = {
        "status": "SHARED_SERVICE_SCOPE_STARTING",
        "current_stage": "CONTRACT_CHECK",
        "owner_pid": os.getpid(),
        "physical_gpu": args.gpu,
        "service_url": args.service_url,
        "service_owner": str(args.service_owner),
        "task": args.task,
        "label": args.label,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "model": str(args.model),
        "model_label": args.model_label,
        "model_sha256": sha256(args.model / "model.safetensors"),
        "grounding_parser_policy": parser_policy,
        "grounding_decoder_policy": "tracking_dense",
        "same_input_parser_decoder_tracker_scorer": True,
        "shared_service_no_extra_scheduler": True,
        "completed": {},
        "failed": None,
    }
    save(owner_path, state)

    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    try:
        state.update(status="SHARED_SERVICE_WAITING", current_stage="SERVICE_CHECK")
        state["service_models"] = service_ready(args.service_url)
        save(owner_path, state)

        output = args.out / args.model_label / args.label
        command = [
            str(PYTHON), str(RUN_VIDEO), "--task", args.task, "--video", str(args.manifest),
            "--backend", "simple", "--grounding-url", args.service_url.rstrip("/") + "/v1",
            "--grounding-parser", parser_policy, "--grounding-decoder-policy", "tracking_dense",
            "--perception-failure-policy", "empty_and_record", "--out", str(output),
        ]
        if args.task == "ovmot":
            if args.ovmot_chunk_size <= 0:
                raise ValueError("--ovmot-chunk-size must be positive")
            command += [
                "--grounding-vocabulary-chunk-size", str(args.ovmot_chunk_size),
                "--category-policy", "native_label",
            ]
        state.update(status=f"RUNNING_{args.model_label}_{args.label}", current_stage="INFERENCE", command=command)
        save(owner_path, state)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "shared_service_inference.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=RUNTIME, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"inference exited with {result.returncode}; see {args.out / 'shared_service_inference.log'}")

        scorer = MOT_SCORER if args.task == "mot" else MULTITASK_SCORER
        scorer_command = [str(PYTHON), str(scorer), "--video", str(args.manifest), "--run", str(output)]
        with (args.out / "shared_service_eval.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(scorer_command, cwd=RUNTIME, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"official scorer exited with {result.returncode}; see {args.out / 'shared_service_eval.log'}")

        result_json = read_json(output / "result.json")
        metrics = read_json(output / "official_metrics.json")
        if result_json is None or metrics is None:
            raise RuntimeError("missing result.json or official_metrics.json")
        observed = [
            json.loads(line)["frame"]
            for line in (output / "frame_decisions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        expected = manifest.get("frame_numbers", [])
        if observed != expected:
            raise RuntimeError(f"frame contract drift: observed {len(observed)} vs expected {len(expected)}")

        state["completed"] = {
            "frames": len(observed),
            "result": result_json,
            "official_metrics": metrics,
            "inference_command": command,
            "scorer_command": scorer_command,
        }
        state.update(status="SHARED_SERVICE_SCOPE_COMPLETE", current_stage="COMPLETE")
        save(owner_path, state)
        (args.out / "receipt.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 0
    except Exception as error:
        state.update(status="SHARED_SERVICE_SCOPE_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner_path, state)
        (args.out / "failure_receipt.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
