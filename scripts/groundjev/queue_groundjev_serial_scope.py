#!/usr/bin/env python3
"""Run one held-out scope after a same-GPU scope has released its service.

This queue keeps at most one GroundAnything serving scheduler from this
project on a physical GPU.  It waits on the predecessor owner, then delegates
the actual inference/scoring contract to ``run_g2_parallel_scope.py``.  The
queue state is stored beside (rather than inside) the child output directory so
the child can create its normal ``owner_state.json`` without being overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
RUNNER = Path("/data2/user/LocateMOT/scripts/groundjev/run_g2_parallel_scope.py")


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "failed")}, ensure_ascii=False), flush=True)


def terminal(path: Path) -> bool:
    state = read(path)
    return bool(state) and str(state.get("status", "")).endswith(("COMPLETE", "FAILED"))


def free_memory(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-owner", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task", choices=("mot", "rmot", "ovmot"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--ovmot-chunk-size", type=int, default=96)
    parser.add_argument("--grounding-parser", choices=("official_scoring", "official_scoring_repaired"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    queue_state_path = args.out.with_name(args.out.name + ".queue_state.json")
    state = {
        "status": "WAITING_PREDECESSOR_AND_GPU",
        "current_stage": "WAITING_PREDECESSOR_AND_GPU",
        "queue_pid": __import__("os").getpid(),
        "wait_owner": str(args.wait_owner),
        "physical_gpu": args.gpu,
        "port": args.port,
        "task": args.task,
        "manifest": str(args.manifest),
        "out": str(args.out),
        "label": args.label,
        "model": str(args.model),
        "model_label": args.model_label,
        "resource_rule": "wait for predecessor terminal and at least 32768 MiB free before starting",
        "failed": None,
    }
    save(queue_state_path, state)
    try:
        while not terminal(args.wait_owner):
            time.sleep(args.poll_seconds)
        while free_memory(args.gpu) < 32768:
            state["current_stage"] = "WAITING_GPU_MEMORY"
            save(queue_state_path, state)
            time.sleep(args.poll_seconds)

        state.update(status="STARTING_SERIAL_SCOPE", current_stage="STARTING_SERIAL_SCOPE")
        save(queue_state_path, state)
        command = [
            "python3", str(RUNNER), "--gpu", str(args.gpu), "--port", str(args.port),
            "--task", args.task, "--manifest", str(args.manifest), "--out", str(args.out),
            "--label", args.label, "--model", str(args.model), "--model-label", args.model_label,
        ]
        if args.ovmot_chunk_size:
            command += ["--ovmot-chunk-size", str(args.ovmot_chunk_size)]
        if args.grounding_parser:
            command += ["--grounding-parser", args.grounding_parser]
        state["command"] = command
        state.update(status="SERIAL_SCOPE_RUNNING", current_stage="SERIAL_SCOPE_RUNNING")
        save(queue_state_path, state)
        result = subprocess.run(command, cwd=RUNTIME, check=False)
        state["last_exit_code"] = result.returncode
        state.update(
            status="SERIAL_SCOPE_COMPLETE" if result.returncode == 0 else "SERIAL_SCOPE_FAILED",
            current_stage="COMPLETE" if result.returncode == 0 else "FAILED",
        )
        save(queue_state_path, state)
        return result.returncode
    except Exception as error:
        state.update(status="SERIAL_SCOPE_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(queue_state_path, state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
