#!/usr/bin/env python3
"""Run the prepared expression-conditioned RMOT pilot after MOT20 releases services.

The waiter is intentionally single-worker: MOT20-05 and this pilot use the same
GroundAnything decoder/feature endpoints, so starting the pilot early would
contend with the protected long export.  A completed query is never rerun.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

ROOT = Path("/data2/user/LocateMOT_OneJev_CVPR")
PY = "/data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python"
DEFAULT_PLAN = ROOT / "outputs/groundjev/full_data_v2/candidate_export_rmot_expression_v1"


def process_identity(pid: int) -> tuple[bool, str]:
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return False, ""
    try:
        command = (proc / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
    except OSError:
        return False, ""
    return True, command


def port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-command-fragment", default="mot20_MOT20-05.json")
    parser.add_argument("--decoder-url", default="http://127.0.0.1:8109/v1")
    parser.add_argument("--feature-url", default="http://127.0.0.1:8112")
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    plan_path = args.plan / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["status"] == "EXPRESSION_CONDITIONED_RMOT_PILOT_PLANNED"
    manifests = sorted((args.plan / "manifests").glob("*.json"))
    assert len(manifests) == len(plan["selected_queries"]) == 24
    output_root = args.plan / "runs"
    output_root.mkdir(exist_ok=True)
    state_path = args.plan / "owner_state.json"
    state: dict[str, Any] = {
        "status": "WAITING_MOT20_RELEASE",
        "owner_pid": os.getpid(),
        "owner_starttick": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        "wait_pid": args.wait_pid,
        "wait_command_fragment": args.wait_command_fragment,
        "decoder_url": args.decoder_url,
        "feature_url": args.feature_url,
        "completed": {},
        "failed": None,
        "model_inference_started": False,
        "new_optimizer_updates": 0,
        "full_goal_complete": False,
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text())
        if previous.get("status") == "EXPRESSION_RMOT_PILOT_COMPLETE":
            print(json.dumps(previous, indent=1), flush=True)
            return 0
        state["completed"] = previous.get("completed", {})

    def save() -> None:
        atomic_write(state_path, state)
        print(json.dumps({
            "status": state["status"],
            "completed": len(state["completed"]),
            "failed": state["failed"],
        }), flush=True)

    save()
    try:
        while True:
            exists, command = process_identity(args.wait_pid)
            if exists and args.wait_command_fragment not in command:
                raise RuntimeError(
                    f"wait pid {args.wait_pid} is alive but is not the protected MOT20 command"
                )
            if not exists:
                break
            time.sleep(args.poll_seconds)
        state["status"] = "WAITING_GROUNDANYTHING_SERVICES"
        save()
        while not (port_ready(8109) and port_ready(8112)):
            time.sleep(args.poll_seconds)
        state["status"] = "RUNNING_EXPRESSION_RMOT_PILOT"
        state["model_inference_started"] = True
        save()
        for manifest_path in manifests:
            query_id = manifest_path.stem
            if query_id in state["completed"]:
                continue
            run_dir = output_root / query_id
            if (run_dir / "result.json").is_file():
                result = json.loads((run_dir / "result.json").read_text())
                state["completed"][query_id] = {
                    "status": "RECOVERED_EXISTING_RESULT",
                    "frames": result.get("frames_processed"),
                    "run": str(run_dir),
                }
                save()
                continue
            log_path = args.plan / f"{query_id}.log"
            command = [
                PY, str(ROOT / "tools/groundjev/run_video.py"),
                "--task", "rmot", "--video", str(manifest_path),
                "--backend", "simple_native", "--record-native-candidates",
                "--native-candidate-format", "packed_npz_f32",
                "--native-feature-url", args.feature_url,
                "--grounding-url", args.decoder_url,
                "--grounding-decoder-policy", "tracking_dense",
                "--perception-failure-policy", "empty_and_record",
                "--out", str(run_dir),
            ]
            started = time.time()
            with log_path.open("w") as log:
                proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            result_path = run_dir / "result.json"
            if proc.returncode != 0 or not result_path.is_file():
                state["status"] = "EXPRESSION_RMOT_PILOT_FAILED"
                state["failed"] = {
                    "query_id": query_id,
                    "returncode": proc.returncode,
                    "log": str(log_path),
                    "run": str(run_dir),
                    "seconds": round(time.time() - started, 1),
                }
                save()
                return 1
            result = json.loads(result_path.read_text())
            state["completed"][query_id] = {
                "status": "COMPLETE",
                "frames": result.get("frames_processed"),
                "rejected_frames": result.get("rejected_frames"),
                "run": str(run_dir),
                "seconds": round(time.time() - started, 1),
                "proposal_source": "native_G0_expression_conditioned",
            }
            save()
        state["status"] = "EXPRESSION_RMOT_PILOT_COMPLETE"
        state["model_inference_completed"] = True
        atomic_write(args.plan / "summary.json", {
            "status": state["status"],
            "queries": len(state["completed"]),
            "completed": state["completed"],
            "proposal_source": "native_G0_expression_conditioned",
            "full_goal_complete": False,
        })
        save()
        return 0
    except Exception as error:
        state["status"] = "EXPRESSION_RMOT_PILOT_FAILED"
        state["failed"] = {"error": repr(error)}
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
