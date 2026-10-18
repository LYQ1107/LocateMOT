#!/usr/bin/env python3
"""Resume-safe gate from the J1 final checkpoint to held-out MOT/RMOT/OVMOT.

This owner does not reserve a GPU while it waits.  It refuses to infer from a
partial checkpoint, a missing MOT20 release receipt, or a failed RMOT pilot.
Every transition is written atomically so a killed owner can be restarted
without deleting a partial result.
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
DEFAULT_TRAINING = Path("/data3/user/LocateMOT-runs/TrackJEV_FullDataV2/trackjev_3task_J1_v4_birth")
DEFAULT_OUT = ROOT / "outputs/groundjev/full_data_v2/post_j1_heldout_gate_v1"
DEFAULT_MANIFEST = ROOT / "outputs/groundjev/full_data_v2/manifests/DT1203_eval_only.json"
DEFAULT_MOT_RELEASE = ROOT / "outputs/groundjev/full_data_v2/candidate_export_v2/mot20_MOT20-05_resumed_v2/result.json"
DEFAULT_RMOT_STATE = ROOT / "outputs/groundjev/full_data_v2/candidate_export_rmot_expression_v1/owner_state.json"
MULTITASK_SCOPES = (
    ("rmot_single8", "rmot", ROOT / "outputs/groundjev/multitask_pilot_data/rmot_single_empty.json", ()),
    ("rmot_multi12", "rmot", ROOT / "outputs/groundjev/multitask_pilot_data/rmot_multi_empty.json", ()),
    ("ovmot_video4", "ovmot", ROOT / "outputs/groundjev/multitask_pilot_data/ovmot_tao_full.json",
     ("--grounding-vocabulary-chunk-size", "32")),
    ("ovmot_video63", "ovmot", ROOT / "outputs/groundjev/multitask_pilot_data/ovmot_tao_base_novel63_full.json",
     ("--grounding-vocabulary-chunk-size", "32")),
)


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pid_command(pid: int) -> str | None:
    path = Path("/proc") / str(pid)
    if not path.exists():
        return None
    try:
        return (path / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
    except OSError:
        return None


def port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


def adapter_file(adapter: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No adapter weights in {adapter}")


def final_training_receipt(training: Path) -> tuple[bool, dict[str, Any]]:
    state_path = training / "latest_training_state.json"
    if not state_path.is_file():
        return False, {"reason": "latest_training_state.json absent"}
    state = json.loads(state_path.read_text())
    step = int(state.get("step", -1))
    adapter = training / "adapter-step1200"
    if step < 1200:
        return False, {"step": step, "reason": "final step not reached"}
    if not adapter.is_dir():
        return False, {"step": step, "reason": "adapter-step1200 absent"}
    try:
        weight = adapter_file(adapter)
    except FileNotFoundError as error:
        return False, {"step": step, "reason": str(error)}
    return True, {"step": step, "adapter": str(adapter), "adapter_weights": str(weight), "state": str(state_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-command-fragment", default="trackjev_3task_J1_v4_birth")
    parser.add_argument("--mot-release", type=Path, default=DEFAULT_MOT_RELEASE)
    parser.add_argument("--rmot-state", type=Path, default=DEFAULT_RMOT_STATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", type=Path, default=ROOT / "models/OneJev-0.8B")
    parser.add_argument("--decoder-url", default="http://127.0.0.1:8109/v1")
    parser.add_argument("--feature-url", default="http://127.0.0.1:8112")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=float, default=96.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state_path = args.out / "owner_state.json"
    state: dict[str, Any] = {
        "status": "WAITING_J1_FINAL",
        "owner_pid": os.getpid(),
        "wait_pid": args.wait_pid,
        "wait_command_fragment": args.wait_command_fragment,
        "training_root": str(args.training_root),
        "mot_release": str(args.mot_release),
        "rmot_state": str(args.rmot_state),
        "manifest": str(args.manifest),
        "out": str(args.out),
        "gpu": args.gpu,
        "decoder_url": args.decoder_url,
        "feature_url": args.feature_url,
        "completed": {},
        "failed": None,
        "full_goal_complete": False,
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text())
        if previous.get("status") == "HELDOUT_DT1203_COMPLETE":
            print(json.dumps(previous, indent=1), flush=True)
            return 0
        state["completed"] = previous.get("completed", {})

    def save() -> None:
        atomic_write(state_path, state)
        print(json.dumps({"status": state["status"], "completed": state["completed"], "failed": state["failed"]}), flush=True)

    def fail(reason: str, **extra: Any) -> int:
        state["status"] = "HELDOUT_GATE_FAILED"
        state["failed"] = {"reason": reason, **extra}
        save()
        return 1

    save()
    if args.dry_run:
        state["status"] = "DRY_RUN_VALIDATED"
        state["completed"]["configuration"] = {
            "training_final_receipt": final_training_receipt(args.training_root),
            "mot_release_exists": args.mot_release.is_file(),
            "rmot_state_exists": args.rmot_state.is_file(),
            "manifest_exists": args.manifest.is_file(),
        }
        save()
        return 0

    deadline = time.monotonic() + args.max_wait_hours * 3600
    try:
        while True:
            ready, receipt = final_training_receipt(args.training_root)
            if ready:
                state["completed"]["j1_final"] = receipt
                break
            command = pid_command(args.wait_pid)
            if command is None:
                return fail("J1 owner exited before step 1200", receipt=receipt)
            if args.wait_command_fragment not in command:
                return fail("wait PID is no longer the registered J1 command", command=command)
            if time.monotonic() > deadline:
                return fail("J1 wait exceeded max-wait-hours", receipt=receipt)
            time.sleep(args.poll_seconds)
        state["status"] = "WAITING_MOT20_RELEASE"
        save()

        while not args.mot_release.is_file():
            if time.monotonic() > deadline:
                return fail("MOT20 release wait exceeded max-wait-hours")
            time.sleep(args.poll_seconds)
        state["completed"]["mot20_release"] = json.loads(args.mot_release.read_text())
        state["status"] = "WAITING_RMOT_PILOT"
        save()

        while True:
            if not args.rmot_state.is_file():
                status = None
            else:
                rmot = json.loads(args.rmot_state.read_text())
                status = rmot.get("status")
            if status == "EXPRESSION_RMOT_PILOT_COMPLETE":
                state["completed"]["rmot_pilot"] = rmot
                break
            if status == "EXPRESSION_RMOT_PILOT_FAILED":
                return fail("expression-conditioned RMOT pilot failed", rmot=rmot)
            if time.monotonic() > deadline:
                return fail("RMOT pilot wait exceeded max-wait-hours", rmot_status=status)
            time.sleep(args.poll_seconds)

        while not (port_ready(8109) and port_ready(8112)):
            if time.monotonic() > deadline:
                return fail("GroundAnything services did not become ready")
            time.sleep(args.poll_seconds)
        state["status"] = "MERGING_J1"
        save()

        adapter = args.training_root / "adapter-step1200"
        merged = args.out / "merged_3task_J1_v4_birth"
        merge_receipt = merged / "merge_receipt.json"
        if not merge_receipt.is_file():
            if merged.exists() and any(merged.iterdir()):
                return fail("merged output exists without a receipt", merged=str(merged))
            command = [PY, str(ROOT / "tools/groundjev/merge_lora_adapter.py"),
                       "--base", str(args.base), "--adapter", str(adapter), "--out", str(merged)]
            proc = subprocess.run(command, cwd=ROOT)
            if proc.returncode != 0 or not merge_receipt.is_file():
                return fail("J1 merge failed", returncode=proc.returncode, command=command)
        receipt = json.loads(merge_receipt.read_text())
        if receipt.get("unmatched_count") != 0 or receipt.get("tensors_updated", 0) <= 0:
            return fail("J1 merge receipt failed tensor gate", merge_receipt=receipt)
        state["completed"]["merge"] = receipt

        eval_out = args.out / "dt1203_j1_v4_birth"
        if eval_out.exists() and any(eval_out.iterdir()) and not (eval_out / "result.json").is_file():
            return fail("held-out output exists without a result receipt", output=str(eval_out))
        if not (eval_out / "result.json").is_file():
            state["status"] = "RUNNING_HELDOUT_DT1203"
            save()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTORCH_ALLOC_CONF="expandable_segments:True")
            command = [PY, str(ROOT / "tools/groundjev/run_video.py"),
                       "--task", "mot", "--video", str(args.manifest),
                       "--checkpoint", str(merged), "--backend", "trackjev",
                       "--trackjev-arm", "B4", "--trackjev-device", "cuda:0",
                       "--native-feature-url", args.feature_url,
                       "--grounding-url", args.decoder_url,
                       "--grounding-decoder-policy", "tracking_dense",
                       "--perception-failure-policy", "empty_and_record",
                       "--out", str(eval_out)]
            log_path = args.out / "dt1203_j1_v4_birth.log"
            with log_path.open("a") as log:
                proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0 or not (eval_out / "result.json").is_file():
                return fail("held-out DT1203 inference failed", returncode=proc.returncode, log=str(log_path))
        state["completed"]["dt1203_inference"] = {"run": str(eval_out), "result": str(eval_out / "result.json")}

        metrics_log = args.out / "dt1203_official_metrics.log"
        metrics_command = [PY, str(ROOT / "tools/groundjev/evaluate_video.py"),
                           "--video", str(args.manifest), "--run", str(eval_out)]
        with metrics_log.open("a") as log:
            metrics_proc = subprocess.run(metrics_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if metrics_proc.returncode != 0:
            return fail("held-out DT1203 evaluator failed", returncode=metrics_proc.returncode, log=str(metrics_log))
        state["completed"]["dt1203_metrics"] = {"log": str(metrics_log), "returncode": metrics_proc.returncode}
        multitask_root = args.out / "multitask"
        for name, task, manifest, extra in MULTITASK_SCOPES:
            if not manifest.is_file():
                return fail("registered multitask manifest is missing", name=name, manifest=str(manifest))
            eval_out = multitask_root / name
            result_file = eval_out / "result.json"
            metrics_file = eval_out / "official_metrics.json"
            if eval_out.exists() and any(eval_out.iterdir()) and not result_file.is_file():
                return fail("multitask output exists without a result receipt", name=name, output=str(eval_out))
            if not result_file.is_file():
                state["status"] = f"RUNNING_HELDOUT_{name.upper()}"
                save()
                command = [PY, str(ROOT / "tools/groundjev/run_video.py"),
                           "--task", task, "--video", str(manifest),
                           "--checkpoint", str(merged), "--backend", "trackjev",
                           "--trackjev-arm", "B4", "--trackjev-device", "cuda:0",
                           "--native-feature-url", args.feature_url,
                           "--grounding-url", args.decoder_url,
                           "--grounding-decoder-policy", "tracking_dense",
                           "--perception-failure-policy", "empty_and_record",
                           "--out", str(eval_out)] + list(extra)
                log_path = args.out / f"{name}.log"
                with log_path.open("a") as log:
                    proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if proc.returncode != 0 or not result_file.is_file():
                    return fail("multitask held-out inference failed", name=name,
                                returncode=proc.returncode, log=str(log_path))
            if not metrics_file.is_file():
                metrics_log = args.out / f"{name}_official_metrics.log"
                metrics_command = [PY, str(ROOT / "tools/groundjev/evaluate_multitask.py"),
                                   "--video", str(manifest), "--run", str(eval_out)]
                with metrics_log.open("a") as log:
                    metrics_proc = subprocess.run(metrics_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if metrics_proc.returncode != 0 or not metrics_file.is_file():
                    return fail("multitask held-out evaluator failed", name=name,
                                returncode=metrics_proc.returncode, log=str(metrics_log))
            state["completed"][name] = {"run": str(eval_out), "result": str(result_file),
                                        "official_metrics": str(metrics_file)}
            save()
        state["status"] = "HELDOUT_ALL_REGISTERED_TASKS_COMPLETE"
        state["full_goal_complete"] = False
        save()
        return 0
    except Exception as error:
        state["status"] = "HELDOUT_GATE_FAILED"
        state["failed"] = {"error": repr(error)}
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
