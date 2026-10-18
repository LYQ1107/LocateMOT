#!/usr/bin/env python3
"""Resume-safe gate from the J1 final checkpoint to held-out MOT/RMOT/OVMOT.

This owner does not reserve a GPU while it waits.  It refuses to infer from a
partial checkpoint, a missing MOT20 release receipt, or a failed RMOT pilot.
Every transition is written atomically so a killed owner can be restarted
without deleting a partial result.
"""
from __future__ import annotations

import argparse
import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_eval_env(gpu: int) -> dict[str, str]:
    """Build one immutable environment for every held-out subprocess."""

    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )


def _resolve_training_file(training: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    # ``args.json`` stores paths relative to the OneJev repository, while
    # checkpoints live on /data3.  Keep that historical launcher contract.
    return (ROOT / candidate).resolve()


def final_training_receipt(
    training: Path,
    *,
    training_manifest: Path | None = None,
    expected_training_manifest_sha256: str | None = None,
    training_receipt: Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    state_path = training / "latest_training_state.json"
    if not state_path.is_file():
        return False, {"reason": "latest_training_state.json absent"}
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, {"reason": "latest_training_state.json invalid", "error": str(error)}
    step = int(state.get("step", -1))
    adapter = training / "adapter-step1200"
    if step != 1200:
        return False, {"step": step, "reason": "durable final step must equal 1200"}
    required_state = {
        "format": "groundjev-full-fp32-adam-rng-v1",
        "FP32_masters_and_Adam_moments_saved": True,
        "scheduler_and_all_RNG_saved": True,
    }
    for key, expected in required_state.items():
        if state.get(key) != expected:
            return False, {"step": step, "reason": f"training state contract failed: {key}", "value": state.get(key)}
    state_file = Path(str(state.get("path", "")))
    if not state_file.is_file() or state_file.stat().st_size <= 0:
        return False, {"step": step, "reason": "durable training state path missing or empty", "path": str(state_file)}
    if int(state.get("bytes", 0)) <= 0:
        return False, {"step": step, "reason": "durable training state byte receipt is not positive"}
    if not adapter.is_dir():
        return False, {"step": step, "reason": "adapter-step1200 absent"}
    try:
        weight = adapter_file(adapter)
    except FileNotFoundError as error:
        return False, {"step": step, "reason": str(error)}
    if weight.stat().st_size <= 0:
        return False, {"step": step, "reason": "adapter weights are empty", "adapter_weights": str(weight)}
    config_path = adapter / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, {"step": step, "reason": "adapter_config.json invalid", "error": str(error)}
    expected_config = {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            return False, {"step": step, "reason": f"adapter config contract failed: {key}", "value": config.get(key)}
    if not str(config.get("base_model_name_or_path", "")):
        return False, {"step": step, "reason": "adapter base_model_name_or_path missing"}
    if not isinstance(config.get("target_modules"), list) or not config["target_modules"]:
        return False, {"step": step, "reason": "adapter target_modules missing or empty"}

    args_path = training / "args.json"
    if not args_path.is_file():
        return False, {"step": step, "reason": "args.json absent"}
    try:
        training_args = json.loads(args_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, {"step": step, "reason": "args.json invalid", "error": str(error)}
    if int(training_args.get("max_steps", -1)) != 1200:
        return False, {"step": step, "reason": "args.json max_steps is not 1200"}
    resolved_manifest = training_manifest or _resolve_training_file(training, str(training_args.get("train_file", "")))
    if not resolved_manifest.is_file() or resolved_manifest.stat().st_size <= 0:
        return False, {"step": step, "reason": "training manifest missing or empty", "path": str(resolved_manifest)}
    manifest_sha = sha256(resolved_manifest)
    if expected_training_manifest_sha256 and manifest_sha != expected_training_manifest_sha256:
        return False, {"step": step, "reason": "training manifest SHA mismatch", "path": str(resolved_manifest), "sha256": manifest_sha}

    terminal = None
    if training_receipt is not None:
        try:
            terminal = json.loads(training_receipt.read_text())
        except (OSError, json.JSONDecodeError) as error:
            return False, {"step": step, "reason": "training terminal receipt invalid", "error": str(error)}
        if terminal.get("status") not in {"J1_TRAINING_COMPLETE", "TRAINING_COMPLETE", "COMPLETED"}:
            return False, {"step": step, "reason": "training terminal receipt is not complete", "status": terminal.get("status")}
        if int(terminal.get("step", 1200)) != 1200:
            return False, {"step": step, "reason": "training terminal receipt step drift"}
    return True, {
        "step": step,
        "adapter": str(adapter),
        "adapter_weights": str(weight),
        "state": str(state_path),
        "state_bytes": int(state_file.stat().st_size),
        "training_manifest": str(resolved_manifest),
        "training_manifest_sha256": manifest_sha,
        "adapter_config": config,
        "training_terminal_receipt": str(training_receipt) if training_receipt else None,
    }


def manifest_frame_count(manifest: Path) -> int:
    payload = json.loads(manifest.read_text())
    images = payload.get("images")
    frame_numbers = payload.get("frame_numbers")
    if isinstance(images, list):
        return len(images)
    if isinstance(frame_numbers, list):
        return len(frame_numbers)
    raise ValueError(f"manifest has no images/frame_numbers list: {manifest}")


def valid_inference_receipt(
    result_file: Path,
    manifest: Path,
    *,
    expected_task: str | None = None,
    expected_scope: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    try:
        result = json.loads(result_file.read_text())
        frame_count = manifest_frame_count(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, {"reason": "invalid inference receipt or manifest", "error": str(error)}
    if result.get("status") not in {"VIDEO_INFERENCE_COMPLETE", "VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION"}:
        return False, {"reason": "inference status is not terminal", "status": result.get("status")}
    if expected_task is not None and result.get("task") != expected_task:
        return False, {"reason": "inference task mismatch", "task": result.get("task")}
    if expected_scope is not None and result.get("scope") != expected_scope:
        return False, {"reason": "inference scope mismatch", "scope": result.get("scope")}
    if int(result.get("frames", -1)) != frame_count:
        return False, {"reason": "inference frame count mismatch", "frames": result.get("frames"), "manifest_frames": frame_count}
    if result.get("all_input_frames_processed") is not True:
        return False, {"reason": "all_input_frames_processed is not true"}
    if result.get("gt_read_by_runtime") is True or result.get("future_frames_read_by_runtime") is True:
        return False, {"reason": "runtime receipt indicates GT/future-frame access"}
    rejected = result.get("perception_rejected_frames", [])
    if not isinstance(rejected, list) or any(not isinstance(frame, int) or frame < 0 or frame >= frame_count for frame in rejected):
        return False, {"reason": "rejected-frame receipt is malformed"}
    latencies = result.get("latencies_seconds")
    if isinstance(latencies, list) and len(latencies) != frame_count:
        return False, {"reason": "latency list does not retain every input frame", "latencies": len(latencies), "frames": frame_count}
    return True, {
        "status": result.get("status"),
        "task": result.get("task"),
        "scope": result.get("scope"),
        "frames": frame_count,
        "rejected_frames": len(rejected),
        "gt_read_by_runtime": False,
        "future_frames_read_by_runtime": False,
    }


def valid_metrics_receipt(metrics_file: Path, manifest: Path, *, expected_task: str | None = None) -> tuple[bool, dict[str, Any]]:
    try:
        metrics = json.loads(metrics_file.read_text())
        frame_count = manifest_frame_count(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, {"reason": "invalid metrics receipt or manifest", "error": str(error)}
    if metrics.get("status") not in {"OFFICIAL_SCOPE_EVALUATION_COMPLETE", "OFFICIAL_REGISTERED_MULTITASK_SCOPE_EVALUATION_COMPLETE"}:
        return False, {"reason": "metrics receipt is not terminal", "status": metrics.get("status")}
    if expected_task is not None and metrics.get("task", "mot") != expected_task:
        return False, {"reason": "metrics task mismatch", "task": metrics.get("task")}
    if int(metrics.get("frames", -1)) != frame_count:
        return False, {"reason": "metrics frame count mismatch", "frames": metrics.get("frames"), "manifest_frames": frame_count}
    if not isinstance(metrics.get("metrics"), dict) or not metrics["metrics"]:
        return False, {"reason": "metrics values missing"}
    return True, {"status": metrics.get("status"), "task": metrics.get("task", "mot"), "frames": frame_count}


def inspect_eval_output(
    eval_out: Path,
    manifest: Path,
    *,
    expected_task: str,
    expected_scope: str | None,
) -> dict[str, Any]:
    """Classify a resumable inference/metric directory without launching GPU.

    This is shared by recovery tests and by operators auditing an interrupted
    gate.  A non-empty directory without a valid result is explicitly partial;
    it is never treated as a completed task.
    """

    result_file = eval_out / "result.json"
    metrics_file = eval_out / "official_metrics.json"
    if eval_out.exists() and any(eval_out.iterdir()) and not result_file.is_file():
        return {"status": "INVALID_PARTIAL_OUTPUT", "reason": "output exists without result.json"}
    if not result_file.is_file():
        return {"status": "RUN_INFERENCE", "result": str(result_file)}
    valid, detail = valid_inference_receipt(
        result_file, manifest, expected_task=expected_task, expected_scope=expected_scope
    )
    if not valid:
        return {"status": "INVALID_RESULT", "result": str(result_file), "detail": detail}
    if not metrics_file.is_file():
        return {"status": "RUN_METRICS", "result": str(result_file), "metrics": str(metrics_file), "detail": detail}
    valid_metrics, metrics_detail = valid_metrics_receipt(metrics_file, manifest, expected_task=expected_task)
    if not valid_metrics:
        return {"status": "INVALID_METRICS", "result": str(result_file), "metrics": str(metrics_file), "detail": metrics_detail}
    return {"status": "COMPLETE", "result": str(result_file), "metrics": str(metrics_file), "detail": detail, "metrics_detail": metrics_detail}


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
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--expected-training-manifest-sha256", default=None)
    parser.add_argument("--training-receipt", type=Path, default=None)
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
        if previous.get("status") == "HELDOUT_ALL_REGISTERED_TASKS_COMPLETE":
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
            "training_final_receipt": final_training_receipt(
                args.training_root,
                training_manifest=args.training_manifest,
                expected_training_manifest_sha256=args.expected_training_manifest_sha256,
                training_receipt=args.training_receipt,
            ),
            "mot_release_exists": args.mot_release.is_file(),
            "rmot_state_exists": args.rmot_state.is_file(),
            "manifest_exists": args.manifest.is_file(),
        }
        save()
        return 0

    deadline = time.monotonic() + args.max_wait_hours * 3600
    # A resumed owner may already have a DT1203 result and proceed directly to
    # RMOT/OVMOT, so construct the shared environment before that branch.
    env = build_eval_env(args.gpu)
    try:
        while True:
            ready, receipt = final_training_receipt(
                args.training_root,
                training_manifest=args.training_manifest,
                expected_training_manifest_sha256=args.expected_training_manifest_sha256,
                training_receipt=args.training_receipt,
            )
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
        if not args.manifest.is_file():
            return fail("held-out DT1203 manifest is missing", manifest=str(args.manifest))
        expected_dt_scope = json.loads(args.manifest.read_text()).get("scope")
        result_file = eval_out / "result.json"
        if eval_out.exists() and any(eval_out.iterdir()) and not result_file.is_file():
            return fail("held-out output exists without a result receipt", output=str(eval_out))
        if result_file.is_file():
            valid, detail = valid_inference_receipt(
                result_file, args.manifest, expected_task="mot", expected_scope=expected_dt_scope
            )
            if not valid:
                return fail("held-out DT1203 result receipt failed validation", detail=detail)
        else:
            state["status"] = "RUNNING_HELDOUT_DT1203"
            save()
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
            valid, detail = valid_inference_receipt(
                result_file, args.manifest, expected_task="mot", expected_scope=expected_dt_scope
            ) if result_file.is_file() else (False, {"reason": "result.json absent"})
            if proc.returncode != 0 or not valid:
                return fail("held-out DT1203 inference failed", returncode=proc.returncode, log=str(log_path))
        state["completed"]["dt1203_inference"] = {"run": str(eval_out), "result": str(result_file)}

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
            expected_scope = json.loads(manifest.read_text()).get("scope")
            if eval_out.exists() and any(eval_out.iterdir()) and not result_file.is_file():
                return fail("multitask output exists without a result receipt", name=name, output=str(eval_out))
            if result_file.is_file():
                valid, detail = valid_inference_receipt(
                    result_file, manifest, expected_task=task, expected_scope=expected_scope
                )
                if not valid:
                    return fail("multitask result receipt failed validation", name=name, detail=detail)
            else:
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
                valid, detail = valid_inference_receipt(
                    result_file, manifest, expected_task=task, expected_scope=expected_scope
                ) if result_file.is_file() else (False, {"reason": "result.json absent"})
                if proc.returncode != 0 or not valid:
                    return fail("multitask held-out inference failed", name=name,
                                returncode=proc.returncode, log=str(log_path), detail=detail)
            if metrics_file.is_file():
                valid_metrics, metrics_detail = valid_metrics_receipt(metrics_file, manifest, expected_task=task)
                if not valid_metrics:
                    return fail("multitask official metrics receipt failed validation", name=name, detail=metrics_detail)
            else:
                metrics_log = args.out / f"{name}_official_metrics.log"
                metrics_command = [PY, str(ROOT / "tools/groundjev/evaluate_multitask.py"),
                                   "--video", str(manifest), "--run", str(eval_out)]
                with metrics_log.open("a") as log:
                    metrics_proc = subprocess.run(metrics_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                valid_metrics, metrics_detail = valid_metrics_receipt(metrics_file, manifest, expected_task=task) if metrics_file.is_file() else (False, {"reason": "official_metrics.json absent"})
                if metrics_proc.returncode != 0 or not valid_metrics:
                    return fail("multitask held-out evaluator failed", name=name,
                                returncode=metrics_proc.returncode, log=str(metrics_log), detail=metrics_detail)
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
