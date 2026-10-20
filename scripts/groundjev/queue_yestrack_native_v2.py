#!/usr/bin/env python3
"""Queue the real native YesTrack validation run on an exclusive GPU7.

The corpus and CPU rendering plan are prepared before this owner starts.  The
owner waits internally for GPU7 to have no compute clients and at least 32 GiB
free, then performs one real LoRA training run followed by frozen-base and
fine-tuned validation on the same held-out rows.  It never stops or reuses an
unrelated process and it writes an explicit terminal receipt only after all
three stages complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
QUEUE_ROOT = RUNTIME_ROOT / "outputs/groundjev/coupled/yestrack_native_v2_queue"
TRAIN_ROOT = Path("/data3/user/LocateMOT-checkpoints/yestrack_native_v2_lora")
CONFIG = PROJECT_ROOT / "configs/groundjev/yestrack_native_v2_runtime.json"
TRAIN_LOG = TRAIN_ROOT / "log.jsonl"
ADAPTER = TRAIN_ROOT / "adapter"
BASE_MODEL = RUNTIME_ROOT / "outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600"
TRAIN_ROWS = RUNTIME_ROOT / "outputs/groundjev/coupled/native_yesno_v2_phaseb.jsonl"
VAL_ROWS = RUNTIME_ROOT / "outputs/groundjev/coupled/native_yesno_v2_phaseb_val.jsonl"
GPU = 7
EXPECTED_TRAIN_ROWS = 726
EXPECTED_VAL_ROWS = 390
EXPECTED_EPOCHS = 3
EXPECTED_TOTAL_STEPS = 273
EXPECTED_TRAIN_EXAMPLES = EXPECTED_TRAIN_ROWS * EXPECTED_EPOCHS
EXPECTED_TRAIN_SHA = "065337d2fedca2377d02b94e57c9eb3778aeca09aeb1a8b9463429fa15bbee66"
EXPECTED_VAL_SHA = "d63a9fb0d54b8a1dc5d12b0960a48cb75c56ab786671f955fb317e196b3033d8"
VENV_PYTHON = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(json.dumps({k: state.get(k) for k in (
        "status", "current_stage", "last_exit_code", "failed"
    )}, ensure_ascii=False), flush=True)


def compute_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(GPU), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def free_memory_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(GPU), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"unexpected GPU7 memory response: {result.stdout!r}")
    return values[0]


def resource_probe() -> tuple[bool, dict]:
    pids = compute_pids()
    free = free_memory_mib()
    return (not pids and free >= 32768), {
        "physical_gpu": GPU,
        "compute_pids": pids,
        "free_memory_MiB": free,
        "rule": "no compute PIDs and at least 32768 MiB free",
    }


def validate_inputs() -> dict:
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    if not BASE_MODEL.is_dir():
        raise FileNotFoundError(BASE_MODEL)
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(VENV_PYTHON)
    if not TRAIN_ROWS.is_file() or sha256(TRAIN_ROWS) != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"YesTrack train corpus SHA mismatch: {TRAIN_ROWS}")
    if not VAL_ROWS.is_file() or sha256(VAL_ROWS) != EXPECTED_VAL_SHA:
        raise RuntimeError(f"YesTrack validation corpus SHA mismatch: {VAL_ROWS}")
    train_rows = 0
    source_total = 0
    with TRAIN_ROWS.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            source_total += 1
            if json.loads(line).get("split") == "train":
                train_rows += 1
    val_rows = sum(1 for line in VAL_ROWS.open() if line.strip())
    if train_rows != EXPECTED_TRAIN_ROWS or val_rows != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"unexpected YesTrack row counts: train={train_rows} val={val_rows}")
    return {
        "train_rows": train_rows, "source_total_rows": source_total, "validation_rows": val_rows,
        "train_sha256": EXPECTED_TRAIN_SHA, "validation_sha256": EXPECTED_VAL_SHA,
        "official_test_labels_read": False, "future_evidence": False,
        "gt_used_for_prompt": False,
    }


def records_from_log() -> list[dict]:
    if not TRAIN_LOG.is_file():
        return []
    rows = []
    for line in TRAIN_LOG.read_text(errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run_wrapped(command: list[str], log_path: Path) -> int:
    wrapped = [str(RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"),
               "--gpus", str(GPU), "--", *command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(wrapped, cwd=RUNTIME_ROOT, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, default=QUEUE_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    owner = args.queue_root / "owner_state.json"
    old = read_json(owner)
    if old and str(old.get("status", "")).endswith(("_COMPLETE", "_FAILED")):
        raise FileExistsError(f"terminal YesTrack queue already exists: {owner}")
    if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
        raise FileExistsError(f"YesTrack queue owner is already live: {old['owner_pid']}")
    if args.queue_root.exists():
        if not old or str(old.get("status", "")) != "WAITING_GPU7_EXCLUSIVE":
            raise FileExistsError(f"refusing to overwrite an existing YesTrack queue: {args.queue_root}")
        unexpected_queue = [
            child.name for child in args.queue_root.iterdir()
            if child.name not in {"owner_state.json", "OWNER.log"}
        ]
        if unexpected_queue:
            raise FileExistsError(
                f"refusing to resume a partial YesTrack queue; unexpected={unexpected_queue}"
            )
    if TRAIN_ROOT.exists():
        unexpected_train = [child.name for child in TRAIN_ROOT.iterdir() if child.name != "args.json"]
        if unexpected_train:
            raise FileExistsError(
                f"refusing to overwrite YesTrack training artifacts; unexpected={unexpected_train}"
            )

    inputs = validate_inputs()
    state = {
        "status": "WAITING_GPU7_EXCLUSIVE",
        "current_stage": "WAITING_GPU7_EXCLUSIVE",
        "owner_pid": os.getpid(),
        "project_root": str(PROJECT_ROOT), "runtime_root": str(RUNTIME_ROOT),
        "physical_gpu": GPU, "config": str(CONFIG), "base_model": str(BASE_MODEL),
        "train_rows": str(TRAIN_ROWS), "validation_rows": str(VAL_ROWS),
        "training_root": str(TRAIN_ROOT), "adapter": str(ADAPTER),
        "expected_train_rows": EXPECTED_TRAIN_ROWS, "expected_validation_rows": EXPECTED_VAL_ROWS,
        "epochs": EXPECTED_EPOCHS, "expected_total_steps": EXPECTED_TOTAL_STEPS,
        "expected_terminal_examples": EXPECTED_TRAIN_EXAMPLES,
        "inputs": inputs, "completed": {}, "failed": None,
        "other_active_jobs_untouched": True, "real_gpu_training_required": True,
        "resumed_wait_owner": bool(old),
    }
    atomic_save(owner, state)
    try:
        while True:
            ready, evidence = resource_probe()
            state["resource_probe"] = evidence
            state["resource_gate"] = {
                "ready": ready,
                "checked_at_epoch_s": time.time(),
                "rule": "GPU7 has no compute PIDs and >=32768 MiB free",
            }
            if ready:
                break
            atomic_save(owner, state)
            time.sleep(args.poll_seconds)

        train_command = [
            str(VENV_PYTHON), str(RUNTIME_ROOT / "tools/groundjev/run_source_manifest_sft.py"),
            "--source-training-backend", "fla", "--config", str(CONFIG),
            "--deterministic-execution", "--seed-global-rngs",
        ]
        state.update(status="REAL_YESTRACK_TRAINING_RUNNING",
                     current_stage="REAL_YESTRACK_TRAINING_RUNNING", command=train_command)
        atomic_save(owner, state)
        train_exit = run_wrapped(train_command, args.queue_root / "TRAINING.log")
        state["train_exit_code"] = train_exit
        records = records_from_log()
        done = next((row for row in reversed(records) if row.get("event") == "done"), None)
        state["training_terminal_record"] = done
        if (train_exit != 0 or done is None
                or int(done.get("examples", -1)) != EXPECTED_TRAIN_EXAMPLES
                or int(done.get("steps", -1)) != EXPECTED_TOTAL_STEPS):
            state.update(status="REAL_YESTRACK_TRAINING_FAILED", current_stage="FAILED",
                         failed={"reason": "training_nonzero_or_incomplete",
                                 "expected_examples": EXPECTED_TRAIN_EXAMPLES,
                                 "expected_steps": EXPECTED_TOTAL_STEPS})
            atomic_save(owner, state)
            return train_exit or 3
        if not ADAPTER.is_dir() or not (ADAPTER / "adapter_model.safetensors").is_file():
            state.update(status="REAL_YESTRACK_TRAINING_FAILED", current_stage="FAILED",
                         failed={"reason": "final_adapter_missing", "adapter": str(ADAPTER)})
            atomic_save(owner, state)
            return 4

        evaluator = str(PROJECT_ROOT / "scripts/groundjev/eval_yestrack_native.py")
        base_out = args.queue_root / "yestrack_base_val.json"
        tuned_out = args.queue_root / "yestrack_finetuned_val.json"
        base_command = [str(VENV_PYTHON), evaluator, "--runtime-root", str(RUNTIME_ROOT),
                        "--base-model", str(BASE_MODEL), "--rows", str(VAL_ROWS),
                        "--media-root", str(RUNTIME_ROOT), "--device", "cuda:0",
                        "--batch-tokens", "32768", "--max-rows", "1", "--max-len", "32768",
                        "--out", str(base_out)]
        state.update(status="YESTRACK_BASE_EVAL_RUNNING", current_stage="YESTRACK_BASE_EVAL_RUNNING",
                     base_eval_command=base_command)
        atomic_save(owner, state)
        base_exit = run_wrapped(base_command, args.queue_root / "BASE_EVAL.log")
        state["base_eval_exit_code"] = base_exit
        if base_exit != 0 or not base_out.is_file():
            state.update(status="REAL_YESTRACK_EVALUATION_FAILED", current_stage="FAILED",
                         failed={"reason": "base_eval_failed", "exit_code": base_exit})
            atomic_save(owner, state)
            return base_exit or 5

        tuned_command = [str(VENV_PYTHON), evaluator, "--runtime-root", str(RUNTIME_ROOT),
                         "--base-model", str(BASE_MODEL), "--checkpoint", str(ADAPTER),
                         "--rows", str(VAL_ROWS), "--media-root", str(RUNTIME_ROOT),
                         "--device", "cuda:0", "--batch-tokens", "32768", "--max-rows", "1",
                         "--max-len", "32768", "--out", str(tuned_out)]
        state.update(status="YESTRACK_FINETUNED_EVAL_RUNNING", current_stage="YESTRACK_FINETUNED_EVAL_RUNNING",
                     finetuned_eval_command=tuned_command)
        atomic_save(owner, state)
        tuned_exit = run_wrapped(tuned_command, args.queue_root / "FINETUNED_EVAL.log")
        state["finetuned_eval_exit_code"] = tuned_exit
        if tuned_exit != 0 or not tuned_out.is_file():
            state.update(status="REAL_YESTRACK_EVALUATION_FAILED", current_stage="FAILED",
                         failed={"reason": "finetuned_eval_failed", "exit_code": tuned_exit})
            atomic_save(owner, state)
            return tuned_exit or 6

        base = read_json(base_out) or {}
        tuned = read_json(tuned_out) or {}
        if int(base.get("row_count", -1)) != EXPECTED_VAL_ROWS or int(tuned.get("row_count", -1)) != EXPECTED_VAL_ROWS:
            state.update(status="REAL_YESTRACK_EVALUATION_FAILED", current_stage="FAILED",
                         failed={"reason": "evaluation_row_count_mismatch", "base": base.get("row_count"),
                                 "finetuned": tuned.get("row_count")})
            atomic_save(owner, state)
            return 7
        state.update(status="REAL_YESTRACK_TRAINING_AND_EVALUATION_COMPLETE", current_stage="COMPLETE",
                     completed={"training": done, "base_eval": base.get("metrics"),
                                "finetuned_eval": tuned.get("metrics")},
                     terminal_receipts={"base": str(base_out), "finetuned": str(tuned_out)},
                     full_goal_complete=False)
        atomic_save(owner, state)
        return 0
    except Exception as error:
        state.update(status="REAL_YESTRACK_TRAINING_FAILED", current_stage="FAILED",
                     failed={"error": repr(error)})
        atomic_save(owner, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
