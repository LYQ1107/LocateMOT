#!/usr/bin/env python3
"""Create a read-only audit receipt for the current GroundJEV submission.

The command reads contracts, event logs, durable checkpoint metadata and
registered owner states once. It never stops or restarts a worker and never
mutates an active training directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_alive(pid: int | None) -> bool | None:
    if pid is None or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_jsonl(path: Path) -> tuple[int, dict[str, Any] | None, list[str]]:
    count = 0
    latest: dict[str, Any] | None = None
    anomalies: list[str] = []
    if not path.is_file():
        return count, latest, anomalies
    for line_no, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except Exception as error:
            anomalies.append(f"invalid_jsonl_line:{line_no}:{type(error).__name__}")
            continue
        if not isinstance(event, dict):
            anomalies.append(f"non_object_jsonl_line:{line_no}")
            continue
        count += 1
        latest = event
        for field in ("loss", "gradient_norm", "grad_norm", "lr", "slot_acc"):
            value = event.get(field)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                anomalies.append(f"non_finite:{field}:line{line_no}")
        for field in ("skipped", "skipped_rows", "truncated", "truncated_rows"):
            value = event.get(field)
            if isinstance(value, (int, float)) and value:
                anomalies.append(f"{field}={value}:line{line_no}")
    return count, latest, anomalies


def log_anomalies(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    patterns = [
        (re.compile(r"cuda out of memory|out of memory|oom", re.I), "oom_text"),
        (re.compile(r"nan|non[- ]finite", re.I), "non_finite_text"),
        (re.compile(r"truncat|skip(?:ped|ping)", re.I), "skip_or_truncation_text"),
    ]
    for path in paths:
        if not path.is_file():
            continue
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for pattern, label in patterns:
                if pattern.search(line):
                    findings.append(f"{label}:{path.name}:line{line_no}:{line[:240]}")
    return findings


def checkpoint_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "mtime_ns": stat.st_mtime_ns,
    }


def classify(
    *,
    terminal: bool,
    failed: bool,
    alive: bool | None,
    has_checkpoint: bool,
) -> str:
    if terminal:
        return "COMPLETED"
    if failed:
        return "FAILED"
    if alive is True:
        return "RUNNING"
    if has_checkpoint:
        return "RECOVERABLE"
    return "UNKNOWN"


def audit_g2(root: Path) -> dict[str, Any]:
    contract = read_json(root / "trainer_contract.json") or {}
    launch = read_json(root / "launch_contract.json") or {}
    receipt = read_json(root / "final_receipt.json")
    manifest = contract.get("manifest") if isinstance(contract.get("manifest"), dict) else {}
    event_count, latest_event, event_anomalies = read_jsonl(root / "training_events.jsonl")
    latest_state = root / "checkpoints" / "latest_training_state.pt"
    checkpoint_paths = sorted(root.glob("checkpoints/adapter-step*.pt"))
    log_findings = log_anomalies([root / "launch.log"])
    terminal = bool(receipt and str(receipt.get("status", "")).endswith("COMPLETE"))
    failed = bool(str(contract.get("status", "")).endswith("FAILED"))
    rows_path = Path(str(manifest.get("rows_path", contract.get("rows", ""))))
    base_path = Path(str(launch.get("base_checkpoint", ""))) / "model.safetensors"
    return {
        "status": classify(
            terminal=terminal,
            failed=failed,
            alive=None if terminal else not failed,
            has_checkpoint=latest_state.is_file(),
        ),
        "contract_status": contract.get("status"),
        "launch_status": launch.get("status"),
        "terminal_receipt": receipt,
        "physical_gpus": launch.get("physical_gpus") or contract.get("requested_physical_gpus"),
        "world_size": contract.get("world_size"),
        "max_steps": contract.get("max_steps") or launch.get("max_steps"),
        "latest_event": latest_event,
        "event_count": event_count,
        "checkpoint_count": len(checkpoint_paths),
        "latest_checkpoint": checkpoint_summary(latest_state),
        "latest_named_checkpoint": checkpoint_summary(checkpoint_paths[-1] if checkpoint_paths else None),
        "manifest": {
            "path": str(rows_path),
            "recorded_sha256": manifest.get("rows_sha256"),
            "actual_sha256": sha256(rows_path),
            "row_count": manifest.get("row_count"),
            "reject_count": manifest.get("reject_count"),
            "coverage_mode": contract.get("training_contract", {}).get("coverage_mode"),
            "effective_global_batch": contract.get("training_contract", {}).get("effective_global_batch"),
            "grad_accum_steps": contract.get("training_contract", {}).get("grad_accum_steps"),
            "sampler_audit": contract.get("sampler", {}).get("audit"),
        },
        "base_model": {
            "path": str(base_path.parent),
            "recorded_sha256": manifest.get("base_model_safetensors_sha256"),
            "actual_sha256": sha256(base_path),
        },
        "training_contract": contract.get("training_contract"),
        "official_dlm_memory_contract": contract.get("official_dlm_memory_contract"),
        "anomalies": event_anomalies + log_findings,
    }


def audit_j1(root: Path, pid: int | None) -> dict[str, Any]:
    state = read_json(root / "latest_training_state.json") or {}
    receipt = read_json(root / "final_receipt.json")
    event_count, latest_event, event_anomalies = read_jsonl(root / "log.jsonl")
    checkpoint = Path(str(state.get("path", root / "latest_training_state.pt")))
    terminal = bool(receipt and str(receipt.get("status", "")).endswith("COMPLETE"))
    failed = bool(state.get("failed"))
    return {
        "status": classify(
            terminal=terminal,
            failed=failed,
            alive=process_alive(pid) if not terminal else False,
            has_checkpoint=checkpoint.is_file(),
        ),
        "pid": pid,
        "process_alive": process_alive(pid),
        "terminal_receipt": receipt,
        "latest_observed_event": latest_event,
        "event_count": event_count,
        "durable_state": {
            "step": state.get("step"),
            "seen": state.get("seen"),
            "epoch": state.get("epoch"),
            "next_step_in_epoch": state.get("next_step_in_epoch"),
            "format": state.get("format"),
            "fp32_adam_saved": state.get("FP32_masters_and_Adam_moments_saved"),
            "scheduler_rng_saved": state.get("scheduler_and_all_RNG_saved"),
            "feature_cache": state.get("feature_cache"),
            "checkpoint": checkpoint_summary(checkpoint),
        },
        "anomalies": event_anomalies + log_anomalies([root / "log.jsonl"]),
    }


def audit_owner(path: Path, pid_key: str | None = None) -> dict[str, Any]:
    state = read_json(path) or {}
    pid = state.get(pid_key) if pid_key else state.get("owner_pid")
    pid = int(pid) if isinstance(pid, (int, float, str)) and str(pid).isdigit() else None
    status = str(state.get("status", "UNKNOWN"))
    terminal = status.endswith("COMPLETE") or status in {"G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED"}
    failed = "FAILED" in status or "REFUSED" in status
    return {
        "path": str(path),
        "status": classify(terminal=terminal, failed=failed, alive=process_alive(pid), has_checkpoint=False),
        "owner_status": status,
        "pid": pid,
        "process_alive": process_alive(pid),
        "state": state,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--j1-root", type=Path, required=True)
    parser.add_argument("--j1-pid", type=int, default=None)
    parser.add_argument("--g1-state", type=Path, required=True)
    parser.add_argument("--g2-export-state", type=Path, required=True)
    parser.add_argument("--j1-gate-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "format": "groundjev-latest-submission-audit-v1",
        "generated_unix": time.time(),
        "generated_localtime": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "read_only": True,
        "g2": audit_g2(args.g2_root),
        "j1": audit_j1(args.j1_root, args.j1_pid),
        "g1_queue": audit_owner(args.g1_state),
        "g2_export_queue": audit_owner(args.g2_export_state),
        "j1_heldout_gate": audit_owner(args.j1_gate_state, "owner_pid"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "status": "AUDIT_WRITTEN",
        "out": str(args.out),
        "g2_status": report["g2"]["status"],
        "g2_step": (report["g2"]["latest_event"] or {}).get("step"),
        "j1_status": report["j1"]["status"],
        "j1_observed_step": (report["j1"]["latest_observed_event"] or {}).get("step"),
        "j1_durable_step": report["j1"]["durable_state"].get("step"),
        "g1_queue_status": report["g1_queue"]["owner_status"],
        "g2_export_status": report["g2_export_queue"]["owner_status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
