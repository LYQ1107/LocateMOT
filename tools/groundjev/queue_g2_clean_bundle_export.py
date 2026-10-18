#!/usr/bin/env python3
"""Export the clean G2 adapter into a deployable bundle after its owners finish.

The formal G2 trainer intentionally keeps a strict FP32/Adam/RNG recovery
checkpoint and only writes ``adapter_projector_final.pt`` at the terminal step.
This owner converts that state into a versioned GroundAnything bundle without
touching the source model or allocating a GPU.  It waits for the registered
GPU5 G1 continuation as well so the large CPU merge cannot overlap the G1
GPU5/control window.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
G2_DEFAULT = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "formal_clean_coverage_v2_final"
)
G1_DEFAULT = ROOT / "outputs/groundjev/ground_adaptation/native_tracking200_gpu5_g1_v1/queue_owner_state.json"
OUT_DEFAULT = ROOT / "outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1"
BUNDLE_DEFAULT = ROOT / "models/GroundAnything-DLM-G2-CLEAN-V2"
MANIFEST_SHA = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
G2_STEPS = 18_227


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(
        json.dumps(
            {
                "status": state.get("status"),
                "current_stage": state.get("current_stage"),
                "completed": state.get("completed"),
                "failed": state.get("failed"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def g2_terminal(root: Path) -> tuple[bool, dict | None, str | None]:
    receipt_path = root / "final_receipt.json"
    receipt = read(receipt_path)
    if receipt is None:
        contract = read(root / "trainer_contract.json")
        if contract is not None and contract.get("status") not in {"G2_FORMAL_TRAINING_RUNNING"}:
            return False, contract, "G2 owner became terminal without a final receipt"
        return False, None, None
    if receipt.get("status") != "G2_EXACT_COVERAGE_TRAINING_COMPLETE":
        return False, receipt, "G2 terminal receipt is not the exact-once clean contract"
    if int(receipt.get("steps", -1)) != G2_STEPS:
        return False, receipt, "G2 terminal step does not match the registered coverage budget"
    if receipt.get("manifest_sha256") != MANIFEST_SHA:
        return False, receipt, "G2 terminal manifest SHA differs from the clean manifest"
    adapter = root / "adapter_projector_final.pt"
    if not adapter.is_file():
        return False, receipt, "G2 terminal receipt exists but adapter_projector_final.pt is missing"
    return True, receipt, None


def g1_terminal(path: Path) -> tuple[bool, dict | None, str | None]:
    state = read(path)
    if state is None:
        return False, None, None
    status = state.get("status")
    if status == "G1_GPU5_QUEUE_FAILED" or status == "G1_GPU5_NOT_STARTED_G2_DEPENDENCY_FAILED":
        return False, state, f"G1 owner failed: {status}"
    if status != "G1_GPU5_ADAPTATION_AND_CONTROLS_COMPLETE":
        return False, state, None
    bundle = Path(str(state.get("bundle", "")))
    if not bundle.is_dir() or not (bundle / "model.safetensors").is_file():
        return False, state, "G1 owner is marked complete but its bundle is missing"
    return True, state, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-root", type=Path, default=G2_DEFAULT)
    parser.add_argument("--g1-state", type=Path, default=G1_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--bundle", type=Path, default=BUNDLE_DEFAULT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.out.exists() or args.bundle.exists():
        raise FileExistsError("refusing to overwrite an existing G2 clean export")

    owner_path = args.out / "owner_state.json"
    state = {
        "status": "WAITING_G2_AND_G1_TERMINAL",
        "current_stage": "WAITING_G2_AND_G1_TERMINAL",
        "owner_pid": os.getpid(),
        "g2_root": str(args.g2_root),
        "g1_state": str(args.g1_state),
        "out": str(args.out),
        "bundle": str(args.bundle),
        "manifest_sha256": MANIFEST_SHA,
        "expected_g2_steps": G2_STEPS,
        "physical_gpu": None,
        "feature_cache": False,
        "completed": {},
        "failed": None,
        "source_bundle_untouched": True,
    }
    save(owner_path, state)
    try:
        while True:
            g2_ok, g2, g2_error = g2_terminal(args.g2_root)
            if g2_error:
                state.update(
                    status="G2_CLEAN_BUNDLE_NOT_EXPORTED_G2_FAILED",
                    current_stage="G2_DEPENDENCY_FAILED",
                    failed={"reason": g2_error, "evidence": g2},
                )
                save(owner_path, state)
                return 2
            g1_ok, g1, g1_error = g1_terminal(args.g1_state)
            if g1_error:
                state.update(
                    status="G2_CLEAN_BUNDLE_WAIT_REFUSED_G1_FAILED",
                    current_stage="G1_DEPENDENCY_FAILED",
                    failed={"reason": g1_error, "evidence": g1},
                )
                save(owner_path, state)
                return 2
            if g2_ok and g1_ok:
                state["completed"]["g2_training"] = g2
                state["completed"]["g1_gpu5"] = {
                    "status": g1.get("status"),
                    "bundle": g1.get("bundle"),
                    "controls": g1.get("controls"),
                }
                break
            time.sleep(args.poll_seconds)

        state.update(status="EXPORTING_G2_CLEAN_DEPLOYMENT_BUNDLE", current_stage="EXPORTING_G2_CLEAN_DEPLOYMENT_BUNDLE")
        save(owner_path, state)
        adapter = args.g2_root / "adapter_projector_final.pt"
        receipt = args.out / "export_receipt.json"
        args.out.mkdir(parents=True, exist_ok=True)
        command = [
            "python3",
            "tools/groundjev/g2/export_g2_checkpoint.py",
            "--adapter",
            str(adapter),
            "--out",
            str(args.bundle),
            "--receipt",
            str(receipt),
            "--step",
            str(G2_STEPS),
            "--manifest-sha256",
            MANIFEST_SHA,
        ]
        with (args.out / "G2_EXPORT.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"G2 exporter exited {result.returncode}")
        exported = read(receipt)
        if exported is None or exported.get("status") != "G2_NATIVE_MERGED_BUNDLE_EXPORTED":
            raise RuntimeError("G2 exporter did not write a valid terminal receipt")
        if exported.get("native_tensor_count") != 569 or exported.get("source_bundle_overwritten"):
            raise RuntimeError("G2 bundle tensor-preservation gate failed")
        state["completed"]["export"] = exported
        state.update(status="G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED", current_stage="COMPLETE")
        save(owner_path, state)
        return 0
    except Exception as error:
        state.update(status="G2_CLEAN_BUNDLE_EXPORT_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
