#!/usr/bin/env python3
"""Queue the registered G1 projector adaptation behind the clean G2 owner.

The current clean G2 course owns GPU5.  This owner therefore waits for an
authoritative successful G2 completion receipt before acquiring GPU5, trains a
fresh G1 projector from G0, exports a new bundle without touching older G1/G200
artifacts, and runs the fixed isolated native controls.  A G2 failure is
recorded and never converted into a G1 launch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
G2_DEFAULT = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final")
G1_PLAN = ROOT / "configs/groundjev/native_ground_tracking200_20261014.json"
G1_BASE = ROOT / "models/GroundAnything-DLM"
G1_OUT = ROOT / "outputs/groundjev/ground_adaptation/native_tracking200_gpu5_g1_v1"
G1_BUNDLE = ROOT / "models/GroundAnything-DLM-TrackProjG1-GPU5-V2"
G1_CONTROLS = ROOT / "outputs/groundjev/ground_adaptation/g1_gpu5_isolated96_v1"


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "completed", "failed")}, ensure_ascii=False), flush=True)


def run(command: list[str], log: Path, *, cwd: Path = ROOT) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"command exited {proc.returncode}: {' '.join(command)}")


def successful_g2(g2: Path) -> tuple[bool, dict | None, str | None]:
    receipt_path = g2 / "final_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("status") == "G2_EXACT_COVERAGE_TRAINING_COMPLETE"
            and int(receipt.get("steps", -1)) == 18227
            and receipt.get("manifest_sha256") == "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
            and (g2 / "adapter_projector_final.pt").is_file()
        ):
            return True, receipt, None
        return False, receipt, "G2 final receipt exists but failed the clean-contract gate"
    contract_path = g2 / "trainer_contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text())
        status = contract.get("status")
        if status not in ("G2_FORMAL_TRAINING_RUNNING",):
            return False, contract, f"G2 owner became terminal without a valid final receipt: {status}"
    return False, None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-root", type=Path, default=G2_DEFAULT)
    parser.add_argument("--out", type=Path, default=G1_OUT)
    parser.add_argument("--bundle", type=Path, default=G1_BUNDLE)
    parser.add_argument("--controls", type=Path, default=G1_CONTROLS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    owner = args.out / "queue_owner_state.json"
    previous = json.loads(owner.read_text()) if owner.is_file() else None
    if previous and previous.get("owner_pid") and Path(f"/proc/{previous['owner_pid']}").exists():
        raise FileExistsError(f"G1 GPU5 owner is already live: {previous['owner_pid']}")
    if args.bundle.exists() or args.controls.exists():
        raise FileExistsError("refusing to overwrite an existing G1 GPU5 artifact")
    if args.out.exists():
        unfinished = previous and str(previous.get("status", "")) == "WAITING_G2_CLEAN_SUCCESS"
        unexpected = [child for child in args.out.iterdir() if child.name != "queue_owner_state.json"]
        if not unfinished or unexpected:
            raise FileExistsError("refusing to overwrite an existing or partial G1 GPU5 artifact")
    state = {
        "status": "WAITING_G2_CLEAN_SUCCESS",
        "current_stage": "WAITING_G2_CLEAN_SUCCESS",
        "owner_pid": os.getpid(),
        "g2_root": str(args.g2_root),
        "physical_gpu": 5,
        "g1_plan": str(G1_PLAN),
        "base_model": str(G1_BASE),
        "out": str(args.out),
        "bundle": str(args.bundle),
        "controls": str(args.controls),
        "completed": {},
        "failed": None,
        "other_active_jobs_untouched": True,
        "feature_cache": False,
        "resumed_wait_owner": bool(previous),
    }
    save(owner, state)
    try:
        while True:
            ok, evidence, error = successful_g2(args.g2_root)
            if error:
                state.update(status="G1_GPU5_NOT_STARTED_G2_DEPENDENCY_FAILED", current_stage="G2_DEPENDENCY_FAILED",
                             failed={"reason": error, "evidence": evidence})
                save(owner, state)
                return 2
            if ok:
                state["completed"]["g2"] = evidence
                break
            time.sleep(args.poll_seconds)

        state.update(status="STARTING_G1_GPU5_TRAINING", current_stage="G1_GPU5_TRAINING")
        save(owner, state)
        # The queue state itself already created ``args.out``.  Keep the
        # directory and place all mutable training artifacts in its child.
        args.out.mkdir(parents=True, exist_ok=True)
        train_out = args.out / "training"
        run([
            "python3", "tools/groundjev/run_with_resources.py", "--gpus", "5", "--",
            str(ROOT / ".venv-ground-train-py312/bin/python"),
            "tools/groundjev/train_native_ground_tracking200.py",
            "--plan", str(G1_PLAN), "--out", str(train_out),
            "--static-flex-shapes", "--debug-kernel-sync", "--checkpoint-every", "25",
        ], args.out / "G1_GPU5_TRAINING.log")
        trained = json.loads((train_out / "owner_state.json").read_text())
        if trained.get("status") != "NATIVE_GROUND_TRACKING200_TRAINING_COMPLETE" or trained.get("steps") != 200:
            raise RuntimeError(f"unexpected G1 training receipt: {trained.get('status')} / {trained.get('steps')}")
        state["completed"]["training"] = {
            "run": str(train_out), "status": trained["status"], "steps": trained["steps"],
            "trainable_parameters": trained.get("trainable_parameters"),
            "changed_BF16_elements": trained.get("changed_BF16_elements"),
            "checkpoint_reload_max_difference": trained.get("checkpoint_reload_max_difference"),
            "all_registered_pairs_visited": trained.get("all_registered_pairs_visited"),
        }
        save(owner, state)

        state.update(status="EXPORTING_G1_GPU5_BUNDLE", current_stage="EXPORTING_G1_GPU5_BUNDLE")
        save(owner, state)
        receipt = args.out / "export_receipt.json"
        run([
            str(ROOT / ".venv-ground-train-py312/bin/python"),
            "tools/groundjev/export_native_projector_delta.py",
            "--delta", str(train_out / "projector_delta.pt"),
            "--out", str(args.bundle), "--receipt", str(receipt),
        ], args.out / "G1_GPU5_EXPORT.log")
        exported = json.loads(receipt.read_text())
        if exported.get("native_tensor_count") != 569 or not exported.get("all_other_native_tensors_unmodified"):
            raise RuntimeError("G1 export tensor-preservation gate failed")
        state["completed"]["export"] = exported
        save(owner, state)

        state.update(status="RUNNING_G1_GPU5_ISOLATED_CONTROLS", current_stage="RUNNING_G1_GPU5_ISOLATED_CONTROLS")
        save(owner, state)
        plan = {
            "name": "G0/G200/G1-GPU5 isolated native controls",
            "physical_gpu": 5,
            "GPU_model": "registered physical GPU5",
            "port": 8121,
            "requested_sampling_seed": 42,
            "execution": "fresh service per variant, fixed manifests and official scorers",
            "metrics": "unchanged official MOT/RMOT metrics with empty/rejected frames retained",
            "limits": "96 native frames across dense MOT4 and RMOT single8/multi12; not full-dataset claim",
            "variants": [
                {"name": "original", "checkpoint": "models/GroundAnything-DLM"},
                {"name": "projector200", "checkpoint": "models/GroundAnything-DLM-TrackProj200"},
                {"name": "g1_gpu5", "checkpoint": str(args.bundle)},
                {"name": "g1_gpu5_repeat", "checkpoint": str(args.bundle)},
            ],
            "repeat_pair": ["g1_gpu5", "g1_gpu5_repeat"],
            "clips": [
                {"name": "dense_mot4", "task": "mot", "manifest": "outputs/groundjev/pilot_data/mot17_first4.json"},
                {"name": "refer_single8", "task": "rmot", "manifest": "outputs/groundjev/multitask_pilot_data/rmot_single_empty.json"},
                {"name": "refer_multi12", "task": "rmot", "manifest": "outputs/groundjev/multitask_pilot_data/rmot_multi_empty.json"},
            ],
        }
        plan_path = args.controls.parent / "g1_gpu5_controls_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        run(["python3", "tools/groundjev/compare_ground_checkpoints_isolated.py",
             "--plan", str(plan_path), "--out", str(args.controls)], args.out / "G1_GPU5_CONTROLS.log")
        controls = json.loads((args.controls / "owner_state.json").read_text())
        if controls.get("status") != "ISOLATED_ORDERED_GROUND_CONTROLS_COMPLETE":
            raise RuntimeError(f"G1 controls did not complete: {controls.get('status')}")
        state["completed"]["isolated_controls"] = {
            "run": str(args.controls), "status": controls["status"],
            "completed_units": len(controls.get("completed", {})),
            "repeat_checks": controls.get("repeat_checks"),
            "scope": "96 registered native frames, not full-dataset evidence",
        }
        state.update(status="G1_GPU5_ADAPTATION_AND_CONTROLS_COMPLETE", current_stage="COMPLETE")
        save(owner, state)
        return 0
    except Exception as error:
        state.update(status="G1_GPU5_QUEUE_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
