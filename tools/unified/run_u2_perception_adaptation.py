#!/usr/bin/env python3
"""One blocking controller for the existing F0 waits and the bounded U2-P run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.unified.train_u2_perception_adaptation import DIRECTORY, digest, preflight
from tools.unified.build_u2_portability_artifact import read_measurement


def publish(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)
    summary_path = ROOT / "outputs/unified/u2/final_status.json"
    summary = json.loads(summary_path.read_text())
    adaptation = summary.setdefault("perception_adaptation", {})
    adaptation.update(state=state["state"], phase=state["phase"], run_state=str(path),
                      controller_pid=state["controller_pid"],
                      optimization_launched=state.get("optimization_launched", False))
    for key in ("checkpoint", "foundation_sha256", "selection", "selection_status"):
        if key in state:
            adaptation[key] = state[key]
    if state["state"] == "PERCEPTION_SELECTED":
        summary["perception_portability"] = str(Path(state["selection"]).parent / "portability.json")
        summary["track_training_eligible"] = True
        summary["status"] = "LOCATEMOT_U_U2_IN_PROGRESS"
    if "blocked_status" in state:
        summary["status"] = state["blocked_status"]
        summary["track_training_eligible"] = False
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(state), flush=True)


def wait_existing(pids: dict[str, int]) -> None:
    # The actual still-live parent PIDs are supplied at launch. This command
    # owns the wait; the agent never needs repeated process/log/GPU polling.
    remaining = dict(pids)
    while remaining:
        for job, pid in tuple(remaining.items()):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                if job == "replay":
                    manifest = DIRECTORY / "replay_manifest.json"
                    if not manifest.is_file() or json.loads(manifest.read_text())["rows_per_epoch"] != 20000:
                        raise RuntimeError("Existing replay preparation ended without its complete 20,000-row manifest")
                else:
                    datasets = ("tao", "ctao") if job == "tao_audit" else ("bdd100k",)
                    for name in datasets:
                        if read_measurement(ROOT / "outputs/unified/u2/perception/measurements", name) is None:
                            raise RuntimeError(f"Existing {job} ended without a complete {name} measurement")
                del remaining[job]
                print(json.dumps({"state": "EXISTING_JOB_VALIDATED", "job": job}), flush=True)
        if remaining:
            time.sleep(10)


def step(command: list[str], log: Path, state_path: Path, state: dict, phase: str) -> None:
    state.update(state="RUNNING", phase=phase, command=command, log=str(log))
    if phase.startswith("TRAIN_"):
        state["optimization_launched"] = True
    publish(state_path, state)
    with log.open("a") as handle:
        subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tao-audit-pid", type=int, required=True)
    parser.add_argument("--bdd-audit-pid", type=int, required=True)
    parser.add_argument("--replay-pid", type=int, required=True)
    parser.add_argument("--extraction-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.out.resolve() != DIRECTORY.resolve():
        parser.error("The registered replay and training work directory use the fixed U2-P output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "run_state.json"
    work_dir = args.out / "training"
    pids = {"tao_audit": args.tao_audit_pid, "bdd_audit": args.bdd_audit_pid, "replay": args.replay_pid}
    state = {"format": "locatemot-u2-bounded-perception-run-v1", "state": "WAITING_EXISTING_JOBS",
             "phase": "F0_AND_SOURCE_DATA", "wait_pids": pids, "controller_pid": os.getpid(),
             "world_size": 8, "maximum_epochs": 2, "maximum_optimizer_steps": 2500,
             "track_training_launched": False, "official_test_labels_read": False}
    publish(state_path, state)
    try:
        wait_existing(pids)
        eligibility = preflight(argparse.Namespace(epoch=1, epoch1_selection=None, resume=None))
        if eligibility["state"] != "READY":
            state.update(state="PRECONDITIONS_NOT_MET", phase="F0_AND_SOURCE_DATA", eligibility=eligibility)
            publish(state_path, state)
            return 1
        for epoch in (1, 2):
            checkpoint = work_dir / f"epoch_{epoch}.pth"
            if not checkpoint.exists():
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                           str(ROOT / "tools/unified/train_u2_perception_adaptation.py"),
                           "--epoch", str(epoch), "--work-dir", str(work_dir)]
                if epoch == 2:
                    command += ["--resume", str(work_dir / "epoch_1.pth"),
                                "--epoch1-selection", str(args.out / "eval_epoch_1/selection.json")]
                step(command, args.out / f"train_epoch_{epoch}.log", state_path, state, f"TRAIN_EPOCH_{epoch}")
            if not checkpoint.is_file():
                raise RuntimeError(f"Training ended without the registered checkpoint: {checkpoint}")
            sha = digest(checkpoint)
            state["optimization_launched"] = True
            step([sys.executable, "-u", str(ROOT / "tools/unified/eval_u2_perception_adaptation.py"),
                  "--checkpoint", str(checkpoint), "--epoch", str(epoch),
                  "--extraction-pid", str(args.extraction_pid)],
                 args.out / f"eval_epoch_{epoch}.log", state_path, state, f"FULL_AUDIT_EPOCH_{epoch}")
            selection_path = args.out / f"eval_epoch_{epoch}/selection.json"
            selection = json.loads(selection_path.read_text())
            if selection["foundation_sha256"] != sha:
                raise ValueError("Checkpoint selection belongs to different weights")
            state.update(epoch=epoch, checkpoint=str(checkpoint), foundation_sha256=sha,
                         completed_optimizer_steps=epoch * 1250,
                         selection=str(selection_path), selection_status=selection["status"])
            if selection["track_training_eligible"]:
                state.update(state="PERCEPTION_SELECTED", phase="COMPLETE",
                             next_stage="U2_A_LABEL_FREE_CANDIDATE_CACHE")
                publish(state_path, state)
                return 0
            if epoch == 1 and not selection["next_epoch_eligible"]:
                raise ValueError("Epoch-1 full selection does not permit the second bounded epoch")
        state.update(state="PERCEPTION_FAILED_AFTER_TWO_EPOCHS", phase="COMPLETE",
                     blocked_status="LOCATEMOT_U_BLOCKED_U2_FOUNDATION_FORGETTING" if not selection["retention_pass"]
                                    else "LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY")
        publish(state_path, state)
        return 0
    except Exception as error:
        state.update(state="FAILED", error=str(error))
        publish(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
