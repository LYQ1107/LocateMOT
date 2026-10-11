#!/usr/bin/env python3
"""One blocking owner: wait for simple metrics, then run the gated 6k baseline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.unified.build_u2_candidate_cache import write_json
from tools.unified.queue_u2_ordinary_baseline import wait_process
from tools.unified.train_u2_mot_only import preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-ordinary-pid", type=int, required=True)
    args = parser.parse_args()
    directory = ROOT / "outputs/unified/u2/mot_only_queue"
    state_path = directory / "run_state.json"
    state = {"format": "locatemot-u2-mot-only-queue-v1", "state": "WAITING_ORDINARY_CACHE_AND_SIMPLE",
             "controller_pid": os.getpid(), "ordinary_controller_pid": args.wait_ordinary_pid,
             "formal_track_training_launched": False, "maximum_optimizer_steps": 6000}
    write_json(state_path, state)
    print(json.dumps(state), flush=True)
    wait_process(args.wait_ordinary_pid)
    ordinary = json.loads((ROOT / "outputs/unified/u2/ordinary_baseline/run_state.json").read_text())
    if ordinary["state"] != "COMPLETE":
        state.update(state="HELD_ORDINARY_BASELINE_INCOMPLETE", ordinary_terminal_state=ordinary["state"])
        write_json(state_path, state)
        print(json.dumps(state), flush=True)
        return 0
    simple_path = Path(ordinary["result"])
    simple = json.loads(simple_path.read_text())
    cache_path = Path(simple["candidate_cache_manifest"])
    try:
        preflight(cache_path, simple_path)
        # Registration is exact and refuses a changed formal protocol.
        subprocess.run([sys.executable, str(ROOT / "tools/unified/train_u2_mot_only.py"), "--register-only"],
                       cwd=ROOT, check=True)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                   str(ROOT / "tools/unified/train_u2_mot_only.py"), "--cache-manifest", str(cache_path),
                   "--simple-baseline", str(simple_path)]
        latest = ROOT / "outputs/unified/u2/mot_only/latest_checkpoint.json"
        if latest.is_file():
            command.extend(["--resume", json.loads(latest.read_text())["path"]])
        state.update(state="RUNNING_MOT_ONLY", command=command, formal_track_training_launched=True)
        write_json(state_path, state)
        print(json.dumps(state), flush=True)
        with (directory / "training.log").open("a") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        selection_path = ROOT / "outputs/unified/u2/mot_only/selection.json"
        selection = json.loads(selection_path.read_text())
        if selection["optimizer_steps"] != 6000:
            raise ValueError("MOT-only controller ended without the complete registered budget")
        state.update(state="COMPLETE", selection=str(selection_path), selection_status=selection["status"])
        write_json(state_path, state)
        summary_path = ROOT / "outputs/unified/u2/final_status.json"
        summary = json.loads(summary_path.read_text())
        summary.update(mot_only_baseline=selection["selected"], mot_only_selection=str(selection_path),
                       execution_state="MOT_ONLY_COMPLETE", ready_for_u3=False)
        summary["status"] = ("LOCATEMOT_U_U2_IN_PROGRESS" if selection["selected"]
                             else "LOCATEMOT_U_BLOCKED_U2_IDENTITY_CORE")
        summary["stop"] = "JOINT_OVMOT_NEXT" if selection["selected"] else "IDENTITY_CORE_ELIGIBILITY_FAILED"
        write_json(summary_path, summary)
        print(json.dumps(state), flush=True)
        return 0
    except Exception as error:
        state.update(state="FAILED", error=str(error))
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
