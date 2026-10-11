#!/usr/bin/env python3
"""One blocking owner for the selected ordinary cache and simple baseline."""
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
from tools.unified.build_u2_candidate_cache import DIRECTORY, write_json
from tools.unified.build_u2_portability_artifact import read_measurement


def wait_process(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-perception-pid", type=int, required=True)
    parser.add_argument("--kitti-audit-pid", type=int, required=True)
    parser.add_argument("--extraction-pid", type=int, required=True)
    args = parser.parse_args()
    state_path = ROOT / "outputs/unified/u2/ordinary_baseline/run_state.json"
    state = {"format": "locatemot-u2-ordinary-baseline-queue-v1", "state": "WAITING_PERCEPTION_SELECTION",
             "perception_pid": args.wait_perception_pid, "controller_pid": os.getpid(),
             "kitti_audit_pid": args.kitti_audit_pid,
             "formal_cache_started": False, "formal_track_training_launched": False}
    write_json(state_path, state)
    print(json.dumps(state), flush=True)
    wait_process(args.kitti_audit_pid)
    if read_measurement(ROOT / "outputs/unified/u2/perception/measurements", "kitti_legal_development") is None:
        raise ValueError("Original KITTI audit ended without the complete registered measurement")
    wait_process(args.wait_perception_pid)
    perception = json.loads((ROOT / "outputs/unified/u2/perception_adaptation/run_state.json").read_text())
    # The already-running perception owner may reach preflight before the new
    # missing original-KITTI audit finishes. Resume only after it has actually
    # terminated before any optimizer launch and the full preflight is READY.
    if perception["state"] == "PRECONDITIONS_NOT_MET" and not perception.get("optimization_launched", False):
        from tools.unified.train_u2_perception_adaptation import preflight
        eligibility = preflight(argparse.Namespace(epoch=1, epoch1_selection=None, resume=None))
        if eligibility["state"] == "READY":
            pids = perception["wait_pids"]
            command = [sys.executable, "-u", str(ROOT / "tools/unified/run_u2_perception_adaptation.py"),
                       "--tao-audit-pid", str(pids["tao_audit"]), "--bdd-audit-pid", str(pids["bdd_audit"]),
                       "--replay-pid", str(pids["replay"]), "--extraction-pid", str(args.extraction_pid)]
            state.update(state="RESUMING_TERMINAL_PREOPTIMIZER_PREFLIGHT", command=command)
            write_json(state_path, state)
            print(json.dumps(state), flush=True)
            with (state_path.parent / "perception_resume.log").open("a") as handle:
                subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
            perception = json.loads((ROOT / "outputs/unified/u2/perception_adaptation/run_state.json").read_text())
    if perception["state"] != "PERCEPTION_SELECTED":
        state.update(state="HELD_PERCEPTION_NOT_SELECTED", perception_terminal_state=perception["state"])
        write_json(state_path, state)
        print(json.dumps(state), flush=True)
        return 0
    state.update(state="BUILDING_ORDINARY_CACHE", foundation_sha256=perception["foundation_sha256"],
                 formal_cache_started=True)
    write_json(state_path, state)
    try:
        steps = [([sys.executable, "-u", str(ROOT / "tools/unified/build_u2_candidate_cache.py"),
                   "--checkpoint", perception["checkpoint"], "--selection", perception["selection"],
                   "--phase", "ordinary", "--gpus", *map(str, range(8))], "cache.log"),
                 ([sys.executable, "-u", str(ROOT / "tools/unified/run_u2_simple_baseline.py"),
                   "--cache-manifest", str(DIRECTORY / perception["foundation_sha256"] / "ordinary/complete_manifest.json")],
                  "simple_baseline.log")]
        for command, name in steps:
            if name == "simple_baseline.log":
                state["state"] = "RUNNING_SIMPLE_BASELINE"
            state.update(command=command, log=str(state_path.parent / name))
            write_json(state_path, state)
            print(json.dumps(state), flush=True)
            with (state_path.parent / name).open("a") as handle:
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        result = ROOT / "outputs/unified/u2/simple_baseline/result.json"
        if json.loads(result.read_text())["status"] != "COMPLETE":
            raise ValueError("Simple baseline ended without complete five-dataset evaluation")
        state.update(state="COMPLETE", result=str(result), next_stage="U2_MOT_ONLY_6000_STEP_BASELINE")
        write_json(state_path, state)
        summary_path = ROOT / "outputs/unified/u2/final_status.json"
        summary = json.loads(summary_path.read_text())
        summary.update(mot_simple_baseline=str(result), execution_state="ORDINARY_SIMPLE_BASELINE_COMPLETE_MOT_ONLY_NEXT")
        write_json(summary_path, summary)
        print(json.dumps(state), flush=True)
        return 0
    except Exception as error:
        state.update(state="FAILED", error=str(error))
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
