#!/usr/bin/env python3
"""One blocking owner: eligible MOT-only -> full OV cache -> exact joint U2-B."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.unified.build_u2_candidate_cache import DIRECTORY, write_json
from tools.unified.queue_u2_ordinary_baseline import wait_process
from tools.unified.train_u2_joint import mot_reference, preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-mot-only-pid", type=int, required=True)
    args = parser.parse_args()
    directory = ROOT / "outputs/unified/u2/joint_queue"
    state_path = directory / "run_state.json"
    state = {"format": "locatemot-u2-joint-queue-v1", "state": "WAITING_ELIGIBLE_MOT_ONLY",
             "controller_pid": os.getpid(), "mot_only_controller_pid": args.wait_mot_only_pid,
             "formal_joint_training_launched": False, "ovmot_cache_started": False,
             "maximum_optimizer_steps": 4000}
    write_json(state_path, state)
    print(json.dumps(state), flush=True)
    wait_process(args.wait_mot_only_pid)
    try:
        mot_state = json.loads((ROOT / "outputs/unified/u2/mot_only_queue/run_state.json").read_text())
        if mot_state["state"] != "COMPLETE" or mot_state["selection_status"] != "MOT_ONLY_SELECTED":
            state.update(state="HELD_MOT_ONLY_NOT_ELIGIBLE", mot_only_terminal_state=mot_state["state"])
            write_json(state_path, state)
            print(json.dumps(state), flush=True)
            return 0
        ordinary_state = json.loads((ROOT / "outputs/unified/u2/ordinary_baseline/run_state.json").read_text())
        simple_path = Path(ordinary_state["result"])
        simple = json.loads(simple_path.read_text())
        inputs = argparse.Namespace(cache_manifest=Path(simple["candidate_cache_manifest"]), simple_baseline=simple_path,
                                    mot_only_selection=Path(mot_state["selection"]))
        ordinary, _, reference = mot_reference(inputs)
        spec = ordinary["spec"]
        inputs.ovmot_cache = DIRECTORY / spec["foundation_sha256"] / "ovmot/complete_manifest.json"
        inputs.ovmot_supervision = ROOT / "outputs/unified/u2/ovmot_supervision/manifest.json"
        inputs.class_text = ROOT / "outputs/unified/u2/ovmot_semantics" / spec["foundation_sha256"] / "class_text.pt"
        inputs.tao_gt_view = ROOT / "outputs/unified/u2/teta/gt/gt_view.json"

        def run(name: str, command: list[str]) -> None:
            state.update(state=name, command=command)
            write_json(state_path, state)
            print(json.dumps(state), flush=True)
            with (directory / (name.lower() + ".log")).open("a") as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)

        # Every cache worker retains completed shards on a real interrupted
        # run. Re-enter only after the original MOT owner is truly terminal.
        if not inputs.ovmot_supervision.is_file():
            run("PREPARING_BASE_SUPERVISION", [sys.executable, "tools/unified/prepare_u2_ovmot_supervision.py"])
        state["ovmot_cache_started"] = True
        run("BUILDING_FULL_OVMOT_CACHE", [sys.executable, "tools/unified/build_u2_candidate_cache.py", "--phase", "ovmot",
                                        "--checkpoint", spec["checkpoint"], "--selection", spec["selection"]])
        run("BUILDING_SELECTED_CLASS_TEXT", [sys.executable, "tools/unified/build_u2_class_text_cache.py",
                                            "--checkpoint", spec["checkpoint"], "--selection", spec["selection"],
                                            "--expected-foundation-sha256", spec["foundation_sha256"],
                                            "--device", "cuda:0", "--out", str(inputs.class_text)])
        run("PREPARING_OFFICIAL_TETA", [sys.executable, "tools/unified/prepare_u2_teta.py"])
        preflight(inputs)
        subprocess.run([sys.executable, "tools/unified/train_u2_joint.py", "--register-only"], cwd=ROOT, check=True)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                   "tools/unified/train_u2_joint.py"]
        for name in ("cache_manifest", "simple_baseline", "mot_only_selection", "ovmot_cache", "ovmot_supervision", "class_text", "tao_gt_view"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(inputs, name))])
        latest = ROOT / "outputs/unified/u2/joint/latest_checkpoint.json"
        if latest.is_file():
            command.extend(["--resume", json.loads(latest.read_text())["path"]])
        state.update(formal_joint_training_launched=True, mot_only_reference_sha256=reference["checkpoint_sha256"])
        run("RUNNING_JOINT_U2_B", command)
        selection_path = ROOT / "outputs/unified/u2/joint/selection.json"
        selection = json.loads(selection_path.read_text())
        if selection["optimizer_steps"] != 4000:
            raise ValueError("Joint owner ended without all4,000 registered optimizer steps")
        state.update(state="COMPLETE", selection=str(selection_path), selection_status=selection["status"])
        write_json(state_path, state)
        summary_path = ROOT / "outputs/unified/u2/final_status.json"
        summary = json.loads(summary_path.read_text())
        summary.update(joint_mot_ovmot=selection["selected"], joint_selection=str(selection_path),
                       execution_state="U2_B_JOINT_COMPLETE", ready_for_u3=False)
        summary["status"] = "LOCATEMOT_U_U2_IN_PROGRESS" if selection["selected"] else selection["status"]
        summary["stop"] = selection["pending"]
        write_json(summary_path, summary)
        print(json.dumps(state), flush=True)
        return 0
    except Exception as error:
        state.update(state="FAILED", error=str(error))
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
