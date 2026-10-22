#!/usr/bin/env python3
"""Write a compact, reproducible snapshot of the active GroundJEV goal."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


ROOT = Path("/data2/user/LocateMOT")
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
OUT = ROOT / "docs/GROUNDJEV/ACTIVE_GOAL_LIVE_20261022.json"
JOINT_RESUME = RUNTIME / "outputs/groundjev/training/full_joint_source_epoch1_resume_after_heldout_v1/owner_state.json"

OWNERS = {
    "G0": RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu3_v1/owner_state.json",
    "G2_GPU5": RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu5_v1/owner_state.json",
    "G2_GPU6": RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu6_v1/owner_state.json",
    "G2_DANCE": RUNTIME / "outputs/groundjev/heldout/g2_domain_dancetrack_gpu5_v1/owner_state.json",
    "G2_ANIMAL": RUNTIME / "outputs/groundjev/heldout/g2_domain_animaltrack_gpu6_v1/owner_state.json",
    "G1_ADAPTATION": RUNTIME / "outputs/groundjev/ground_adaptation/g1_gpu5_isolated96_v1/owner_state.json",
    "G2_TAO": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu5_v1/owner_state.json",
    "G2_TAO_FAST": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_fast_v1/owner_state.json",
    "G2_TAO105_REPAIRED": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_repaired_v1/owner_state.json",
    "G2_TAO63": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video63_v1/owner_state.json",
    "G2_TAO63_REPAIRED": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video63_repaired_v1/owner_state.json",
    "G1_DANCE": RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_gpu3_v1/owner_state.json",
    "G1_ANIMAL": RUNTIME / "outputs/groundjev/heldout/g1_animaltrack_gpu6_v1/owner_state.json",
    "G200_DANCE": RUNTIME / "outputs/groundjev/heldout/g200_dancetrack_gpu3_v2/owner_state.json",
    "RMOT_SINGLE": RUNTIME / "outputs/groundjev/heldout/g2_parallel_rmot_single_gpu7_v1/owner_state.json",
    "RMOT_MULTI": RUNTIME / "outputs/groundjev/heldout/g2_parallel_rmot_multi_gpu7_v1/owner_state.json",
}

FRAME_FILES = {
    "G0_RMOT": RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu3_v1/G0/rmot_single0007/frame_decisions.jsonl",
    "G2_MOT_GPU5": RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu5_v1/G2_CLEAN_V2/mot20_05_validation1657/frame_decisions.jsonl",
    "G2_MOT_GPU6": RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu6_v1/G2_CLEAN_V2/mot20_05_validation1657/frame_decisions.jsonl",
    "G2_DANCE": RUNTIME / "outputs/groundjev/heldout/g2_domain_dancetrack_gpu5_v1/G2_CLEAN_V2/dancetrack0004_val1203/frame_decisions.jsonl",
    "G2_ANIMAL": RUNTIME / "outputs/groundjev/heldout/g2_domain_animaltrack_gpu6_v1/G2_CLEAN_V2/animaltrack_chicken1_test306/frame_decisions.jsonl",
    "G2_TAO": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu5_v1/G2_CLEAN_V2/tao_video105_independent40/frame_decisions.jsonl",
    "G2_TAO_FAST": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_fast_v1/G2_CLEAN_V2/tao_video105_independent40/frame_decisions.jsonl",
    "G2_TAO105_REPAIRED": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_repaired_v1/G2_CLEAN_V2/tao_video105_independent40_repaired/frame_decisions.jsonl",
    "G2_TAO63": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video63_v1/G2_CLEAN_V2/tao_video63_base_novel40/frame_decisions.jsonl",
    "G2_TAO63_REPAIRED": RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video63_repaired_v1/G2_CLEAN_V2/tao_video63_base_novel40_repaired/frame_decisions.jsonl",
    "G1_DANCE": RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_gpu3_v1/G1_GPU5/dancetrack0004_val1203/frame_decisions.jsonl",
    "G1_ANIMAL": RUNTIME / "outputs/groundjev/heldout/g1_animaltrack_gpu6_v1/G1_GPU5/animaltrack_chicken1_test306/frame_decisions.jsonl",
    "G200_DANCE": RUNTIME / "outputs/groundjev/heldout/g200_dancetrack_gpu3_v2/G200_TRACKING200/dancetrack0004_val1203/frame_decisions.jsonl",
    "RMOT_SINGLE": RUNTIME / "outputs/groundjev/heldout/g2_parallel_rmot_single_gpu7_v1/G2_CLEAN_V2/rmot_single0007/frame_decisions.jsonl",
    "RMOT_MULTI": RUNTIME / "outputs/groundjev/heldout/g2_parallel_rmot_multi_gpu7_v1/G2_CLEAN_V2/rmot_multi0007/frame_decisions.jsonl",
}


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def line_count(path: Path) -> int:
    try:
        with path.open() as stream:
            return sum(1 for _ in stream)
    except OSError:
        return 0


def gpu_snapshot() -> list[dict]:
    command = ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 4:
            rows.append({"index": int(fields[0]), "memory_used_MiB": int(fields[1]),
                         "memory_free_MiB": int(fields[2]), "utilization_percent": int(fields[3])})
    return rows


def main() -> int:
    owners = {}
    for name, path in OWNERS.items():
        state = read(path)
        owners[name] = {"path": str(path), "status": "MISSING" if state is None else state.get("status"),
                        "current_stage": None if state is None else state.get("current_stage"),
                        "failed": None if state is None else state.get("failed"),
                        "completed": [] if state is None else list((state.get("completed") or {}).keys()),
                        "physical_gpu": None if state is None else state.get("physical_gpu")}
    metrics = {}
    for path in RUNTIME.glob("outputs/groundjev/heldout/**/official_metrics.json"):
        data = read(path)
        if data is not None:
            metrics[str(path)] = data
    report = {
        "schema": "groundjev.active_goal.live.v1",
        "project_root": str(ROOT),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(),
        "owners": owners,
        "frame_progress_rows": {name: line_count(path) for name, path in FRAME_FILES.items()},
        "official_metrics": metrics,
        "gpu_snapshot": gpu_snapshot(),
        "joint_checkpoint": str(Path("/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_expansion_gpu7_v3/groundjev-native-three-task-step001200.pt")),
        "joint_receipt": str(Path("/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_expansion_gpu7_v3/receipt.json")),
        "onejev_full_joint_resume": {"path": str(JOINT_RESUME), "state": read(JOINT_RESUME)},
        "claim_boundary": "Only complete official_metrics receipts are metrics; frame counts and running states are progress evidence.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(OUT), "owner_count": len(owners), "metric_count": len(metrics)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
