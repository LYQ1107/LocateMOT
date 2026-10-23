#!/usr/bin/env python3
"""Finalize G0/G1/G2/G200/RMOT plus the fast-chunk TAO OVMOT control."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/data2/user/LocateMOT")
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
PYTHON = RUNTIME / ".venv-onejev-py310/bin/python"
COLLECTOR = ROOT / "scripts/groundjev/collect_g2_heldout_receipts.py"
OWNERS = (
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu3_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu5_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu6_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_domain_dancetrack_gpu5_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_domain_animaltrack_gpu6_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu5_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_fast_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_repaired_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video63_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video63_repaired_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_dancetrack_gpu3_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_animaltrack_gpu6_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_dancetrack_gpu3_v2/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_dancetrack_gpu3_serial_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_dancetrack_gpu3_serial_v2/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video105_serial_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_ovmot_gpu7_video105_serial_v2/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_rmot_single_gpu7_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_parallel_rmot_multi_gpu7_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_mot20_05_gpu4_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_mot20_05_gpu5_resume_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_mot20_05_gpu7_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g1_rmot_shared_gpu5_v2/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g200_ovmot_shared_gpu7_v2/owner_state.json",
)
OUT = RUNTIME / "outputs/groundjev/heldout/groundjev_heldout_fast_ovmot_finalization_receipt.json"
COLLECTION = RUNTIME / "outputs/groundjev/heldout/groundjev_heldout_fast_ovmot_receipt_collection.json"


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def terminal(state: dict | None) -> bool:
    return bool(state) and str(state.get("status", "")).endswith(("COMPLETE", "FAILED"))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    while True:
        states = {str(path): read(path) for path in OWNERS}
        if all(terminal(state) for state in states.values()):
            break
        time.sleep(60)
    command = [str(PYTHON), str(COLLECTOR), "--out", str(COLLECTION)]
    for owner in OWNERS:
        command.extend(("--owner", str(owner)))
    subprocess.run(command, cwd=ROOT, check=True)
    collection = read(COLLECTION) or {}
    report = {
        "schema": "groundjev.heldout.fast_ovmot_finalization.v1",
        "project_root": str(ROOT),
        "owners": states,
        "collection": str(COLLECTION),
        "accepted_scope_count": collection.get("accepted_scope_count", 0),
        "finalized_by_pid": os.getpid(),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(OUT), "accepted_scope_count": report["accepted_scope_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
