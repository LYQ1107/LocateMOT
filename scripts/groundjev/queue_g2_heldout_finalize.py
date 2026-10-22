#!/usr/bin/env python3
"""Finalize the G2 held-out evidence once every registered owner is terminal."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/data2/user/LocateMOT")
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
OWNERS = (
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu3_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu5_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_clean_independent_gpu6_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_domain_dancetrack_gpu5_v1/owner_state.json",
    RUNTIME / "outputs/groundjev/heldout/g2_domain_animaltrack_gpu6_v1/owner_state.json",
)
OUT = RUNTIME / "outputs/groundjev/heldout/g2_heldout_finalization_receipt.json"
COLLECTOR = ROOT / "scripts/groundjev/collect_g2_heldout_receipts.py"
PYTHON = RUNTIME / ".venv-onejev-py310/bin/python"


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
    collection = OUT.with_name("g2_heldout_receipt_collection_final.json")
    subprocess.run(
        [str(PYTHON), str(COLLECTOR), "--out", str(collection)],
        cwd=ROOT,
        check=True,
    )
    report = {
        "schema": "groundjev.g2.heldout.finalization.v1",
        "project_root": str(ROOT),
        "owners": states,
        "collection": str(collection),
        "accepted_scope_count": (read(collection) or {}).get("accepted_scope_count", 0),
        "finalized_by_pid": os.getpid(),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(OUT), "accepted_scope_count": report["accepted_scope_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
