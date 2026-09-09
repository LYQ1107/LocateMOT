#!/usr/bin/env python3
"""Validate and, only when valid, construct an R0 dense training index.

The current L49 fit artifact is deliberately checked against the exact R0
canonical-query contract.  A failure is materialized as evidence and exits;
the script never repairs frame-varying target IDs by taking a union.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r0_dense_data import (  # noqa: E402
    FIT_DATASETS,
    R0DataContractError,
    canonical_query_diagnostics,
    contract_descriptor,
    load_fit_rows,
    validate_canonical_query_contract,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 dense-index output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    base = {
        "format": "locatemot-r0-dense-train-index-v1",
        "status": "incomplete",
        "command": command,
        "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD,
        "seed": SEED,
        "dataset": args.dataset,
        "inputs": contract_descriptor(),
        "outputs": {"root": str(out)},
        "failure_root_cause": None,
        "next_action": "repair or explicitly re-register the canonical-query label contract before R0 training",
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "training_run": False,
        "hota_trackeval_run": False,
    }
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"R0 dense-index wrong cwd: {Path.cwd()}")
        rows = load_fit_rows()
        selected = rows if args.dataset == "all" else [row for row in rows if row["dataset"] == args.dataset]
        if not selected:
            raise R0DataContractError(f"no selected fit rows for {args.dataset}")
        contract = validate_canonical_query_contract(selected)
        payload = {
            **base, "status": "complete", "canonical_contract": contract,
            "row_count": len(selected), "wall_seconds": time.perf_counter() - started,
            "failure_root_cause": None, "next_action": "build visual cache only after the contract remains valid",
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text(
            "# R0 dense train index — INCOMPLETE\n\n" + trace, encoding="utf-8"
        )
        diagnostics = None
        try:
            if "rows" not in locals():
                rows = load_fit_rows()
            diagnostics = canonical_query_diagnostics(
                rows if args.dataset == "all" else [row for row in rows if row["dataset"] == args.dataset]
            )
        except Exception as diagnostic_exc:
            diagnostics = {"status": "diagnostic_failed", "error": f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"}
        payload = {
            **base,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback_path": str((out / "INCOMPLETE.md").resolve()),
            "wall_seconds": time.perf_counter() - started,
            "canonical_contract_diagnostics": diagnostics,
        }
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", *FIT_DATASETS), default="all")
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
