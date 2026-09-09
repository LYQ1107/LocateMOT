#!/usr/bin/env python3
"""Merge corrected Stage-S rows with immutable, already-consistent T/J rows."""
from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from l89d_fullvideo_common import (  # noqa: E402
    MANIFEST_SHA,
    SEED,
    THREAD,
    WORK_ROOT,
    command_line,
    manifest_assertion,
    sha256_file,
    standard_flags,
    write_json,
)


FORMAT = "locatemot-l89e-phase-consistent-merged-score-v1"
S_EPOCHS = {2, 4, 6, 8}
TJ_EPOCHS = set(range(10, 41, 2))
ALL_EPOCHS = S_EPOCHS | TJ_EPOCHS


def _read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.resolve().read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("format") != "locatemot-l89-score-record-v1":
            raise AssertionError(f"unexpected score format at {path}:{line_number}")
        n = int(value["candidate_count"])
        for name in ("score", "candidate_energy", "r_static", "r_total", "candidate_prior", "labels", "candidate_gt", "row_keys"):
            if len(value.get(name, [])) != n:
                raise AssertionError(f"score length drift {name}: {value.get('unit_key')}")
        if not bool(value.get("finite_scores")) or value.get("candidate_deletion") or value.get("candidate_truncation"):
            raise AssertionError(f"invalid score flags: {value.get('unit_key')}")
        if not np.isfinite(np.asarray(value["score"], dtype=np.float64)).all():
            raise FloatingPointError(f"nonfinite score: {value.get('unit_key')}")
        rows.append(value)
    return rows


def _epoch(row: dict[str, Any]) -> int:
    checkpoint = row.get("checkpoint") or {}
    return int(checkpoint["epoch"])


def _validate_epoch_rows(rows: list[dict[str, Any]], epoch: int, source: str) -> None:
    if len(rows) != 498:
        raise AssertionError(f"{source} epoch {epoch} record count {len(rows)} != 498")
    keys = [str(row["unit_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError(f"duplicate {source} unit keys at epoch {epoch}")
    if source == "stage_s":
        for row in rows:
            if not bool(row.get("phase_consistent_temporal")) or row.get("history_contract") != "zero_history":
                raise AssertionError(f"Stage-S phase metadata missing: {row['unit_key']}")
            if bool(row.get("temporal_enabled", False)) or int(row.get("history_valid_count", -1)) != 0:
                raise AssertionError(f"Stage-S history drift: {row['unit_key']}")
    else:
        checkpoint_phase = str(rows[0].get("checkpoint", {}).get("phase"))
        expected_phase = "T" if epoch <= 20 else "J"
        if checkpoint_phase != expected_phase:
            raise AssertionError(f"historical T/J checkpoint phase drift: epoch={epoch} phase={checkpoint_phase}")
        for row in rows:
            if str(row.get("checkpoint", {}).get("phase")) != expected_phase:
                raise AssertionError(f"historical phase drift in epoch {epoch}")


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty merged score output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89E worktree cwd: {Path.cwd()}")
        manifest_assertion()
        stage_rows = _read(args.stage_s_scores)
        historical_rows = _read(args.historical_scores)
        stage_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        old_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in stage_rows:
            stage_by_epoch[_epoch(row)].append(row)
        for row in historical_rows:
            old_by_epoch[_epoch(row)].append(row)
        if set(stage_by_epoch) != S_EPOCHS:
            raise AssertionError(f"Stage-S epoch set drift: {sorted(stage_by_epoch)}")
        if set(old_by_epoch) != TJ_EPOCHS:
            raise AssertionError(f"historical T/J epoch set drift: {sorted(old_by_epoch)}")
        for epoch in sorted(S_EPOCHS):
            _validate_epoch_rows(stage_by_epoch[epoch], epoch, "stage_s")
        for epoch in sorted(TJ_EPOCHS):
            _validate_epoch_rows(old_by_epoch[epoch], epoch, "historical_tj")
        epoch_rows = {epoch: (stage_by_epoch[epoch] if epoch in S_EPOCHS else old_by_epoch[epoch]) for epoch in sorted(ALL_EPOCHS)}
        reference = [str(row["unit_key"]) for row in epoch_rows[2]]
        if any([str(row["unit_key"]) for row in epoch_rows[epoch]] != reference for epoch in sorted(ALL_EPOCHS)):
            raise AssertionError("merged epoch unit-key order drift")
        merged_path = out / "score_records.jsonl"
        with merged_path.open("w", encoding="utf-8") as handle:
            for epoch in sorted(ALL_EPOCHS):
                for row in epoch_rows[epoch]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload = {
            "format": FORMAT,
            "status": "complete",
            "stage": "L89E merged phase-consistent fit/dev score pool",
            "command": command,
            "cwd": str(WORK_ROOT),
            "luna_thread": THREAD,
            "seed": SEED,
            "stage_s_source": str(args.stage_s_scores.resolve()),
            "stage_s_source_sha256": sha256_file(args.stage_s_scores.resolve()),
            "historical_tj_source": str(args.historical_scores.resolve()),
            "historical_tj_source_sha256": sha256_file(args.historical_scores.resolve()),
            "stage_s_epochs": sorted(S_EPOCHS),
            "historical_tj_epochs": sorted(TJ_EPOCHS),
            "merged_epochs": sorted(ALL_EPOCHS),
            "record_count_per_epoch": {str(epoch): len(epoch_rows[epoch]) for epoch in sorted(ALL_EPOCHS)},
            "record_count": int(sum(len(rows) for rows in epoch_rows.values())),
            "score_records": str(merged_path.resolve()),
            "score_records_sha256": sha256_file(merged_path),
            "phase_consistent": True,
            "stage_s_rescored": True,
            "tj_sparse_scores_reused": True,
            "tj_history_contract": "last4_causal_history via historical build_frame(..., temporal_enabled=True)",
            "fixed_calibration_read": False,
            "fixed_validation_read": False,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "all_candidate_rows_scored": True,
            "manifest_sha256": MANIFEST_SHA,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            **standard_flags(hota_trackeval_run=False),
            "failure_root_cause": None,
            "next_action": "build the registered maximum-five phase-consistent shortlist",
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-l89e-merged-score-provenance-v1"})
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "complete",
            "record_count": payload["record_count"],
            "phase_consistent": True,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "zero_training": True,
        })
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89E merged score pool — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "incomplete",
            "command": command,
            "cwd": str(WORK_ROOT),
            "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair the first merge contract error and use a new output",
            **standard_flags(hota_trackeval_run=False),
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-s-scores", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
