#!/usr/bin/env python3
"""Build and print the terminal held-out evaluation result.

This is deliberately a receipt reader.  It never recomputes metrics and it
never treats a partial ``frame_decisions.jsonl`` as an official result.  The
orchestrator invokes it only after every registered owner and retry/control
queue has reached a terminal state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from queue_onejev_full_joint_resume_after_heldout import OWNER_DIRS


RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
HELDOUT = RUNTIME / "outputs/groundjev/heldout"
EXTRA_STATE_FILES = (
    HELDOUT / "g200_animaltrack_shared_gpu7_v1/owner_state.json",
    HELDOUT / "g1_dancetrack_shared_gpu5_retry.queue_state.json",
    HELDOUT / "g200_ovmot_dedicated_gpu7_retry.queue_state.json",
    HELDOUT / "extra_shared_controls_collection.queue_state.json",
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def terminal(state: dict[str, Any] | None) -> bool:
    return bool(state) and str(state.get("status", "")).endswith(
        ("COMPLETE", "FAILED", "STOPPED", "CANCELLED")
    )


def nested_paths(value: Any) -> Iterable[Path]:
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            yield candidate
    elif isinstance(value, dict):
        for child in value.values():
            yield from nested_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_paths(child)


def line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            return sum(1 for _ in stream)
    except OSError:
        return None


def metric_files(owner_path: Path, state: dict[str, Any]) -> list[Path]:
    candidates: set[Path] = set()
    for raw in nested_paths(state):
        # Read explicit metric-file references, but do not recursively scan
        # arbitrary model/data directories found in the receipt.
        if raw.is_file() and raw.name == "official_metrics.json":
            candidates.add(raw)
    out = state.get("out")
    if isinstance(out, str):
        output_root = Path(out)
        if output_root.is_dir():
            candidates.update(output_root.rglob("official_metrics.json"))
    # Owner directories are usually the most reliable scope root.  The
    # bounded fallback also handles old receipts that omitted ``out``.
    if not candidates and owner_path.is_dir() and owner_path != HELDOUT:
        candidates.update(owner_path.rglob("official_metrics.json"))
    return sorted(path for path in candidates if path.is_file())


def record(owner_path: Path, state_path: Path, state: dict[str, Any], metrics_path: Path) -> dict[str, Any] | None:
    metrics = read_json(metrics_path)
    if not metrics or not isinstance(metrics.get("metrics"), dict):
        return None
    run_root = metrics_path.parent
    frame_decisions = run_root / "frame_decisions.jsonl"
    predictions = run_root / "predictions.jsonl"
    rejected = metrics.get("perception_rejected_frames") or []
    metric_values = metrics["metrics"]
    row: dict[str, Any] = {
        "owner": str(owner_path),
        "owner_state": str(state_path),
        "owner_status": state.get("status"),
        "model_label": state.get("model_label") or state.get("model") or metrics.get("model_label"),
        "task": metrics.get("task") or state.get("task"),
        "dataset": metrics.get("dataset"),
        "sequence": metrics.get("sequence"),
        "scope": metrics.get("scope"),
        "frames": metrics.get("frames"),
        "official_metrics": str(metrics_path),
        "metrics": metric_values,
        "metric_scale": metrics.get("metric_scale"),
        "fps": metrics.get("end_to_end_fps"),
        "perception_rejected_frames": len(rejected),
        "predictions": line_count(predictions),
        "decision_rows": line_count(frame_decisions),
    }
    return row


def collect() -> tuple[list[dict[str, Any]], dict[str, str]]:
    state_paths = [path / "owner_state.json" for path in OWNER_DIRS]
    state_paths.extend(EXTRA_STATE_FILES)
    rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    seen: set[str] = set()
    for state_path in state_paths:
        state = read_json(state_path)
        statuses[str(state_path)] = (state or {}).get("status", "MISSING")
        if state is None:
            continue
        owner_path = state_path.parent
        for metrics_path in metric_files(owner_path, state):
            key = str(metrics_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            item = record(owner_path, state_path, state, metrics_path)
            if item is not None:
                rows.append(item)
    rows.sort(key=lambda item: (str(item.get("task")), str(item.get("dataset")), str(item.get("sequence")), str(item.get("official_metrics"))))
    return rows, statuses


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GroundJEV held-out consolidated result",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "All rows below are read from terminal owner receipts and official metric files. Partial JSONL output is not presented as an official metric.",
        "",
        "| Task | Dataset / sequence | Frames | HOTA/TETA | DetA/LocA | AssA/AssocA | IDF1/ClsA | IDSW | FP | FN | FPS | rejected |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        values = row["metrics"]
        lines.append(
            "| {task} | {dataset} / {sequence} | {frames} | {a:.6f} | {b:.6f} | {c:.6f} | {d:.6f} | {idsw} | {fp} | {fn} | {fps} | {rejected} |".format(
                task=row.get("task") or "?",
                dataset=row.get("dataset") or "?",
                sequence=row.get("sequence") or "?",
                frames=row.get("frames") or "?",
                a=float(values.get("HOTA", values.get("TETA", 0.0))),
                b=float(values.get("DetA", values.get("LocA", 0.0))),
                c=float(values.get("AssA", values.get("AssocA", 0.0))),
                d=float(values.get("IDF1", values.get("ClsA", 0.0))),
                idsw=values.get("IDSW", "—"),
                fp=values.get("FP", "—"),
                fn=values.get("FN", "—"),
                fps=(f"{float(row['fps']):.6f}" if row.get("fps") is not None else "—"),
                rejected=row.get("perception_rejected_frames", "—"),
            )
        )
    lines.extend(["", f"Official metric rows: `{len(report['rows'])}`", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    rows, statuses = collect()
    report = {
        "schema": "groundjev.heldout.consolidated_result.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_owner_statuses": statuses,
        "rows": rows,
        "official_metric_rows": len(rows),
        "all_evaluation_owners_terminal": all(status.endswith(("COMPLETE", "FAILED", "STOPPED", "CANCELLED")) for status in statuses.values()),
        "partial_outputs_are_not_official": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.markdown_out.write_text(markdown(report) + "\n")
    print("HELDOUT_CONSOLIDATED_RESULT_BEGIN", flush=True)
    print(markdown(report), flush=True)
    print("HELDOUT_CONSOLIDATED_RESULT_JSON", args.out, flush=True)
    print("HELDOUT_CONSOLIDATED_RESULT_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
