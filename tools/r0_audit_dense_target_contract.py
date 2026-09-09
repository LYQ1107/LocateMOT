#!/usr/bin/env python3
"""Audit all 5,314 L49 fit rows against safe frame-specific target records."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r0_dense_data import (  # noqa: E402
    EXPECTED_FIT_ROWS,
    MANIFEST,
    R0DataContractError,
    load_fit_rows,
    native_frame_ids,
    sha256_file,
)
from locatemot.rmot.r0_safe_target_source import (  # noqa: E402
    load_safe_query_records,
)
from locatemot.rmot.r0_dense_data import R0FrameTargetSource  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty audit output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {
        "format": "locatemot-r0a-dense-target-contract-audit-v1",
        "status": "incomplete",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD,
        "seed": SEED,
        "inputs": {"safe_target_artifact": str(args.safe_targets.resolve()), "manifest_sha256": sha256_file(MANIFEST), "expected_manifest_sha256": MANIFEST_SHA},
        "outputs": {"root": str(out)},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "failure_root_cause": None,
        "next_action": "build dense native-frame indexes only after exact zero mismatch result",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0A worktree cwd: {Path.cwd()}")
        if base["inputs"]["manifest_sha256"] != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        rows = load_fit_rows()
        if len(rows) != EXPECTED_FIT_ROWS:
            raise R0DataContractError(f"expected {EXPECTED_FIT_ROWS} rows, got {len(rows)}")
        records = load_safe_query_records(args.safe_targets.resolve())
        pairs = {(str(row["dataset"]), str(row["video"])) for row in rows}
        source = R0FrameTargetSource(purpose="audit", allowed_video_pairs=pairs, records=records)
        valid_frames: dict[tuple[str, str], set[int]] = {}
        counters = Counter()
        examples: dict[str, list[dict[str, Any]]] = {name: [] for name in ("sentence_mismatch", "target_mismatch", "missing_query", "invalid_frame")}
        checked = 0
        for row in rows:
            checked += 1
            dataset, video, query_id, frame_id = str(row["dataset"]), str(row["video"]), int(row["query_id"]), int(row["frame_id"])
            pair = (dataset, video)
            if pair not in valid_frames:
                valid_frames[pair] = set(native_frame_ids(video))
            if frame_id not in valid_frames[pair]:
                counters["invalid_frame"] += 1
                if len(examples["invalid_frame"]) < 8:
                    examples["invalid_frame"].append({"unit_key": row.get("unit_key"), "frame_id": frame_id})
                continue
            try:
                expected_sentence = source.query_sentence(dataset, video, query_id)
                expected_targets = set(source.target_ids(dataset, video, query_id, frame_id))
            except KeyError as exc:
                counters["missing_query"] += 1
                if len(examples["missing_query"]) < 8:
                    examples["missing_query"].append({"unit_key": row.get("unit_key"), "error": str(exc)})
                continue
            observed_sentence = str(row.get("sentence") or row.get("expression") or "")
            observed_targets = {str(value) for value in row.get("target_ids", [])}
            if observed_sentence != expected_sentence:
                counters["sentence_mismatch"] += 1
                if len(examples["sentence_mismatch"]) < 8:
                    examples["sentence_mismatch"].append({"unit_key": row.get("unit_key"), "expected": expected_sentence, "observed": observed_sentence})
            if observed_targets != expected_targets:
                counters["target_mismatch"] += 1
                if len(examples["target_mismatch"]) < 8:
                    examples["target_mismatch"].append({"unit_key": row.get("unit_key"), "expected": sorted(expected_targets), "observed": sorted(observed_targets)})
        passed = checked == EXPECTED_FIT_ROWS and all(counters[name] == 0 for name in ("sentence_mismatch", "target_mismatch", "missing_query", "invalid_frame"))
        payload = {
            **base,
            "status": "complete" if passed else "invalid",
            "rows_checked": checked,
            "sentence_mismatch": int(counters["sentence_mismatch"]),
            "target_mismatch": int(counters["target_mismatch"]),
            "missing_query": int(counters["missing_query"]),
            "invalid_frame": int(counters["invalid_frame"]),
            "examples": examples,
            "safe_source": source.source_descriptor(),
            "source_record_count": len(records),
            "native_bank_video_count": len(valid_frames),
            "passed": passed,
            "wall_seconds": time.perf_counter() - started,
            "next_action": "build separate V1/V2 dense native-frame indexes" if passed else "R0A_DATA_CONTRACT_STILL_INVALID; stop before dense indexing",
        }
        for name in ("audit.json", "provenance.json", "status.json"):
            write_json(out / name, payload)
        if not passed:
            (out / "INCOMPLETE.md").write_text("# R0A dense target contract — INVALID\n\n" + json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0 if passed else 2
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0A dense target contract — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback_path": str((out / "INCOMPLETE.md").resolve()), "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
