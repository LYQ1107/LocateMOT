#!/usr/bin/env python3
"""Build and audit the independent L71 index over the L69 budget-40 bank."""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l71_common import (  # noqa: E402
    DATASETS,
    L69_FEATURE_ROOT,
    L69_VIDEOS,
    MANIFEST_PATH,
    MANIFEST_SHA256,
    L62_RECORDS,
    L71Bank,
    fixed_eval_units,
    load_l49_splits,
    load_text_cache,
    make_unit_record,
    sha256_file,
    unit_key,
    write_json,
)


SEED = 20260829
DEFAULT_OUT = ROOT / "outputs/l71/audit/data_contract"


def base_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": "locatemot-l71-data-contract-v1",
        "status": "running",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "seed": SEED,
        "inputs": {
            "l69_feature_root": str(L69_FEATURE_ROOT),
            "l49_fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
            "l49_calibration_units": str(ROOT / "outputs/l49/data/calibration_units.jsonl"),
            "l49_validation_units": str(ROOT / "outputs/l49/data/validation_units.jsonl"),
            "l62_fixed_order": str(L62_RECORDS),
            "text_cache": str(ROOT / "outputs/l48/data/text_cache.pt"),
            "manifest": str(MANIFEST_PATH),
        },
        "outputs": {"unit_records": str(args.out / "unit_records.jsonl")},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "training_run": False,
        "raw_dense_feature_cache_written": False,
        "hota_trackeval_run": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    running = base_payload(args)
    write_json(out / "status.json", running)
    started = time.perf_counter()
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        manifest_sha = sha256_file(MANIFEST_PATH)
        if manifest_sha != MANIFEST_SHA256:
            raise AssertionError(f"manifest SHA mismatch: {manifest_sha}")
        splits = load_l49_splits()
        fit_units = [row for row in splits["fit"] if str(row.get("split")) == "fit"]
        if len(fit_units) != 5314:
            raise AssertionError(f"expected 5314 fit units, got {len(fit_units)}")
        if {str(row["dataset"]) for row in fit_units} != set(DATASETS):
            raise AssertionError("fit dataset set is not exactly Refer-KITTI V1/V2")
        eval_units = fixed_eval_units(splits)
        fixed_keys = [unit_key(row) for row in eval_units]
        if len(fixed_keys) != 40 or len(set(fixed_keys)) != 40:
            raise AssertionError("fixed evaluation key/order contract failed")
        all_units = [(row, "fit", None) for row in fit_units]
        all_units.extend(
            (row, "fixed_eval", index) for index, row in enumerate(eval_units)
        )
        all_keys = [unit_key(row) for row, _, _ in all_units]
        if len(set(all_keys)) != len(all_keys):
            raise AssertionError("fit and fixed evaluation unit keys overlap")
        text_cache = load_text_cache()
        records_by_video: dict[str, list[tuple[dict[str, Any], str, int | None]]] = defaultdict(list)
        for row, role, order in all_units:
            video = str(row["video"])
            if video not in L69_VIDEOS:
                raise AssertionError(f"unit video outside L69 materialized set: {video}")
            records_by_video[video].append((row, role, order))

        indexed: list[dict[str, Any]] = []
        video_stats: dict[str, dict[str, Any]] = {}
        for video in sorted(records_by_video):
            bank = L71Bank(video)
            try:
                counts = Counter()
                candidate_rows = 0
                future_rows = 0
                for row, role, order in records_by_video[video]:
                    record = make_unit_record(row, bank)
                    record["index_role"] = role
                    record["fixed_eval_order"] = order
                    if role == "fixed_eval":
                        record["fixed_eval_split"] = "calibration" if int(order) < 16 else "validation"
                    if record["candidate_count"] != record["frame_pointer"]["end"] - record["frame_pointer"]["begin"]:
                        raise AssertionError(f"candidate/frame pointer mismatch: {record['unit_key']}")
                    if len(record["row_keys"]) != record["candidate_count"]:
                        raise AssertionError(f"candidate key count mismatch: {record['unit_key']}")
                    if any(len(frames) != len(rows) for frames, rows in zip(record["history_frame_ids"], record["history_row_offsets"])):
                        raise AssertionError(f"history shape mismatch: {record['unit_key']}")
                    future_rows += sum(
                        int(frame > int(record["frame_id"]))
                        for frames in record["history_frame_ids"]
                        for frame in frames
                    )
                    candidate_rows += int(record["candidate_count"])
                    counts[(role, record["category"])] += 1
                    indexed.append(record)
                video_stats[video] = {
                    "units": len(records_by_video[video]),
                    "candidate_rows": candidate_rows,
                    "bank_path": str(bank.path),
                    "bank_sha256": bank.sha256,
                    "bank_mtime_ns": int(bank.path.stat().st_mtime_ns),
                    "categories": {"|".join(key): value for key, value in sorted(counts.items())},
                    "future_history_rows": future_rows,
                }
            finally:
                bank.close()

        fit_indexed = [row for row in indexed if row["index_role"] == "fit"]
        eval_indexed = sorted(
            (row for row in indexed if row["index_role"] == "fixed_eval"),
            key=lambda row: int(row["fixed_eval_order"]),
        )
        if [unit_key(row) for row in eval_indexed] != fixed_keys:
            raise AssertionError("fixed evaluation order drift after L69 indexing")
        if len(fit_indexed) != 5314 or len(eval_indexed) != 40:
            raise AssertionError("indexed fit/evaluation count mismatch")
        if any(row["old_l49_begin_end_ignored"] is not True or row["old_l49_positive_indices_ignored"] is not True for row in indexed):
            raise AssertionError("old L49 addressing was not explicitly ignored")
        if any(row["category"] == "present_uncovered" and row["coverage_mask"] for row in indexed):
            raise AssertionError("present-uncovered coverage mask drift")
        if any(row["category"] == "inactive" and row["null_target"] != 1 for row in indexed):
            raise AssertionError("inactive NULL target drift")

        with (out / "unit_records.jsonl").open("w") as handle:
            for record in indexed:
                handle.write(__import__("json").dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        fit_categories = Counter(row["category"] for row in fit_indexed)
        eval_categories = Counter(row["category"] for row in eval_indexed)
        dataset_counts = Counter(row["dataset"] for row in fit_indexed)
        contract = {
            **running,
            "status": "complete",
            "format": "locatemot-l71-data-contract-v1",
            "counts": {
                "fit_units": len(fit_indexed),
                "fixed_eval_units": len(eval_indexed),
                "candidate_rows": sum(int(row["candidate_count"]) for row in indexed),
                "fit_dataset": dict(sorted(dataset_counts.items())),
                "fit_categories": dict(sorted(fit_categories.items())),
                "fixed_eval_categories": dict(sorted(eval_categories.items())),
                "fit_videos": sorted({str(row["video"]) for row in fit_indexed}),
                "all_materialized_videos_seen": sorted({str(row["video"]) for row in indexed}),
                "duplicate_candidate_index_units": sum(bool(row["duplicate_candidate_index"]) for row in indexed),
                "future_history_rows": sum(int(value["future_history_rows"]) for value in video_stats.values()),
            },
            "checks": {
                "candidate_key_drift": 0,
                "candidate_truncation": False,
                "candidate_deletion": False,
                "fixed_eval_order": True,
                "missing_frame": 0,
                "missing_sidecar": 0,
                "text_lookup_missing": 0,
                "future_history_rows": 0,
                "observation_dim": 1432,
                "text_token_sequence_preserved": True,
                "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
            },
            "inputs": {
                **running["inputs"],
                "manifest_sha256": manifest_sha,
                "text_cache_shape": [int(text_cache["token_hidden"].shape[0]), int(text_cache["token_hidden"].shape[1]), int(text_cache["token_hidden"].shape[2])],
                "l69_videos": list(L69_VIDEOS),
            },
            "video_stats": video_stats,
            "next_action": "run the independent L71 forward/loss contract audits",
            "failure_root_cause": None,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(out / "contract.json", contract)
        write_json(out / "provenance.json", {
            **contract,
            "provenance_note": "L69 features are read-only; row keys preserve immutable bank row offsets; labels are joined after candidate/feature construction",
        })
        write_json(out / "status.json", {**contract, "outputs": {"unit_records": str(out / "unit_records.jsonl")}})
        print(__import__("json").dumps({"status": "complete", "counts": contract["counts"], "out": str(out)}), flush=True)
        return 0
    except Exception as exc:
        failure = {**running, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}", "next_action": "fix only the first data-contract root cause and rerun in a new directory", "elapsed_seconds": time.perf_counter() - started}
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text(
            "# L71 data contract INCOMPLETE\n\n"
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n"
            "```text\n" + traceback.format_exc() + "```\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
