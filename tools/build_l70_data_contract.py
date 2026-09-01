#!/usr/bin/env python3
"""Build the L70 index from the L69 budget-40 bank.

The script writes only compact row metadata.  It never writes feature tensors,
images, or a derived bank.  Labels are joined after the L69 frame rows and
causal history indices have been constructed by :mod:`l70_common`.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from l70_common import (
    DATASETS,
    L49_DATA_ROOT,
    L62_RECORDS,
    L69_FEATURE_ROOT,
    L69_VIDEOS,
    MANIFEST_PATH,
    MANIFEST_SHA256,
    TEXT_CACHE_PATH,
    L69Bank,
    dataset_video_counts,
    fixed_eval_units,
    load_l49_splits,
    load_l62_order,
    load_text_cache,
    make_unit_record,
    safe_torch_load,
    sha256_file,
    unit_key,
    write_json,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
DEFAULT_OUT = ROOT / "outputs/l70/audit/data_contract"


def _stat(path: Path) -> dict[str, object]:
    info = path.stat()
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "exists": True,
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _write_lines(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    common = {
        "format": "locatemot-l70-data-contract-v1",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(__import__("sys").argv),
        "seed": 20260829,
        "status": "running",
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_dense_feature_cache_written": False,
    }
    write_json(status_path, common)
    try:
        if str(ROOT.resolve()) != os.getcwd():
            raise RuntimeError(f"cwd mismatch: {os.getcwd()}")
        manifest_sha = sha256_file(MANIFEST_PATH)
        if manifest_sha != MANIFEST_SHA256:
            raise RuntimeError(f"manifest SHA mismatch: {manifest_sha}")
        splits = load_l49_splits()
        text_cache = load_text_cache()
        fit_units = [row for row in splits["fit"] if row["dataset"] in DATASETS]
        if len(fit_units) != 5314:
            raise AssertionError(f"expected 5314 fit units, got {len(fit_units)}")
        eval_units = fixed_eval_units(splits)
        eval_order = load_l62_order()
        if [unit_key(row) for row in eval_units] != [str(row["unit_key"]) for row in eval_order]:
            raise AssertionError("fixed L62 order did not map exactly to L49 units")

        all_records: list[dict[str, object]] = []
        stats = {
            "fit": Counter(),
            "calibration": Counter(),
            "validation": Counter(),
            "fit_dataset": Counter(),
            "eval_dataset": Counter(),
            "fit_video": Counter(),
            "eval_video": Counter(),
        }
        rows_by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in fit_units:
            rows_by_video[str(row["video"])].append(row)
        for row in eval_units:
            rows_by_video[str(row["video"])].append(row)

        bank_provenance: dict[str, object] = {}
        for video in L69_VIDEOS:
            video_units = rows_by_video.get(video, [])
            if not video_units:
                continue
            bank = L69Bank(video)
            try:
                bank_provenance[video] = {
                    "feature_bank": _stat(bank.path),
                    "label_sidecar": _stat(bank.label_path),
                    "rows": bank.count,
                    "frames": bank.frame_count,
                    "metadata": bank.blob.get("metadata", {}),
                }
                for unit in video_units:
                    record = make_unit_record(unit, bank)
                    # A text lookup is a contract check, not a model feature
                    # selection rule.  The token sequence itself is not saved.
                    sentence = str(record["sentence"])
                    if sentence not in text_cache["sentence_to_index"]:
                        raise KeyError(f"text sentence absent: {sentence!r}")
                    record["text_cache_index"] = int(text_cache["sentence_to_index"][sentence])
                    record["text_hidden_shape"] = [int(x) for x in text_cache["token_hidden"].shape[1:]]
                    record["text_mask_true_count"] = int(
                        text_cache["attention_mask"][record["text_cache_index"]].sum()
                    )
                    all_records.append(record)
                    split = str(record["split"])
                    category = str(record["category"])
                    stats[split][category] += 1
                    stats["fit_dataset" if split == "fit" else "eval_dataset"][str(record["dataset"])] += 1
                    stats["fit_video" if split == "fit" else "eval_video"][video] += 1
            finally:
                bank.close()

        fit_records = [row for row in all_records if row["split"] == "fit"]
        eval_records = [row for row in all_records if row["split"] != "fit"]
        expected_keys = [unit_key(row) for row in fit_units] + [unit_key(row) for row in eval_units]
        if [str(row["unit_key"]) for row in all_records] != expected_keys:
            # The iteration is video-grouped for bounded memory, so write a
            # stable global order rather than accepting incidental grouping.
            by_key = {str(row["unit_key"]): row for row in all_records}
            if len(by_key) != len(all_records) or any(key not in by_key for key in expected_keys):
                raise AssertionError("unit key drift/duplicate while building index")
            all_records = [by_key[key] for key in expected_keys]
            fit_records = [row for row in all_records if row["split"] == "fit"]
            eval_records = [row for row in all_records if row["split"] != "fit"]
        if len(fit_records) != 5314 or len(eval_records) != 40:
            raise AssertionError(f"record counts fit={len(fit_records)} eval={len(eval_records)}")
        expected_fit_videos = {str(row["video"]) for row in fit_units}
        if set(str(row["video"]) for row in fit_records) != expected_fit_videos:
            raise AssertionError("fit records do not cover the videos represented by L49 split=fit")
        if not expected_fit_videos.issubset(set(L69_VIDEOS)):
            raise AssertionError("L49 fit contains a video outside the fixed L69 materialization union")
        for name in ("positive", "multi_positive", "inactive", "present_uncovered"):
            if not stats["fit"][name]:
                raise AssertionError(f"fit category missing: {name}")
            if not any(row["category"] == name for row in eval_records):
                raise AssertionError(f"fixed eval category missing: {name}")
        for record in all_records:
            if int(record["candidate_count"]) != int(record["frame_pointer"]["end"]) - int(record["frame_pointer"]["begin"]):
                raise AssertionError("candidate count/frame pointer mismatch")
            if len(record["row_keys"]) != int(record["candidate_count"]):
                raise AssertionError("row key count mismatch")
            for history in record["history_frame_ids"]:
                if any(int(value) > int(record["frame_id"]) for value in history):
                    raise AssertionError("future history found")

        _write_lines(out / "unit_records.jsonl", all_records)
        write_json(out / "provenance.json", {
            **common,
            "status": "complete",
            "inputs": {
                "l69_feature_root": str(L69_FEATURE_ROOT),
                "l69_videos": list(L69_VIDEOS),
                "banks": bank_provenance,
                "l49_data": {name: _stat(L49_DATA_ROOT / filename) for name, filename in {
                    "fit": "train_units.jsonl", "calibration": "calibration_units.jsonl",
                    "validation": "validation_units.jsonl",
                }.items()},
                "l62_order": _stat(L62_RECORDS),
                "text_cache": _stat(TEXT_CACHE_PATH),
                "manifest": {"path": str(MANIFEST_PATH), "sha256": manifest_sha},
            },
            "protocol": {
                "old_l49_begin_end_used": False,
                "old_l49_positive_indices_used": False,
                "l69_frame_ptr_source_of_truth": True,
                "candidate_set_complete": True,
                "duplicate_candidate_index_retained": True,
                "row_key": "(dataset,video,query_id,frame_id,bank_path,row_offset)",
                "history_max": 8,
                "history_causal": True,
                "feature_dim": 1432,
                "feature_fields": ["clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness"],
                "text_shape": [int(x) for x in text_cache["token_hidden"].shape[1:]],
                "token_region_alignment": "UNALIGNED",
            },
        })
        write_json(out / "contract.json", {
            **common,
            "status": "complete",
            "inputs": {"fit_units": 5314, "fixed_eval_units": 40, "fixed_eval_order": True},
            "outputs": {"unit_records": str(out / "unit_records.jsonl")},
            "counts": {
                "fit_units": len(fit_records),
                "fixed_eval_units": len(eval_records),
                "fit_categories": dict(stats["fit"]),
                "calibration_categories": dict(stats["calibration"]),
                "validation_categories": dict(stats["validation"]),
                "fit_dataset": dict(stats["fit_dataset"]),
                "eval_dataset": dict(stats["eval_dataset"]),
                "fit_video": dict(stats["fit_video"]),
                "eval_video": dict(stats["eval_video"]),
                "candidate_rows": int(sum(int(row["candidate_count"]) for row in all_records)),
                "duplicate_candidate_index_units": int(sum(bool(row["duplicate_candidate_index"]) for row in all_records)),
                "future_history_rows": 0,
            },
            "checks": {
                "manifest_sha256": manifest_sha,
                "candidate_key_drift": 0,
                "missing_frame": 0,
                "missing_sidecar": 0,
                "text_lookup_missing": 0,
                "candidate_truncation": False,
                "no_test_or_screening_labels": True,
            },
        })
        complete = {**common, "status": "complete", "failure_root_cause": None, "next_action": "run L70 CPU/forward contract then B0 smoke"}
        write_json(status_path, complete)
        return 0
    except Exception as exc:
        failure = {
            **common,
            "status": "INCOMPLETE",
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "fix only the first data-contract root cause and rerun in a new attempt",
        }
        write_json(status_path, failure)
        (out / "INCOMPLETE.md").write_text(
            "# L70 data contract INCOMPLETE\n\n"
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n"
            "The traceback is retained below; no semantic/training result is valid.\n\n"
            "```text\n" + traceback.format_exc() + "```\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
