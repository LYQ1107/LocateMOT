#!/usr/bin/env python3
"""Materialize R0A label-bearing target artifacts from the safe source only.

The process reads L49 unit metadata to define legal scopes and then obtains
target records exclusively through ``r0_safe_target_source``.  No historical
full-file L49 loader is imported or called here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r0_dense_data import (  # noqa: E402
    ASSET_ROOT,
    FIT_DATASETS,
    FORBIDDEN_SCOPE_VIDEOS,
    L49_DATA,
    load_fit_rows,
    load_l82_video_split,
    read_jsonl,
    sha256_file,
)
from locatemot.rmot.r0_safe_target_source import (  # noqa: E402
    R0SafeLoadResult,
    R0SafeQueryRecord,
    R0SafeSourceError,
    load_safe_query_records_with_manifest,
    write_safe_target_artifact,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
TRAIN_UNITS = L49_DATA / "train_units.jsonl"
CALIBRATION_UNITS = L49_DATA / "calibration_units.jsonl"
VALIDATION_UNITS = L49_DATA / "validation_units.jsonl"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def pairs_from_rows(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs = {(str(row["dataset"]), str(row["video"])) for row in rows}
    if not pairs or any(dataset not in FIT_DATASETS for dataset, _ in pairs):
        raise R0SafeSourceError(f"invalid empty/dataset scope: {sorted(pairs)}")
    if {video for _, video in pairs} & FORBIDDEN_SCOPE_VIDEOS:
        raise R0SafeSourceError("forbidden official video in target scope")
    return pairs


def split_pairs(payload: dict[str, Any], field: str, dataset: str) -> set[tuple[str, str]]:
    values = payload.get(field)
    if not isinstance(values, list):
        raise R0SafeSourceError(f"missing split field {field}")
    result = set()
    for value in values:
        if not isinstance(value, str) or value.count("|") != 1:
            raise R0SafeSourceError(f"invalid split value {value!r}")
        found_dataset, video = value.split("|", 1)
        if found_dataset == dataset:
            result.add((found_dataset, video))
    return result


def records_for_pairs(pairs: set[tuple[str, str]], purpose: str) -> R0SafeLoadResult:
    return load_safe_query_records_with_manifest(pairs, purpose=purpose)


def selected_key_summary(rows: list[dict[str, Any]], pairs: set[tuple[str, str]]) -> dict[str, Any]:
    selected = [row for row in rows if (str(row.get("dataset")), str(row.get("video"))) in pairs]
    keys = sorted({(str(row["dataset"]), str(row["video"]), int(row["query_id"])) for row in selected})
    return {
        "unit_rows": len(selected),
        "unique_unit_query_keys": len(keys),
        "query_keys": [list(key) for key in keys],
    }


def build_one(
    out_root: Path,
    name: str,
    pairs: set[tuple[str, str]],
    *,
    purpose: str,
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    out = out_root / name
    result = records_for_pairs(pairs, purpose)
    write_safe_target_artifact(
        result.records,
        out,
        scope_name=name,
        allowed_video_pairs=pairs,
        source_manifest=result.manifest,
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["reference_unit_summary"] = selected_key_summary(reference_rows, pairs)
    manifest["safe_record_count"] = len(result.records)
    manifest["format"] = "locatemot-r0a-safe-target-artifact-v2"
    manifest["status"] = "complete"
    manifest["forbidden_payload_deserialized"] = False
    manifest["official_test_labels_read"] = False
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "name": name,
        "purpose": purpose,
        "allowed_video_pairs": [list(pair) for pair in sorted(pairs)],
        "safe_record_count": len(result.records),
        "reference_unit_summary": manifest["reference_unit_summary"],
        "forbidden_payload_deserialized": False,
        "official_test_labels_read": False,
        "path": str(out.resolve()),
    }


def run(args: argparse.Namespace) -> int:
    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"refusing nonempty safe-target root: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {
        "format": "locatemot-r0a-safe-target-artifacts-build-v1",
        "status": "incomplete",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD,
        "seed": SEED,
        "manifest": {"path": str(MANIFEST.resolve()), "sha256": sha256_file(MANIFEST), "expected_sha256": MANIFEST_SHA},
        "inputs": {
            "train_units": {"path": str(TRAIN_UNITS.resolve()), "sha256": sha256_file(TRAIN_UNITS)},
            "calibration_units": {"path": str(CALIBRATION_UNITS.resolve()), "sha256": sha256_file(CALIBRATION_UNITS)},
            "validation_units": {"path": str(VALIDATION_UNITS.resolve()), "sha256": sha256_file(VALIDATION_UNITS)},
            "l82_split": str((WORK_ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json").resolve()),
        },
        "outputs": {"root": str(out_root)},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "forbidden_payload_deserialized": False,
        "failure_root_cause": None,
        "next_action": "run the 5314-row sparse-vs-authoritative-frame audit",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0A worktree cwd: {Path.cwd()}")
        if base["manifest"]["sha256"] != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        train_rows = load_fit_rows()
        calibration_rows = read_jsonl(CALIBRATION_UNITS)
        validation_rows = read_jsonl(VALIDATION_UNITS)
        split = load_l82_video_split()
        train_pairs_all = pairs_from_rows(train_rows)
        train_pairs_by_dataset = {dataset: {pair for pair in train_pairs_all if pair[0] == dataset} for dataset in FIT_DATASETS}
        split_train = set()
        split_dev = set()
        for dataset in FIT_DATASETS:
            split_train |= split_pairs(split, "train_videos", dataset)
            split_dev |= split_pairs(split, "dev_videos", dataset)
        if split_train & split_dev:
            raise AssertionError(f"L82 train/dev overlap: {sorted(split_train & split_dev)}")
        scopes = [
            ("fit_audit", train_pairs_all, "audit", train_rows),
            ("v1_train", {pair for pair in split_train if pair[0] == "refer_kitti_v1"}, "train", train_rows),
            ("v2_train", {pair for pair in split_train if pair[0] == "refer_kitti_v2"}, "train", train_rows),
            ("v1_dev", {pair for pair in split_dev if pair[0] == "refer_kitti_v1"}, "dev", train_rows),
            ("v2_dev", {pair for pair in split_dev if pair[0] == "refer_kitti_v2"}, "dev", train_rows),
            ("v1_internal", {("refer_kitti_v1", video) for video in ("0004", "0018")}, "internal", validation_rows),
            ("v2_internal", {("refer_kitti_v2", video) for video in ("0016", "0017", "0020")}, "internal", validation_rows),
        ]
        summaries = [build_one(out_root, name, pairs, purpose=purpose, reference_rows=rows) for name, pairs, purpose, rows in scopes]
        payload = {
            **base,
            "status": "complete",
            "artifact_summaries": summaries,
            "train_rows": len(train_rows),
            "calibration_rows_read_for_internal_protocol_only": len(calibration_rows),
            "validation_rows_read_for_internal_protocol_only": len(validation_rows),
            "wall_seconds": time.perf_counter() - started,
            "next_action": "run exact 5314-row sparse-vs-authoritative-frame audit before dense indexes",
        }
        for name in ("build_summary.json", "provenance.json", "status.json"):
            write_json(out_root / name, payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out_root / "INCOMPLETE.md").write_text("# R0A safe target artifacts — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback_path": str((out_root / "INCOMPLETE.md").resolve()), "wall_seconds": time.perf_counter() - started}
        write_json(out_root / "provenance.json", payload)
        write_json(out_root / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/r0/data/safe_targets"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
