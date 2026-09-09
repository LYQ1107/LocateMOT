#!/usr/bin/env python3
"""Audit R0 dense native-frame indexes and their L82 scope contract."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r0_dense_data import (  # noqa: E402
    FIT_DATASETS,
    MANIFEST,
    R0DataContractError,
    load_fit_rows,
    load_l82_video_split,
    load_l69_bank,
    native_frame_slice,
    sha256_file,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_pairs(split: dict[str, Any], field: str, dataset: str) -> set[tuple[str, str]]:
    result = set()
    for value in split[field]:
        found_dataset, video = str(value).split("|", 1)
        if found_dataset == dataset:
            result.add((found_dataset, video))
    return result


def audit_one(dataset: str, index_root: Path, fit_rows: list[dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    index_root = index_root.resolve()
    summary = json.loads((index_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise R0DataContractError(f"{dataset}: dense summary is not complete")
    train_pairs = split_pairs(split, "train_videos", dataset)
    dev_pairs = split_pairs(split, "dev_videos", dataset)
    queries = read_jsonl(index_root / "queries.jsonl")
    frames = read_jsonl(index_root / "frames.jsonl")
    labels = read_jsonl(index_root / "query_frame_labels.jsonl")
    selected_fit = [row for row in fit_rows if (str(row["dataset"]), str(row["video"])) in train_pairs]
    expected_query_keys = {(dataset, str(row["video"]), int(row["query_id"])) for row in selected_fit}
    query_keys = {(str(row["dataset"]), str(row["video"]), int(row["query_id"])) for row in queries}
    if query_keys != expected_query_keys:
        raise R0DataContractError(f"{dataset}: query key mismatch missing={len(expected_query_keys-query_keys)} extra={len(query_keys-expected_query_keys)}")
    if any((str(row["dataset"]), str(row["video"])) not in train_pairs for row in queries):
        raise R0DataContractError(f"{dataset}: query outside train scope")
    if set((str(row["dataset"]), str(row["video"])) for row in queries) & dev_pairs:
        raise R0DataContractError(f"{dataset}: dev pair entered dense train index")
    frame_map: dict[tuple[str, int], dict[str, Any]] = {}
    bank_stats: list[dict[str, Any]] = []
    duplicate_candidate_rows = 0
    for video in sorted({pair[1] for pair in train_pairs}):
        bank_path, blob = load_l69_bank(video)
        tensors = blob["tensors"]
        expected_frames = [int(value) for value in tensors["frame_ids"].long().tolist()]
        video_frames = [row for row in frames if str(row["video"]) == video]
        if [int(row["frame_id"]) for row in video_frames] != expected_frames:
            raise R0DataContractError(f"{dataset}|{video}: native frame order/count drift")
        video_rows = 0
        duplicate_count = 0
        for position, row in enumerate(video_frames):
            frame_id, begin, end = native_frame_slice(tensors, position)
            if int(row["frame_index"]) != position or int(row["bank_row_begin"]) != begin or int(row["bank_row_end"]) != end or int(row["candidate_count"]) != end - begin:
                raise R0DataContractError(f"{dataset}|{video}|{frame_id}: native pointer drift")
            candidate_indices = [int(value) for value in tensors["candidate_index"][begin:end].tolist()]
            track_ids = [int(value) for value in tensors["track_id"][begin:end].tolist()]
            pool_ids = [int(value) for value in tensors["pool_id"][begin:end].tolist()]
            if row["candidate_indices"] != candidate_indices or row["track_ids"] != track_ids or row["pool_ids"] != pool_ids:
                raise R0DataContractError(f"{dataset}|{video}|{frame_id}: candidate/order metadata drift")
            boxes = torch.tensor(row["boxes_xyxy"], dtype=torch.float32)
            expected_boxes = tensors["box"][begin:end].float()
            if boxes.shape != expected_boxes.shape or not torch.equal(boxes, expected_boxes):
                raise R0DataContractError(f"{dataset}|{video}|{frame_id}: box drift")
            if not bool(torch.isfinite(boxes).all()):
                raise FloatingPointError(f"{dataset}|{video}|{frame_id}: nonfinite boxes")
            duplicate_count += len(candidate_indices) - len(set(candidate_indices))
            video_rows += end - begin
            frame_map[(video, frame_id)] = row
        bank_stats.append({"dataset": dataset, "video": video, "frame_count": len(video_frames), "candidate_rows": video_rows, "duplicate_candidate_index_rows": duplicate_count, "bank_path": str(bank_path.resolve()), "bank_sha256": sha256_file(bank_path)})
        duplicate_candidate_rows += duplicate_count
        del blob
    label_keys = [(str(row["dataset"]), str(row["video"]), int(row["query_id"]), int(row["frame_id"])) for row in labels]
    if len(label_keys) != len(set(label_keys)):
        raise R0DataContractError(f"{dataset}: duplicate query-frame labels")
    category_counts: Counter[str] = Counter()
    query_frames: Counter[tuple[str, int]] = Counter()
    for row in labels:
        key = (str(row["dataset"]), str(row["video"]), int(row["query_id"]))
        if key not in query_keys:
            raise R0DataContractError(f"{dataset}: label query outside dense query set: {key}")
        frame_key = (str(row["video"]), int(row["frame_id"]))
        frame = frame_map.get(frame_key)
        if frame is None or int(row["candidate_count"]) != int(frame["candidate_count"]) or int(row["bank_row_begin"]) != int(frame["bank_row_begin"]) or int(row["bank_row_end"]) != int(frame["bank_row_end"]):
            raise R0DataContractError(f"{dataset}: label/frame row mismatch: {key}|{frame_key}")
        if len(row["membership_mask"]) != int(row["candidate_count"]):
            raise R0DataContractError(f"{dataset}: membership mask length drift")
        positive_offsets = [int(value) for value in row["positive_row_offsets"]]
        if positive_offsets != sorted(set(positive_offsets)):
            raise R0DataContractError(f"{dataset}: positive offsets duplicated/unsorted")
        if any(value < int(row["bank_row_begin"]) or value >= int(row["bank_row_end"]) for value in positive_offsets):
            raise R0DataContractError(f"{dataset}: positive offset out of range")
        expected_mask_value = not bool(row["present_uncovered"])
        expected_mask = [expected_mask_value for _ in range(int(row["bank_row_begin"]), int(row["bank_row_end"]))]
        if [bool(value) for value in row["membership_mask"]] != expected_mask:
            raise R0DataContractError(f"{dataset}: dense membership mask/positive offsets drift")
        category = str(row["category"])
        category_counts[category] += 1
        visible, covered = int(row["visible_target_count"]), int(row["covered_target_count"])
        expected_category = "inactive" if visible == 0 else "present_uncovered" if covered == 0 else "positive" if covered == 1 else "multi_positive"
        if category != expected_category or int(row["positive_row_count"]) != len(positive_offsets) or bool(row["candidate_present"]) != (covered > 0) or bool(row["present_uncovered"]) != (visible > 0 and covered == 0):
            raise R0DataContractError(f"{dataset}: supervision category/coverage drift")
        query_frames[(str(row["video"]), int(row["query_id"]))] += 1
    expected_label_count = sum(sum(1 for frame in frames if str(frame["video"]) == video) for _dataset, video, _qid in query_keys)
    if len(labels) != expected_label_count or any(value != sum(1 for frame in frames if str(frame["video"]) == video) for (video, _qid), value in query_frames.items()):
        raise R0DataContractError(f"{dataset}: query-frame cardinality drift")
    return {"dataset": dataset, "status": "complete", "query_count": len(queries), "fit_unit_rows": len(selected_fit), "frame_count": len(frames), "query_frame_count": len(labels), "candidate_rows": int(sum(item["candidate_rows"] for item in bank_stats)), "duplicate_candidate_index_rows": duplicate_candidate_rows, "category_counts": dict(sorted(category_counts.items())), "bank_stats": bank_stats, "train_pairs": [list(pair) for pair in sorted(train_pairs)], "dev_pairs": [list(pair) for pair in sorted(dev_pairs)], "candidate_deletion": False, "candidate_truncation": False}


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty dense audit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {"format": "locatemot-r0-dense-index-contract-audit-v1", "status": "incomplete", "command": " ".join([sys.executable, *sys.argv]), "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD, "inputs": {"v1": str(args.v1.resolve()), "v2": str(args.v2.resolve()), "manifest_sha256": sha256_file(MANIFEST), "expected_manifest_sha256": MANIFEST_SHA}, "outputs": {"root": str(out)}, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False, "failure_root_cause": None, "next_action": "run R0-only compile/boundary/forward contract smoke"}
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0A cwd: {Path.cwd()}")
        if base["inputs"]["manifest_sha256"] != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        fit_rows = load_fit_rows()
        split = load_l82_video_split()
        per_dataset = [audit_one(dataset, path, fit_rows, split) for dataset, path in (("refer_kitti_v1", args.v1), ("refer_kitti_v2", args.v2))]
        payload = {**base, "status": "complete", "per_dataset": per_dataset, "wall_seconds": time.perf_counter() - started, "next_action": "run compile, boundary guard, and R0 forward/backward contract smoke"}
        for name in ("audit.json", "provenance.json", "status.json"):
            write_json(out / name, payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 dense index audit — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback_path": str((out / "INCOMPLETE.md").resolve()), "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
