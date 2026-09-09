#!/usr/bin/env python3
"""Build a real R0 native-frame dense index from safe target artifacts."""
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
    FIT_DATASETS,
    MANIFEST,
    R0BankStore,
    R0DataContractError,
    R0FrameTargetSource,
    load_fit_rows,
    load_l82_video_split,
    load_l69_bank,
    native_frame_slice,
    sha256_file,
)
from locatemot.rmot.r0_safe_target_source import load_safe_query_records  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
FORBIDDEN = {"0005", "0011", "0013", "0019"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def pair_set(split: dict[str, Any], field: str, dataset: str) -> set[tuple[str, str]]:
    values = split.get(field)
    if not isinstance(values, list):
        raise R0DataContractError(f"invalid L82 {field}")
    result = set()
    for value in values:
        if not isinstance(value, str) or value.count("|") != 1:
            raise R0DataContractError(f"invalid L82 pair: {value!r}")
        found_dataset, video = value.split("|", 1)
        if found_dataset == dataset:
            result.add((found_dataset, video))
    if {video for _dataset, video in result} & FORBIDDEN:
        raise R0DataContractError(f"forbidden video in L82 {field}: {result}")
    return result


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty dense index: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {
        "format": "locatemot-r0-dense-native-frame-index-v2",
        "status": "incomplete",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD,
        "seed": SEED,
        "dataset": args.dataset,
        "inputs": {"safe_targets": str(args.safe_targets.resolve()), "manifest": {"path": str(MANIFEST.resolve()), "sha256": sha256_file(MANIFEST), "expected_sha256": MANIFEST_SHA}},
        "outputs": {"root": str(out)},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "labels_in_visual_cache": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "failure_root_cause": None,
        "next_action": "audit dense index integrity and train/dev leakage",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0A worktree cwd: {Path.cwd()}")
        if base["inputs"]["manifest"]["sha256"] != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if args.dataset not in FIT_DATASETS:
            raise ValueError(args.dataset)
        rows = load_fit_rows()
        split = load_l82_video_split()
        train_pairs = pair_set(split, "train_videos", args.dataset)
        dev_pairs = pair_set(split, "dev_videos", args.dataset)
        if train_pairs & dev_pairs:
            raise AssertionError("train/dev video overlap")
        selected_rows = [row for row in rows if str(row["dataset"]) == args.dataset and (str(row["dataset"]), str(row["video"])) in train_pairs]
        if not selected_rows:
            raise R0DataContractError("no fit rows for selected L82 train pairs")
        selected_keys = {(args.dataset, str(row["video"]), int(row["query_id"])) for row in selected_rows}
        records = load_safe_query_records(args.safe_targets.resolve())
        record_map = {(record.dataset, record.video, int(record.query_id)): record for record in records}
        missing_records = sorted(selected_keys - set(record_map))
        if missing_records:
            raise R0DataContractError(f"safe artifact missing selected fit queries: {missing_records[:8]}")
        selected_records = [record_map[key] for key in sorted(selected_keys)]
        source = R0FrameTargetSource(purpose="train", allowed_video_pairs=train_pairs, records=selected_records)
        queries = [{"dataset": args.dataset, "video": record.video, "query_id": int(record.query_id), "sentence": record.sentence, "split": "fit"} for record in selected_records]
        (out / "queries.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in queries), encoding="utf-8")
        frame_path = out / "frames.jsonl"
        label_path = out / "query_frame_labels.jsonl"
        counters: Counter[str] = Counter()
        video_summaries: list[dict[str, Any]] = []
        store = R0BankStore()
        with frame_path.open("w", encoding="utf-8") as frame_handle, label_path.open("w", encoding="utf-8") as label_handle:
            for _dataset, video in sorted(train_pairs):
                store.load_video(video)
                tensors = store.tensors
                bank_path = (store._path or Path("")).resolve()
                frame_ids = [int(value) for value in tensors["frame_ids"].long().tolist()]
                video_queries = [record for record in selected_records if record.video == video]
                if not video_queries:
                    raise R0DataContractError(f"no selected train query for {args.dataset}|{video}")
                frame_count = 0
                candidate_rows = 0
                category_counts: Counter[str] = Counter()
                for frame_position, frame_id in enumerate(frame_ids):
                    _frame, begin, end = native_frame_slice(tensors, frame_position)
                    batch = store.build_frame(args.dataset, video, -1, frame_id, "__frame_manifest__")
                    frame_record = {
                        "dataset": args.dataset, "video": video, "frame_index": int(frame_position), "frame_id": int(frame_id),
                        "bank_path": str(bank_path), "bank_row_begin": int(begin), "bank_row_end": int(end), "candidate_count": int(end - begin),
                        "row_offsets": list(range(begin, end)), "candidate_indices": list(batch.candidate_indices), "track_ids": list(batch.track_ids), "pool_ids": list(batch.pool_ids),
                        "boxes_xyxy": batch.boxes.tolist(), "image_size": list(batch.image_size),
                    }
                    frame_handle.write(json.dumps(frame_record, ensure_ascii=False, sort_keys=True) + "\n")
                    frame_count += 1
                    candidate_rows += end - begin
                    counters["frames"] += 1
                    counters["candidate_rows"] += end - begin
                    for record in video_queries:
                        current = store.build_frame(args.dataset, video, record.query_id, frame_id, record.sentence)
                        if current.row_offsets != batch.row_offsets or current.candidate_indices != batch.candidate_indices or current.track_ids != batch.track_ids or current.pool_ids != batch.pool_ids:
                            raise R0DataContractError(f"row key drift at {args.dataset}|{video}|{record.query_id}|{frame_id}")
                        supervision = store.attach_frame_labels(current, source.target_ids(args.dataset, video, record.query_id, frame_id))
                        category = str(supervision["category"])
                        category_counts[category] += 1
                        counters[category] += 1
                        positive_offsets = [int(offset) for offset, value in zip(current.row_offsets, supervision["labels"].tolist()) if bool(value)]
                        label_record = {
                            "dataset": args.dataset, "video": video, "query_id": int(record.query_id), "sentence": record.sentence,
                            "frame_index": int(frame_position), "frame_id": int(frame_id), "bank_path": str(bank_path),
                            "bank_row_begin": int(begin), "bank_row_end": int(end), "candidate_count": int(current.candidate_count),
                            "positive_row_offsets": positive_offsets, "target_ids": supervision["target_ids"], "covered_target_ids": supervision["covered_target_ids"],
                            "visible_target_count": int(supervision["visible_target_count"]), "covered_target_count": int(supervision["covered_target_count"]),
                            "positive_row_count": int(supervision["positive_row_count"]), "category": category,
                            "target_present": bool(supervision["target_present"]), "candidate_present": bool(supervision["candidate_present"]),
                            "present_uncovered": bool(supervision["present_uncovered"]), "partially_covered": bool(supervision["partially_covered"]),
                            "coverage_fraction": float(supervision["coverage_fraction"]), "membership_mask": [bool(value) for value in supervision["membership_mask"].tolist()],
                        }
                        if len(label_record["membership_mask"]) != int(current.candidate_count):
                            raise AssertionError("dense label mask length drift")
                        label_handle.write(json.dumps(label_record, ensure_ascii=False, sort_keys=True) + "\n")
                        counters["query_frames"] += 1
                video_summaries.append({"dataset": args.dataset, "video": video, "query_count": len(video_queries), "frame_count": frame_count, "candidate_rows": candidate_rows, "category_counts": dict(sorted(category_counts.items())), "bank_path": str(bank_path), "bank_sha256": sha256_file(bank_path)})
                store._blob = None
                store._candidate_gt = None
        summary = {
            **base, "status": "complete", "train_video_pairs": [list(pair) for pair in sorted(train_pairs)], "dev_video_pairs": [list(pair) for pair in sorted(dev_pairs)],
            "selected_fit_unit_rows": len(selected_rows), "selected_query_count": len(selected_records), "frame_count": int(counters["frames"]), "query_frame_count": int(counters["query_frames"]),
            "candidate_rows": int(counters["candidate_rows"]), "category_counts": dict(sorted((key, int(value)) for key, value in counters.items() if key in {"inactive", "positive", "multi_positive", "present_uncovered"})),
            "video_summaries": video_summaries, "source_descriptor": source.source_descriptor(), "wall_seconds": time.perf_counter() - started,
            "next_action": "audit dense native-frame index integrity and train/dev leakage",
        }
        write_json(out / "summary.json", summary); write_json(out / "provenance.json", summary); write_json(out / "status.json", summary)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 dense native-frame index — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback_path": str((out / "INCOMPLETE.md").resolve()), "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=FIT_DATASETS, required=True)
    parser.add_argument("--safe-targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
