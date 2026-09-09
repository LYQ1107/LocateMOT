#!/usr/bin/env python3
"""Build compact native-frame dev/internal indexes for the isolated R0 path.

The training indexes are intentionally separate from this tool.  Evaluation
uses only safe, already-isolated target artifacts and the native L69 frame
universe; it never invokes the historical L49 loader and never changes a
candidate row.
"""
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
    L49_DATA,
    FORBIDDEN_SCOPE_VIDEOS,
    MANIFEST,
    R0BankStore,
    R0DataContractError,
    R0FrameTargetSource,
    load_l82_video_split,
    load_l69_bank,
    native_frame_slice,
    sha256_file,
)
from locatemot.rmot.r0_safe_target_source import load_safe_query_records  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
INTERNAL_PAIRS = {
    "refer_kitti_v1": ("0004", "0018"),
    "refer_kitti_v2": ("0016", "0017", "0020"),
}


def validation_query_keys(dataset: str, videos: tuple[str, ...]) -> set[tuple[str, str, int]]:
    """Read only the legal internal query identity from validation metadata.

    The target-bearing fields are not used here.  Internal target maps are
    supplied later by the already-isolated safe artifact.
    """
    result: set[tuple[str, str, int]] = set()
    allowed = set(str(value) for value in videos)
    for line in (L49_DATA / "validation_units.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if str(raw.get("dataset")) != str(dataset) or str(raw.get("video")) not in allowed:
            continue
        result.add((str(raw["dataset"]), str(raw["video"]), int(raw["query_id"])))
    if not result:
        raise R0DataContractError(f"no fixed validation query keys for internal {dataset}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _pairs(scope: str, dataset: str) -> set[tuple[str, str]]:
    if dataset not in FIT_DATASETS:
        raise ValueError(dataset)
    if scope == "internal":
        result = {(dataset, video) for video in INTERNAL_PAIRS[dataset]}
    elif scope == "dev":
        split = load_l82_video_split()
        result = {
            (found_dataset, video)
            for value in split["dev_videos"]
            for found_dataset, video in [str(value).split("|", 1)]
            if found_dataset == dataset
        }
    else:
        raise ValueError(f"unsupported evaluation scope: {scope}")
    if not result or any(video in FORBIDDEN_SCOPE_VIDEOS for _dataset, video in result):
        raise R0DataContractError(f"invalid or forbidden {scope} pairs: {sorted(result)}")
    return result


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 evaluation index: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest_sha = sha256_file(MANIFEST)
    base = {
        "format": "locatemot-r0-native-eval-index-v1",
        "status": "incomplete",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD,
        "seed": SEED,
        "scope": str(args.scope),
        "dataset": str(args.dataset),
        "inputs": {
            "safe_targets": str(args.safe_targets.resolve()),
            "manifest": {"path": str(MANIFEST.resolve()), "sha256": manifest_sha, "expected_sha256": MANIFEST_SHA},
        },
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
        "next_action": "audit R0 evaluation index before scoring",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 evaluation-index cwd: {Path.cwd()}")
        if manifest_sha != MANIFEST_SHA:
            raise AssertionError(f"fixed manifest SHA drift: {manifest_sha}")
        pairs = _pairs(args.scope, args.dataset)
        records = load_safe_query_records(args.safe_targets.resolve())
        record_map = {(record.dataset, record.video, int(record.query_id)): record for record in records}
        selected_keys = {key for key in record_map if key[:2] in pairs}
        if args.scope == "internal":
            selected_keys &= validation_query_keys(args.dataset, INTERNAL_PAIRS[args.dataset])
        selected = [record_map[key] for key in sorted(selected_keys)]
        if not selected:
            raise R0DataContractError(f"no safe queries for {args.scope} {args.dataset}")
        source = R0FrameTargetSource(purpose=args.scope, allowed_video_pairs=pairs, records=selected)
        queries = [
            {
                "dataset": record.dataset,
                "video": record.video,
                "query_id": int(record.query_id),
                "sentence": record.sentence,
                "split": args.scope,
            }
            for record in selected
        ]
        (out / "queries.jsonl").write_text(
            "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in queries),
            encoding="utf-8",
        )
        frames_count = 0
        pair_count = 0
        query_frame_count = 0
        candidate_rows = 0
        category_counts: Counter[str] = Counter()
        video_summaries: list[dict[str, Any]] = []
        frame_path = out / "frames.jsonl"
        label_path = out / "query_frame_labels.jsonl"
        store = R0BankStore()
        with frame_path.open("w", encoding="utf-8") as frame_handle, label_path.open("w", encoding="utf-8") as label_handle:
            for _dataset, video in sorted(pairs):
                bank_path, blob = load_l69_bank(video)
                store.load_video(video)
                tensors = blob["tensors"]
                video_records = [record for record in selected if record.video == video]
                if not video_records:
                    raise R0DataContractError(f"safe artifact has no {args.scope} queries for {args.dataset}|{video}")
                frame_count = 0
                video_candidates = 0
                video_categories: Counter[str] = Counter()
                frame_ids = [int(value) for value in tensors["frame_ids"].long().tolist()]
                for frame_position, frame_id in enumerate(frame_ids):
                    _frame, begin, end = native_frame_slice(tensors, frame_position)
                    frame_batch = store.build_frame(args.dataset, video, -1, frame_id, "__r0_frame_manifest__")
                    if frame_batch.row_offsets != list(range(begin, end)):
                        raise R0DataContractError(f"native frame row offset drift: {args.dataset}|{video}|{frame_id}")
                    frame_record = {
                        "dataset": args.dataset,
                        "video": video,
                        "frame_index": int(frame_position),
                        "frame_id": int(frame_id),
                        "bank_path": str(bank_path.resolve()),
                        "bank_row_begin": int(begin),
                        "bank_row_end": int(end),
                        "candidate_count": int(end - begin),
                        "row_offsets": list(range(begin, end)),
                        "candidate_indices": list(frame_batch.candidate_indices),
                        "track_ids": list(frame_batch.track_ids),
                        "pool_ids": list(frame_batch.pool_ids),
                        "boxes_xyxy": frame_batch.boxes.tolist(),
                        "image_size": list(frame_batch.image_size),
                    }
                    frame_handle.write(json.dumps(frame_record, ensure_ascii=False, sort_keys=True) + "\n")
                    frames_count += 1
                    frame_count += 1
                    video_candidates += end - begin
                    for record in video_records:
                        current = store.build_frame(args.dataset, video, record.query_id, frame_id, record.sentence)
                        if current.row_offsets != frame_batch.row_offsets or current.candidate_indices != frame_batch.candidate_indices:
                            raise R0DataContractError(f"candidate order drift: {args.dataset}|{video}|{record.query_id}|{frame_id}")
                        supervision = store.attach_frame_labels(
                            current, source.target_ids(args.dataset, video, record.query_id, frame_id)
                        )
                        label = {
                            "format": "locatemot-r0-native-eval-label-v1",
                            "dataset": args.dataset,
                            "video": video,
                            "query_id": int(record.query_id),
                            "sentence": record.sentence,
                            "frame_index": int(frame_position),
                            "frame_id": int(frame_id),
                            "bank_path": str(bank_path.resolve()),
                            "bank_row_begin": int(begin),
                            "bank_row_end": int(end),
                            "row_offsets": list(range(begin, end)),
                            "candidate_count": int(current.candidate_count),
                            "positive_row_offsets": [
                                int(offset)
                                for offset, positive in zip(current.row_offsets, supervision["labels"].tolist())
                                if bool(positive)
                            ],
                            "target_ids": list(supervision["target_ids"]),
                            "covered_target_ids": list(supervision["covered_target_ids"]),
                            "visible_target_count": int(supervision["visible_target_count"]),
                            "covered_target_count": int(supervision["covered_target_count"]),
                            "positive_row_count": int(supervision["positive_row_count"]),
                            "category": str(supervision["category"]),
                            "target_present": bool(supervision["target_present"]),
                            "candidate_present": bool(supervision["candidate_present"]),
                            "present_uncovered": bool(supervision["present_uncovered"]),
                            "partially_covered": bool(supervision["partially_covered"]),
                            "coverage_fraction": float(supervision["coverage_fraction"]),
                            "membership_mask": [bool(value) for value in supervision["membership_mask"].tolist()],
                        }
                        if len(label["membership_mask"]) != int(current.candidate_count):
                            raise R0DataContractError(f"evaluation membership length drift: {label}")
                        label_handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
                        query_frame_count += 1
                        category_counts[str(label["category"])] += 1
                        video_categories[str(label["category"])] += 1
                pair_count += 1
                candidate_rows += video_candidates
                video_summaries.append(
                    {
                        "dataset": args.dataset,
                        "video": video,
                        "query_count": len(video_records),
                        "frame_count": frames_count,
                        "candidate_rows": video_candidates,
                        "category_counts": dict(sorted(video_categories.items())),
                        "bank_path": str(bank_path.resolve()),
                        "bank_sha256": sha256_file(bank_path),
                    }
                )
                del blob, tensors
                store._blob = None
                store._candidate_gt = None
        summary = {
            **base,
            "status": "complete",
            "pairs": [list(pair) for pair in sorted(pairs)],
            "pair_count": pair_count,
            "query_count": len(selected),
            "frame_count": frames_count,
            "query_frame_count": query_frame_count,
            "candidate_rows": candidate_rows,
            "category_counts": dict(sorted(category_counts.items())),
            "video_summaries": video_summaries,
            "source_descriptor": source.source_descriptor(),
            "labels_are_posthoc_for_scoring": True,
            "wall_seconds": time.perf_counter() - started,
            "next_action": "audit R0 evaluation index before scoring",
        }
        write_json(out / "summary.json", summary)
        write_json(out / "provenance.json", summary)
        write_json(out / "status.json", summary)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 evaluation index — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {
            **base,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback_path": str((out / "INCOMPLETE.md").resolve()),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--dataset", choices=FIT_DATASETS, required=True)
    parser.add_argument("--safe-targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
