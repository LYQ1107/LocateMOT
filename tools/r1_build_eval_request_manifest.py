#!/usr/bin/env python3
"""Build a label-free R1 legal-dev/fixed-semantic request manifest.

Only key/text metadata from the isolated R0A query indexes and the immutable
40-unit order is used.  Frame ids and candidate counts come from native L69
frame pointers.  No target, category, candidate-GT or score field is copied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r0_dense_data import load_l69_bank  # noqa: E402
from tools.r1_common import (  # noqa: E402
    DEV_ROOTS, FORBIDDEN_VIDEOS, MANIFEST, MANIFEST_SHA, ROOT, SEED, THREAD, check_manifest,
    file_meta, load_fixed_l62_key_order, write_json,
)

L82_SPLIT = ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
FORBIDDEN_FIELDS = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "candidate_gt", "track_id", "state_key", "source_id", "pool_id",
}


def _key(dataset: str, video: str, query_id: int, frame_id: int) -> str:
    return f"{dataset}|{video}|{int(query_id)}|{int(frame_id)}"


def _public_query(raw: dict[str, Any], dataset: str, video: str) -> dict[str, Any]:
    forbidden = FORBIDDEN_FIELDS.intersection(raw)
    if forbidden:
        raise AssertionError(f"forbidden query fields: {sorted(forbidden)}")
    sentence = str(raw.get("sentence") or raw.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty sentence: {dataset}|{video}|{raw.get('query_id')}")
    return {
        "dataset": str(dataset), "video": str(video), "query_id": int(raw["query_id"]),
        "sentence": sentence, "split": "legal_dev",
    }


def _dev_videos() -> list[tuple[str, str]]:
    payload = json.loads(L82_SPLIT.read_text(encoding="utf-8"))
    values = [tuple(str(value).split("|", 1)) for value in payload.get("dev_videos", [])]
    if len(values) != 6 or len(set(values)) != 6:
        raise AssertionError(f"legal dev video contract drift: {values}")
    if any(video in FORBIDDEN_VIDEOS for _dataset, video in values):
        raise AssertionError(f"forbidden legal dev video: {values}")
    return sorted(values)


def _dev_queries(dataset: str, video: str) -> list[dict[str, Any]]:
    path = DEV_ROOTS[dataset] / "queries.jsonl"
    found: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if str(raw.get("dataset")) != dataset or str(raw.get("video")) != video:
            continue
        public = _public_query(raw, dataset, video)
        qid = int(public["query_id"])
        if qid in found and found[qid]["sentence"] != public["sentence"]:
            raise AssertionError(f"query sentence drift: {dataset}|{video}|{qid}")
        found[qid] = public
    if not found:
        raise AssertionError(f"no legal-dev query metadata: {dataset}|{video}")
    return [found[qid] for qid in sorted(found)]


def _native_frames(video: str) -> tuple[list[int], list[int], Path]:
    path, blob = load_l69_bank(video)
    tensors = blob["tensors"]
    frame_ids = [int(value) for value in tensors["frame_ids"].tolist()]
    frame_ptr = [int(value) for value in tensors["frame_ptr"].tolist()]
    if len(frame_ptr) != len(frame_ids) + 1 or frame_ids != sorted(set(frame_ids)):
        raise AssertionError(f"native frame-pointer contract drift: {video}")
    counts = [right - left for left, right in zip(frame_ptr, frame_ptr[1:])]
    if any(value < 0 for value in counts):
        raise AssertionError(f"negative native candidate count: {video}")
    del blob
    return frame_ids, counts, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty request output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([str(sys.executable), *sys.argv])
    started = time.perf_counter()
    records: dict[str, dict[str, Any]] = {}
    video_summary: list[dict[str, Any]] = []
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 request-manifest cwd: {Path.cwd()}")
        manifest_sha = check_manifest()
        for dataset, video in _dev_videos():
            queries = _dev_queries(dataset, video)
            frames, counts, bank_path = _native_frames(video)
            for frame_id, candidate_count in zip(frames, counts):
                for query in queries:
                    value = {
                        "dataset": dataset, "video": video, "query_id": int(query["query_id"]),
                        "frame_id": int(frame_id), "sentence": str(query["sentence"]),
                        "candidate_count": int(candidate_count), "bank_path": str(bank_path.resolve()),
                        "unit_key": _key(dataset, video, int(query["query_id"]), frame_id),
                        "split": "legal_dev",
                    }
                    if FORBIDDEN_FIELDS.intersection(value):
                        raise AssertionError(f"forbidden eval request field: {value['unit_key']}")
                    if value["unit_key"] in records:
                        raise AssertionError(f"duplicate eval request key: {value['unit_key']}")
                    records[value["unit_key"]] = value
            video_summary.append({"dataset": dataset, "video": video, "query_count": len(queries),
                                  "frame_count": len(frames), "query_frame_pairs": len(queries) * len(frames),
                                  "candidate_count_min": min(counts), "candidate_count_max": max(counts),
                                  "bank_path": str(bank_path.resolve())})

        fixed = load_fixed_l62_key_order()
        fixed_keys: list[str] = []
        for raw in fixed:
            key = str(raw["unit_key"])
            value = {
                "dataset": str(raw["dataset"]), "video": str(raw["video"]),
                "query_id": int(raw["query_id"]), "frame_id": int(raw["frame_id"]),
                "sentence": str(raw["sentence"]), "split": str(raw["split"]),
                "unit_key": key,
            }
            if key in records:
                existing = records[key]
                if existing["sentence"] != value["sentence"]:
                    raise AssertionError(f"fixed sentence drift: {key}")
                fixed_keys.append(key)
                continue
            path, blob = load_l69_bank(value["video"])
            tensors = blob["tensors"]
            frame_ids = [int(x) for x in tensors["frame_ids"].tolist()]
            pointers = [int(x) for x in tensors["frame_ptr"].tolist()]
            if value["frame_id"] not in frame_ids:
                raise AssertionError(f"fixed frame missing from L69: {key}")
            pos = frame_ids.index(value["frame_id"])
            value["candidate_count"] = int(pointers[pos + 1] - pointers[pos])
            value["bank_path"] = str(path.resolve())
            if FORBIDDEN_FIELDS.intersection(value):
                raise AssertionError(f"forbidden fixed request field: {key}")
            records[key] = value
            fixed_keys.append(key)
            del blob

        ordered = sorted(records.values(), key=lambda item: (DATASETS.index(str(item["dataset"])),
                                                              str(item["video"]), int(item["frame_id"]),
                                                              int(item["query_id"]), str(item["unit_key"])))
        payload = {
            "format": "locatemot-r1-eval-request-manifest-v1", "status": "complete",
            "command": command, "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
            "scope": "legal_dev_plus_fixed_40", "datasets": list(DATASETS),
            "legal_dev_videos": [list(value) for value in _dev_videos()],
            "legal_dev_video_summary": video_summary,
            "fixed_order_count": len(fixed), "fixed_order_keys": [str(value["unit_key"]) for value in fixed],
            "fixed_keys_present": len(fixed_keys), "request_count": len(ordered),
            "unique_query_records": {dataset: {str(item["unit_key"]): item for item in ordered if item["dataset"] == dataset}
                                      for dataset in DATASETS},
            "labels_in_manifest": False, "labels_read_for_feature_construction": False,
            "forbidden_fields_checked": sorted(FORBIDDEN_FIELDS),
            "manifest_sha256": manifest_sha,
            "inputs": {"manifest": file_meta(MANIFEST),
                       "l82_split": file_meta(L82_SPLIT),
                       "dev_query_files": {dataset: file_meta(DEV_ROOTS[dataset] / "queries.jsonl") for dataset in DATASETS}},
            "outputs": {"request_manifest": str((out / "request_manifest.json").resolve())},
            "failure_root_cause": None,
            "next_action": "build label-free R1 aligned cache before attaching any legal-dev/fixed labels",
            "training_run": False, "hota_trackeval_run": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "candidate_deletion": False, "candidate_truncation": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(out / "request_manifest.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": payload["format"], "status": "complete", "command": command,
                                          "request_count": len(ordered), "fixed_order_count": len(fixed),
                                          "fixed_keys_present": len(fixed_keys), "manifest_sha256": manifest_sha,
                                          "failure_root_cause": None, "next_action": payload["next_action"],
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False})
        print(json.dumps({"status": "complete", "request_count": len(ordered),
                          "legal_dev_pairs": sum(int(x["query_frame_pairs"]) for x in video_summary),
                          "fixed_order_count": len(fixed), "fixed_keys_present": len(fixed_keys)}, sort_keys=True))
        return 0
    except Exception as exc:
        trace = __import__("traceback").format_exc()
        (out / "INCOMPLETE.md").write_text("# R1 eval request manifest incomplete\n\n```text\n" + trace + "```\n", encoding="utf-8")
        write_json(out / "status.json", {"format": "locatemot-r1-eval-request-manifest-v1", "status": "incomplete",
                                          "command": command, "cwd": str(Path.cwd().resolve()), "thread": THREAD,
                                          "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                          "next_action": "repair only the first request-contract error", "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
