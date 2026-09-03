#!/usr/bin/env python3
"""GT-privileged, existing-track semantic oracle for L86.

This tool does not run a detector or model and does not create boxes or track
IDs.  For each internal validation query/GT target it selects one existing L69
track using matched-frame count, IoU sum, and track-id tie breaks, then emits
only that track's best existing positive observation on target frames.  The
result is an oracle ceiling, never a learned semantic result.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.rmot.l80_data import L80BankStore  # noqa: E402
from locatemot.rmot.l85_fullvideo_bank import INTERNAL_V1, INTERNAL_V2, MANIFEST, EXPECTED_MANIFEST_SHA, file_meta, bank_path, sha256_file  # noqa: E402
from locatemot.rmot.l85_runtime import load_validation_key_rows  # noqa: E402
from locatemot.rmot.l49_data import load_l49_queries  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
VIDEOS = {"refer_kitti_v1": tuple(INTERNAL_V1), "refer_kitti_v2": tuple(INTERNAL_V2)}
ORACLE_NAME = "GT_PRIVILEGED_TRACK_CONSISTENT_ORACLE"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def iou_xyxy(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0])); y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2])); y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return inter / max(1e-12, area_left + area_right - inter)


def load_record(video: str) -> tuple[Path, dict[str, Any]]:
    candidates = [
        ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl",
        ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl",
    ]
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"internal record missing for {video}: {candidates}")
    # Compatibility alias only; the source pickle is never rewritten.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with path.open("rb") as handle:
        return path, pickle.load(handle)


def query_rows() -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows = load_validation_key_rows()
    result: dict[str, dict[str, list[dict[str, Any]]]] = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        for video in VIDEOS[dataset]:
            found: dict[int, dict[str, Any]] = {}
            for row in rows:
                if str(row["dataset"]) != dataset or str(row["video"]) != video:
                    continue
                qid = int(row["query_id"])
                sentence = str(row["sentence"])
                old = found.get(qid)
                if old is not None and old["sentence"] != sentence:
                    raise AssertionError(f"query sentence drift {dataset}|{video}|{qid}")
                found.setdefault(qid, {"dataset": dataset, "video": video, "query_id": qid, "sentence": sentence})
            if not found:
                raise AssertionError(f"no internal validation queries {dataset}|{video}")
            result[dataset][video] = [found[key] for key in sorted(found)]
    return result


def frame_gt(record: dict[str, Any], frame: int, target_id: str) -> list[float] | None:
    frames = {int(value["frame"]): value for value in record["frames"]}
    frame_record = frames.get(int(frame), {})
    boxes = frame_record.get("gt_boxes", {})
    value = boxes.get(str(target_id))
    if value is None and str(target_id).isdigit():
        value = boxes.get(int(str(target_id)))
    if value is None:
        return None
    box = [float(x) for x in value]
    if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise AssertionError(f"invalid GT box {frame}|{target_id}")
    return box


def load_sidecar(path: Path) -> list[str | None]:
    payload = json.loads(path.with_suffix(".labels.json").read_text())
    values = payload.get("candidate_gt")
    if not isinstance(values, list):
        raise AssertionError(f"missing candidate_gt sidecar {path}")
    return [None if value is None else str(value) for value in values]


def prepare_dirs(out: Path, dataset: str) -> dict[str, Path]:
    root = out / dataset
    values = {
        "root": root,
        "gt": root / "gt",
        "trackers": root / "trackers",
        "tracker_data": root / "trackers" / "l86_oracle" / "data",
        "seqmap": root / "seqmap.txt",
    }
    for key, path in values.items():
        if key != "seqmap":
            path.mkdir(parents=True, exist_ok=True)
    return values


def sequence_id(video: str, query_id: int) -> str:
    return f"{video}__q{int(query_id):05d}"


def write_gt(dataset: str, video: str, query: dict[str, Any], record: dict[str, Any], bank_frames: list[int], path: Path) -> dict[str, Any]:
    entry_by_qid = {int(row["query_id"]): row for row in load_l49_queries(dataset)}
    entry = entry_by_qid.get(int(query["query_id"]))
    if entry is None or str(entry.get("sentence")) != str(query["sentence"]):
        raise AssertionError(f"query metadata mismatch {dataset}|{video}|{query['query_id']}")
    target_map = entry.get("target", {})
    width, height = [int(x) for x in record.get("image_size", [0, 0])]
    if width <= 0 or height <= 0:
        raise AssertionError(f"invalid image size {video}")
    lines: list[str] = []
    rows = 0
    present_frames = 0
    for frame in bank_frames:
        targets = target_map.get(int(frame), target_map.get(str(frame), set()))
        targets = sorted(str(x) for x in (targets or set()))
        present_frames += int(bool(targets))
        for target in targets:
            box = frame_gt(record, frame, target)
            if box is None:
                continue
            lines.append(f"{frame + 1},{int(target)},{box[0]:.6f},{box[1]:.6f},{box[2]-box[0]:.6f},{box[3]-box[1]:.6f},1,1,1\n")
            rows += 1
    seq = sequence_id(video, int(query["query_id"]))
    target_dir = path / seq
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "gt.txt").write_text("".join(lines))
    (target_dir / "seqinfo.ini").write_text(
        "[Sequence]\n" f"name={seq}\n" "imDir=img1\n" "frameRate=10\n"
        f"seqLength={len(bank_frames)}\n" f"imWidth={width}\n" f"imHeight={height}\n" "imExt=.png\n"
    )
    return {"sequence": seq, "gt_rows": rows, "target_present_frames": present_frames}


def oracle_video(dataset: str, video: str, queries: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    record_path, record = load_record(video)
    frames_by_id = {int(value["frame"]): value for value in record["frames"]}
    store = L80BankStore(max_history=8)
    store._store.load_video(video)
    tensors = store._store.tensors
    frame_ids = [int(value) for value in tensors["frame_ids"].tolist()]
    frame_ptr = [int(value) for value in tensors["frame_ptr"].tolist()]
    if set(frame_ids) != set(frames_by_id):
        raise AssertionError(f"bank/record frame mismatch {dataset}|{video}")
    bank_file = Path(store.bank_path)
    candidate_gt = load_sidecar(bank_file)
    if len(candidate_gt) != int(tensors["track_id"].numel()):
        raise AssertionError(f"sidecar row mismatch {video}")
    # Frame-local target evidence is built once, then each query uses only its
    # expression's target map.  No candidate rows are filtered before this
    # explicit GT-privileged oracle operation.
    query_audits = []
    for query in queries:
        entry = next((row for row in load_l49_queries(dataset) if int(row["query_id"]) == int(query["query_id"])), None)
        if entry is None:
            raise AssertionError(f"query not in L49 metadata {dataset}|{video}|{query['query_id']}")
        target_map = entry.get("target", {})
        stats: dict[str, dict[int, dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {"frames": set(), "iou_sum": 0.0}))
        positives_per_frame: dict[int, dict[str, list[tuple[float, int, int, list[float]]]]] = defaultdict(lambda: defaultdict(list))
        for index, frame in enumerate(frame_ids):
            start, end = frame_ptr[index], frame_ptr[index + 1]
            frame_targets = target_map.get(int(frame), target_map.get(str(frame), set())) or set()
            if not frame_targets:
                continue
            boxes = tensors["box"][start:end].float().tolist()
            tracks = [int(x) for x in tensors["track_id"][start:end].tolist()]
            for target_raw in sorted(str(x) for x in frame_targets):
                gt_box = frame_gt(record, frame, target_raw)
                if gt_box is None:
                    continue
                for local, (box, track_id) in enumerate(zip(boxes, tracks)):
                    offset = start + local
                    if candidate_gt[offset] != target_raw:
                        continue
                    overlap = iou_xyxy(box, gt_box)
                    positives_per_frame[frame][target_raw].append((overlap, offset, track_id, box))
                    stats[target_raw][track_id]["frames"].add(int(frame))
                    stats[target_raw][track_id]["iou_sum"] += float(overlap)
        chosen: dict[str, int | None] = {}
        for target, values in sorted(stats.items()):
            if not values:
                chosen[target] = None
                continue
            chosen[target] = min(values, key=lambda pair: (-len(values[pair]["frames"]), -float(values[pair]["iou_sum"]), int(pair)))
        query_path = prepare_dirs(out, dataset)
        tracker_lines: list[str] = []
        emitted = 0
        collisions = 0
        for frame in frame_ids:
            for target, track_id in sorted(chosen.items()):
                if track_id is None:
                    continue
                values = [item for item in positives_per_frame[frame].get(target, []) if int(item[2]) == int(track_id)]
                if not values:
                    continue
                overlap, offset, _, box = max(values, key=lambda item: (float(item[0]), -int(item[1])))
                if any(int(row.split(",")[1]) == int(track_id) and int(row.split(",")[0]) == int(frame + 1) for row in tracker_lines):
                    collisions += 1
                tracker_lines.append(f"{frame + 1},{track_id},{box[0]:.6f},{box[1]:.6f},{box[2]-box[0]:.6f},{box[3]-box[1]:.6f},1,1,1\n")
                emitted += 1
        seq = sequence_id(video, int(query["query_id"]))
        prediction_path = query_path["tracker_data"] / f"{seq}.txt"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text("".join(tracker_lines))
        gt_audit = write_gt(dataset, video, query, record, frame_ids, query_path["gt"])
        query_audits.append({
            "query_id": int(query["query_id"]), "sequence": seq, "target_count": len(chosen),
            "target_track_choices": {target: None if track is None else int(track) for target, track in chosen.items()},
            "emitted_rows": emitted, "same_track_frame_collisions": collisions,
            "candidate_rows_only": True, "boxes_modified": False, "track_ids_created": False,
            "gt": gt_audit,
        })
    seqs = [item["sequence"] for item in query_audits]
    query_path["seqmap"].write_text("name\n" + "\n".join(seqs) + "\n")
    store._store._bank = None
    store._store._text_cache = None
    del store
    gc.collect()
    return {
        "dataset": dataset, "video": video, "sequence_count": len(seqs), "sequences": seqs,
        "queries": query_audits, "record": file_meta(record_path), "bank": file_meta(bank_file),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L86 oracle output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        queries = query_rows()
        per_video = []
        for dataset in DATASETS:
            for video in VIDEOS[dataset]:
                per_video.append(oracle_video(dataset, video, queries[dataset][video], out))
        summary = {
            "format": "locatemot-l86-semantic-oracle-inference-v1", "status": "complete",
            "evidence_type": ORACLE_NAME, "scope": "internal full-video validation",
            "full_video": True, "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
            "oracle_rule": "existing track with max matched target frames; tie max IoU sum; tie min track ID; best IoU duplicate row per target/frame",
            "candidate_rows_only": True, "boxes_modified": False, "track_ids_created": False,
            "videos": per_video, "sequence_count": int(sum(item["sequence_count"] for item in per_video)),
            "manifest_sha256": sha256_file(MANIFEST), "bank_source": "immutable L69 budget-40 native frame pointers",
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "wall_seconds": time.perf_counter() - started, "failure_root_cause": None,
            "next_action": "run the independent L86 TrackEval wrapper on this frozen oracle output",
        }
        write_json(out / "summary.json", summary); write_json(out / "provenance.json", summary); write_json(out / "status.json", summary)
        print(json.dumps(summary, indent=2, default=str), flush=True)
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L86 semantic oracle — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l86-semantic-oracle-inference-v1", "status": "incomplete", "command": command, "cwd": str(ROOT), "luna_thread": THREAD, "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
