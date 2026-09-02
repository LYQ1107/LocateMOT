#!/usr/bin/env python3
"""Frozen L85 full-video inference for legal internal validation videos.

The model and emission rule are frozen by the internal-dev selection artifact.
This tool scores every L69 candidate row for every validation expression/frame
and writes only the selected tracker rows required by the local TrackEval
adapter.  It never uses labels while constructing or scoring the model input;
GT files are materialized only after all prediction strategy inputs are frozen
and predictions have been emitted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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

from locatemot.models.l85_full_rmot import L85Config, L85FullRMOT  # noqa: E402
from locatemot.rmot.l49_data import load_l49_queries  # noqa: E402
from locatemot.rmot.l85_fullvideo_bank import (  # noqa: E402
    EXPECTED_MANIFEST_SHA,
    INTERNAL_V1,
    INTERNAL_V2,
    MANIFEST,
    bank_path,
    file_meta,
    sha256_file,
)
from locatemot.rmot.l85_runtime import capture_group_z1_batched, load_validation_key_rows  # noqa: E402
from tools.l85_calibrate_dev import score_group_reuse_first  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
SELECTED_DATASETS = {"refer_kitti_v1", "refer_kitti_v2"}
VAL_VIDEOS = {"refer_kitti_v1": tuple(INTERNAL_V1), "refer_kitti_v2": tuple(INTERNAL_V2)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path, device: torch.device) -> tuple[L85FullRMOT, dict[str, Any]]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    model = L85FullRMOT(L85Config(**package["model_config"]))
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict L85 checkpoint load failed: {loaded}")
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "path": str(path.resolve()),
        "sha256": digest_file(path),
        "epoch": int(package.get("epoch", 0)),
        "step": int(package.get("step", 0)),
        "model_config": package["model_config"],
        "strict_reload": True,
    }


def load_record(video: str) -> tuple[Path, dict[str, Any]]:
    """Load only the legal internal train-pool record with pickle aliases."""
    candidates = [
        ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl",
        ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl",
    ]
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no internal record for {video}: {candidates}")
    # Older records refer to NumPy's private namespace.  This is an in-memory
    # interpreter compatibility alias only; the source pickle is untouched.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with path.open("rb") as handle:
        record = pickle.load(handle)
    return path, record


def key_rows_for_internal_validation() -> list[dict[str, Any]]:
    rows = load_validation_key_rows()
    allowed = {(dataset, video) for dataset, videos in VAL_VIDEOS.items() for video in videos}
    result = [row for row in rows if (str(row["dataset"]), str(row["video"])) in allowed]
    if not result:
        raise AssertionError("no internal validation key rows")
    return result


def query_rows(rows: list[dict[str, Any]], dataset: str, video: str, limit: int = 0) -> list[dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row["dataset"]) != dataset or str(row["video"]) != video:
            continue
        query_id = int(row["query_id"])
        current = found.get(query_id)
        if current is not None and str(current["sentence"]) != str(row["sentence"]):
            raise AssertionError(f"query sentence drift: {dataset}|{video}|{query_id}")
        found.setdefault(query_id, {
            "unit_key": f"{dataset}|{video}|{query_id}|0",
            "dataset": dataset, "video": video, "query_id": query_id,
            "frame_id": 0, "sentence": str(row["sentence"]),
            "expression": str(row.get("expression", row["sentence"])),
        })
    result = [found[key] for key in sorted(found)]
    if limit:
        result = result[: int(limit)]
    if not result:
        raise AssertionError(f"no validation queries for {dataset}|{video}")
    return result


def frame_groups(dataset: str, video: str, queries: list[dict[str, Any]], store: L80BankStore,
                 max_frames: int = 0) -> list[dict[str, Any]]:
    # L80/L79 owns the native frame-pointer index.  Reading frame_ids here is
    # label-free and avoids using any legacy L49 begin/end range.
    store._store.load_video(video)
    frame_ids = [int(value) for value in store._store.tensors["frame_ids"].tolist()]
    if max_frames:
        frame_ids = frame_ids[: int(max_frames)]
    result = []
    for frame in frame_ids:
        result.append({
            "group_key": f"{dataset}|{video}|{frame}",
            "dataset": dataset, "video": video, "frame_id": frame,
            "queries": [dict(row, unit_key=f"{dataset}|{video}|{int(row['query_id'])}|{frame}", frame_id=frame)
                        for row in queries],
        })
    return result


def sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0:
        exp_value = float(np.exp(-value))
        return 1.0 / (1.0 + exp_value)
    exp_value = float(np.exp(value))
    return exp_value / (1.0 + exp_value)


def sequence_id(video: str, query_id: int) -> str:
    return f"{video}__q{int(query_id):05d}"


def prepare_trackeval_dirs(out: Path, dataset: str) -> dict[str, Path]:
    root = out / dataset
    paths = {
        "root": root,
        "gt": root / "gt",
        "trackers": root / "trackers",
        "tracker_data": root / "trackers" / "l85" / "data",
        "predictions": root / "predictions",
        "seqmap": root / "seqmap.txt",
    }
    for path in paths.values():
        if path.suffix != ".txt":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def write_prediction(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for frame, track, x, y, width, height, confidence in rows:
            handle.write(f"{int(frame)},{int(track)},{float(x):.6f},{float(y):.6f},"
                         f"{float(width):.6f},{float(height):.6f},{float(confidence):.8f},1,1,1\n")


def materialize_gt(dataset: str, video_queries: dict[str, list[dict[str, Any]]], paths: dict[str, Path],
                   frame_limits: dict[str, int] | None = None) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach legal internal labels only after all predictions are written."""
    metadata = {int(row["query_id"]): row for row in load_l49_queries(dataset)
                if str(row["video"]) in set(VAL_VIDEOS[dataset])}
    seqs: list[str] = []
    query_audits: list[dict[str, Any]] = []
    record_audits: list[dict[str, Any]] = []
    for video in VAL_VIDEOS[dataset]:
        if video not in video_queries:
            continue
        record_path, record = load_record(video)
        frames = {int(value["frame"]): value for value in record["frames"]}
        bank_store = L80BankStore(max_history=8)
        bank_store._store.load_video(video)
        bank_frames = [int(value) for value in bank_store._store.tensors["frame_ids"].tolist()]
        if set(bank_frames) != set(frames):
            raise AssertionError(f"bank/GT frame set drift for {dataset}|{video}")
        record_audits.append({"video": video, "record": file_meta(record_path),
                              "frame_count": len(frames), "bank_frame_count": len(bank_frames)})
        for query in video_queries[video]:
            query_id = int(query["query_id"])
            if query_id not in metadata:
                raise KeyError(f"validation query metadata missing: {dataset}|{video}|{query_id}")
            entry = metadata[query_id]
            if str(entry["sentence"]) != str(query["sentence"]):
                raise AssertionError(f"sentence mismatch for {dataset}|{video}|{query_id}")
            target = entry.get("target", {})
            seq = sequence_id(video, query_id)
            seqs.append(seq)
            gt_dir = paths["gt"] / seq
            gt_dir.mkdir(parents=True, exist_ok=True)
            width, height = [int(value) for value in record.get("image_size", [0, 0])]
            if width <= 0 or height <= 0:
                width, height = [int(value) for value in bank_store._store.bank["metadata"]["image_size"]]
            gt_lines: list[str] = []
            gt_rows = 0
            target_frames = 0
            frame_ids = bank_frames if not frame_limits else bank_frames[: int(frame_limits.get(video, len(bank_frames)))]
            for frame in frame_ids:
                ids = target.get(int(frame), set())
                if ids:
                    target_frames += 1
                fr = frames[int(frame)]
                boxes = fr.get("gt_boxes", {})
                for raw_id in sorted(str(value) for value in ids):
                    box = boxes.get(raw_id, boxes.get(int(raw_id)) if raw_id.isdigit() else None)
                    if box is None:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box]
                    if not (np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1):
                        raise AssertionError(f"invalid GT box {dataset}|{video}|{query_id}|{frame}|{raw_id}")
                    gt_lines.append(f"{int(frame) + 1},{int(raw_id)},{x1:.6f},{y1:.6f},"
                                    f"{x2 - x1:.6f},{y2 - y1:.6f},1,1,1\n")
                    gt_rows += 1
            (gt_dir / "gt.txt").write_text("".join(gt_lines))
            (gt_dir / "seqinfo.ini").write_text(
                "[Sequence]\n" f"name={seq}\n" "imDir=img1\n" "frameRate=10\n"
                f"seqLength={len(frame_ids)}\n" f"imWidth={width}\n" f"imHeight={height}\n" "imExt=.png\n")
            query_audits.append({"video": video, "query_id": query_id, "sequence": seq,
                                 "gt_rows": gt_rows, "target_present_frames": target_frames,
                                 "labels_attached_after_prediction": True,
                                 "label_source": str(entry["label_source"])})
    paths["seqmap"].write_text("name\n" + "\n".join(seqs) + "\n")
    return seqs, query_audits, record_audits


def run(args: argparse.Namespace) -> int:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L85 full-video output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        manifest_sha = sha256_file(MANIFEST)
        if manifest_sha != EXPECTED_MANIFEST_SHA:
            raise AssertionError(f"fixed manifest SHA drift: {manifest_sha}")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        checkpoint = Path(args.checkpoint).resolve()
        selection_path = Path(args.selection).resolve()
        selection = json.loads(selection_path.read_text())
        selected = selection["selected"]
        selected_rule = dict(selected["rule_fit"])
        checkpoint_info = selected["checkpoint_info"]
        if str(checkpoint) != str(Path(checkpoint_info["path"]).resolve()):
            raise AssertionError("inference checkpoint is not the frozen dev-selected checkpoint")
        model, loaded_info = load_checkpoint(checkpoint, device)
        key_rows = key_rows_for_internal_validation()
        datasets = [args.dataset] if args.dataset != "all" else ["refer_kitti_v1", "refer_kitti_v2"]
        if any(dataset not in SELECTED_DATASETS for dataset in datasets):
            raise ValueError(datasets)
        store = L80BankStore(max_history=8)
        from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime
        runtime = GroundingCandidateReferenceRuntime(device)
        video_queries: dict[str, dict[str, list[dict[str, Any]]]] = {}
        prediction_audits: list[dict[str, Any]] = []
        try:
            for dataset in datasets:
                video_queries[dataset] = {}
                for video in VAL_VIDEOS[dataset]:
                    queries = query_rows(key_rows, dataset, video, args.max_queries)
                    paths = prepare_trackeval_dirs(out, dataset)
                    groups = frame_groups(dataset, video, queries, store, args.max_frames)
                    video_queries[dataset][video] = queries
                    for frame_index, group in enumerate(groups):
                        item = capture_group_z1_batched(
                            group, device, runtime=runtime, bank_store=store,
                            query_batch_size=args.query_batch_size)
                        records = score_group_reuse_first(item, group, store, model, device)
                        if len(records) != len(queries):
                            raise AssertionError(f"query record drift {group['group_key']}")
                        boxes = np.asarray(item["boxes"], dtype=np.float32)
                        if boxes.shape != (int(item["candidate_count"]), 4):
                            raise AssertionError(f"box shape drift {group['group_key']}")
                        for record in records:
                            values = np.asarray(record["score"], dtype=np.float64)
                            if values.shape != (int(record["candidate_count"]),) or not np.isfinite(values).all():
                                raise AssertionError(f"score shape/finite drift {record['unit_key']}")
                            gate = (float(record["presence"]) >= float(selected_rule["presence_threshold"]) and
                                    float(record["presence"]) - float(record["null_logit"]) >= float(selected_rule["null_margin"]))
                            selected_rows = np.flatnonzero((values >= float(selected_rule["candidate_threshold"])) & bool(gate))
                            track_ids = [int(value) for value in record["track_ids"]]
                            if len(track_ids) != len(set(track_ids)):
                                raise AssertionError(f"duplicate tracker IDs in frame {record['unit_key']}")
                            prediction_rows = []
                            for local in selected_rows.tolist():
                                x1, y1, x2, y2 = [float(value) for value in boxes[local]]
                                prediction_rows.append([int(group["frame_id"]) + 1, track_ids[local], x1, y1,
                                                        x2 - x1, y2 - y1, sigmoid(values[local])])
                            seq = sequence_id(video, int(record["query_id"]))
                            prediction_path = paths["tracker_data"] / f"{seq}.txt"
                            # Full-video files are accumulated frame by frame;
                            # a frame is appended exactly once in native order.
                            with prediction_path.open("a") as handle:
                                for row in prediction_rows:
                                    handle.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},"
                                                 f"{row[4]:.6f},{row[5]:.6f},{row[6]:.8f},1,1,1\n")
                            prediction_audits.append({
                                "dataset": dataset, "video": video, "query_id": int(record["query_id"]),
                                "frame_id": int(group["frame_id"]), "unit_key": str(record["unit_key"]),
                                "candidate_rows_scored": int(record["candidate_count"]),
                                "selected_rows": int(len(prediction_rows)), "presence_gate": bool(gate),
                                "candidate_rows_retained": True, "candidate_deletion": False,
                                "candidate_truncation": False,
                            })
                        del item, records
                        if frame_index % 10 == 0:
                            gc.collect()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                    print(f"[l85-infer] {dataset} video={video} frames={len(groups)} queries={len(queries)} "
                          f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
            # Only now attach the expression/GT maps to construct TrackEval
            # ground truth.  The model, checkpoint, and rule were frozen above.
            eval_summary: dict[str, Any] = {}
            for dataset in datasets:
                paths = prepare_trackeval_dirs(out, dataset)
                frame_limits = {video: args.max_frames for video in VAL_VIDEOS[dataset]} if args.max_frames else None
                seqs, query_audits, record_audits = materialize_gt(dataset, video_queries[dataset], paths, frame_limits)
                eval_summary[dataset] = {"trackeval_root": str(paths["root"]), "seqmap": str(paths["seqmap"]),
                                         "sequences": seqs, "query_gt_audits": query_audits,
                                         "record_audits": record_audits, "labels_attached_after_predictions": True}
            full = not args.max_frames and not args.max_queries
            summary = {
                "format": "locatemot-l85-fullvideo-inference-v1", "status": "complete",
                "scope": "internal full-video validation" if full else "targeted internal regression",
                "full_video": full, "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                "seed": SEED, "datasets": datasets, "selected_checkpoint": loaded_info,
                "selection_source": str(selection_path), "selection_sha256": digest_file(selection_path),
                "emission_rule": selected_rule, "prediction_strategy_frozen_before_gt": True,
                "prediction_record_count": len(prediction_audits),
                "prediction_audits_path": str(out / "prediction_audits.jsonl"),
                "eval_summary": eval_summary, "candidate_bank": "immutable L69 budget-40; native frame pointers",
                "query_batch_size": int(args.query_batch_size),
                "all_candidate_rows_scored": True, "candidate_deletion": False, "candidate_truncation": False,
                "model_selection_used_validation": False, "screening_gt_used": False,
                "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                "hota_trackeval_run": False, "trackeval_run": False,
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
                "manifest_sha256": manifest_sha, "bank_sources": [file_meta(bank_path(video))
                    for dataset in datasets for video in VAL_VIDEOS[dataset]],
                "wall_seconds": time.perf_counter() - started,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "failure_root_cause": None,
                "next_action": "run local TrackEval on the frozen internal validation predictions",
            }
            with (out / "prediction_audits.jsonl").open("w") as handle:
                for row in prediction_audits:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            write_json(out / "summary.json", summary)
            write_json(out / "provenance.json", summary)
            write_json(out / "status.json", summary)
            return 0
        finally:
            runtime.close()
            del runtime, store
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L85 full-video inference — INCOMPLETE\n\n" + trace + "\n")
        write_json(out / "status.json", {
            "format": "locatemot-l85-fullvideo-inference-v1", "status": "incomplete", "command": command,
            "cwd": str(ROOT), "luna_thread": THREAD, "failure_root_cause": "first traceback in INCOMPLETE.md",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        raise
    finally:
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--query-batch-size", type=int, default=8)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
