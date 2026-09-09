#!/usr/bin/env python3
"""Emit complete legal dev/internal R0 predictions before attaching GT.

The model is run over every native L69 candidate row.  Safe target artifacts
and per-video GT-box records are opened only after the prediction files for a
candidate have been written, so this tool cannot accidentally use labels to
construct model inputs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.r0_track_grounding import R0Config, R0TrackGroundingHead  # noqa: E402
from locatemot.rmot.r0_dense_data import R0BankStore, native_frame_ids  # noqa: E402
from locatemot.rmot.r0_safe_target_source import load_safe_query_records  # noqa: E402
from tools.r0_common import (  # noqa: E402
    DEFAULT_LANGUAGE_ROOTS,
    EvalIndex,
    MergedLanguageCache,
    R0RuntimeData,
    VisualCacheIndex,
    check_manifest,
    file_meta,
    model_forward,
    sha256_file,
    write_json,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
INTERNAL_VIDEOS = {"refer_kitti_v1": ("0004", "0018"), "refer_kitti_v2": ("0016", "0017", "0020")}


def checkpoint_model(path: Path, device: torch.device) -> tuple[R0TrackGroundingHead, dict[str, Any]]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != "locatemot-r0-tcgh-checkpoint-v2":
        raise AssertionError(f"invalid R0 inference checkpoint: {path}")
    if package.get("driver") != "grouped_ddp_v2" or package.get("grouped_query_training") is not True:
        raise AssertionError(f"R0 inference checkpoint is not grouped training: {path}")
    if package.get("detector_state_included") or package.get("tracker_state_included"):
        raise AssertionError(f"R0 inference checkpoint contains forbidden state: {path}")
    model = R0TrackGroundingHead(R0Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"R0 inference strict reload failed: {result}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    info = {"path": str(path.resolve()), "sha256": sha256_file(path),
            "dataset": str(package["dataset"]), "epoch": int(package["epoch"]),
            "global_step": int(package["global_step"]), "model_config": package["model_config"],
            "format": package["format"], "driver": package["driver"],
            "grouped_query_training": True,
            "primary_selection_eligible": bool(package.get("primary_selection_eligible", False)),
            "strict_reload": True}
    return model, info


def candidates_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.scope == "dev":
        if args.preview_checkpoint is not None:
            checkpoint = args.preview_checkpoint.resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            package = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if package.get("format") != "locatemot-r0-tcgh-checkpoint-v2":
                raise AssertionError(f"invalid R0 preview checkpoint: {checkpoint}")
            return [{"role": f"preview_epoch{int(package['epoch']):03d}",
                     "checkpoint_info": {"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                                         "dataset": str(package["dataset"]), "epoch": int(package["epoch"]),
                                         "global_step": int(package["global_step"]),
                                         "model_config": package["model_config"], "format": package["format"],
                                         "driver": package.get("driver"),
                                         "grouped_query_training": bool(package.get("grouped_query_training", False)),
                                         "primary_selection_eligible": bool(package.get("primary_selection_eligible", False)),
                                         "strict_reload": True},
                     "rule": {"name": "zero", "membership_threshold": 0.0,
                              "presence_threshold": 0.0, "null_logit": 0.0,
                              "description": "membership_logit>=0 and coverage_presence_logit>=0"},
                     "source": {"preview_checkpoint": str(checkpoint)}}]
        payload = json.loads(args.shortlist.resolve().read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or not payload.get("shortlist"):
            raise AssertionError("R0 dev shortlist is incomplete")
        result = []
        for value in payload["shortlist"]:
            result.append({"role": str(value["role"]), "checkpoint_info": value["checkpoint_info"],
                           "rule": value["rule"], "dev_metrics": value.get("metrics"), "source": value})
        return result
    payload = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not isinstance(payload.get("selected"), dict):
        raise AssertionError("R0 internal selection is incomplete")
    value = payload["selected"]
    return [{"role": "frozen_internal_selection", "checkpoint_info": value["checkpoint_info"],
             "rule": value["rule"], "dev_metrics": value.get("dev_metrics"), "source": value}]


def safe_internal_record(video: str) -> tuple[Path, dict[str, Any]]:
    candidates = [
        ASSET_ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl",
        ASSET_ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl",
    ]
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"legal per-video GT record missing for {video}: {candidates}")
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("frames"), list):
        raise AssertionError(f"invalid legal per-video GT record: {path}")
    return path, value


def sequence_id(video: str, query_id: int) -> str:
    return f"{str(video)}__q{int(query_id):05d}"


def prepare_paths(root: Path, dataset: str) -> dict[str, Path]:
    base = root / dataset
    paths = {"root": base, "gt": base / "gt", "trackers": base / "trackers",
             "tracker_data": base / "trackers" / "r0" / "data", "seqmap": base / "seqmap.txt"}
    for key, value in paths.items():
        if key != "seqmap":
            value.mkdir(parents=True, exist_ok=True)
    return paths


def materialize_gt(dataset: str, queries_by_video: dict[str, list[dict[str, Any]]],
                   safe_targets: Path, paths: dict[str, Path], frame_map: dict[str, list[int]]) -> dict[str, Any]:
    """Materialize legal GT after the candidate predictions are complete."""
    records = load_safe_query_records(safe_targets.resolve())
    by_key = {(record.dataset, record.video, int(record.query_id)): record for record in records}
    sequences: list[str] = []
    query_audits: list[dict[str, Any]] = []
    record_audits: list[dict[str, Any]] = []
    for video in sorted(queries_by_video):
        record_path, record = safe_internal_record(video)
        frame_records = {int(value["frame"]): value for value in record["frames"]}
        frames = [int(value) for value in frame_map[video]]
        if set(frame_records) != set(frames):
            raise AssertionError(f"legal GT/native frame mismatch: {dataset}|{video}")
        record_audits.append({"video": video, "record": file_meta(record_path),
                              "frame_count": len(frame_records), "native_frame_count": len(frames)})
        image_size = [int(value) for value in record.get("image_size", [0, 0])]
        if len(image_size) < 2 or image_size[0] <= 0 or image_size[1] <= 0:
            raise AssertionError(f"invalid legal GT image size: {record_path}")
        for query in queries_by_video[video]:
            qid = int(query["query_id"])
            safe = by_key.get((dataset, video, qid))
            if safe is None or str(safe.sentence) != str(query["sentence"]):
                raise AssertionError(f"safe target query mismatch: {dataset}|{video}|{qid}")
            sequence = sequence_id(video, qid)
            sequences.append(sequence)
            gt_dir = paths["gt"] / sequence
            gt_dir.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            target_present_frames = 0
            gt_rows = 0
            for frame in frames:
                targets = tuple(str(value) for value in safe.target.get(int(frame), ()))
                target_present_frames += int(bool(targets))
                boxes = frame_records[frame].get("gt_boxes", {})
                for target in sorted(set(targets)):
                    value = boxes.get(target)
                    if value is None and target.isdigit():
                        value = boxes.get(int(target))
                    if value is None:
                        raise AssertionError(f"missing legal GT box: {dataset}|{video}|{qid}|{frame}|{target}")
                    box = [float(item) for item in value]
                    if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                        raise AssertionError(f"invalid legal GT box: {dataset}|{video}|{qid}|{frame}|{target}")
                    if not target.isdigit():
                        raise AssertionError(f"non-integral TrackEval GT ID: {target}")
                    lines.append(f"{int(frame)+1},{int(target)},{box[0]:.6f},{box[1]:.6f},"
                                 f"{box[2]-box[0]:.6f},{box[3]-box[1]:.6f},1,1,1\n")
                    gt_rows += 1
            (gt_dir / "gt.txt").write_text("".join(lines), encoding="utf-8")
            (gt_dir / "seqinfo.ini").write_text(
                "[Sequence]\n" f"name={sequence}\n" "imDir=img1\n" "frameRate=10\n"
                f"seqLength={max(frames)+1}\n" f"imWidth={image_size[0]}\n"
                f"imHeight={image_size[1]}\n" "imExt=.png\n", encoding="utf-8")
            query_audits.append({"dataset": dataset, "video": video, "query_id": qid,
                                 "sequence": sequence, "gt_rows": gt_rows,
                                 "target_present_frames": target_present_frames,
                                 "label_source": safe.label_source,
                                 "labels_attached_after_predictions": True})
    paths["seqmap"].write_text("name\n" + "\n".join(sequences) + "\n", encoding="utf-8")
    return {"sequence_count": len(sequences), "sequences": sequences,
            "query_gt_audits": query_audits, "record_audits": record_audits,
            "labels_attached_after_predictions": True, "safe_target_artifact": str(safe_targets.resolve())}


def run_candidate(candidate: dict[str, Any], args: argparse.Namespace, index: EvalIndex,
                  runtime: R0RuntimeData, out: Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(str(candidate["checkpoint_info"]["path"])).resolve()
    model, info = checkpoint_model(checkpoint_path, device)
    epoch = int(info["epoch"])
    candidate_root = out / f"candidate_epoch{epoch:03d}" / "zero"
    paths = prepare_paths(candidate_root, args.dataset)
    queries_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in index.query_by_key.values():
        if str(value["dataset"]) == args.dataset:
            queries_by_video[str(value["video"])].append({"dataset": args.dataset, "video": str(value["video"]),
                                                            "query_id": int(value["query_id"]), "sentence": str(value["sentence"])})
    for video in queries_by_video:
        unique = {int(value["query_id"]): value for value in queries_by_video[video]}
        queries_by_video[video] = [unique[key] for key in sorted(unique)]
    prediction_files: dict[tuple[str, int], Path] = {}
    for video, queries in queries_by_video.items():
        for query in queries:
            path = paths["tracker_data"] / f"{sequence_id(video, int(query['query_id']))}.txt"
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text("", encoding="utf-8")
            prediction_files[(video, int(query["query_id"]))] = path

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in index.records:
        grouped[(str(record["video"]), int(record["frame_id"]))].append(record)
    group_keys = sorted(grouped)
    prediction_audits: list[dict[str, Any]] = []
    candidate_rows_scored = selected_rows = record_count = 0
    per_video = Counter()
    try:
        with torch.inference_mode():
            for group_index, key in enumerate(group_keys):
                group = sorted(grouped[key], key=lambda value: int(value["query_id"]))
                for start in range(0, len(group), int(args.query_batch_size)):
                    chunk = group[start:start + int(args.query_batch_size)]
                    sample = runtime.prepare_group(chunk, attach_labels=False)
                    output = model_forward(model, sample)
                    if output["membership_logit"].shape != (len(chunk), int(sample["candidate_count"])):
                        raise AssertionError(f"R0 full-video model shape drift at {key}")
                    scores = output["membership_logit"].detach().float().cpu()
                    presence = output["coverage_presence_logit"].detach().float().cpu()
                    boxes = sample["batch"].boxes.float().cpu().tolist()
                    track_ids = [int(value) for value in sample["batch"].track_ids]
                    if len(track_ids) != len(set(track_ids)):
                        raise AssertionError(f"duplicate native track IDs in a frame: {key}")
                    for query_index, record in enumerate(chunk):
                        row_scores = [float(value) for value in scores[query_index].tolist()]
                        present = float(presence[query_index])
                        if not math.isfinite(present) or not all(math.isfinite(value) for value in row_scores):
                            raise FloatingPointError(f"nonfinite R0 full-video scores: {record['unit_key']}")
                        selected = [index for index, value in enumerate(row_scores) if present >= 0.0 and value >= 0.0]
                        path = prediction_files[(str(record["video"]), int(record["query_id"]))]
                        with path.open("a", encoding="utf-8") as handle:
                            for local in selected:
                                x1, y1, x2, y2 = boxes[local]
                                confidence = float(torch.sigmoid(torch.tensor(row_scores[local])))
                                handle.write(f"{int(record['frame_id'])+1},{track_ids[local]},{x1:.6f},{y1:.6f},"
                                             f"{x2-x1:.6f},{y2-y1:.6f},{confidence:.8f},1,1,1\n")
                        prediction_audits.append({"dataset": args.dataset, "video": str(record["video"]),
                                                  "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]),
                                                  "unit_key": str(record["unit_key"]),
                                                  "candidate_rows_scored": len(row_scores), "selected_rows": len(selected),
                                                  "presence_logit": present, "candidate_rows_retained": True,
                                                  "candidate_deletion": False, "candidate_truncation": False,
                                                  "labels_used_for_prediction": False})
                        record_count += 1; candidate_rows_scored += len(row_scores); selected_rows += len(selected)
                        per_video[str(record["video"])] += 1
                    del sample, output, scores, presence
                if (group_index + 1) % 20 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prediction_audit_path = candidate_root / "prediction_audits.jsonl"
    with prediction_audit_path.open("w", encoding="utf-8") as handle:
        for value in prediction_audits:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    frame_map = {video: sorted({int(value["frame_id"]) for value in index.records if str(value["video"]) == video})
                 for video in queries_by_video}
    gt_audit = materialize_gt(args.dataset, queries_by_video, args.safe_targets, paths, frame_map)
    rule = {"name": "zero", "membership_threshold": 0.0, "presence_threshold": 0.0,
            "null_logit": 0.0, "description": "membership_logit>=0 and coverage_presence_logit>=0"}
    return {"role": candidate["role"], "checkpoint_info": info, "rule": rule,
            "root": str(candidate_root.resolve()), "dataset": args.dataset,
            "query_count": sum(len(value) for value in queries_by_video.values()),
            "record_count": record_count, "candidate_rows_scored": candidate_rows_scored,
            "selected_rows": selected_rows, "per_video_record_counts": dict(sorted(per_video.items())),
            "prediction_audits": str(prediction_audit_path.resolve()), "gt_audit": gt_audit,
            "prediction_before_gt": True, "labels_used_for_prediction": False,
            "candidate_deletion": False, "candidate_truncation": False}


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 full-video output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); command = " ".join([sys.executable, *sys.argv])
    base = {"format": "locatemot-r0-fullvideo-inference-v1", "status": "incomplete",
            "command": command, "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD,
            "seed": SEED, "scope": args.scope, "dataset": args.dataset,
            "dense_root": str(args.dense_root.resolve()), "visual_cache": str(args.visual_cache.resolve()),
            "safe_targets": str(args.safe_targets.resolve()), "manifest_sha256": check_manifest(),
            "outputs": {"root": str(out)}, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "candidate_deletion": False, "candidate_truncation": False,
            "preview_only": bool(args.preview_checkpoint is not None),
            "checkpoint_selection_run": False if args.preview_checkpoint is not None else None,
            "labels_used_for_prediction": False, "prediction_before_gt": True,
            "failure_root_cause": None, "next_action": "run legal R0 TrackEval matrix or finalize selection"}
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 inference cwd: {Path.cwd()}")
        index = EvalIndex(args.dense_root)
        if args.scope == "internal" and set(INTERNAL_VIDEOS[args.dataset]) != {str(v) for v in {r["video"] for r in index.records}}:
            raise AssertionError("R0 internal dense index video scope drift")
        candidates = candidates_from_args(args)
        if not candidates or len(candidates) > 3:
            raise AssertionError(f"invalid R0 candidate count: {len(candidates)}")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("R0 inference requested CUDA but it is unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        visual = VisualCacheIndex(args.visual_cache)
        language = MergedLanguageCache([Path(value) for value in args.language_root])
        runtime = R0RuntimeData(index, visual, language, device)
        results = [run_candidate(candidate, args, index, runtime, out, device) for candidate in candidates]
        runtime.close()
        payload = {**base, "status": "complete", "full_video": True,
                   "scope_key": args.scope, "candidate_count": len(results), "candidates": results,
                   "query_frame_record_count": len(index.records), "query_count": len(index.query_by_key),
                   "visual_cache_entries": len(visual.entries), "language_cache_entries": language.entry_count,
                   "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                   "wall_seconds": time.perf_counter() - started,
                   "labels_attached_after_predictions": True,
                   "next_action": "run legal R0 TrackEval matrix or finalize selection"}
        write_json(out / "summary.json", payload); write_json(out / "provenance.json", payload | {"format": "locatemot-r0-fullvideo-provenance-v1"}); write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 full-video inference — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2"), required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--safe-targets", type=Path, required=True)
    parser.add_argument("--shortlist", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--preview-checkpoint", type=Path, default=None,
                        help="run one legal dev preview checkpoint without shortlist selection")
    parser.add_argument("--language-root", type=Path, action="append", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.language_root is None:
        args.language_root = [Path(value) for value in DEFAULT_LANGUAGE_ROOTS]
    if args.scope == "dev" and args.shortlist is None and args.preview_checkpoint is None:
        parser.error("--shortlist is required for dev inference")
    if args.scope == "dev" and args.shortlist is not None and args.preview_checkpoint is not None:
        parser.error("--shortlist and --preview-checkpoint are mutually exclusive")
    if args.scope == "internal" and args.selection is None:
        parser.error("--selection is required for internal inference")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
