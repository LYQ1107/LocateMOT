#!/usr/bin/env python3
"""Run the corrected L88 candidate-vs-NULL rule on a full internal scope.

The detector/sidecar forward is inherited from the already audited L88
runtime.  This file is intentionally separate from the old full-video helper:
the only emission change is the L88C equation, where candidate energy is
compared with NULL per row and presence remains an independent gate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from l88_eval_common import (
    L85_CACHE, L88_CACHE, MANIFEST, MANIFEST_SHA, SEED, THREAD, EncoderCacheReader,
    L88ClipStore, load_checkpoint_into, make_runtime, sha256, write_json,
)
from l88_infer_fullvideo_matrix import (
    INTERNAL_DATASETS, SPLIT, cache_for_frame, frame_ids, make_frame,
    materialize_gt, scope_queries, scope_videos, score_frame_queries,
    sequence_id, strategy_root, prepare_strategy, sigmoid,
)
from l88c_eval_metrics import corrected_emission_mask


WORK_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
RULES = ("B", "R", "P")


def _checkpoint_sha(candidate: dict[str, Any]) -> dict[str, Any]:
    info = dict(candidate["checkpoint_info"])
    path = Path(str(info["path"])).resolve()
    if sha256(path) != str(info["sha256"]):
        raise AssertionError(f"shortlist checkpoint SHA drift: {path}")
    info["path"] = str(path)
    return info


def _write_prediction_row(path: Path, frame_id: int, track_id: int,
                          box: list[float], score: float) -> None:
    x1, y1, x2, y2 = [float(value) for value in box]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{int(frame_id) + 1},{int(track_id)},{x1:.6f},{y1:.6f},"
            f"{x2 - x1:.6f},{y2 - y1:.6f},{sigmoid(score):.8f},1,1,1\n"
        )


def _safe_key_digest(record: dict[str, Any]) -> bytes:
    return json.dumps(
        {"unit_key": record["unit_key"], "row_keys": record["row_keys"]},
        sort_keys=False,
    ).encode("utf-8")


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88C full-video output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    runtime = None
    store = None
    reader = None
    try:
        if args.scope not in {"dev", "internal"}:
            raise ValueError(args.scope)
        datasets = list(DATASETS if args.dataset == "all" else (str(args.dataset),))
        split = json.loads(SPLIT.read_text())
        videos_by_dataset = scope_videos(args.scope, datasets, split)
        if args.video:
            if len(datasets) != 1:
                raise ValueError("--video requires one dataset")
            if str(args.video) not in videos_by_dataset.get(datasets[0], ()):
                raise ValueError(f"video is outside selected scope: {datasets[0]}|{args.video}")
            videos_by_dataset = {datasets[0]: (str(args.video),)}

        shortlist_path = args.shortlist.resolve()
        shortlist = json.loads(shortlist_path.read_text())
        if shortlist.get("status") != "complete":
            raise AssertionError("corrected shortlist is not complete")
        candidates = list(shortlist.get("shortlist", []))
        if not candidates:
            raise AssertionError("corrected shortlist is empty")
        if args.candidate_index >= 0:
            if args.candidate_index >= len(candidates):
                raise IndexError(f"candidate index out of range: {args.candidate_index}")
            candidates = [candidates[int(args.candidate_index)]]
        if args.max_candidates:
            candidates = candidates[: int(args.max_candidates)]
        if args.max_frames < 0 or args.max_queries < 0:
            raise ValueError("negative scope limit")

        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        reader = EncoderCacheReader(args.cache)
        store = L88ClipStore(L85_CACHE, load_cache_into_ram=False)
        runtime, injector, base_digest = make_runtime(device)

        text_cache: dict[str, tuple[dict[str, Any], list[str], list[Any]]] = {}
        video_queries: dict[str, dict[str, list[dict[str, Any]]]] = {dataset: {} for dataset in datasets}
        for dataset, videos in videos_by_dataset.items():
            for video in videos:
                queries = scope_queries(store, args.scope, dataset, video, split)
                if args.max_queries:
                    queries = queries[: int(args.max_queries)]
                if not queries:
                    raise AssertionError(f"empty query scope: {dataset}|{video}")
                video_queries[dataset][video] = queries

        summaries: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            checkpoint_info = _checkpoint_sha(candidate)
            checkpoint_path = Path(str(checkpoint_info["path"])).resolve()
            sidecar, loaded_info = load_checkpoint_into(runtime, injector, checkpoint_path, device)
            strategy_paths: dict[str, dict[str, dict[str, Path]]] = {}
            for rule_name in RULES:
                strategy_paths[rule_name] = prepare_strategy(
                    strategy_root(out, candidate, rule_name), datasets, video_queries
                )
            audit_handles = {
                rule_name: (strategy_root(out, candidate, rule_name) / "prediction_audits.jsonl").open("w", encoding="utf-8")
                for rule_name in RULES
            }
            counters = {
                rule_name: {
                    "frames": 0, "queries": 0, "candidate_rows": 0,
                    "selected_rows": 0, "key_digest": hashlib.sha256(),
                    "presence_gate_pass": 0, "presence_gate_fail": 0,
                }
                for rule_name in RULES
            }
            try:
                for dataset in datasets:
                    for video in videos_by_dataset[dataset]:
                        queries = video_queries[dataset][video]
                        frames = frame_ids(store, video)
                        if args.max_frames:
                            frames = frames[: int(args.max_frames)]
                        for frame_index, frame_id in enumerate(frames):
                            first, _ = make_frame(store, dataset, video, frame_id, queries[0])
                            records = score_frame_queries(
                                runtime, reader, sidecar, store, first, queries, device,
                                int(args.query_tile), text_cache,
                            )
                            if len(records) != len(queries):
                                raise AssertionError(f"query count drift: {dataset}|{video}|{frame_id}")
                            boxes = first.boxes.detach().cpu().numpy().tolist()
                            for record in records:
                                values = np.asarray(record["score"], dtype=np.float64)
                                track_ids = [int(value) for value in record["track_ids"]]
                                if values.shape != (int(record["candidate_count"]),):
                                    raise AssertionError(f"score shape drift: {record['unit_key']}")
                                if len(track_ids) != len(set(track_ids)):
                                    raise AssertionError(f"track row drift: {record['unit_key']}")
                                if len(record["row_keys"]) != int(record["candidate_count"]):
                                    raise AssertionError(f"row-key count drift: {record['unit_key']}")
                                key_digest_value = _safe_key_digest(record)
                                for rule_name in RULES:
                                    rule = candidate["rule_fits"][rule_name]
                                    presence = float(record["presence_logit"])
                                    null = float(record["null_logit"])
                                    selected_mask = corrected_emission_mask(
                                        values, presence, null,
                                        float(rule["candidate_threshold"]),
                                        float(rule["presence_threshold"]),
                                        float(rule["null_margin"]),
                                    )
                                    selected = np.flatnonzero(selected_mask)
                                    tracker_path = (
                                        strategy_paths[rule_name][dataset]["tracker_data"]
                                        / f"{sequence_id(video, int(record['query_id']))}.txt"
                                    )
                                    for local in selected.tolist():
                                        _write_prediction_row(
                                            tracker_path, frame_id, track_ids[local], boxes[local], float(values[local])
                                        )
                                    counter = counters[rule_name]
                                    counter["frames"] += 1
                                    counter["queries"] += 1
                                    counter["candidate_rows"] += int(record["candidate_count"])
                                    counter["selected_rows"] += int(selected.size)
                                    counter["key_digest"].update(key_digest_value)
                                    if presence >= float(rule["presence_threshold"]):
                                        counter["presence_gate_pass"] += 1
                                    else:
                                        counter["presence_gate_fail"] += 1
                                    audit_handles[rule_name].write(json.dumps({
                                        "dataset": dataset, "video": video,
                                        "query_id": int(record["query_id"]), "frame_id": int(frame_id),
                                        "unit_key": record["unit_key"],
                                        "candidate_rows_scored": int(record["candidate_count"]),
                                        "selected_rows": int(selected.size),
                                        "presence_logit": presence, "null_logit": null,
                                        "candidate_threshold": float(rule["candidate_threshold"]),
                                        "presence_threshold": float(rule["presence_threshold"]),
                                        "null_margin": float(rule["null_margin"]),
                                        "candidate_vs_null_applied": True,
                                        "candidate_rows_retained": True,
                                        "candidate_deletion": False, "candidate_truncation": False,
                                        "labels_attached": False,
                                    }, sort_keys=True) + "\n")
                            del first, records
                            store.release_loaded_cache_items()
                            gc.collect()
                            if device.type == "cuda" and frame_index % 10 == 0:
                                torch.cuda.empty_cache()
                        print(
                            f"[l88c-fullvideo] scope={args.scope} epoch={checkpoint_info['epoch']} "
                            f"{dataset} video={video} frames={len(frames)} queries={len(queries)} "
                            f"elapsed={time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
            finally:
                for handle in audit_handles.values():
                    handle.close()

            for rule_name in RULES:
                for dataset in datasets:
                    for video, queries in video_queries[dataset].items():
                        tracker_dir = strategy_paths[rule_name][dataset]["tracker_data"]
                        for query in queries:
                            (tracker_dir / f"{sequence_id(video, int(query['query_id']))}.txt").touch(exist_ok=True)

            eval_summary: dict[str, Any] = {}
            for rule_name in RULES:
                eval_summary[rule_name] = {}
                for dataset in datasets:
                    eval_summary[rule_name][dataset] = materialize_gt(
                        args.scope, dataset, video_queries[dataset],
                        strategy_paths[rule_name][dataset], store,
                    )
            summaries.append({
                "candidate_index": candidate_index,
                "candidate_name": candidate.get("candidate_name"),
                "checkpoint": loaded_info,
                "rules": {
                    rule: {
                        **{key: value for key, value in counters[rule].items() if key != "key_digest"},
                        "key_digest": counters[rule]["key_digest"].hexdigest(),
                        "candidate_rows_retained": True, "candidate_deletion": False,
                        "candidate_truncation": False,
                    }
                    for rule in RULES
                },
                "rule_fits": candidate["rule_fits"],
                "strategy_paths": {
                    rule: {
                        dataset: {name: str(path.resolve()) for name, path in paths.items()}
                        for dataset, paths in per_dataset.items()
                    }
                    for rule, per_dataset in strategy_paths.items()
                },
                "eval_summary": eval_summary,
                "prediction_strategy_frozen_before_gt": True,
                "corrected_candidate_vs_null": True,
            })
            del sidecar
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        full = not args.max_frames and not args.max_queries and not args.video and args.candidate_index < 0 and not args.max_candidates
        payload = {
            "format": "locatemot-l88c-fullvideo-matrix-v1", "status": "complete",
            "scope_key": args.scope,
            "scope": "internal full-video dev selection" if args.scope == "dev" else "internal V1/V2 full-video validation",
            "evidence_type": "zero-training corrected candidate-vs-NULL full-video prediction matrix",
            "full_video": bool(full), "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "seed": SEED, "datasets": datasets, "videos": videos_by_dataset,
            "shortlist_source": str(shortlist_path), "shortlist_sha256": sha256(shortlist_path),
            "candidates": summaries, "candidate_index_filter": int(args.candidate_index),
            "video_filter": str(args.video) if args.video else None,
            "cache": str(args.cache.resolve()), "cache_summary_sha256": reader.summary_sha256,
            "base_detector_digest": base_digest, "query_tile": int(args.query_tile),
            "all_candidate_rows_scored": True, "candidate_deletion": False, "candidate_truncation": False,
            "model_selection_used_validation": False, "labels_attached_after_predictions": True,
            "fit_dev_labels_only": args.scope == "dev", "internal_validation_labels_only": args.scope == "internal",
            "zero_training": True, "backward_called": False, "optimizer_step_called": False,
            "new_checkpoint_written": False, "lora_update": False, "sidecar_update": False,
            "corrected_candidate_vs_null": True, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True,
            "groundingdino_lora_used": True, "groundingdino_backbone_trainable": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "manifest_sha256": MANIFEST_SHA, "persistent_dense_cache_written": False,
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "failure_root_cause": None,
            "next_action": "run L88C TrackEval matrix on this frozen internal strategy output",
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-fullvideo-status-v1", "status": "complete",
            "scope": args.scope, "full_video": bool(full), "candidate_count": len(summaries),
            "zero_training": True, "corrected_candidate_vs_null": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True,
        })
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88C full-video matrix — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-fullvideo-status-v1", "status": "incomplete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "zero_training": True, "corrected_candidate_vs_null": True,
            "failure_root_cause": "first traceback in INCOMPLETE.md",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True,
        })
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if store is not None:
            store.release_loaded_cache_items()
            store.close()
        if reader is not None:
            del reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=L88_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=("all", "refer_kitti_v1", "refer_kitti_v2"), default="all")
    parser.add_argument("--query-tile", type=int, default=4)
    parser.add_argument("--candidate-index", type=int, default=-1)
    parser.add_argument("--video", default="")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
