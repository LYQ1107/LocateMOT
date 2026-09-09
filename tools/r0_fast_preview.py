#!/usr/bin/env python3
"""Deterministic small legal-dev preview for one grouped R0 checkpoint.

This is an early diagnostic only.  It uses the registered zero-logit rule,
does not fit a threshold, and samples up to 32 uniformly spaced native frames
per legal dev video and up to 32 queries per sampled frame.  It never reads
internal, screening, or official-test labels.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r0_common import (  # noqa: E402
    DEFAULT_LANGUAGE_ROOTS,
    EvalIndex,
    MergedLanguageCache,
    R0RuntimeData,
    VisualCacheIndex,
    check_manifest,
    model_forward,
    sha256_file,
    standard_flags,
    write_json,
)
from tools.r0_metric_utils import ZeroLogitMetrics  # noqa: E402
from tools.r0_score_dev import attach_after_score, checkpoint_info, load_model, make_score_record  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909


def selected_records(index: EvalIndex, max_frames: int, max_queries: int) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    by_video_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in index.records:
        by_video_frame[(str(record["dataset"]), str(record["video"]), int(record["frame_id"]))].append(record)
    groups: list[list[dict[str, Any]]] = []
    sampled: dict[str, dict[str, Any]] = {}
    for dataset, video in sorted({(key[0], key[1]) for key in by_video_frame}):
        frame_ids = sorted(key[2] for key in by_video_frame if key[:2] == (dataset, video))
        if len(frame_ids) <= int(max_frames):
            chosen = frame_ids
        else:
            positions = np.linspace(0, len(frame_ids) - 1, int(max_frames), dtype=int)
            chosen = sorted(set(int(frame_ids[position]) for position in positions.tolist()))
        query_counts = []
        for frame_id in chosen:
            values = sorted(by_video_frame[(dataset, video, frame_id)], key=lambda value: int(value["query_id"]))
            if len(values) > int(max_queries):
                values = values[: int(max_queries)]
            query_counts.append(len(values))
            groups.append(values)
        sampled[f"{dataset}|{video}"] = {
            "native_frame_count": len(frame_ids), "selected_frame_ids": chosen,
            "selected_frame_count": len(chosen), "queries_per_selected_frame": query_counts,
            "query_selection": "all legal queries if <=32, otherwise first 32 by query_id",
        }
    if not groups:
        raise AssertionError("R0 fast preview has no legal dev groups")
    return groups, {"max_frames_per_video": int(max_frames), "max_queries_per_frame": int(max_queries), "videos": sampled}


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 preview output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable, *sys.argv])
    base = {
        "format": "locatemot-r0-fast-dev-preview-v2", "status": "incomplete",
        "command": command, "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD,
        "seed": SEED, "dataset": str(args.dataset), "scope": "dev",
        "checkpoint": str(args.checkpoint.resolve()), "dense_root": str(args.dense_root.resolve()),
        "visual_cache": str(args.visual_cache.resolve()), "manifest_sha256": check_manifest(),
        "outputs": {"root": str(out)}, "registered_rule": "membership_logit>=0 and coverage_presence_logit>=0; no threshold fitting",
        "preview_only": True, "checkpoint_selection_run": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "failure_root_cause": None, "next_action": "inspect preview; resume formal grouped training",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 preview cwd: {Path.cwd()}")
        index = EvalIndex(args.dense_root)
        groups, sample_scope = selected_records(index, args.max_frames_per_video, args.max_queries_per_frame)
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("R0 preview requested CUDA but it is unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        visual = VisualCacheIndex(args.visual_cache)
        language = MergedLanguageCache([Path(value) for value in args.language_root])
        model, info = load_model(args.checkpoint.resolve(), device)
        runtime = R0RuntimeData(index, visual, language, device)
        metrics = ZeroLogitMetrics()
        records_path = out / "score_records.jsonl"
        record_count = 0
        candidate_rows = 0
        selected_rows = 0
        pair_total = 0
        pair_correct = 0
        with records_path.open("w", encoding="utf-8") as handle:
            with torch.inference_mode():
                for group in groups:
                    sample = runtime.prepare_group(group, attach_labels=False)
                    sample["query_batches"] = []
                    sample["supervisions"] = []
                    output = model_forward(model, sample)
                    if output["membership_logit"].shape != (len(group), int(sample["candidate_count"])):
                        raise AssertionError("R0 fast preview output shape drift")
                    attach_after_score(runtime, index, sample, group)
                    for query_index, record in enumerate(group):
                        scored = make_score_record(record, sample, output, query_index, index)
                        handle.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")
                        metrics.add(scored)
                        values = [float(value) for value in scored["score"]]
                        labels = [bool(value) for value in scored["labels"]]
                        positives = [values[i] for i, label in enumerate(labels) if label]
                        negatives = [values[i] for i, label in enumerate(labels) if not label]
                        if positives and negatives:
                            pair_total += 1
                            pair_correct += int(min(positives) > max(negatives))
                        selected_rows += sum(int(value >= 0.0 and scored["presence_logit"] >= 0.0) for value in values)
                        record_count += 1
                        candidate_rows += int(scored["candidate_count"])
                    del sample, output
        summary_metrics = metrics.finish()
        summary_metrics["positive_gt_above_negative_pairwise_accuracy"] = float(pair_correct / pair_total) if pair_total else None
        summary_metrics["positive_gt_above_negative_pairwise_denominator"] = pair_total
        if candidate_rows and selected_rows == candidate_rows:
            raise AssertionError("R0 fast preview selected every candidate row; likely output contract failure")
        runtime.close()
        del model
        payload = {
            **base, "status": "complete", "checkpoint_info": info,
            "sample_scope": sample_scope, "group_count": len(groups),
            "record_count": record_count, "candidate_rows": candidate_rows,
            "selected_rows": selected_rows, "score_records": str(records_path.resolve()),
            "score_records_sha256": sha256_file(records_path), "metrics": summary_metrics,
            "visual_cache_entries": len(visual.entries), "language_cache_entries": language.entry_count,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started, "labels_used_after_prediction": True,
            "candidate_rows_retained": True, "no_persistent_raw_dense_cache_created": True,
        }
        write_json(out / "metrics.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-r0-fast-dev-preview-provenance-v2"})
        write_json(out / "status.json", payload | standard_flags(training_run=False))
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 fast preview — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--language-root", type=Path, action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames-per-video", type=int, default=32)
    parser.add_argument("--max-queries-per-frame", type=int, default=32)
    args = parser.parse_args()
    if args.language_root is None:
        args.language_root = [Path(value) for value in DEFAULT_LANGUAGE_ROOTS]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
