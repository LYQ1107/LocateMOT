#!/usr/bin/env python3
"""Materialize the compact, query-independent GroundingDINO encoder inputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


WORK_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L88").resolve()
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L69_ROOT = ASSET_ROOT / "outputs/l69/attempt9/budget40_features/kitti"
IMAGE_ROOT = ASSET_ROOT / "data/kitti_tracking_training/image_02"

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))
import locatemot.rmot as _rmot_package  # noqa: E402
if str(ASSET_ROOT / "locatemot" / "rmot") not in [str(x) for x in _rmot_package.__path__]:
    _rmot_package.__path__.append(str(ASSET_ROOT / "locatemot" / "rmot"))

from locatemot.rmot.l88_grounding_runtime import (  # noqa: E402
    L88GroundingRuntime, cache_filename, cache_key, file_meta, save_cache_item,
    sha256_file,
)
from locatemot.rmot.l85_runtime import (  # noqa: E402
    load_fit_train_dev_groups, load_internal_eval_groups,
)
from locatemot.rmot.l80_data import load_fixed_key_units  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def collect_requests(include_internal: bool) -> list[dict[str, Any]]:
    """Collect unique dataset/video/frame requests without reading labels."""
    requests: OrderedDict[tuple[str, str, int], dict[str, Any]] = OrderedDict()
    groups, train_keys, dev_keys = load_fit_train_dev_groups()
    for group_key in train_keys + dev_keys:
        group = groups[str(group_key)]
        identity = (str(group["dataset"]), str(group["video"]), int(group["frame_id"]))
        requests.setdefault(identity, {"dataset": identity[0], "video": identity[1], "frame": identity[2], "reason": "fit_or_fit_dev_group"})
    for unit in load_fixed_key_units():
        identity = (str(unit["dataset"]), str(unit["video"]), int(unit["frame_id"]))
        requests.setdefault(identity, {"dataset": identity[0], "video": identity[1], "frame": identity[2], "reason": "fixed_semantic_unit"})
    if include_internal:
        internal_videos = {
            "refer_kitti_v1": ("0004", "0018"),
            "refer_kitti_v2": ("0016", "0017", "0020"),
        }
        for dataset, videos in internal_videos.items():
            for video in videos:
                path = (L69_ROOT / f"{video}.pt").resolve()
                bank = torch.load(path, map_location="cpu", weights_only=False)
                tensors = bank.get("tensors") if isinstance(bank, dict) else None
                if not isinstance(tensors, dict) or "frame_ids" not in tensors:
                    raise AssertionError(f"invalid L69 frame index: {path}")
                for frame in tensors["frame_ids"].long().tolist():
                    identity = (dataset, video, int(frame))
                    requests.setdefault(identity, {"dataset": dataset, "video": video, "frame": int(frame), "reason": "internal_full_video"})
                del bank, tensors
    return [requests[key] for key in sorted(requests)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "outputs/l88/cache/encoder_inputs_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-internal", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="only for a declared contract smoke; zero means all")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 cache output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    runtime = None
    manifest_handle = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json") != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(device)
        requests = collect_requests(bool(args.include_internal))
        if int(args.max_frames) > 0:
            requests = requests[:int(args.max_frames)]
        if not requests:
            raise AssertionError("no cache frame requests")
        runtime = L88GroundingRuntime(device)
        manifest_handle = (out / "manifest.jsonl").open("w", encoding="utf-8")
        records: list[dict[str, Any]] = []
        total_bytes = 0
        for index, request in enumerate(requests):
            image = IMAGE_ROOT / request["video"] / f"{int(request['frame']):06d}.png"
            key = cache_key(request["dataset"], request["video"], request["frame"])
            item = runtime.cache_frame(image)
            item.update({"cache_key": key, "dataset": request["dataset"], "video": request["video"], "frame": int(request["frame"])})
            filename = cache_filename(key)
            saved = save_cache_item(out / filename, item)
            line = {
                "format": "locatemot-l88-cache-manifest-row-v1",
                "cache_key": key, "dataset": request["dataset"], "video": request["video"],
                "frame": int(request["frame"]), "image_path": str(image.resolve()),
                "file": filename, "bytes": int(saved["bytes"]), "sha256": saved["sha256"],
                "reason": request["reason"], "candidate_independent": True,
                "labels_in_cache": False, "query_strings_in_cache": False,
                "semantic_scores_in_cache": False, "query_ids_in_cache": False,
                "finite_tensors": True, "candidate_deletion": False,
                "candidate_truncation": False,
            }
            manifest_handle.write(json.dumps(line, sort_keys=True) + "\n"); manifest_handle.flush()
            records.append(line); total_bytes += int(saved["bytes"])
            del item
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if index % 8 == 0:
                gc.collect()
            print(json.dumps({"index": index + 1, "total": len(requests), "cache_key": key, "bytes": saved["bytes"]}), flush=True)
        manifest_handle.close(); manifest_handle = None
        summary = {
            "format": "locatemot-l88-encoder-input-cache-summary-v1", "status": "complete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "entry_count": len(records), "total_bytes": total_bytes,
            "manifest": str((out / "manifest.jsonl").resolve()),
            "manifest_sha256": sha256_file(out / "manifest.jsonl"),
            "include_internal_full_video": bool(args.include_internal),
            "max_frames": int(args.max_frames),
            "cache_payload": ["feat", "feat_mask", "feat_pos", "spatial_shapes", "level_start_index", "valid_ratios", "metainfo"],
            "labels_in_cache": False, "query_independent": True,
            "candidate_deletion": False, "candidate_truncation": False,
            "groundingdino_weight": file_meta(Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth")),
            "mmdetection_reference": "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a",
            "manifest_sha256_fixed": MANIFEST_SHA,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False, "failure_root_cause": None,
            "next_action": "run L88 zero-init parity/gradient contract before training",
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        }
        write_json(out / "summary.json", summary)
        write_json(out / "provenance.json", summary)
        write_json(out / "status.json", summary)
        return 0
    except Exception:
        trace = traceback.format_exc()
        if manifest_handle is not None:
            manifest_handle.close()
        (out / "INCOMPLETE.md").write_text("# L88 encoder-input cache — INCOMPLETE\n\n" + trace)
        payload = {"format": "locatemot-l88-encoder-input-cache-status-v1", "status": "incomplete", "command": command,
                   "cwd": str(WORK_ROOT), "luna_thread": THREAD, "failure_root_cause": "first traceback in INCOMPLETE.md",
                   "screening_gt_used": False, "official_test_labels_read": False,
                   "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False}
        write_json(out / "status.json", payload)
        raise
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
