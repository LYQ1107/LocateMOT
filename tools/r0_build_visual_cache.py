#!/usr/bin/env python3
"""Build the query-independent R0 visual-token cache.

This is the only R0 script allowed to construct a GroundingDINO runtime.  It
captures ``model.extract_feat`` before text-conditioned fusion, samples every
native L69 row, and writes only compact per-frame token tensors to the
explicit /data2 cache root.  It never loads labels or target metadata.
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

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.l82_grounding_reference import boxes_xyxy_to_normalized  # noqa: E402
from locatemot.rmot.l82_grounding_runtime import build_groundingdino  # noqa: E402
from locatemot.rmot.r0_dense_data import (  # noqa: E402
    FORBIDDEN_SCOPE_VIDEOS,
    FIT_DATASETS,
    L82_SPLIT,
    load_fit_rows,
    load_l69_bank,
    native_frame_slice,
    sha256_file,
)
from locatemot.rmot.r0_visual_tokens import R0VisualTokenConfig, sample_track_visual_tokens  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
IMAGE_ROOT = ASSET_ROOT / "data/kitti_tracking_training/image_02"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
INTERNAL = {"refer_kitti_v1": ("0004", "0018"), "refer_kitti_v2": ("0016", "0017", "0020")}
DEFAULT_CACHE = Path("/data2/usr_for_deadline/locatemot_r0/visual_tokens")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.exists() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
            "sha256": sha256_file(path) if path.is_file() else None}


def _fit_video_pairs() -> list[tuple[str, str]]:
    values = sorted({(str(row["dataset"]), str(row["video"])) for row in load_fit_rows()})
    if any(video in FORBIDDEN_SCOPE_VIDEOS for _dataset, video in values):
        raise AssertionError(f"official video in fit scope: {values}")
    return values


def _scope_pairs(scopes: list[str]) -> list[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for scope in scopes:
        if scope == "train":
            result.update(_fit_video_pairs())
        elif scope == "dev":
            split = json.loads(L82_SPLIT.read_text(encoding="utf-8"))
            result.update(tuple(str(value).split("|", 1)) for value in split.get("dev_videos", []))
        elif scope == "internal":
            result.update((dataset, video) for dataset, videos in INTERNAL.items() for video in videos)
        else:
            raise ValueError(f"unsupported scope {scope}")
    if any(dataset not in FIT_DATASETS or video in FORBIDDEN_SCOPE_VIDEOS for dataset, video in result):
        raise AssertionError(f"illegal R0 cache scope: {sorted(result)}")
    return sorted(result)


class R0FrozenVisualRuntime:
    """Frozen GroundingDINO extract-feature wrapper with no persistent maps."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model, self.model_info = build_groundingdino(device)
        from mmdet.apis import inference_detector
        self.inference_detector = inference_detector
        self.capture: dict[str, Any] = {}
        original_extract = self.model.extract_feat

        def wrapped_extract(batch_inputs: Any) -> Any:
            features = original_extract(batch_inputs)
            self.capture["visual_feats"] = tuple(value.detach().clone() for value in features)
            return features

        self.model.extract_feat = wrapped_extract
        self._original_extract = original_extract
        self.extract_calls = 0

    def capture_frame(self, image: Path, prompt: str) -> tuple[tuple[torch.Tensor, ...], tuple[int, int], Any]:
        self.capture.clear()
        with torch.inference_mode():
            native = self.inference_detector(self.model, str(image), text_prompt=str(prompt), custom_entities=True)
        features = self.capture.get("visual_feats")
        if not isinstance(features, tuple) or len(features) != 4:
            raise AssertionError(f"GroundingDINO extract_feat did not return four levels: {type(features)}")
        for level, value in enumerate(features):
            if value.ndim != 4 or tuple(value.shape[:2]) != (1, 256):
                raise AssertionError(f"visual level {level} shape drift: {tuple(value.shape)}")
            if not bool(torch.isfinite(value.float()).all()):
                raise FloatingPointError(f"nonfinite visual level {level}")
        image_shape = tuple(int(value) for value in native.metainfo["img_shape"][:2])
        scale_factor = native.metainfo["scale_factor"]
        self.extract_calls += 1
        del native
        self.capture.clear()
        return features, image_shape, scale_factor

    def prompt_invariance(self, image: Path) -> dict[str, Any]:
        left, left_shape, left_scale = self.capture_frame(image, "object")
        right, right_shape, right_scale = self.capture_frame(image, "an unrelated empty blue sky")
        if left_shape != right_shape or np.asarray(left_scale).reshape(-1).tolist() != np.asarray(right_scale).reshape(-1).tolist():
            raise AssertionError("prompt-invariance metadata drift")
        deltas = [int((a != b).sum().item()) for a, b in zip(left, right)]
        max_abs = [float((a.float() - b.float()).abs().max().item()) for a, b in zip(left, right)]
        passed = all(delta == 0 for delta in deltas) and all(value == 0.0 for value in max_abs)
        result = {"prompt_a": "object", "prompt_b": "an unrelated empty blue sky",
                  "image": str(image.resolve()), "image_shape": list(left_shape),
                  "scale_factor": np.asarray(left_scale).reshape(-1).tolist(),
                  "per_level_non_equal_count": deltas, "per_level_max_abs_delta": max_abs,
                  "exact_allclose_atol0_rtol0": bool(passed), "extract_calls": int(self.extract_calls)}
        del left, right
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if not passed:
            raise AssertionError("query-independent extract_feat prompt invariance failed")
        return result

    def close(self) -> None:
        del self.model
        self.capture.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _frame_item(
    runtime: R0FrozenVisualRuntime,
    bank_path: Path,
    blob: dict[str, Any],
    video: str,
    frame_position: int,
    cache_path: Path,
    visual_config: R0VisualTokenConfig,
) -> dict[str, Any]:
    tensors = blob["tensors"]
    frame_id, begin, end = native_frame_slice(tensors, frame_position)
    boxes = tensors["box"][begin:end].float().clone()
    metadata = blob["metadata"]
    width, height = [int(value) for value in metadata["image_size"][:2]]
    image = IMAGE_ROOT / str(video) / f"{frame_id:06d}.png"
    if not image.is_file():
        raise FileNotFoundError(image)
    visual_feats, image_shape, scale_factor = runtime.capture_frame(image, "object")
    boxes_normalized = boxes_xyxy_to_normalized(boxes.to(runtime.device), image_shape, scale_factor)
    tokens = sample_track_visual_tokens(visual_feats, boxes_normalized, visual_config)
    candidate_indices = [int(x) for x in tensors["candidate_index"][begin:end].tolist()]
    track_ids = [int(x) for x in tensors["track_id"][begin:end].tolist()]
    pool_ids = [int(x) for x in tensors["pool_id"][begin:end].tolist()]
    raw_ranks = [int(x) for x in tensors["raw_rank"][begin:end].tolist()] if "raw_rank" in tensors else []
    offsets = list(range(begin, end))
    if not (len(offsets) == len(candidate_indices) == int(boxes.shape[0]) == len(track_ids) == len(pool_ids)):
        raise AssertionError(f"R0 row count drift at {video}:{frame_id}")
    item = {
        "format": "locatemot-r0-track-visual-token-v1", "dataset": None, "video": str(video),
        "frame_id": int(frame_id), "group_key": None, "candidate_count": int(len(offsets)),
        "row_offsets": offsets, "candidate_indices": candidate_indices, "track_ids": track_ids,
        "pool_ids": pool_ids, "raw_ranks": raw_ranks, "boxes_xyxy": boxes,
        "boxes_normalized": tokens["boxes_normalized"].float().cpu(),
        "inner_tokens": tokens["inner_tokens"].half().cpu(),
        "context_tokens": tokens["context_tokens"].half().cpu(),
        "visual_level_count": 4, "level_shapes": tokens["level_shapes"].cpu(),
        "grid_size": int(visual_config.grid_size), "context_scale": float(visual_config.context_scale),
        "native_image_shape": list(image_shape), "native_scale_factor": np.asarray(scale_factor).reshape(-1).tolist(),
        "image_size": [width, height], "image_path": str(image.resolve()),
        "query_independent": True, "labels_in_cache": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "groundingdino_frozen": True, "capture_point": "GroundingDINO extract_feat output",
    }
    if not bool(torch.isfinite(item["inner_tokens"].float()).all()) or not bool(torch.isfinite(item["context_tokens"].float()).all()):
        raise FloatingPointError(f"nonfinite cached ROI tokens at {video}:{frame_id}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(item, cache_path)
    del visual_feats, boxes_normalized, tokens, boxes, item
    runtime.capture.clear()
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()
    return {"video": str(video), "frame_id": int(frame_id), "path": str(cache_path.resolve()),
            "candidate_count": int(len(offsets)), "row_offset_start": int(begin), "row_offset_end": int(end),
            "image_path": str(image.resolve()), "finite": True, "candidate_deletion": False,
            "candidate_truncation": False}


def run(args: argparse.Namespace) -> int:
    cache_root = args.out.resolve()
    if cache_root.exists() and any(cache_root.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 visual cache root: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    audit_root = args.audit_out.resolve()
    if audit_root.exists() and any(audit_root.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 visual-cache audit root: {audit_root}")
    audit_root.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    pairs = _scope_pairs(args.scopes)
    base = {
        "format": "locatemot-r0-visual-cache-v1", "status": "incomplete", "command": command,
        "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED, "scopes": list(args.scopes),
        "pairs": [list(value) for value in pairs], "cache_root": str(cache_root),
        "manifest_sha256": sha256_file(MANIFEST), "expected_manifest_sha256": MANIFEST_SHA,
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "tracker_source_changed": False, "uidm_source_changed": False, "l69_source_changed": False,
        "production_entrypoint_changed": False, "labels_in_cache": False, "candidate_deletion": False,
        "candidate_truncation": False, "failure_root_cause": None,
    }
    runtime: R0FrozenVisualRuntime | None = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"R0 cache wrong cwd: {Path.cwd()}")
        if base["manifest_sha256"] != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        runtime = R0FrozenVisualRuntime(device)
        first_train = next(((dataset, video) for dataset, video in pairs if dataset in FIT_DATASETS), None)
        if first_train is None:
            raise AssertionError("R0 cache requires a legal train pair for prompt invariance")
        train_path, train_blob = load_l69_bank(first_train[1])
        train_frame_ids = train_blob["tensors"]["frame_ids"].long().tolist()
        if not train_frame_ids:
            raise AssertionError("empty first R0 train bank")
        prompt_audit = runtime.prompt_invariance(IMAGE_ROOT / first_train[1] / f"{int(train_frame_ids[0]):06d}.png")
        write_json(audit_root / "prompt_invariance.json", prompt_audit)
        del train_blob
        summaries: list[dict[str, Any]] = []
        manifest_lines: list[str] = []
        config = R0VisualTokenConfig()
        for dataset, video in pairs:
            bank_path, blob = load_l69_bank(video)
            tensors = blob["tensors"]
            video_dir = cache_root / dataset / video
            rows: list[dict[str, Any]] = []
            for position in range(int(tensors["frame_ids"].numel())):
                frame_id = int(tensors["frame_ids"][position])
                path = video_dir / f"{frame_id:06d}.pt"
                item_audit = _frame_item(runtime, bank_path, blob, video, position, path, config)
                item_audit["dataset"] = dataset
                item_audit["group_key"] = f"{dataset}|{video}|{frame_id}"
                rows.append(item_audit)
                manifest_lines.append(json.dumps(item_audit, ensure_ascii=False))
            summary = {"dataset": dataset, "video": video, "bank_path": str(bank_path.resolve()),
                       "bank_sha256": sha256_file(bank_path), "frame_count": len(rows),
                       "candidate_rows": int(sum(row["candidate_count"] for row in rows)),
                       "frames": rows, "query_independent": True, "labels_in_cache": False,
                       "candidate_deletion": False, "candidate_truncation": False}
            write_json(video_dir / "video_manifest.json", summary)
            summaries.append({key: value for key, value in summary.items() if key != "frames"})
            del blob, tensors
            gc.collect()
        (cache_root / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        payload = {**base, "status": "complete", "summaries": summaries,
                   "frame_count": int(sum(item["frame_count"] for item in summaries)),
                   "candidate_rows": int(sum(item["candidate_rows"] for item in summaries)),
                   "prompt_invariance": prompt_audit, "model_info": runtime.model_info,
                   "visual_config": {"dim": 256, "levels": 4, "grid_size": 3, "context_scale": 1.75},
                   "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                   "wall_seconds": time.perf_counter() - started,
                   "failure_root_cause": None, "next_action": "build R0 dense train indexes"}
        write_json(cache_root / "summary.json", payload)
        write_json(audit_root / "provenance.json", payload)
        write_json(audit_root / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (cache_root / "INCOMPLETE.md").write_text("# R0 visual cache — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((cache_root / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(audit_root / "provenance.json", payload)
        write_json(audit_root / "status.json", payload)
        return 2
    finally:
        if runtime is not None:
            runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scopes", nargs="+", choices=("train", "dev", "internal"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit-out", type=Path, default=Path("outputs/r0/audit/visual_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
