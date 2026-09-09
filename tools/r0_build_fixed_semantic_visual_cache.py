#!/usr/bin/env python3
"""Build the small query-independent cache needed by R0 fixed diagnostics.

The registered train/dev/internal cache is intentionally not widened in place.
This helper materializes only the two pre-registered calibration videos that
are absent from that cache, using the same frozen GroundingDINO extraction and
L69 row contract.  It reads no query or label payload and writes only compact
visual-token items under the approved /data2 temporary root.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r0_build_visual_cache import (  # noqa: E402
    IMAGE_ROOT,
    R0FrozenVisualRuntime,
    _frame_item,
    meta,
    write_json,
)
from locatemot.rmot.r0_dense_data import (  # noqa: E402
    FIT_DATASETS,
    MANIFEST,
    EXPECTED_MANIFEST_SHA,
    load_l69_bank,
    native_frame_slice,
    sha256_file,
)
from locatemot.rmot.r0_visual_tokens import R0VisualTokenConfig  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
CALIBRATION_PAIRS = (("refer_kitti_v1", "0016"), ("refer_kitti_v2", "0015"))
DEFAULT_OUT = Path("/data2/usr_for_deadline/locatemot_r0a_visual_tokens_calibration_retry1")


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty calibration visual cache: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable, *sys.argv])
    base: dict[str, Any] = {
        "format": "locatemot-r0-fixed-semantic-visual-cache-v1",
        "status": "incomplete", "command": command, "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD, "seed": SEED,
        "pairs": [list(value) for value in CALIBRATION_PAIRS],
        "manifest_sha256": sha256_file(MANIFEST), "expected_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False,
        "hota_trackeval_run": False, "labels_in_cache": False,
        "query_independent": True, "candidate_deletion": False,
        "candidate_truncation": False, "failure_root_cause": None,
        "next_action": "merge with the frozen R0 visual manifest for fixed diagnostics",
    }
    runtime: R0FrozenVisualRuntime | None = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 fixed-cache cwd: {Path.cwd()}")
        if base["manifest_sha256"] != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        runtime = R0FrozenVisualRuntime(device)
        config = R0VisualTokenConfig()
        manifest_lines: list[str] = []
        summaries: list[dict[str, Any]] = []
        for dataset, video in CALIBRATION_PAIRS:
            if dataset not in FIT_DATASETS:
                raise AssertionError(f"invalid calibration dataset: {dataset}")
            bank_path, blob = load_l69_bank(video)
            tensors = blob["tensors"]
            video_dir = out / dataset / video
            rows: list[dict[str, Any]] = []
            for position in range(int(tensors["frame_ids"].numel())):
                frame_id, begin, end = native_frame_slice(tensors, position)
                path = video_dir / f"{frame_id:06d}.pt"
                item = _frame_item(runtime, bank_path, blob, dataset, video, position, path, config)
                rows.append(item)
                manifest_lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            summary = {
                "dataset": dataset, "video": video,
                "bank_path": str(bank_path.resolve()), "bank_sha256": sha256_file(bank_path),
                "frame_count": len(rows), "candidate_rows": int(sum(row["candidate_count"] for row in rows)),
                "query_independent": True, "labels_in_cache": False,
                "candidate_deletion": False, "candidate_truncation": False,
            }
            write_json(video_dir / "video_manifest.json", summary)
            summaries.append(summary)
            del blob, tensors
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        (out / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        payload = {
            **base, "status": "complete", "summaries": summaries,
            "frame_count": int(sum(value["frame_count"] for value in summaries)),
            "candidate_rows": int(sum(value["candidate_rows"] for value in summaries)),
            "model_info": runtime.model_info, "visual_config": {
                "dim": 256, "levels": 4, "grid_size": 3, "context_scale": 1.75,
            },
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload | {"input_metadata": {"manifest": meta(MANIFEST)}})
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 fixed semantic visual cache — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2
    finally:
        if runtime is not None:
            runtime.close()
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
