#!/usr/bin/env python3
"""Build only the fixed-slice frames missing from the frozen R0 visual cache.

This is a label-free R1 supplement.  It reuses the audited query-independent
GroundingDINO ``extract_feat`` contract but never opens candidate-GT or target
sidecars.  The existing base visual cache is referenced, not copied.
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
R0_VISUAL_ROOT = Path("/data2/usr_for_deadline/locatemot_r0a_visual_tokens_retry3_final").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r0_build_visual_cache import R0FrozenVisualRuntime, _frame_item  # noqa: E402
from tools.r0_common import VisualCacheIndex  # noqa: E402
from locatemot.rmot.r0_dense_data import load_l69_bank, native_frame_slice, sha256_file  # noqa: E402
from locatemot.rmot.r0_visual_tokens import R0VisualTokenConfig  # noqa: E402
from tools.r1_common import load_fixed_l62_key_order, file_meta, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 visual supplement: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([str(sys.executable), *sys.argv])
    started = time.perf_counter()
    base: dict[str, Any] = {
        "format": "locatemot-r1-fixed-visual-supplement-v1", "status": "incomplete",
        "command": command, "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
        "base_visual_root": str(R0_VISUAL_ROOT), "manifest_sha256": sha256_file(MANIFEST),
        "expected_manifest_sha256": MANIFEST_SHA, "labels_read": False, "labels_in_cache": False,
        "query_independent": True, "candidate_deletion": False, "candidate_truncation": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False,
        "failure_root_cause": None, "next_action": "merge supplement root after row-level validation",
    }
    runtime: R0FrozenVisualRuntime | None = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 visual supplement cwd: {Path.cwd()}")
        if base["manifest_sha256"] != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        base_index = VisualCacheIndex(R0_VISUAL_ROOT)
        fixed = load_fixed_l62_key_order()
        # The fixed-order helper deliberately returns only dataset/video/query/
        # frame/sentence/split metadata.  No target or candidate label field is
        # retained or used in this label-free supplement.
        missing = []
        for record in fixed:
            key = (str(record["dataset"]), str(record["video"]), int(record["frame_id"]))
            if key not in base_index.entries and key not in missing:
                missing.append(key)
        if not missing:
            raise AssertionError("fixed visual supplement unexpectedly has no missing frames")
        by_video: dict[tuple[str, str], list[int]] = {}
        for dataset, video, frame_id in missing:
            by_video.setdefault((dataset, video), []).append(frame_id)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("R1 visual supplement requires CUDA")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        runtime = R0FrozenVisualRuntime(device)
        config = R0VisualTokenConfig()
        rows: list[dict[str, Any]] = []
        for (dataset, video), frame_ids in sorted(by_video.items()):
            bank_path, blob = load_l69_bank(video)
            tensors = blob["tensors"]
            native_ids = [int(value) for value in tensors["frame_ids"].tolist()]
            for frame_id in sorted(frame_ids):
                position = native_ids.index(int(frame_id))
                path = out / dataset / video / f"{int(frame_id):06d}.pt"
                summary = _frame_item(runtime, bank_path, blob, dataset, video, position, path, config)
                rows.append(summary)
            del blob, tensors
            gc.collect()
            torch.cuda.empty_cache()
        if len(rows) != len(missing) or {(r["dataset"], r["video"], int(r["frame_id"])) for r in rows} != set(missing):
            raise AssertionError("R1 visual supplement frame-set drift")
        (out / "manifest.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
        for row in rows:
            payload = torch.load(Path(row["path"]), map_location="cpu", weights_only=False)
            if payload.get("labels_in_cache") is not False or payload.get("query_independent") is not True:
                raise AssertionError(f"R1 visual supplement label/query flag drift: {row['path']}")
            if int(payload["candidate_count"]) != len(payload["row_offsets"]):
                raise AssertionError(f"R1 visual supplement candidate count drift: {row['path']}")
            for name in ("inner_tokens", "context_tokens", "boxes_normalized"):
                if not torch.is_tensor(payload[name]) or not bool(torch.isfinite(payload[name].float()).all()):
                    raise FloatingPointError(f"R1 visual supplement nonfinite {name}: {row['path']}")
            del payload
        payload = {**base, "status": "complete", "missing_frame_count": len(rows),
                   "missing_frames": [list((r["dataset"], r["video"], int(r["frame_id"]))) for r in rows],
                   "rows": rows, "model_info": runtime.model_info,
                   "visual_config": {"dim": 256, "levels": 4, "grid_size": 3, "context_scale": 1.75},
                   "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                   "wall_seconds": time.perf_counter() - started,
                   "inputs": {"manifest": file_meta(MANIFEST), "base_visual_manifest": file_meta(R0_VISUAL_ROOT / "manifest.jsonl")},
                   "outputs": {"manifest": str(out / "manifest.jsonl"), "root": str(out)}}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        error = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R1 visual supplement incomplete\n\n```text\n" + error + "```\n", encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str(out / "INCOMPLETE.md"), "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2
    finally:
        if runtime is not None:
            runtime.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
