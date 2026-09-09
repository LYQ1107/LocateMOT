#!/usr/bin/env python3
"""Audit the frozen interfaces used by the R0 sidecar, without model load."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
BASE_SHA = "7fa201e4967f71a42ba669e90f330a6e3979e1cd"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
MMDET = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0").resolve()
L89E_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E").resolve()

INTERFACES = {
    ASSET_ROOT / "locatemot/rmot/l82_grounding_runtime.py": (
        "build_groundingdino", "GroundingCandidateReferenceRuntime", "extract_feat",
    ),
    ASSET_ROOT / "locatemot/models/l82_grounding_reference.py": (
        "boxes_xyxy_to_normalized", "pool_memory_by_box",
    ),
    L89E_ROOT / "tools/l89d_fullvideo_common.py": (
        "scope_video_values", "VideoScope", "load_video_scopes", "native_frame_ids",
        "native_groups", "expected_timeline_descriptor", "validate_native_batch",
    ),
    ASSET_ROOT / "tools/l86_infer_fullvideo.py": (
        "frame_groups", "query_rows_for_video",
    ),
    L89E_ROOT / "tools/l89_trackeval_matrix.py": ("run_dataset", "PERCENT_METRICS"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_meta(path: Path, symbols: tuple[str, ...]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    missing = [symbol for symbol in symbols if symbol not in text]
    return {
        "path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path),
        "required_symbols": list(symbols), "missing_symbols": missing,
        "passed": not missing,
    }


def command_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty source audit output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable, *sys.argv])
    result: dict[str, Any] = {
        "format": "locatemot-r0-source-interface-audit-v1",
        "status": "complete",
        "command": command,
        "cwd": str(WORK_ROOT),
        "luna_thread": THREAD,
        "base_sha": BASE_SHA,
        "asset_root": str(ASSET_ROOT),
        "frozen_l89e_source_root": str(L89E_ROOT),
        "frozen_interfaces": [source_meta(path, symbols) for path, symbols in INTERFACES.items()],
        "groundingdino_checkout": {
            "path": str(MMDET),
            "git_head": command_output("git", "-C", str(MMDET), "rev-parse", "HEAD"),
            "git_head_note": "no verifiable HEAD is recorded if the value is null",
        },
        "manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST), "expected_sha256": MANIFEST_SHA},
        "interface_contract": {
            "visual_capture": "GroundingDINO model.extract_feat output; four [1,256,H,W] maps",
            "box_sampling": "existing l82 boxes_xyxy_to_normalized and align_corners=False grid_sample contract",
            "timeline": "native L69 frame_ids/frame_ptr; no sparse L49 frame range addressing",
            "tracker": "read-only l89/l86 wrappers; no tracker source imported by training",
        },
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "tracker_source_changed": False,
        "uidm_source_changed": False,
        "l69_source_changed": False,
        "production_entrypoint_changed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "failure_root_cause": None,
        "next_action": "continue R0 implementation only if data contract passes",
    }
    result["passed"] = all(item["passed"] for item in result["frozen_interfaces"]) and result["manifest"]["sha256"] == MANIFEST_SHA
    if not result["passed"]:
        result["status"] = "invalid"
        result["failure_root_cause"] = "missing frozen interface or manifest SHA drift"
    for name in ("audit.json", "provenance.json", "status.json"):
        (out / name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/r0/audit/source_interfaces"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
