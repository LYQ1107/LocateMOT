#!/usr/bin/env python3
"""Audit the legal L85 bank, split and TrackEval source without labels."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.rmot.l85_fullvideo_bank import (  # noqa: E402
    ALL_VIDEOS, EXPECTED_MANIFEST_SHA, MANIFEST, audit_bank, bank_source_manifest,
    file_meta, sha256_file,
)
from locatemot.rmot.l85_runtime import load_fit_key_rows  # noqa: E402

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SPLIT = ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
L69_MANIFEST = ROOT / "outputs/l69/attempt9/budget40_features/kitti/manifest.json"
TRACK_EVAL = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master").resolve()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l85/audit/protocol")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        manifest_sha = sha256_file(MANIFEST)
        if manifest_sha != EXPECTED_MANIFEST_SHA:
            raise AssertionError(f"fixed manifest SHA drift: {manifest_sha}")
        fit_rows = load_fit_key_rows()
        if len(fit_rows) != 5314:
            raise AssertionError(f"fit row count drift: {len(fit_rows)}")
        split = json.loads(SPLIT.read_text())
        if len(split.get("train_group_keys", [])) != 524 or len(split.get("dev_group_keys", [])) != 138:
            raise AssertionError("video-disjoint split count drift")
        videos = []
        per_video = []
        for video in ALL_VIDEOS:
            record = audit_bank(video)
            per_video.append(record)
            videos.append(str(video))
            write_json(out / f"{video}.json", record)
        trackeval_info = {
            "path": str(TRACK_EVAL), "exists": TRACK_EVAL.is_dir(),
            "git_head": None,
        }
        if TRACK_EVAL.is_dir():
            import subprocess
            try:
                trackeval_info["git_head"] = subprocess.check_output(
                    ["git", "-C", str(TRACK_EVAL), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
            except Exception as exc:
                trackeval_info["git_head_error"] = f"{type(exc).__name__}: {exc}"
            trackeval_info["run_mot_script"] = str(TRACK_EVAL / "scripts/run_mot_challenge.py")
            trackeval_info["package_exists"] = (TRACK_EVAL / "trackeval").is_dir()
        result = {
            "format": "locatemot-l85-protocol-audit-v1", "status": "complete",
            "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
            "inputs": {"manifest": file_meta(MANIFEST), "l69_manifest": file_meta(L69_MANIFEST),
                        "fit_units": {"path": str(ROOT / "outputs/l49/data/train_units.jsonl"), "count": len(fit_rows)},
                        "split": file_meta(SPLIT)},
            "videos": videos, "per_video": per_video, "source_manifest": bank_source_manifest(ALL_VIDEOS),
            "train_group_count": len(split["train_group_keys"]), "dev_group_count": len(split["dev_group_keys"]),
            "trackeval": trackeval_info, "elapsed_sec": time.perf_counter() - started,
            "candidate_bank_gt_conditioned": False, "candidate_bank_query_conditioned": False,
            "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            "failure_root_cause": None, "next_action": "run candidate oracle and label-free memory contract",
        }
        write_json(out / "protocol.json", result)
        write_json(out / "provenance.json", result)
        write_json(out / "status.json", result)
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text("# L85 protocol audit — INCOMPLETE\n\n" + __import__("traceback").format_exc() + "\n")
        write_json(out / "status.json", {"format": "locatemot-l85-protocol-audit-v1", "status": "incomplete",
                                          "command": command, "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "next_action": "fix only the first actionable contract error", "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
