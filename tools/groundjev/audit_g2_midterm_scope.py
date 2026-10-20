#!/usr/bin/env python3
"""Audit historical G2 midterm frames against the Clean TRAIN manifest.

The step-3000 results predate the Clean V2 naming repair and remain historical
G2-PL artifacts.  This script only classifies their input media; it does not
rewrite results or read evaluation labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_G2_ROWS = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "g2_clean_train_rows.jsonl"
)
DEFAULT_MIDTERM_ROOT = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/"
    "g2_midterm_step3000_pair_20261019_r1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def load_train_index(rows_path: Path) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    with rows_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source", "unknown"))
            for image in row.get("images", []):
                image_path = image.get("path") if isinstance(image, dict) else image
                if image_path:
                    index.setdefault(canonical(str(image_path)), set()).add(source)
    return index


def load_scope(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError(f"invalid video manifest: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-rows", type=Path, default=DEFAULT_G2_ROWS)
    parser.add_argument("--midterm-root", type=Path, default=DEFAULT_MIDTERM_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scopes = [
        ("MOT17-02-FRCNN", "G2-PL", args.midterm_root / "G2-PL/dense_mot4", "outputs/groundjev/pilot_data/mot17_first4.json"),
        ("Refer-KITTI-0007-single", "G2-PL", args.midterm_root / "G2-PL/refer_single8", "outputs/groundjev/multitask_pilot_data/rmot_single_empty.json"),
        ("Refer-KITTI-0007-multi", "G2-PL", args.midterm_root / "G2-PL/refer_multi12", "outputs/groundjev/multitask_pilot_data/rmot_multi_empty.json"),
    ]
    train_index = load_train_index(args.g2_rows)
    all_frames: list[dict[str, Any]] = []
    scope_reports: list[dict[str, Any]] = []
    for name, model_label, result_dir, manifest_hint in scopes:
        manifest_path = Path("/data2/user/LocateMOT_GroundJEV") / manifest_hint
        manifest = load_scope(manifest_path)
        frames: list[dict[str, Any]] = []
        for frame_number, raw_path in zip(manifest["frame_numbers"], manifest["images"]):
            path = Path(canonical(str(raw_path)))
            sources = sorted(train_index.get(str(path), set()))
            record = {
                "frame": int(frame_number),
                "image": str(path),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "g2_train_overlap": bool(sources),
                "g2_train_sources": sources,
            }
            frames.append(record)
            all_frames.append({"scope": name, **record})
        overlap = [f for f in frames if f["g2_train_overlap"]]
        scope_reports.append(
            {
                "scope": name,
                "historical_model_label": model_label,
                "manifest": str(manifest_path),
                "dataset": manifest.get("dataset", "mot17" if "MOT" in name else "refer_kitti_v2"),
                "sequence": manifest.get("sequence"),
                "frames": len(frames),
                "overlap_frames": len(overlap),
                "overlap_images": sorted({f["image"] for f in overlap}),
                "same_video_or_sequence_in_g2_train": bool(overlap),
                "frames_detail": frames,
            }
        )

    result = {
        "status": "G2_MIDTERM_TRAIN_SCOPE_AUDIT_COMPLETE",
        "historical_result_label": "G2-PL",
        "historical_result_unchanged": True,
        "g2_train_manifest": str(args.g2_rows),
        "g2_train_row_count": 583251,
        "g2_train_manifest_sha256": "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71",
        "total_frames": len(all_frames),
        "overlap_frames": sum(int(x["g2_train_overlap"]) for x in all_frames),
        "independent_frames": sum(not x["g2_train_overlap"] for x in all_frames),
        "independent_generalization_claim_allowed": False,
        "claim_boundary": "MOT4 is TRAIN-scope; both RMOT clips are independent of this G2 Clean image-path index but remain short diagnostic clips, not full validation results.",
        "scopes": scope_reports,
    }
    output = args.output or Path("docs/GROUNDJEV/G2_MIDTERM_SCOPE_AUDIT_20261021.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "total_frames", "overlap_frames", "independent_frames", "output")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
