#!/usr/bin/env python3
"""Prepare a legal, expression-conditioned Refer-KITTI native-export pilot.

The earlier ``candidate_export_rmot_v2`` pass used one class-agnostic pedestrian
request per video.  That is a valid MOT detector cache, but it cannot represent
RMOT expressions such as automobiles, vans, or cyclists.  This planner keeps the
old cache untouched and emits one ``task=rmot`` manifest per selected real query.
It only reads legal expression metadata and already-exported frame names; model
inference is performed later by the native exporter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path("/data2/user/LocateMOT_OneJev_CVPR")
PROJECT_ROOT = Path("/data2/user/LocateMOT")
EXPORT_ROOT = ROOT / "outputs/groundjev/full_data_v2/candidate_export_rmot_v2"
DEFAULT_OUT = ROOT / "outputs/groundjev/full_data_v2/candidate_export_rmot_expression_v1"
IMAGE_ROOT = PROJECT_ROOT / "data/kitti_tracking/training/image_02"
EXPRESSION_ROOT = PROJECT_ROOT / "data/refer_kitti_v2/expression"
sys.path.insert(0, str(ROOT))


def legal_videos() -> list[str]:
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS

    split = LEGAL_SPLITS["refer_kitti_v2"]
    held_out = set(split.get("validation", [])) | {
        "0005", "0007", "0011", "0013", "0019"
    }
    videos: list[str] = []
    for part in ("fit", "calibration"):
        for video in split.get(part, []):
            if video not in held_out and video not in videos:
                videos.append(video)
    return videos


def load_native_frames(directory: Path) -> list[int]:
    path = directory / "frame_candidates.jsonl"
    if not path.is_file():
        return []
    frames: set[int] = set()
    for line in path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("frame") is not None:
            frames.add(int(row["frame"]))
    return sorted(frames)


def load_expression(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("sentence"), str) or not payload["sentence"].strip():
        raise ValueError(f"expression has no sentence: {path}")
    labels = payload.get("label") or {}
    if not isinstance(labels, dict):
        raise ValueError(f"expression labels are not a mapping: {path}")
    return payload


def select_frames(available: list[int], payload: dict[str, Any], max_frames: int) -> list[int]:
    labels = payload.get("label") or {}
    labelled = sorted(int(frame) for frame in labels if int(frame) in set(available))
    if not labelled:
        return []
    low, high = labelled[0], labelled[-1]
    span = [frame for frame in available if low <= frame <= high]
    if len(span) <= max_frames:
        return span
    step = max(1, len(span) // max_frames)
    return span[::step][:max_frames]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=Path, default=EXPORT_ROOT)
    parser.add_argument("--queries-per-video", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=250)
    args = parser.parse_args()
    if args.queries_per_video < 1 or args.max_frames < 1:
        raise SystemExit("queries-per-video and max-frames must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    manifests = args.out / "manifests"
    manifests.mkdir()

    selected: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for video in legal_videos():
        native_dir = args.source / f"kitti_{video}"
        available = load_native_frames(native_dir)
        if not available:
            skipped[video] = 0
            continue
        expression_dir = EXPRESSION_ROOT / video
        candidates: list[tuple[int, str, Path, dict[str, Any], list[int]]] = []
        for expression_file in sorted(expression_dir.glob("*.json")):
            payload = load_expression(expression_file)
            frames = select_frames(available, payload, args.max_frames)
            if frames:
                positive = sum(bool((payload.get("label") or {}).get(str(frame), [])) for frame in frames)
                candidates.append((-positive, expression_file.name, expression_file, payload, frames))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for query_index, (_neg_positive, _name, expression_file, payload, frames) in enumerate(
            candidates[: args.queries_per_video]
        ):
            query_id = f"{video}_q{query_index:02d}_{expression_file.stem}"
            image_dir = IMAGE_ROOT / video
            images = [str(image_dir / f"{frame:06d}.png") for frame in frames]
            missing_images = [path for path in images if not Path(path).is_file()]
            if missing_images:
                raise FileNotFoundError(missing_images[0])
            manifest = {
                "task": "rmot",
                "dataset": "refer_kitti_v2",
                "sequence": video,
                "query_id": query_id,
                "fps": 10,
                "images": images,
                "frame_numbers": frames,
                "vocabulary": ["person", "car", "van", "cyclist"],
                "expression": payload["sentence"],
                "expression_file": str(expression_file),
                "scope": "legal Refer-KITTI-V2 fit/calibration expression-conditioned native pilot",
                "training_on_this_scope": True,
                "runtime_gt_fields": False,
                "official_test_labels_read": False,
                "proposal_source": "native_G0_expression_conditioned_pending",
                "native_export_source": str(native_dir),
                "positive_label_frames_in_manifest": [
                    frame for frame in frames if (payload.get("label") or {}).get(str(frame), [])
                ],
                "frame_selection": "contiguous native-export span around real query labels; labels are metadata only",
            }
            manifest_path = manifests / f"{query_id}.json"
            manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
            selected.append({
                "query_id": query_id,
                "video": video,
                "expression_file": str(expression_file),
                "expression": payload["sentence"],
                "frames": len(frames),
                "positive_frames": len(manifest["positive_label_frames_in_manifest"]),
                "manifest": str(manifest_path.relative_to(args.out)),
            })
        if not candidates:
            skipped[video] = len(available)

    plan = {
        "status": "EXPRESSION_CONDITIONED_RMOT_PILOT_PLANNED",
        "source_native_export": str(args.source),
        "output_root": str(args.out),
        "proposal_source": "native_G0_expression_conditioned_pending",
        "legal_videos_considered": legal_videos(),
        "queries_per_video_cap": args.queries_per_video,
        "max_frames_per_query": args.max_frames,
        "selected_queries": selected,
        "skipped_videos": skipped,
        "heldout_videos_excluded": ["0005", "0007", "0011", "0013", "0019"],
        "model_inference_run": False,
        "full_goal_complete": False,
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": plan["status"],
        "selected_queries": len(selected),
        "selected_frames": sum(item["frames"] for item in selected),
        "videos_with_queries": len({item["video"] for item in selected}),
        "skipped_videos": skipped,
        "out": str(args.out),
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
