#!/usr/bin/env python3
"""Build a deterministic, video-disjoint LocateMOT-U legal split.

The script reads only public development videos.  Reserved Refer-KITTI videos
are rejected by the shared legal-scope module before any expression or label
path is opened.  It records enough per-video evidence to audit whether the
three partitions differ materially in size, expression volume, target volume,
and motion proxy.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from locatemot.paths import PROJECT_ROOT, V1_ROOT, V2_ROOT
from locatemot.rmot.l49_data import L49_SPLITS, load_l49_queries
from locatemot.unified.data.legal_scope import FORBIDDEN_VIDEOS, assert_legal_path, assert_legal_video

SEED = 20261001
MANIFEST_PATH = PROJECT_ROOT / "outputs/unified/protocol/data_manifest.json"
IMAGE_ROOT = PROJECT_ROOT / "data/kitti_tracking/training/image_02"
DATASET_ROOTS = {"refer_kitti_v1": V1_ROOT, "refer_kitti_v2": V2_ROOT}
SPLIT_NAMES = ("fit", "calibration", "validation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_key(dataset: str, video: str) -> str:
    return hashlib.sha256(f"{SEED}|{dataset}|{video}".encode()).hexdigest()


def parse_label(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            track_id = str(fields[1])
            cx, cy, width, height = (float(value) for value in fields[2:6])
        except ValueError:
            continue
        result[track_id] = {
            "class_id": str(fields[0]),
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
        }
    return result


def image_size(video: str) -> tuple[int, int]:
    frames = sorted((IMAGE_ROOT / video).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no legal KITTI images for {video}")
    # KITTI is fixed-resolution, so opening one image is enough for the audit.
    from PIL import Image
    with Image.open(assert_legal_path(frames[0])) as image:
        return int(image.width), int(image.height)


def video_stats(dataset: str, video: str, queries: list[dict[str, Any]]) -> dict[str, Any]:
    video = assert_legal_video(video, dataset=dataset)
    root = DATASET_ROOTS[dataset]
    label_root = root / "labels_with_ids" / "image_02" / video
    labels_by_frame: dict[int, dict[str, dict[str, Any]]] = {}
    for label_path in sorted(label_root.glob("*.txt")):
        assert_legal_path(label_path)
        labels_by_frame[int(label_path.stem)] = parse_label(label_path)
    width, height = image_size(video)

    occurrences = 0
    valid_occurrences = 0
    source_missing_occurrences = 0
    missing_physical: set[tuple[str, int, str]] = set()
    valid_physical: set[tuple[str, int, str]] = set()
    expression_frames = 0
    single_target_frames = 0
    multi_target_frames = 0
    inactive_frames = 0
    for query in queries:
        for frame, target_ids in sorted(query["target"].items()):
            target_ids = {str(target) for target in target_ids}
            expression_frames += 1
            if not target_ids:
                inactive_frames += 1
            elif len(target_ids) == 1:
                single_target_frames += 1
            else:
                multi_target_frames += 1
            labels = labels_by_frame.get(int(frame), {})
            for target_id in sorted(target_ids):
                occurrences += 1
                physical = (str(query["query_id"]), int(frame), target_id)
                if target_id in labels:
                    valid_occurrences += 1
                    valid_physical.add((video, int(frame), target_id))
                else:
                    source_missing_occurrences += 1
                    missing_physical.add((video, int(frame), target_id))

    class_counts = Counter()
    size_counts = Counter()
    track_centers: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for frame, labels in labels_by_frame.items():
        for track_id, value in labels.items():
            class_counts[value["class_id"]] += 1
            box_height = value["height"] * height
            size_counts["small" if box_height < 32 else "medium" if box_height <= 96 else "large"] += 1
            track_centers[track_id].append(
                (int(frame), value["cx"] * width, value["cy"] * height)
            )
    displacements: list[float] = []
    moving_tracks = 0
    for points in track_centers.values():
        points.sort()
        local = [math.hypot(x1 - x0, y1 - y0) for (_, x0, y0), (_, x1, y1) in zip(points, points[1:])]
        if local:
            displacements.extend(local)
            if max(local) > 2.0:
                moving_tracks += 1
    track_count = len(track_centers)
    return {
        "dataset": dataset,
        "video": video,
        "hash_key": hash_key(dataset, video),
        "frame_count": len(sorted((IMAGE_ROOT / video).glob("*.png"))),
        "label_frame_count": len(labels_by_frame),
        "image_width": width,
        "image_height": height,
        "expression_count": len(queries),
        "expression_frame_count": expression_frames,
        "single_target_expression_frames": single_target_frames,
        "multi_target_expression_frames": multi_target_frames,
        "inactive_expression_frames": inactive_frames,
        "target_references": occurrences,
        "valid_target_references": valid_occurrences,
        "source_target_id_missing_references": source_missing_occurrences,
        "source_target_id_missing_physical": len(missing_physical),
        "unique_physical_targets": len(valid_physical),
        "class_distribution": dict(sorted(class_counts.items())),
        "size_distribution": dict(sorted(size_counts.items())),
        "track_count": track_count,
        "moving_track_ratio": moving_tracks / max(1, track_count),
        "median_consecutive_center_displacement_px": statistics.median(displacements) if displacements else 0.0,
    }


def assign_splits(stats: list[dict[str, Any]]) -> dict[str, list[str]]:
    count = len(stats)
    validation_count = max(2, round(count * 0.20))
    calibration_count = max(1, round(count * 0.20))
    quotas = {
        "fit": count - validation_count - calibration_count,
        "calibration": calibration_count,
        "validation": validation_count,
    }
    # Greedily distribute high-volume videos first.  The objective balances
    # expression count, target volume and motion proxy against fixed quotas;
    # the hash key is the deterministic tie breaker.
    ordered = sorted(
        stats,
        key=lambda row: (
            -(math.log1p(row["expression_count"]) + math.log1p(row["unique_physical_targets"])),
            -row["moving_track_ratio"],
            row["hash_key"],
        ),
    )
    totals = {name: {"expression_count": 0.0, "unique_physical_targets": 0.0, "moving_track_ratio": 0.0} for name in SPLIT_NAMES}
    assigned = {name: [] for name in SPLIT_NAMES}
    global_totals = {
        key: sum(float(row[key]) for row in stats)
        for key in ("expression_count", "unique_physical_targets", "moving_track_ratio")
    }
    for row in ordered:
        candidates = [name for name in SPLIT_NAMES if len(assigned[name]) < quotas[name]]
        scored = []
        for name in candidates:
            projected = dict(totals[name])
            for key in projected:
                projected[key] += float(row[key])
            target_ratio = quotas[name] / count
            score = sum(
                abs((projected[key] / max(1e-9, global_totals[key])) - target_ratio)
                for key in ("expression_count", "unique_physical_targets", "moving_track_ratio")
            )
            scored.append((score, name))
        _, chosen = min(scored)
        assigned[chosen].append(row["video"])
        for key in totals[chosen]:
            totals[chosen][key] += float(row[key])
    for name in assigned:
        assigned[name].sort()
    return assigned


def main() -> int:
    source_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_manifest_sha = sha256_file(MANIFEST_PATH)
    all_stats: dict[str, list[dict[str, Any]]] = {}
    split_map: dict[str, dict[str, list[str]]] = {}
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        queries = load_l49_queries(dataset)
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for query in queries:
            assert_legal_video(query["video"], dataset=dataset)
            by_video[query["video"]].append(query)
        stats = [video_stats(dataset, video, by_video[video]) for video in sorted(by_video)]
        all_stats[dataset] = stats
        split_map[dataset] = assign_splits(stats)

    payload: dict[str, Any] = {
        "format": "locatemot-u-legal-video-split-v2",
        "status": "FROZEN_BEFORE_U1_FORMAL_EVALUATION",
        "seed": SEED,
        "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "source_manifest_sha256": source_manifest_sha,
        "datasets": {
            dataset: {
                "fit": split_map[dataset]["fit"],
                "calibration": split_map[dataset]["calibration"],
                "validation": split_map[dataset]["validation"],
                "official_eval_reserved": list(L49_SPLITS[dataset]["official_eval"]),
                "video_stats": all_stats[dataset],
            }
            for dataset in ("refer_kitti_v1", "refer_kitti_v2")
        },
        "forbidden_videos": sorted(FORBIDDEN_VIDEOS),
        "v2_source_target_id_missing_count": sum(
            row["source_target_id_missing_references"]
            for row in all_stats["refer_kitti_v2"]
        ),
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_started": False,
    }
    payload["split_sha256"] = canonical_sha(payload)
    out = PROJECT_ROOT / "outputs/unified/protocol/legal_video_split_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# LocateMOT-U legal split audit (v2)",
        "",
        f"The split is deterministic with seed `{SEED}` and source manifest SHA256 `{source_manifest_sha}`.",
        "It is frozen before any U1 formal evaluation or adaptation training.",
        "Official evaluation videos are represented as metadata and were not opened.",
        "",
        "## Assignment contract",
        "",
        "Video quotas are fit 60%, calibration 20%, validation 20% (with at least two validation and one calibration video per dataset). High-volume videos are assigned first by a fixed objective over log expression volume, unique physical target volume, and moving-track ratio; SHA256(seed, dataset, video) breaks ties.",
        "",
    ]
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        report_lines.extend([f"## {dataset}", "", "| split | videos | expressions | valid target refs | missing source refs | unique physical targets |", "|---|---:|---:|---:|---:|---:|"])
        stats_by_video = {row["video"]: row for row in all_stats[dataset]}
        for split in ("fit", "calibration", "validation"):
            rows = [stats_by_video[video] for video in split_map[dataset][split]]
            report_lines.append(
                f"| {split} | {len(rows)} | {sum(row['expression_count'] for row in rows)} | {sum(row['valid_target_references'] for row in rows)} | {sum(row['source_target_id_missing_references'] for row in rows)} | {sum(row['unique_physical_targets'] for row in rows)} |"
            )
        report_lines.extend(["", "| video | frames | expressions | valid refs | missing refs | moving track ratio | size small/medium/large |", "|---|---:|---:|---:|---:|---:|---|"])
        for row in all_stats[dataset]:
            sizes = row["size_distribution"]
            report_lines.append(
                f"| {row['video']} | {row['frame_count']} | {row['expression_count']} | {row['valid_target_references']} | {row['source_target_id_missing_references']} | {row['moving_track_ratio']:.3f} | {sizes.get('small', 0)}/{sizes.get('medium', 0)}/{sizes.get('large', 0)} |"
            )
        report_lines.append("")
    report_lines.extend([
        "## V2 anomaly contract",
        "",
        f"V2 source-target-ID-missing references counted from legal expression/label joins: `{payload['v2_source_target_id_missing_count']}`. Missing references remain in the source rows and are excluded only from valid-positive denominators; they are not replaced, converted to negatives, or silently dropped.",
        "",
        "Machine-readable output: `outputs/unified/protocol/legal_video_split_v2.json`.",
        "",
        "`official_test_labels_read=false`, `screening_gt_used=false`, and `training_started=false`.",
    ])
    (PROJECT_ROOT / "reports/unified/LEGAL_SPLIT_AUDIT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "format": payload["format"],
        "split_sha256": payload["split_sha256"],
        "v2_source_target_id_missing_count": payload["v2_source_target_id_missing_count"],
        "datasets": {dataset: split_map[dataset] for dataset in split_map},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
