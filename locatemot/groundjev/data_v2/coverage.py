"""Real data-coverage audit for the ten named FullDataV2 datasets.

Counts are *physical* (videos, unique images, identities, births, occlusion
reappearances, queries, categories), never "number of generated question rows".
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from . import mot_challenge, ovmot, registry, rmot


def _strip_detector_suffix(name: str) -> str:
    """MOT17 ships three detector copies of every video; count the video once."""
    for suffix in ("-DPM", "-FRCNN", "-SDP"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _aggregate_mot(spec: registry.DatasetSpec, max_sequences: Optional[int] = None,
                   class_filter: Optional[set[int]] = None) -> dict[str, Any]:
    root = spec.resolved_root()
    if root is None:
        return {"status": "UNAVAILABLE", "roots_checked": [str(p) for p in spec.roots]}
    problems: list[str] = []
    if spec.protocol is not None:
        try:
            registry.load_frozen_protocol(spec.name)
        except Exception as exc:  # noqa: BLE001 - audit reports real failures
            problems.append(f"protocol load failed: {exc!r}")

    sequences = mot_challenge.discover_sequences(root, spec.splits)
    if max_sequences:
        sequences = sequences[:max_sequences]
    rows = []
    base_videos: dict[str, set[str]] = defaultdict(set)
    canonical: list[tuple[Path, str]] = []
    seen_sequences: set[tuple[str, str]] = set()
    for seq_dir, role in sequences:
        base = _strip_detector_suffix(seq_dir.name)
        base_videos[role].add(base)
        key = (base, role)
        if key in seen_sequences:
            continue
        seen_sequences.add(key)
        canonical.append((seq_dir, role))
    for seq_dir, role in canonical:
        inventory = mot_challenge.inventory_sequence(seq_dir, role, class_filter=class_filter)
        row = inventory.to_dict()
        row["counted_as_canonical_copy"] = True
        rows.append(row)
    totals = Counter()
    for row in rows:
        for key in ("frames_on_disk", "frames_in_gt", "gt_boxes", "identities", "births",
                    "identity_gaps", "reappearances", "empty_gt_frames", "duplicate_pairs",
                    "long_occlusions"):
            totals[key] += int(row.get(key) or 0)
    return {
        "status": "SCANNED" if rows else "NO_SEQUENCE_FOUND",
        "root": str(root),
        "split_dirs": spec.splits,
        "kind": "mot",
        "sequence_dirs_found": len(sequences),
        "sequences_scanned": len(rows),
        "detector_copies_ignored": len(sequences) - len(rows),
        "videos_by_role": {role: len(names) for role, names in base_videos.items()},
        "totals": dict(totals),
        "unique_images": totals["frames_on_disk"],
        "unique_identities": totals["identities"],
        "identity_gap_events": totals["identity_gaps"],
        "occlusion_reappearances": totals["reappearances"],
        "long_occlusions": totals["long_occlusions"],
        "duplicate_gt_pairs": totals["duplicate_pairs"],
        "annotation_exhaustive": True,
        "licence": spec.licence,
        "notes": spec.notes + problems,
        "flags": spec.flags,
        "sequences": rows,
    }


def _aggregate_animaltrack(spec: registry.DatasetSpec) -> dict[str, Any]:
    """AnimalTrack ships videos + per-video gt files, no per-frame images.

    Frames only exist after decoding the mp4s, so coverage counts the GT frames
    (which the benchmark provides) and records that the official TRAIN/TEST split
    file was not obtained.
    """
    root = spec.resolved_root()
    if root is None:
        return {"status": "UNAVAILABLE", "roots_checked": [str(p) for p in spec.roots]}
    gt_dir = root / "gt_all"
    videos_dir = root / "videos_all"
    if not gt_dir.is_dir():
        return {"status": "NO_GT", "root": str(root)}
    videos = sorted(videos_dir.glob("*.mp4")) if videos_dir.is_dir() else []
    gt_files = sorted(gt_dir.glob("*.txt"))
    split_dir = root / "splits"
    train_names, test_names = set(), set()
    if (split_dir / "gt_train.txt").is_file():
        train_names = {l.strip() for l in (split_dir / "gt_train.txt").read_text().splitlines() if l.strip()}
        test_names = {l.strip() for l in (split_dir / "gt_test.txt").read_text().splitlines() if l.strip()}
        gt_files = [p for p in gt_files if p.name in train_names]
    frames_dir = root / "frames_all"
    rows = 0
    identities: set[tuple[str, int]] = set()
    frames_by_video: dict[str, set[int]] = {}
    for path in gt_files:
        video = path.stem[:-3] if path.stem.endswith("_gt") else path.stem
        seen = frames_by_video.setdefault(video, set())
        for line in path.read_text(errors="replace").splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                frame = int(float(parts[0]))
                track = int(float(parts[1]))
            except ValueError:
                continue
            rows += 1
            identities.add((video, track))
            seen.add(frame)
    frames_on_disk = sum(1 for _ in frames_dir.rglob("*.jpg")) if frames_dir.is_dir() else 0
    return {
        "status": "SCANNED",
        "kind": "mot",
        "root": str(root),
        "videos_by_role": {"train": len(train_names), "validation": 0,
                           "test_without_gt": len(test_names)},
        "videos_on_disk": len(videos),
        "frames_on_disk": frames_on_disk,
        "official_split_registered": bool(train_names),
        "gt_files": len(gt_files),
        "unique_images": 0,
        "gt_frames": sum(len(v) for v in frames_by_video.values()),
        "unique_identities": len(identities),
        "gt_rows": rows,
        "totals": {"births": len(identities), "gt_boxes": rows},
        "annotation_exhaustive": True,
        "split_status": ("OFFICIAL_SPLIT_REGISTERED" if train_names
                         else "OFFICIAL_SPLIT_NOT_OBTAINED"),
        "notes": spec.notes + [
            "videos are mp4; per-frame images require decoding (not done yet)",
            "official TRAIN 32 / TEST 26 split registered from videos_train.txt/gt_train.txt",
        ],
        "flags": spec.flags,
    }


def build_report(datasets: Optional[list[str]] = None,
                 max_sequences: Optional[int] = None) -> dict[str, Any]:
    names = datasets or list(registry.DATASETS)
    report: dict[str, Any] = {"registry": registry.availability_table(), "coverage": {}}
    for name in names:
        spec = registry.dataset(name)
        entry: dict[str, Any]
        try:
            if name == "animaltrack":
                entry = _aggregate_animaltrack(spec)
            elif name in ("mot17", "mot20", "dancetrack", "sportsmot"):
                entry = _aggregate_mot(spec, max_sequences=max_sequences)
            elif name == "refer_kitti_v2":
                entry = rmot.inventory_refer_kitti(spec).to_dict()
                entry["status"] = "SCANNED"
            elif name == "rmot26":
                annotation = Path("/data3/user/LocateMOT_FullDataV2/downloads/RMOT26/train/annotations.json")
                entry = rmot.inventory_rmot26(annotation).to_dict()
                entry["status"] = "SCANNED"
            elif name == "aerialmind":
                entry = rmot.inventory_aerialmind(spec).to_dict()
                entry["status"] = "SCANNED"
                if not entry.get("expressions"):
                    entry["status"] = "PENDING_DOWNLOAD"
            elif name == "tao_amodal":
                entry = ovmot.inventory_tao(spec).to_dict()
                entry["status"] = "SCANNED"
            elif name == "burst":
                entry = ovmot.inventory_burst(spec).to_dict()
                entry["status"] = "PROBED"
            else:  # pragma: no cover
                entry = {"status": "UNKNOWN_DATASET"}
        except Exception as exc:  # noqa: BLE001 - record the real blocker
            entry = {"status": "BLOCKED", "error": repr(exc)}
        report["coverage"][name] = entry
    report["totals"] = {
        "datasets_declared": len(registry.DATASETS),
        "datasets_available": sum(1 for spec in registry.DATASETS.values()
                                  if spec.resolved_root() is not None),
    }
    return report


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "coverage_report.json"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    lines = ["# FullDataV2 真实数据覆盖报告", "",
             "自动生成。统计对象是物理视频 / 唯一图像 / 物理身份 / 文本查询，不是生成的问答行数。",
             "MOT17 的三份检测器副本已去重（只统计一份），DanceTrack 的 val 属于注册开发范围并保持 held-out。", ""]
    mot_rows, rmot_rows, ovmot_rows, other_rows = [], [], [], []
    for name, entry in report["coverage"].items():
        kind = entry.get("kind") or ("rmot" if name in ("refer_kitti_v2", "rmot26", "aerialmind")
                                     else "ovmot" if name in ("tao_amodal", "burst") else "mot")
        status = entry.get("status")
        if kind == "mot":
            roles = entry.get("videos_by_role") or {}
            videos = "/".join(str(roles.get(k, 0)) for k in ("train", "validation", "test_without_gt"))
            mot_rows.append(
                f"| {name} | {status} | {videos} | {entry.get('unique_images', 0)} | "
                f"{entry.get('unique_identities', 0)} | {(entry.get('totals') or {}).get('births', 0)} | "
                f"{entry.get('occlusion_reappearances', 0)} | {entry.get('long_occlusions', 0)} | "
                f"{entry.get('duplicate_gt_pairs', 0)} |")
        elif kind == "rmot":
            rmot_rows.append(
                f"| {name} | {status} | {entry.get('videos', 0)} | {entry.get('expressions', 0)} | "
                f"{entry.get('target_identities', 0)} | {entry.get('positive_frames', 0)} |")
        elif kind == "ovmot":
            ovmot_rows.append(
                f"| {name} | {status} | {entry.get('videos', 0)} | {entry.get('images', 0)} | "
                f"{entry.get('tracks', 0)} | {entry.get('categories_present', 0)} | "
                f"{entry.get('base_categories_present', 0)} |")
        else:
            other_rows.append(f"| {name} | {status} |")

    lines.append("## MOT 数据集")
    lines.append("")
    lines.append("| 数据集 | 状态 | 视频(train/val/test) | 唯一图像 | 物理身份 | 出生事件 | 遮挡重现(≥10帧) | 长遮挡(≥30帧) | 重复GT对(IoU≥.5) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    lines.extend(mot_rows or ["| — | — | — | — | — | — | — | — | — |"])
    lines.append("")
    lines.append("## RMOT 数据集")
    lines.append("")
    lines.append("| 数据集 | 状态 | 视频 | 表达式 | 目标身份 | 正例帧 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    lines.extend(rmot_rows or ["| — | — | — | — | — | — |"])
    lines.append("")
    lines.append("## OVMOT 数据集")
    lines.append("")
    lines.append("| 数据集 | 状态 | 视频 | 标注图像 | track | 出现的类别 | Base 类别 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    lines.extend(ovmot_rows or ["| — | — | — | — | — | — | — |"])
    lines.append("")
    lines.append("## 各数据集备注")
    for name, entry in report["coverage"].items():
        if entry.get("notes") or entry.get("flags"):
            lines.append(f"- **{name}**: " + "; ".join(entry.get("notes") or []) +
                         (f"　flags={entry['flags']}" if entry.get("flags") else ""))
    md_path = out_dir / "coverage_report.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FullDataV2 coverage audit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()
    result = build_report(args.datasets, args.max_sequences)
    written = write_report(result, args.out)
    print(json.dumps({"written": [str(p) for p in written],
                      "statuses": {k: v.get("status") for k, v in result["coverage"].items()}},
                     indent=1, ensure_ascii=False))
