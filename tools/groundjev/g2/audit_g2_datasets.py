#!/usr/bin/env python3
"""Audit the ten G2 data sources without opening held-out media.

The first environment audit used ``Path.rglob`` and therefore under-counted
symlinked trees.  This receipt follows the resolved TRAIN roots, records
annotation/image intersections, and explicitly distinguishes complete data
from annotation-only or partially downloaded sources.  It never copies,
decodes, or modifies a dataset.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OUT_DEFAULT = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/"
    "G2_PROJECTOR_QWEN3_LORA_FULLDATA/g2_dataset_coverage_audit.json"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt", ".json", ".jsonl"}

DATASETS = {
    "MOT17": {"root": Path("/data2/user/LocateMOT/data/MOT17").resolve()},
    "MOT20": {"root": Path("/data2/user/LocateMOT/data/MOT20").resolve()},
    "DanceTrack": {"root": Path("/data2/user/LocateMOT/data/DanceTrack").resolve()},
    "SportsMOT": {
        "root": Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT").resolve()
    },
    "AnimalTrack": {
        "root": Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/AnimalTrack").resolve()
    },
    "Refer-KITTI-V2": {
        "root": Path("/data2/user/LocateMOT/data/refer_kitti_v2").resolve(),
        "image_root": Path("/data2/user/LocateMOT/data/kitti_tracking_training/image_02").resolve(),
    },
    "AerialMind": {
        "root": Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/AerialMind").resolve()
    },
    "RMOT26": {
        "root": Path("/data3/user/LocateMOT_FullDataV2/downloads/RMOT26").resolve()
    },
    "TAO-Amodal": {
        "root": Path("/data2/user/TAO-Amodal").resolve(),
        "annotation_root": Path("/data2/user/LocateMOT/data/TAO-Amodal/annotations").resolve(),
        # The user's TAO extraction is deliberately distributed across the
        # four data volumes.  Resolve every file against these roots instead
        # of assuming one volume contains all 500 videos.
        "frame_roots": [
            Path(f"/data{i}/user/TAO-Amodal/frames") for i in range(4)
        ],
    },
    "BURST": {
        "root": Path("/data3/user/LocateMOT_FullDataV2/datasets/BURST").resolve()
    },
}


def iter_files(root: Path) -> Iterable[Path]:
    """Yield real files while following the dataset's symlinked trees."""

    if not root.is_dir():
        return
    seen_dirs: set[tuple[int, int]] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.stat(directory, follow_symlinks=True)
        key = (int(real.st_dev), int(real.st_ino))
        if key in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(key)
        dirnames[:] = sorted(
            name for name in dirnames if not name.startswith(".") and name != "__MACOSX"
        )
        for name in sorted(filenames):
            if name.startswith(".") or name.startswith("._"):
                continue
            path = Path(directory) / name
            if path.is_file():
                yield path


def basic_stats(root: Path) -> dict[str, Any]:
    counts = Counter()
    examples: dict[str, list[str]] = {}
    total = 0
    for path in iter_files(root):
        total += 1
        ext = path.suffix.lower() or "<none>"
        counts[ext] += 1
        examples.setdefault(ext, []).append(str(path))
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "total_files": total,
        "image_files": sum(counts[e] for e in IMAGE_EXTS),
        "extension_counts": dict(sorted(counts.items())),
        "examples": {key: value[:3] for key, value in sorted(examples.items())},
    }


def parse_seqinfo(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(path)
    values = dict(parser.defaults())
    for section in parser.sections():
        values.update({key: value for key, value in parser.items(section)})
    return values


def standard_sequences(root: Path) -> dict[str, Any]:
    sequences = []
    seqinfo_paths: list[Path] = []
    for directory, _, filenames in os.walk(root, followlinks=True):
        if "seqinfo.ini" in filenames:
            seqinfo_paths.append(Path(directory) / "seqinfo.ini")
    for seqinfo in sorted(seqinfo_paths):
        relative = seqinfo.relative_to(root)
        # Only the declared TRAIN tree is legal for G2.  This also prevents a
        # MOT20 test directory from entering the row manifest by accident.
        if "train" not in {part.lower() for part in relative.parts}:
            continue
        seq = seqinfo.parent
        image_dir = next((seq / name for name in ("img1", "image_02") if (seq / name).is_dir()), None)
        gt = seq / "gt" / "gt.txt"
        image_files = [p for p in iter_files(image_dir)] if image_dir else []
        image_files = [p for p in image_files if p.suffix.lower() in IMAGE_EXTS]
        label_lines = 0
        frame_ids: set[int] = set()
        track_ids: set[int] = set()
        if gt.is_file():
            for line in gt.read_text(errors="replace").splitlines():
                fields = line.split(",")
                if len(fields) >= 6:
                    label_lines += 1
                    try:
                        frame_ids.add(int(fields[0]))
                        track_ids.add(int(fields[1]))
                    except ValueError:
                        pass
        sequences.append(
            {
                "sequence": seq.name,
                "path": str(seq),
                "images": len(image_files),
                "gt_file": str(gt),
                "gt_exists": gt.is_file(),
                "gt_rows": label_lines,
                "gt_frames": len(frame_ids),
                "gt_tracks": len(track_ids),
                "seqinfo": parse_seqinfo(seqinfo),
            }
        )
    return {
        "train_sequences": len(sequences),
        "image_frames": sum(item["images"] for item in sequences),
        "gt_rows": sum(item["gt_rows"] for item in sequences),
        "sequences": sequences,
    }


def animaltrack(root: Path) -> dict[str, Any]:
    split = root / "splits" / "videos_train.txt"
    train_names = [line.strip() for line in split.read_text().splitlines() if line.strip()] if split.is_file() else []
    records = []
    for name in train_names:
        stem = Path(name).stem
        frames = root / "frames_all" / stem
        gt = root / "gt_all" / f"{stem}_gt.txt"
        images = [p for p in iter_files(frames) if p.suffix.lower() in IMAGE_EXTS]
        rows = sum(1 for line in gt.read_text(errors="replace").splitlines() if line.strip()) if gt.is_file() else 0
        records.append({"video": name, "frames": len(images), "gt_file": str(gt), "gt_exists": gt.is_file(), "gt_rows": rows})
    return {
        "train_videos": len(records),
        "train_frames": sum(item["frames"] for item in records),
        "gt_rows": sum(item["gt_rows"] for item in records),
        "videos": records,
        "source_media": "decoded TRAIN frames are present; test split is not opened",
    }


def refer_kitti(root: Path, image_root: Path) -> dict[str, Any]:
    try:
        from locatemot.unified.data.legal_scope import LEGAL_SPLITS

        legal_fit = list(LEGAL_SPLITS["refer_kitti_v2"]["fit"])
        calibration = list(LEGAL_SPLITS["refer_kitti_v2"]["calibration"])
        validation = list(LEGAL_SPLITS["refer_kitti_v2"]["validation"])
        reserved = list(LEGAL_SPLITS["refer_kitti_v2"]["official_eval_reserved"])
    except Exception:
        legal_fit, calibration, validation, reserved = [], [], [], []
    records = []
    for video in sorted(p.name for p in (root / "labels_with_ids" / "image_02").iterdir() if p.is_dir()):
        label_dir = root / "labels_with_ids" / "image_02" / video
        image_dir = image_root / video
        labels = sorted(label_dir.glob("*.txt"))
        images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
        records.append({
            "video": video,
            "scope": "fit" if video in legal_fit else "calibration" if video in calibration else "validation" if video in validation else "official_eval_reserved" if video in reserved else "unregistered",
            "images": len(images),
            "labels": len(labels),
            "image_label_intersection": len({p.stem for p in images} & {p.stem for p in labels}),
            "expressions": len(list((root / "expression" / video).glob("*.json"))),
        })
    return {"videos": len(records), "legal_fit": legal_fit, "calibration": calibration, "validation": validation, "official_eval_reserved": reserved, "records": records}


def aerial(root: Path) -> dict[str, Any]:
    image_root = root / "image_02"
    label_root = root / "labels_with_ids"
    expression_root = root / "expression"
    records = []
    for image_dir in sorted(p for p in image_root.iterdir() if p.is_dir()) if image_root.is_dir() else []:
        name = image_dir.name
        images = [p for p in iter_files(image_dir) if p.suffix.lower() in IMAGE_EXTS]
        labels = [p for p in iter_files(label_root / name) if p.suffix.lower() == ".txt"] if (label_root / name).is_dir() else []
        expressions = [p for p in iter_files(expression_root / name) if p.suffix.lower() == ".json"] if (expression_root / name).is_dir() else []
        records.append({"sequence": name, "images": len(images), "labels": len(labels), "image_label_intersection": len({p.stem for p in images} & {p.stem for p in labels}), "expressions": len(expressions)})
    return {"sequences": len(records), "images": sum(x["images"] for x in records), "labels": sum(x["labels"] for x in records), "expressions": sum(x["expressions"] for x in records), "records": records}


def tao(annotation_root: Path, frame_roots: list[Path]) -> dict[str, Any]:
    annotation = annotation_root / "train.json"
    if not annotation.is_file():
        return {"annotation_exists": False, "frame_root": str(frame_root)}
    data = json.loads(annotation.read_text())
    image_records = data.get("images", [])
    existing = 0
    resolved_by_root = Counter()
    missing: list[str] = []
    for image in image_records:
        path = None
        root_used = None
        for candidate in frame_roots:
            candidate_path = candidate / image["file_name"]
            if candidate_path.is_file():
                path = candidate_path
                root_used = candidate
                break
        if path is not None:
            existing += 1
            resolved_by_root[str(root_used)] += 1
        elif len(missing) < 20:
            missing.append(image["file_name"])
    return {"annotation_exists": True, "annotation": str(annotation), "frame_roots": [str(path) for path in frame_roots], "videos": len(data.get("videos", [])), "images": len(image_records), "annotations": len(data.get("annotations", [])), "tracks": len(data.get("tracks", [])), "categories": len(data.get("categories", [])), "existing_train_images": existing, "resolved_by_root": dict(resolved_by_root), "missing_examples": missing, "train_media_complete": existing == len(image_records)}


def rmot26(root: Path) -> dict[str, Any]:
    annotation = root / "train" / "annotations.json"
    rows = json.loads(annotation.read_text()) if annotation.is_file() else []
    existing = 0
    referenced = 0
    missing: list[str] = []
    for row in rows:
        paths = row.get("images", row.get("input_images", []))
        if row.get("ref_image"):
            paths = [row["ref_image"], *paths]
        for relative in paths:
            referenced += 1
            path = root / "train" / relative
            if path.is_file():
                existing += 1
            elif len(missing) < 20:
                missing.append(str(path))
    failed_manifest = root / ".failed.txt"
    failed_entries = [line.strip() for line in failed_manifest.read_text(errors="replace").splitlines() if line.strip()] if failed_manifest.is_file() else []
    return {"annotation_exists": annotation.is_file(), "annotation": str(annotation), "rows": len(rows), "referenced_images": referenced, "existing_images": existing, "missing_images": referenced - existing, "missing_examples": missing, "failed_manifest": str(failed_manifest) if failed_manifest.is_file() else None, "failed_manifest_entries": len(failed_entries), "failed_examples": failed_entries[:20], "train_media_complete": referenced == existing and not failed_entries, "status": "TRAIN_MEDIA_PARTIAL_FAILED_DOWNLOADS" if failed_entries else "TRAIN_MEDIA_COMPLETE"}


def burst(root: Path) -> dict[str, Any]:
    annotation = root / "annotations" / "train" / "train.json"
    data = json.loads(annotation.read_text()) if annotation.is_file() else {}
    sequences = data.get("sequences", [])
    referenced = sum(len(item.get("all_image_paths", [])) for item in sequences)
    # The annotation payload carries paths but this checkout has no legal TRAIN
    # image tree.  Do not substitute TAO test frames, which would be leakage.
    return {"annotation_exists": annotation.is_file(), "annotation": str(annotation), "sequences": len(sequences), "referenced_train_images": referenced, "train_image_root_found": False, "train_media_complete": False, "status": "ANNOTATIONS_ONLY_NO_LEGAL_TRAIN_MEDIA"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    receipt: dict[str, Any] = {
        "status": "G2_TEN_DATASET_COVERAGE_AUDIT_COMPLETE",
        "legal_policy": "TRAIN only; no official evaluation videos, validation/test media, or annotation-only rows enter the formal corpus",
        "datasets": {},
    }
    for name, spec in DATASETS.items():
        root = spec["root"]
        item: dict[str, Any] = {"root": str(root), "root_exists": root.is_dir(), "basic": basic_stats(root)}
        if name in {"MOT17", "MOT20", "DanceTrack", "SportsMOT"}:
            item["standard_sequences"] = standard_sequences(root)
        elif name == "AnimalTrack":
            item["animaltrack"] = animaltrack(root)
        elif name == "Refer-KITTI-V2":
            item["refer_kitti"] = refer_kitti(root, spec["image_root"])
        elif name == "AerialMind":
            item["aerial"] = aerial(root)
        elif name == "RMOT26":
            item["rmot26"] = rmot26(root)
        elif name == "TAO-Amodal":
            item["tao"] = tao(spec["annotation_root"], spec["frame_roots"])
        elif name == "BURST":
            item["burst"] = burst(root)
        receipt["datasets"][name] = item
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": receipt["status"], "out": str(args.out), "datasets": {name: item.get("standard_sequences", item.get("animaltrack", item.get("aerial", item.get("rmot26", item.get("tao", item.get("burst", {})))))) for name, item in receipt["datasets"].items()}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
