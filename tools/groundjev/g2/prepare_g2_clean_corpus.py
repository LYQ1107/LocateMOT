#!/usr/bin/env python3
"""Prepare the corrected G2 clean-coverage corpus.

The original G2 corpus is immutable evidence.  This builder creates a new
version with the frozen MOT20-05 holdout removed, MOT validity/class flags
respected, TAO Base categories expanded per image, all legal Refer-KITTI
expression frames retained, and the original RMOT26 query text preserved.
BURST remains an explicit source gap until a legal TRAIN image root exists.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/GroundAnything"))
sys.path.insert(0, "/data2/user/LocateMOT")

from grounding_anything.protocol import canonicalize_phrase  # noqa: E402
from locatemot.unified.data.legal_scope import LEGAL_SPLITS  # noqa: E402


def load_legacy_builder():
    source = Path(__file__).with_name("prepare_g2_corpus.py")
    spec = importlib.util.spec_from_file_location("locatemot_g2_legacy_builder", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy row builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_legacy_builder()
IMAGE_EXTS = BASE.IMAGE_EXTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_query(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("Question:", "", 1)).strip()


def parse_mot_valid(path: Path) -> tuple[dict[int, list[tuple[float, float, float, float]]], collections.Counter[str]]:
    by_frame: dict[int, list[tuple[float, float, float, float]]] = collections.defaultdict(list)
    audit: collections.Counter[str] = collections.Counter()
    if not path.is_file():
        return by_frame, audit
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            audit["malformed"] += 1
            continue
        try:
            frame = int(float(fields[0]))
            mark = int(float(fields[6])) if len(fields) > 6 else 1
            category = int(float(fields[7])) if len(fields) > 7 else 1
            x, y, width, height = [float(value) for value in fields[2:6]]
        except ValueError:
            audit["malformed"] += 1
            continue
        audit["raw"] += 1
        if mark != 1:
            audit["ignored_mark"] += 1
            continue
        if category != 1:
            audit[f"excluded_class_{category}"] += 1
            continue
        if width <= 0 or height <= 0:
            audit["invalid_box"] += 1
            continue
        by_frame[frame].append((x, y, x + width, y + height))
        audit["valid_person"] += 1
    return by_frame, audit


def standard_sequences(root: Path) -> list[Path]:
    return list(BASE.standard_sequence_paths(root))


def add_mot_clean(
    *,
    name: str,
    root: Path,
    excluded_sequences: set[str],
    rows: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    counts: collections.Counter[str],
    coverage: dict[str, set[str]],
) -> None:
    for sequence in standard_sequences(root):
        if sequence.name in excluded_sequences:
            rejects.append({"dataset": name, "sequence": sequence.name, "reason": "frozen_holdout_sequence"})
            continue
        info = BASE.parse_seqinfo(sequence / "seqinfo.ini")
        image_dir = next((sequence / candidate for candidate in ("img1", "image_02") if (sequence / candidate).is_dir()), None)
        if image_dir is None:
            rejects.append({"dataset": name, "sequence": sequence.name, "reason": "missing_image_dir"})
            continue
        image_paths = sorted(
            (path for path in BASE.iter_files(image_dir) if path.suffix.lower() in IMAGE_EXTS),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
        )
        if not image_paths:
            rejects.append({"dataset": name, "sequence": sequence.name, "reason": "empty_image_dir"})
            continue
        width = int(info.get("imwidth", 0) or 0)
        height = int(info.get("imheight", 0) or 0)
        if not width or not height:
            width, height = BASE.image_size(image_paths[0])
        gt, gt_audit = parse_mot_valid(sequence / "gt" / "gt.txt")
        counts[f"{name}.raw_gt_rows"] += sum(gt_audit.values())
        counts[f"{name}.valid_person_boxes"] += gt_audit["valid_person"]
        counts[f"{name}.ignored_or_nonperson_boxes"] += sum(value for key, value in gt_audit.items() if key.startswith("ignored_") or key.startswith("excluded_class_"))
        for path in image_paths:
            if not path.stem.isdigit():
                rejects.append({"dataset": name, "sequence": sequence.name, "path": str(path), "reason": "non_numeric_frame"})
                continue
            frame = int(path.stem)
            try:
                row = BASE.make_row(
                    row_id=f"{name.lower()}_{sequence.name}_{frame:06d}_dense_clean_v2",
                    source=f"{name}:train:clean_v2",
                    path=path,
                    phrase="person",
                    boxes=gt.get(frame, ()),
                    family="dense",
                    width=width,
                    height=height,
                )
                rows.append(row)
                counts[f"{name}.dense"] += 1
                counts[f"{name}.empty_valid_frame"] += int(not gt.get(frame))
                coverage.setdefault(name, set()).add(sequence.name)
            except Exception as error:
                rejects.append({"dataset": name, "sequence": sequence.name, "frame": frame, "reason": type(error).__name__, "detail": str(error)[:300]})


def add_tao_clean(
    annotation_root: Path,
    rows: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    counts: collections.Counter[str],
    coverage: dict[str, set[str]],
) -> None:
    data = json.loads((annotation_root / "train.json").read_text())
    categories = {int(item["id"]): item for item in data.get("categories", [])}
    by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in data.get("annotations", []):
        by_image[int(item["image_id"])].append(item)
    novel_annotations = 0
    base_rows = 0
    for image in data.get("images", []):
        image_id = int(image["id"])
        relative = image["file_name"]
        path = next((root / relative for root in BASE.TAO_FRAME_ROOTS if (root / relative).is_file()), None)
        if path is None:
            rejects.append({"dataset": "TAO-Amodal", "image_id": image_id, "reason": "missing_train_image", "file_name": relative})
            continue
        grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for annotation in by_image.get(image_id, []):
            category_id = int(annotation["category_id"])
            category = categories.get(category_id, {})
            if category.get("frequency") == "r":
                novel_annotations += 1
                continue
            if int(annotation.get("iscrowd", 0)) != 0:
                counts["TAO-Amodal.crowd_excluded"] += 1
                continue
            grouped[category_id].append(annotation)
        for category_id, annotations in sorted(grouped.items()):
            category = categories.get(category_id, {})
            try:
                phrase = canonicalize_phrase(str(category.get("name", "object")))
                boxes = []
                for annotation in annotations:
                    x, y, width, height = [float(value) for value in annotation["bbox"]]
                    if width > 0 and height > 0:
                        boxes.append((x, y, x + width, y + height))
                if not boxes:
                    raise ValueError("category has no valid non-crowd boxes")
                row = BASE.make_row(
                    row_id=f"tao_amodal_{image_id:07d}_cat{category_id:04d}_dense_clean_v2",
                    source="TAO-Amodal:train:base_clean_v2",
                    path=path,
                    phrase=phrase,
                    boxes=boxes,
                    family="dense",
                    width=int(image["width"]),
                    height=int(image["height"]),
                )
                rows.append(row)
                base_rows += 1
                counts["TAO-Amodal.dense_base"] += 1
                coverage.setdefault("TAO-Amodal", set()).add(str(image_id))
            except Exception as error:
                rejects.append({"dataset": "TAO-Amodal", "image_id": image_id, "category_id": category_id, "reason": type(error).__name__, "detail": str(error)[:300]})
    counts["TAO-Amodal.novel_annotations_excluded"] += novel_annotations
    counts["TAO-Amodal.base_category_rows"] += base_rows


def add_animaltrack_clean(
    root: Path,
    rows: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    counts: collections.Counter[str],
    coverage: dict[str, set[str]],
) -> None:
    split = root / "splits" / "videos_train.txt"
    if not split.is_file():
        rejects.append({"dataset": "AnimalTrack", "reason": "missing_train_split"})
        return
    for line in split.read_text().splitlines():
        video = line.strip()
        if not video:
            continue
        stem = Path(video).stem
        frame_dir = root / "frames_all" / stem
        gt_path = root / "gt_all" / f"{stem}_gt.txt"
        frame_paths = sorted(path for path in BASE.iter_files(frame_dir) if path.suffix.lower() in IMAGE_EXTS)
        if not frame_paths or not gt_path.is_file():
            rejects.append({"dataset": "AnimalTrack", "video": video, "reason": "missing_train_frames_or_gt"})
            continue
        width, height = BASE.image_size(frame_paths[0])
        gt, gt_audit = parse_mot_valid(gt_path)
        counts["AnimalTrack.raw_gt_rows"] += sum(gt_audit.values())
        for path in frame_paths:
            if not path.stem.isdigit():
                rejects.append({"dataset": "AnimalTrack", "video": video, "path": str(path), "reason": "non_numeric_frame"})
                continue
            frame = int(path.stem)
            try:
                row = BASE.make_row(
                    row_id=f"animaltrack_{stem}_{frame:06d}_dense_clean_v2",
                    source="AnimalTrack:train:clean_v2",
                    path=path,
                    phrase=stem.rsplit("_", 1)[0],
                    boxes=gt.get(frame, ()),
                    family="dense",
                    width=width,
                    height=height,
                )
                rows.append(row)
                counts["AnimalTrack.dense"] += 1
                coverage.setdefault("AnimalTrack", set()).add(stem)
            except Exception as error:
                rejects.append({"dataset": "AnimalTrack", "video": video, "frame": frame, "reason": type(error).__name__, "detail": str(error)[:300]})


def add_refer_clean(
    root: Path,
    image_root: Path,
    rows: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    counts: collections.Counter[str],
    coverage: dict[str, set[str]],
) -> None:
    for video in sorted(LEGAL_SPLITS["refer_kitti_v2"]["fit"]):
        expression_dir = root / "expression" / video
        labels_dir = root / "labels_with_ids" / "image_02" / video
        images_dir = image_root / video
        for expression in sorted(expression_dir.glob("*.json")):
            try:
                record = json.loads(expression.read_text())
                if record.get("ignore"):
                    continue
                phrase = canonicalize_phrase(str(record["sentence"]))
                label_map = record.get("label", {})
                emitted = 0
                for key in sorted(label_map, key=lambda value: int(value) if str(value).isdigit() else str(value)):
                    frame = int(key)
                    image_path = next((images_dir / f"{frame:06d}{ext}" for ext in (".png", ".jpg", ".jpeg") if (images_dir / f"{frame:06d}{ext}").is_file()), None)
                    ids = {int(value) for value in label_map.get(key, [])}
                    if image_path is None or not ids:
                        continue
                    width, height = BASE.image_size(image_path)
                    boxes_by_id: dict[int, tuple[float, float, float, float]] = {}
                    label_file = labels_dir / f"{frame:06d}.txt"
                    if label_file.is_file():
                        for line in label_file.read_text(errors="replace").splitlines():
                            fields = line.split()
                            if len(fields) < 6:
                                continue
                            try:
                                track = int(fields[1])
                                x, y, box_width, box_height = [float(value) for value in fields[2:6]]
                            except ValueError:
                                continue
                            boxes_by_id[track] = (x * width, y * height, (x + box_width) * width, (y + box_height) * height)
                    target = [boxes_by_id[track] for track in sorted(ids) if track in boxes_by_id]
                    if not target:
                        continue
                    row = BASE.make_row(
                        row_id=f"refer_kitti_v2_{video}_{expression.stem}_{frame:06d}_ref_clean_v2",
                        source="Refer-KITTI-V2:train:expression_clean_v2",
                        path=image_path,
                        phrase=phrase,
                        boxes=target,
                        family="referring",
                        width=width,
                        height=height,
                    )
                    rows.append(row)
                    emitted += 1
                    coverage.setdefault("Refer-KITTI-V2", set()).add(video)
                if emitted == 0:
                    raise ValueError("expression has no valid legal labeled frame")
                counts["Refer-KITTI-V2.referring"] += emitted
                counts["Refer-KITTI-V2.expressions"] += 1
            except Exception as error:
                rejects.append({"dataset": "Refer-KITTI-V2", "video": video, "expression": str(expression), "reason": type(error).__name__, "detail": str(error)[:300]})


def add_rmot26_clean(
    root: Path,
    rows: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    counts: collections.Counter[str],
    coverage: dict[str, set[str]],
) -> None:
    annotation = root / "train" / "annotations.json"
    if not annotation.is_file():
        rejects.append({"dataset": "RMOT26", "reason": "missing_train_annotations"})
        return
    for record in json.loads(annotation.read_text()):
        try:
            record_id = str(record["id"])
            if "MOT20-05" in record_id or "MOT20-05" in str(record.get("ref_image", "")):
                rejects.append({"dataset": "RMOT26", "id": record_id, "reason": "frozen_MOT20_05_holdout_overlap"})
                continue
            objects = record.get("objects", [])
            if not objects:
                raise ValueError("row has no target objects")
            relative = record.get("ref_image") or record.get("images", [None])[0]
            path = root / "train" / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            width, height = BASE.image_size(path)
            boxes = []
            for obj in objects:
                x, y, box_width, box_height = [float(value) for value in obj["initial_bbox"]]
                boxes.append((x, y, x + box_width, y + box_height))
            phrase = canonicalize_phrase(normalized_query(str(record["question"])))
            row = BASE.make_row(
                row_id=f"rmot26_{record_id}_ref_clean_v2",
                source="RMOT26:train:expression_clean_v2",
                path=path,
                phrase=phrase,
                boxes=boxes,
                family="referring",
                width=width,
                height=height,
            )
            rows.append(row)
            counts["RMOT26.referring"] += 1
            counts["RMOT26.target_objects"] += len(boxes)
            coverage_key = Path(str(relative)).parent.as_posix() or record_id
            coverage.setdefault("RMOT26", set()).add(coverage_key)
        except Exception as error:
            rejects.append({"dataset": "RMOT26", "id": record.get("id"), "reason": type(error).__name__, "detail": str(error)[:300]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=ROOT / "models/GroundAnything-DLM")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite clean corpus: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    coverage: dict[str, set[str]] = {}
    add_mot_clean(name="MOT17", root=Path("/data2/user/LocateMOT/data/MOT17").resolve(), excluded_sequences=set(), rows=rows, rejects=rejects, counts=counts, coverage=coverage)
    add_mot_clean(name="MOT20", root=Path("/data2/user/LocateMOT/data/MOT20").resolve(), excluded_sequences={"MOT20-05"}, rows=rows, rejects=rejects, counts=counts, coverage=coverage)
    add_mot_clean(name="DanceTrack", root=Path("/data2/user/LocateMOT/data/DanceTrack").resolve(), excluded_sequences=set(), rows=rows, rejects=rejects, counts=counts, coverage=coverage)
    add_mot_clean(name="SportsMOT", root=Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT").resolve(), excluded_sequences=set(), rows=rows, rejects=rejects, counts=counts, coverage=coverage)
    add_animaltrack_clean(Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/AnimalTrack").resolve(), rows, rejects, counts, coverage)
    add_refer_clean(Path("/data2/user/LocateMOT/data/refer_kitti_v2").resolve(), Path("/data2/user/LocateMOT/data/kitti_tracking_training/image_02").resolve(), rows, rejects, counts, coverage)
    # AerialMind labels already carry the official expression text and track IDs;
    # retain that adapter while registering the corrected split policy here.
    BASE.add_aerial(Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/AerialMind").resolve(), rows, counts, rejects)
    add_rmot26_clean(Path("/data3/user/LocateMOT_FullDataV2/downloads/RMOT26").resolve(), rows, rejects, counts, coverage)
    add_tao_clean(Path("/data2/user/LocateMOT/data/TAO-Amodal/annotations").resolve(), rows, rejects, counts, coverage)

    row_path = args.out_dir / "g2_clean_train_rows.jsonl"
    reject_path = args.out_dir / "g2_clean_train_rejects.jsonl"
    seen: set[str] = set()
    with row_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in seen:
                raise RuntimeError(f"duplicate clean row id: {row['id']}")
            seen.add(row["id"])
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with reject_path.open("w", encoding="utf-8") as handle:
        for item in rejects:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "status": "G2_CLEAN_FULL_COVERAGE_V2_PREPARED_WITH_EXPLICIT_SOURCE_GAPS",
        "rows_path": str(row_path.resolve()),
        "rows_sha256": sha256(row_path),
        "rejects_path": str(reject_path.resolve()),
        "rejects_sha256": sha256(reject_path),
        "row_count": len(rows),
        "reject_count": len(rejects),
        "source_counts": dict(sorted(counts.items())),
        "coverage_videos_or_sequences": {key: sorted(value) for key, value in sorted(coverage.items())},
        "datasets_in_rows": sorted({row["source"].split(":", 1)[0] for row in rows}),
        "datasets_missing_or_excluded": {
            "BURST": "TRAIN annotations exist but no legal TRAIN image root; excluded",
            "MOT20-05": "frozen held-out MOT20 long-export sequence; excluded from all clean sources including RMOT26 copies",
        },
        "supervision_policy": {
            "MOT": "mark==1, class==1, positive boxes; ignored/distractor classes are not person positives",
            "TAO-Amodal": "frequency!=r Base categories only, one row per annotated category, no sparse negative labels",
            "Refer-KITTI-V2": "legal fit videos, one row per valid labeled frame per expression",
            "RMOT26": "normalized exact source question and all initial target boxes; no placeholder phrase",
        },
        "media_policy": "absolute paths, bytes=null, online decode by official DLM collator",
        "split_policy": "TRAIN only; MOT20-05 excluded; Refer-KITTI-V2 fit only; TAO Base-only; no evaluation labels",
        "coverage_epoch": "g2-clean-coverage-v2-20261019",
        "source_audit": str(args.audit.resolve()),
        "base_model_safetensors_sha256": sha256(args.base_checkpoint / "model.safetensors"),
    }
    manifest_path = args.out_dir / "g2_clean_train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
