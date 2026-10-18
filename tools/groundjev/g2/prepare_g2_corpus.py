#!/usr/bin/env python3
"""Build the legal TRAIN-only GroundAnything G2 row corpus.

Rows are emitted in the official ``id/messages/images/source`` schema and keep
media as absolute paths (no duplicated image bytes).  The builder uses the
official GroundAnything protocol for every answer and validates each row
before it is written.  It intentionally records incomplete sources rather
than manufacturing media or silently reading a held-out split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "external/GroundAnything"
sys.path[:0] = [str(OFFICIAL), "/data2/user/LocateMOT"]

from grounding_anything.protocol import (  # noqa: E402
    GroundingEntry,
    build_dense_bbox_prompt,
    build_refer_bbox_prompt,
    canonicalize_phrase,
    serialize_answer,
    validate_prompt_answer_pair,
)
from train.data.row_contract import RecordReject, validate_uniform_row  # noqa: E402


OUT_DEFAULT = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/"
    "G2_PROJECTOR_QWEN3_LORA_FULLDATA"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TAO_FRAME_ROOTS = [Path(f"/data{i}/user/TAO-Amodal/frames") for i in range(4)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    seen_dirs: set[tuple[int, int]] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            stat = os.stat(directory, follow_symlinks=True)
        except OSError:
            continue
        key = (int(stat.st_dev), int(stat.st_ino))
        if key in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(key)
        dirnames[:] = sorted(name for name in dirnames if not name.startswith(".") and name != "__MACOSX")
        for name in sorted(filenames):
            if name.startswith(".") or name.startswith("._"):
                continue
            path = Path(directory) / name
            if path.is_file():
                yield path


def parse_seqinfo(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    section = False
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line.lower() == "[sequence]"
            continue
        if section and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip()
    return values


def image_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": None}


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def clipped_xyxy(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(0.0, min(x2, float(width - 1)))
    y2 = max(0.0, min(y2, float(height - 1)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def grid_box(box: Iterable[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    clipped = clipped_xyxy(box, width, height)
    if clipped is None:
        return None
    values = []
    for index, value in enumerate(clipped):
        denominator = max(1, width - 1 if index % 2 == 0 else height - 1)
        values.append(max(0, min(999, round(value / denominator * 999))))
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return tuple(values)  # type: ignore[return-value]


def make_row(
    *,
    row_id: str,
    source: str,
    path: Path,
    phrase: str,
    boxes: Iterable[Iterable[float]],
    family: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    phrase = canonicalize_phrase(phrase)
    coordinates = tuple(
        result
        for result in (grid_box(box, width, height) for box in boxes)
        if result is not None
    )
    answer = serialize_answer([GroundingEntry(phrase, coordinates or None)], "bbox")
    if family == "dense":
        prompt = build_dense_bbox_prompt([phrase])
        prompt_family = "dense_bbox"
    elif family == "referring":
        prompt = build_refer_bbox_prompt(phrase)
        prompt_family = "refer_bbox"
    else:
        raise ValueError(family)
    validate_prompt_answer_pair(prompt, answer, prompt_family)
    row = {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "<image>\n" + prompt, "loss": None},
            {"role": "assistant", "content": answer, "loss": None},
        ],
        "images": [image_record(path)],
        "source": source,
    }
    validate_uniform_row(row)
    return row


def standard_sequence_paths(root: Path) -> list[Path]:
    paths = []
    seen: set[str] = set()
    for directory, _, filenames in os.walk(root, followlinks=True):
        if "seqinfo.ini" not in filenames:
            continue
        raw_path = Path(directory)
        parts = {part.lower() for part in raw_path.parts}
        if not ("train" in parts or "train1" in parts):
            continue
        path = raw_path.resolve()
        key = str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return sorted(paths)


def parse_mot_gt(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    if not path.is_file():
        return by_frame
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            continue
        try:
            frame, track = int(float(fields[0])), int(float(fields[1]))
            x, y, width, height = [float(value) for value in fields[2:6]]
        except ValueError:
            continue
        if track < 0 or width <= 0 or height <= 0:
            continue
        by_frame[frame].append((x, y, x + width, y + height))
    return by_frame


def add_standard_dataset(
    *,
    name: str,
    root: Path,
    rows: list[dict[str, Any]],
    counts: Counter,
    rejects: list[dict[str, Any]],
) -> None:
    for sequence in standard_sequence_paths(root):
        info = parse_seqinfo(sequence / "seqinfo.ini")
        image_dir = next((sequence / candidate for candidate in ("img1", "image_02") if (sequence / candidate).is_dir()), None)
        if image_dir is None:
            rejects.append({"dataset": name, "sequence": sequence.name, "reason": "missing_image_dir"})
            continue
        image_paths = sorted(
            (path for path in iter_files(image_dir) if path.suffix.lower() in IMAGE_EXTS),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
        )
        gt = parse_mot_gt(sequence / "gt" / "gt.txt")
        width = int(info.get("imwidth", 0) or 0)
        height = int(info.get("imheight", 0) or 0)
        if not width or not height:
            width, height = image_size(image_paths[0])
        for path in image_paths:
            frame = int(path.stem) if path.stem.isdigit() else None
            if frame is None:
                rejects.append({"dataset": name, "sequence": sequence.name, "path": str(path), "reason": "non_numeric_frame"})
                continue
            try:
                rows.append(make_row(row_id=f"{name.lower()}_{sequence.name}_{frame:06d}_dense", source=f"{name}:train", path=path, phrase="person", boxes=gt.get(frame, ()), family="dense", width=width, height=height))
                counts[f"{name}.dense"] += 1
            except Exception as error:
                rejects.append({"dataset": name, "sequence": sequence.name, "frame": frame, "reason": type(error).__name__, "detail": str(error)[:300]})


def parse_yolo(path: Path, width: int, height: int) -> dict[int, tuple[float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            track = int(fields[1])
            cx, cy, bw, bh = [float(value) for value in fields[2:6]]
        except ValueError:
            continue
        result[track] = ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)
    return result


def add_aerial(root: Path, rows: list[dict[str, Any]], counts: Counter, rejects: list[dict[str, Any]]) -> None:
    image_root, label_root, expression_root = root / "image_02", root / "labels_with_ids", root / "expression"
    for sequence in sorted(path for path in image_root.iterdir() if path.is_dir()):
        image_paths = sorted(path for path in iter_files(sequence) if path.suffix.lower() in IMAGE_EXTS)
        if not image_paths:
            continue
        width, height = image_size(image_paths[0])
        by_stem = {path.stem: path for path in image_paths}
        for path in image_paths:
            boxes = parse_yolo(label_root / sequence.name / f"{path.stem}.txt", width, height)
            try:
                rows.append(make_row(row_id=f"aerialmind_{sequence.name}_{path.stem}_dense", source="AerialMind:train", path=path, phrase="object", boxes=boxes.values(), family="dense", width=width, height=height))
                counts["AerialMind.dense"] += 1
            except Exception as error:
                rejects.append({"dataset": "AerialMind", "sequence": sequence.name, "path": str(path), "reason": type(error).__name__, "detail": str(error)[:300]})
        for expression in sorted((expression_root / sequence.name).glob("*.json")) if (expression_root / sequence.name).is_dir() else []:
            if expression.name.startswith("."):
                continue
            try:
                record = json.loads(expression.read_text())
                phrase = canonicalize_phrase(record.get("sentence", "object"))
                label = record.get("label", {})
                selected = None
                for frame_key in sorted(label, key=lambda value: int(value) if str(value).isdigit() else str(value)):
                    frame_path = by_stem.get(str(frame_key).zfill(7)) or by_stem.get(str(frame_key).zfill(6)) or by_stem.get(str(frame_key))
                    ids = [int(value) for value in label.get(frame_key, [])]
                    if frame_path is not None and ids:
                        selected = (frame_path, ids)
                        break
                if selected is None:
                    raise ValueError("expression has no labeled TRAIN frame")
                frame_path, ids = selected
                width_i, height_i = image_size(frame_path)
                boxes = parse_yolo(label_root / sequence.name / f"{frame_path.stem}.txt", width_i, height_i)
                target_boxes = [boxes[index] for index in ids if index in boxes]
                if not target_boxes:
                    raise ValueError("expression IDs are absent from frame labels")
                rows.append(make_row(row_id=f"aerialmind_{sequence.name}_{expression.stem}_ref", source="AerialMind:train:expression", path=frame_path, phrase=phrase, boxes=target_boxes, family="referring", width=width_i, height=height_i))
                counts["AerialMind.referring"] += 1
            except Exception as error:
                rejects.append({"dataset": "AerialMind", "expression": str(expression), "reason": type(error).__name__, "detail": str(error)[:300]})


def add_animaltrack(root: Path, rows: list[dict[str, Any]], counts: Counter, rejects: list[dict[str, Any]]) -> None:
    split = root / "splits" / "videos_train.txt"
    if not split.is_file():
        return
    for line in split.read_text().splitlines():
        video = line.strip()
        if not video:
            continue
        stem = Path(video).stem
        frame_dir, gt_path = root / "frames_all" / stem, root / "gt_all" / f"{stem}_gt.txt"
        frame_paths = sorted(path for path in iter_files(frame_dir) if path.suffix.lower() in IMAGE_EXTS)
        if not frame_paths or not gt_path.is_file():
            rejects.append({"dataset": "AnimalTrack", "video": video, "reason": "missing_train_frames_or_gt"})
            continue
        width, height = image_size(frame_paths[0])
        gt = parse_mot_gt(gt_path)
        phrase = stem.rsplit("_", 1)[0]
        for path in frame_paths:
            frame = int(path.stem) if path.stem.isdigit() else None
            if frame is None:
                continue
            try:
                rows.append(make_row(row_id=f"animaltrack_{stem}_{frame:06d}_dense", source="AnimalTrack:train", path=path, phrase=phrase, boxes=gt.get(frame, ()), family="dense", width=width, height=height))
                counts["AnimalTrack.dense"] += 1
            except Exception as error:
                rejects.append({"dataset": "AnimalTrack", "video": video, "frame": frame, "reason": type(error).__name__, "detail": str(error)[:300]})


def add_tao(annotation_root: Path, rows: list[dict[str, Any]], counts: Counter, rejects: list[dict[str, Any]]) -> None:
    annotation_path = annotation_root / "train.json"
    data = json.loads(annotation_path.read_text())
    categories = {int(item["id"]): item.get("name", "object") for item in data.get("categories", [])}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in data.get("annotations", []):
        by_image[int(item["image_id"])].append(item)
    for image in data.get("images", []):
        relative = image["file_name"]
        path = next((root / relative for root in TAO_FRAME_ROOTS if (root / relative).is_file()), None)
        if path is None:
            rejects.append({"dataset": "TAO-Amodal", "image_id": image.get("id"), "reason": "missing_train_image", "file_name": relative})
            continue
        annotations = by_image.get(int(image["id"]), [])
        if not annotations:
            boxes, phrase = [], "object"
        else:
            chosen = None
            for candidate in annotations:
                try:
                    phrase = canonicalize_phrase(str(categories.get(int(candidate["category_id"]), "object")))
                    chosen = [item for item in annotations if int(item["category_id"]) == int(candidate["category_id"])]
                    break
                except Exception:
                    continue
            if chosen is None:
                phrase, chosen = "object", annotations
            boxes = []
            for item in chosen:
                x, y, width, height = [float(value) for value in item["bbox"]]
                boxes.append((x, y, x + width, y + height))
        try:
            rows.append(make_row(row_id=f"tao_amodal_{int(image['id']):07d}_dense", source="TAO-Amodal:train", path=path, phrase=phrase, boxes=boxes, family="dense", width=int(image["width"]), height=int(image["height"])))
            counts["TAO-Amodal.dense"] += 1
        except Exception as error:
            rejects.append({"dataset": "TAO-Amodal", "image_id": image.get("id"), "reason": type(error).__name__, "detail": str(error)[:300]})


def add_refer_kitti(root: Path, image_root: Path, rows: list[dict[str, Any]], counts: Counter, rejects: list[dict[str, Any]]) -> None:
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS

    fit = set(LEGAL_SPLITS["refer_kitti_v2"]["fit"])
    for video in sorted(fit):
        expression_dir, labels_dir, images_dir = root / "expression" / video, root / "labels_with_ids" / "image_02" / video, image_root / video
        for expression in sorted(expression_dir.glob("*.json")):
            try:
                record = json.loads(expression.read_text())
                if record.get("ignore"):
                    continue
                phrase = canonicalize_phrase(record["sentence"])
                label_map = record.get("label", {})
                selected = None
                for key in sorted(label_map, key=lambda value: int(value) if str(value).isdigit() else str(value)):
                    frame = int(key)
                    image_path = next((images_dir / f"{frame:06d}{ext}" for ext in (".png", ".jpg", ".jpeg") if (images_dir / f"{frame:06d}{ext}").is_file()), None)
                    ids = {int(value) for value in label_map.get(key, [])}
                    if image_path is not None and ids:
                        selected = (frame, image_path, ids)
                        break
                if selected is None:
                    raise ValueError("expression has no labeled legal TRAIN frame")
                frame, image_path, ids = selected
                width, height = image_size(image_path)
                boxes = {}
                label_file = labels_dir / f"{frame:06d}.txt"
                if label_file.is_file():
                    for line in label_file.read_text(errors="replace").splitlines():
                        fields = line.split()
                        if len(fields) < 6:
                            continue
                        try:
                            track = int(fields[1]); x, y, bw, bh = [float(value) for value in fields[2:6]]
                        except ValueError:
                            continue
                        boxes[track] = (x * width, y * height, (x + bw) * width, (y + bh) * height)
                target = [boxes[track] for track in ids if track in boxes]
                if not target:
                    raise ValueError("expression IDs are absent from legal TRAIN labels")
                rows.append(make_row(row_id=f"refer_kitti_v2_{video}_{expression.stem}_ref", source="Refer-KITTI-V2:train:expression", path=image_path, phrase=phrase, boxes=target, family="referring", width=width, height=height))
                counts["Refer-KITTI-V2.referring"] += 1
            except Exception as error:
                rejects.append({"dataset": "Refer-KITTI-V2", "video": video, "expression": str(expression), "reason": type(error).__name__, "detail": str(error)[:300]})


def add_rmot26(root: Path, rows: list[dict[str, Any]], counts: Counter, rejects: list[dict[str, Any]]) -> None:
    annotation = root / "train" / "annotations.json"
    if not annotation.is_file():
        return
    for record in json.loads(annotation.read_text()):
        try:
            objects = record.get("objects", [])
            if not objects:
                raise ValueError("row has no target object")
            relative = record.get("ref_image") or record.get("images", [None])[0]
            path = root / "train" / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            width, height = image_size(path)
            x, y, bw, bh = [float(value) for value in objects[0]["initial_bbox"]]
            rows.append(make_row(row_id=f"rmot26_{record['id']}_ref", source="RMOT26:train", path=path, phrase="the selected target", boxes=[(x, y, x + bw, y + bh)], family="referring", width=width, height=height))
            counts["RMOT26.referring"] += 1
        except Exception as error:
            rejects.append({"dataset": "RMOT26", "id": record.get("id"), "reason": type(error).__name__, "detail": str(error)[:300]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "g2_train_rows.jsonl"
    rejects_path = args.out_dir / "g2_train_rejects.jsonl"
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    counts: Counter = Counter()
    add_standard_dataset(name="MOT17", root=Path("/data2/user/LocateMOT/data/MOT17").resolve(), rows=rows, counts=counts, rejects=rejects)
    add_standard_dataset(name="MOT20", root=Path("/data2/user/LocateMOT/data/MOT20").resolve(), rows=rows, counts=counts, rejects=rejects)
    add_standard_dataset(name="DanceTrack", root=Path("/data2/user/LocateMOT/data/DanceTrack").resolve(), rows=rows, counts=counts, rejects=rejects)
    add_standard_dataset(name="SportsMOT", root=Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT").resolve(), rows=rows, counts=counts, rejects=rejects)
    add_animaltrack(Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/AnimalTrack").resolve(), rows, counts, rejects)
    add_refer_kitti(Path("/data2/user/LocateMOT/data/refer_kitti_v2").resolve(), Path("/data2/user/LocateMOT/data/kitti_tracking_training/image_02").resolve(), rows, counts, rejects)
    add_aerial(Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/AerialMind").resolve(), rows, counts, rejects)
    add_rmot26(Path("/data3/user/LocateMOT_FullDataV2/downloads/RMOT26").resolve(), rows, counts, rejects)
    add_tao(Path("/data2/user/LocateMOT/data/TAO-Amodal/annotations").resolve(), rows, counts, rejects)

    # BURST is intentionally absent from rows: its TRAIN annotation payload is
    # present but this checkout has no legal TRAIN image tree.  The coverage
    # audit records this as an explicit data gap; using TAO test frames would
    # violate the held-out boundary.
    seen_ids: set[str] = set()
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in seen_ids:
                raise RuntimeError(f"duplicate G2 row id: {row['id']}")
            seen_ids.add(row["id"])
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with rejects_path.open("w", encoding="utf-8") as handle:
        for item in rejects:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "status": "G2_TRAIN_CORPUS_PREPARED_WITH_EXPLICIT_SOURCE_GAPS",
        "rows_path": str(rows_path),
        "rows_sha256": sha256(rows_path),
        "rejects_path": str(rejects_path),
        "rejects_sha256": sha256(rejects_path),
        "row_count": len(rows),
        "reject_count": len(rejects),
        "source_counts": dict(sorted(counts.items())),
        "datasets_in_rows": sorted({row["source"].split(":", 1)[0] for row in rows}),
        "datasets_missing_or_excluded": {
            "BURST": "TRAIN annotations available but no legal TRAIN image root; excluded to prevent TAO test leakage",
        },
        "media_policy": "absolute paths, bytes=null, online decode by official DLM collator",
        "split_policy": "TRAIN only; Refer-KITTI-V2 uses fit sequences only; no official evaluation videos opened",
        "coverage_epoch": "g2-coverage-audit-20261019-v1",
    }
    manifest_path = args.out_dir / "g2_train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
