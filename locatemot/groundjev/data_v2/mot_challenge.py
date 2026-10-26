"""MOTChallenge-style dataset adapter (MOT17, MOT20, DanceTrack, SportsMOT, AnimalTrack).

All five MOT datasets share the same on-disk format::

    <root>/<split>/<sequence>/img1/*.jpg
    <root>/<split>/<sequence>/seqinfo.ini
    <root>/<split>/<sequence>/gt/gt.txt      (train/val only)

The adapter therefore parses format, not dataset identity, and records the
dataset-specific caveats (half-resolution MOT17/20, three detector copies per
MOT17 video, SportsMOT challenge non-compliance) in the returned metadata.
"""
from __future__ import annotations

import configparser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .registry import DatasetSpec, load_frozen_protocol
from .schema import AnnotationQuality, IdentityDecision, TrainingRecord

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def read_seqinfo(seq_dir: Path) -> dict[str, Any]:
    info = seq_dir / "seqinfo.ini"
    if not info.is_file():
        return {}
    parser = configparser.ConfigParser()
    parser.read(info)
    section = parser["Sequence"] if "Sequence" in parser else {}
    out: dict[str, Any] = {}
    for key in ("name", "imDir", "frameRate", "seqLength", "imWidth", "imHeight"):
        if key in section:
            value = section[key].strip()
            out[key] = float(value) if key == "frameRate" else (
                int(value) if value.isdigit() else value)
    return out


def image_dir(seq_dir: Path) -> Path:
    for candidate in (seq_dir / "img1", seq_dir / "image_02", seq_dir / "img"):
        if candidate.is_dir():
            return candidate
    return seq_dir / "img1"


def list_images(directory: Path) -> list[Path]:
    """List real frame images, skipping macOS AppleDouble resource forks.

    Several archives (SportsMOT) were packed on macOS and contain a ``._name.jpg``
    sidecar for every real image; counting those doubles every image count.
    """
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
        and not p.name.startswith("._")
        and not p.name.startswith(".")
    )


@dataclass
class GtBox:
    frame: int
    identity: int
    box: tuple[float, float, float, float]
    confidence: float
    object_class: int
    visibility: float


def parse_mot_gt(gt_path: Path, *, classes: Optional[set[int]] = None,
                 min_confidence: float = 0.0) -> list[GtBox]:
    """Parse a MOTChallenge gt.txt.

    MOT17/20 ``gt.txt`` columns: frame,id,x,y,w,h,conf,class,visibility.
    DanceTrack/SportsMOT use the same layout with class always 1.
    """
    boxes: list[GtBox] = []
    if not gt_path.is_file():
        return boxes
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        try:
            frame = int(float(parts[0]))
            identity = int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
        except ValueError:
            continue
        confidence = float(parts[6]) if len(parts) > 6 and parts[6] else 1.0
        object_class = int(float(parts[7])) if len(parts) > 7 and parts[7] else 1
        visibility = float(parts[8]) if len(parts) > 8 and parts[8] else 1.0
        if classes is not None and object_class not in classes:
            continue
        if confidence < min_confidence:
            continue
        boxes.append(GtBox(frame, identity, (x, y, w, h), confidence, object_class, visibility))
    return boxes


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class SequenceInventory:
    sequence: str
    split: str
    image_dir: str
    gt_path: Optional[str]
    frames_on_disk: int
    frames_in_gt: int
    gt_boxes: int
    identities: int
    births: int
    identity_gaps: int            # number of >1-frame absences per identity
    reappearances: int            # identities that return after a gap of >=10 frames
    empty_gt_frames: int
    duplicate_pairs: int          # same-frame pairs of different ids with IoU>=0.5
    long_occlusions: int          # identities with a >=30 frame unreported span
    mean_visibility: float
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def inventory_sequence(seq_dir: Path, split: str, *, gap_for_reappear: int = 10,
                       gap_for_long_occlusion: int = 30,
                       class_filter: Optional[set[int]] = None) -> SequenceInventory:
    images = list_images(image_dir(seq_dir))
    info = read_seqinfo(seq_dir)
    gt_path = seq_dir / "gt" / "gt.txt"
    boxes = parse_mot_gt(gt_path, classes=class_filter)
    per_frame: dict[int, list[GtBox]] = defaultdict(list)
    per_identity: dict[int, list[int]] = defaultdict(list)
    for box in boxes:
        per_frame[box.frame].append(box)
        per_identity[box.identity].append(box.frame)

    gaps = 0
    reappearances = 0
    long_occlusions = 0
    for frames in per_identity.values():
        frames.sort()
        for previous, current in zip(frames, frames[1:]):
            delta = current - previous
            if delta > 1:
                gaps += 1
                if delta >= gap_for_reappear:
                    reappearances += 1
                if delta >= gap_for_long_occlusion:
                    long_occlusions += 1

    duplicates = 0
    for group in per_frame.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i].identity != group[j].identity and \
                        _iou(group[i].box, group[j].box) >= 0.5:
                    duplicates += 1

    gt_frames = set(per_frame)
    on_disk = set()
    for image in images:
        stem = image.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            on_disk.add(int(digits))
    empty_gt_frames = len(on_disk - gt_frames) if on_disk else 0
    visibility = [box.visibility for box in boxes]

    notes: list[str] = []
    if gt_path.is_file() and on_disk and gt_frames - on_disk:
        notes.append(f"{len(gt_frames - on_disk)} gt frames have no image on disk")

    return SequenceInventory(
        sequence=seq_dir.name,
        split=split,
        image_dir=str(image_dir(seq_dir)),
        gt_path=str(gt_path) if gt_path.is_file() else None,
        frames_on_disk=len(images),
        frames_in_gt=len(gt_frames),
        gt_boxes=len(boxes),
        identities=len(per_identity),
        births=len(per_identity),
        identity_gaps=gaps,
        reappearances=reappearances,
        empty_gt_frames=empty_gt_frames,
        duplicate_pairs=duplicates,
        long_occlusions=long_occlusions,
        mean_visibility=round(sum(visibility) / len(visibility), 4) if visibility else 0.0,
        fps=info.get("frameRate"),
        width=info.get("imWidth"),
        height=info.get("imHeight"),
        notes=notes,
    )


def discover_sequences(root: Path, split_dirs: dict[str, str]) -> list[tuple[Path, str]]:
    """Return (sequence_dir, role) pairs for every sequence directory found."""
    found: list[tuple[Path, str]] = []
    for split_dir, role in split_dirs.items():
        container = root / split_dir
        if not container.is_dir():
            # some roots already are the split container itself
            if root.is_dir() and any((root / sub).is_dir() for sub in ("img1", "gt")):
                found.append((root, role))
            continue
        for sequence in sorted(container.iterdir()):
            if sequence.is_dir() and (sequence / "img1").is_dir():
                found.append((sequence, role))
    if not found and root.is_dir() and (root / "img1").is_dir():
        found.append((root, "train"))
    return found


def detector_family(sequence_name: str) -> str:
    """MOT17 train stores three detector copies of every video."""
    for suffix in ("-DPM", "-FRCNN", "-SDP"):
        if sequence_name.endswith(suffix):
            return suffix.lstrip("-").lower()
    return "single"


def iter_records(spec: DatasetSpec, *, max_frames_per_sequence: Optional[int] = None,
                 class_filter: Optional[set[int]] = None) -> Iterator[TrainingRecord]:
    """Yield per-GT-box training records for a MOT-style dataset."""
    root = spec.resolved_root()
    if root is None:
        raise FileNotFoundError(f"{spec.name}: no root available")
    for seq_dir, role in discover_sequences(root, spec.splits):
        if role.startswith("test"):
            continue
        info = read_seqinfo(seq_dir)
        fps = float(info.get("frameRate") or 30.0)
        for box in parse_mot_gt(seq_dir / "gt" / "gt.txt", classes=class_filter):
            if max_frames_per_sequence is not None and box.frame > max_frames_per_sequence:
                continue
            yield TrainingRecord(
                dataset_name=spec.name,
                source_video_id=seq_dir.name,
                canonical_video_uid=f"{spec.name}:{seq_dir.name}",
                split=role,
                frame_index=box.frame,
                timestamp=(box.frame - 1) / fps,
                physical_identity=box.identity,
                physical_identity_valid=True,
                identity_decision=IdentityDecision.MATCH.value,
                annotation_quality=AnnotationQuality.HUMAN_EXHAUSTIVE.value,
                annotation_exhaustive=True,
                identity_match_label_valid=True,
                birth_label_valid=True,
                memory_write_label_valid=True,
                candidate_duplicate_label_valid=True,
                source_annotation_path=str(seq_dir / "gt" / "gt.txt"),
                extra={"detector": detector_family(seq_dir.name),
                       "visibility": box.visibility, "confidence": box.confidence,
                       "object_class": box.object_class},
            )
