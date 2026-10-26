"""OVMOT adapters: TAO-Amodal and BURST.

TAO provides box/amodal annotations over 500 train videos; BURST extends a
subset of the same videos with pixel masks.  The two must never be counted as
independent video sources, and BURST annotations are not class-exhaustive.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .registry import DatasetSpec

VOCAB_PATH = Path("/data2/user/LocateMOT/outputs/unified/u2/protocol/lvis_v1_vocabulary.json")


@dataclass
class CategoryInventory:
    dataset: str
    videos: int
    images: int
    annotations: int
    tracks: int
    categories_present: int
    base_categories_present: int
    novel_categories_present: int
    images_resolvable_on_disk: int
    annotation_frame_identity_pairs: int
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_vocabulary() -> dict[str, Any]:
    return json.loads(VOCAB_PATH.read_text())


def base_novel_sets(vocabulary: dict[str, Any]) -> tuple[set[int], set[int]]:
    base, novel = set(), set()
    for category in vocabulary["categories"]:
        if category.get("frequency") in ("f", "c"):
            base.add(int(category["id"]))
        else:
            novel.add(int(category["id"]))
    return base, novel


def inventory_tao(spec: DatasetSpec, frame_roots: Optional[list[Path]] = None) -> CategoryInventory:
    annotation_path = Path("/data2/user/TAO-Amodal/annotations/train_with_freeform.json")
    payload = json.loads(annotation_path.read_text())
    vocabulary = load_vocabulary()
    base, novel = base_novel_sets(vocabulary)
    roots = frame_roots or [Path("/data2/user/TAO-Amodal/frames"),
                            Path("/data3/user/TAO-Amodal/frames"),
                            Path("/data0/user/TAO-Amodal/frames"),
                            Path("/data1/user/TAO-Amodal/frames")]
    images = payload.get("images") or []
    resolvable = 0
    for image in images:
        relative = image["file_name"]
        if any((root / relative).is_file() for root in roots):
            resolvable += 1
    category_ids = {int(a["category_id"]) for a in payload.get("annotations") or []}
    track_frames: Counter[int] = Counter()
    for annotation in payload.get("annotations") or []:
        track_frames[int(annotation["track_id"])] += 1
    return CategoryInventory(
        dataset=spec.name,
        videos=len(payload.get("videos") or []),
        images=len(images),
        annotations=len(payload.get("annotations") or []),
        tracks=len(payload.get("tracks") or []),
        categories_present=len(category_ids),
        base_categories_present=len(category_ids & base),
        novel_categories_present=len(category_ids & novel),
        images_resolvable_on_disk=resolvable,
        annotation_frame_identity_pairs=sum(1 for _ in track_frames),
        notes=[
            f"input vocabulary {len(vocabulary['categories'])} categories "
            f"({len(base)} base / {len(novel)} novel)",
            "non-Base categories are not converted into negatives",
        ],
        detail={"frame_roots": [str(r) for r in roots],
                "tracks_with_frames": len(track_frames)},
    )


def decode_rle_bbox(rle: str, size: list[int]) -> Optional[list[float]]:
    """Decode a COCO compressed RLE mask and return its xywh bbox.

    Boxes derived from masks are *mask-derived* supervision; they must never be
    reported as official BURST box scores.
    """
    try:
        import numpy as np
        from pycocotools import mask as mask_utils
    except Exception:
        return None
    counts = rle.encode("ascii") if isinstance(rle, str) else rle
    decoded = mask_utils.decode({"counts": counts, "size": [int(size[0]), int(size[1])]})
    ys, xs = np.nonzero(decoded)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def _scan_official_burst(spec: DatasetSpec, train_json: Path) -> CategoryInventory:
    payload = json.loads(train_json.read_text())
    info_dir = train_json.parent.parent / "info"
    class_split = json.loads((info_dir / "class_split.json").read_text()) if (info_dir / "class_split.json").is_file() else {}
    common = set(class_split.get("common") or [])
    uncommon = set(class_split.get("uncommon") or [])
    sequences = payload.get("sequences") or []
    images = 0
    tracks: set[tuple[str, int]] = set()
    masks = 0
    gt_masks = 0
    auto_masks = 0
    categories: set[int] = set()
    sample_bbox = None
    for sequence in sequences:
        images += len(sequence.get("all_image_paths") or [])
        track_categories = sequence.get("track_category_ids") or {}
        for track_id in track_categories:
            tracks.add((sequence.get("seq_name", ""), int(track_id)))
        categories.update(int(c) for c in track_categories.values())
        seg_per_image = sequence.get("segmentations") or []
        for frame_entry in seg_per_image:
            if not isinstance(frame_entry, dict):
                continue
            for _track, mask in frame_entry.items():
                if not isinstance(mask, dict):
                    continue
                masks += 1
                if mask.get("is_gt"):
                    gt_masks += 1
                else:
                    auto_masks += 1
                if sample_bbox is None and mask.get("rle"):
                    sample_bbox = decode_rle_bbox(mask["rle"],
                                                  [sequence.get("height"), sequence.get("width")])
    base_categories = categories & common
    return CategoryInventory(
        dataset=spec.name,
        videos=len(sequences),
        images=images,
        annotations=masks,
        tracks=len(tracks),
        categories_present=len(categories),
        base_categories_present=len(base_categories),
        novel_categories_present=len(categories & uncommon),
        images_resolvable_on_disk=0,
        annotation_frame_identity_pairs=masks,
        notes=[
            f"official train.json: {masks} masks, {gt_masks} is_gt=True, {auto_masks} auto-generated",
            "masks are COCO compressed RLE; boxes can be derived from them (mask-derived, not official box scores)",
            "annotations are not class-exhaustive; neg_category_ids / not_exhaustive_category_ids must be honoured",
            "BURST TRAIN videos are a subset of TAO sources (never independent videos)",
        ],
        flags=["SHARES_VIDEOS_WITH_TAO", "ANNOTATION_NOT_CLASS_EXHAUSTIVE", "MASK_DERIVED_BOXES"],
        detail={"train_json": str(train_json),
                "split_field": payload.get("split"),
                "sample_mask_bbox_xywh": sample_bbox,
                "common_classes": len(common), "uncommon_classes": len(uncommon)},
    )


def inventory_burst(spec: DatasetSpec) -> CategoryInventory:
    """Probe whichever BURST annotation copy is present (official zip or visibility json)."""
    candidates = {
        "official_zip": Path("/data3/user/LocateMOT_FullDataV2/downloads/BURST/annotations.zip"),
        "historical_train_visibility": Path("/data2/user/TAO-Amodal/BURST_annotations/train/train_visibility.json"),
        "historical_train_all_classes": Path("/data2/user/TAO-Amodal/BURST_annotations/train/all_classes.json"),
    }
    official_train = Path("/data3/user/LocateMOT_FullDataV2/datasets/BURST/annotations/train/train.json")
    if official_train.is_file():
        return _scan_official_burst(spec, official_train)
    present = {name: path for name, path in candidates.items() if path.is_file()}
    notes: list[str] = []
    detail: dict[str, Any] = {"present_copies": {k: str(v) for k, v in present.items()}}
    sequences = 0
    images = 0
    tracks = 0
    if "historical_train_visibility" in present:
        payload = json.loads(present["historical_train_visibility"].read_text())
        seqs = payload.get("sequences") or []
        sequences = len(seqs)
        images = sum(len(s.get("all_image_paths") or []) for s in seqs)
        track_ids: set[tuple[str, int]] = set()
        for sequence in seqs:
            for annotation in sequence.get("annotations") or []:
                if "track_id" in annotation:
                    track_ids.add((sequence.get("seq_name", ""), int(annotation["track_id"])))
        tracks = len(track_ids)
        notes.append("historical copy only contains the visibility variant")
        source_datasets = Counter(s.get("dataset") for s in seqs)
        detail["source_datasets"] = dict(source_datasets)
    if "official_zip" in present and "official" not in " ".join(notes):
        notes.append("official annotations.zip present but not yet extracted at this probe")
    return CategoryInventory(
        dataset=spec.name, videos=sequences, images=images, annotations=0, tracks=tracks,
        categories_present=0, base_categories_present=0, novel_categories_present=0,
        images_resolvable_on_disk=0, annotation_frame_identity_pairs=0,
        notes=notes + [
            "BURST TRAIN videos are a subset of TAO sources (never independent videos)",
            "annotations are not class-exhaustive; mask-derived boxes are not official BURST scores",
        ],
        flags=["SHARES_VIDEOS_WITH_TAO", "ANNOTATION_NOT_CLASS_EXHAUSTIVE"],
        detail=detail,
    )
