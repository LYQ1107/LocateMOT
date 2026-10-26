"""Canonical registry of the ten named FullDataV2 datasets.

Every path here is *resolved at runtime* against a list of candidates, because
the historical project stores the same dataset in more than one place (for
example DanceTrack lives in ``data/downloads/DanceTrack/extracted`` while the
frozen protocol still points at the now-empty ``data/DanceTrack/train``).

The registry is the only place allowed to know physical paths.  Adapters ask
:func:`dataset` and :func:`resolve` and must never hard-code a root.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DATA_V2_ROOT = Path("/data3/user/LocateMOT_FullDataV2")
PROJECT_DATA = Path("/data2/user/LocateMOT/data")
U2_PROTOCOL_ROOT = Path("/data2/user/LocateMOT/outputs/unified/u2/protocol")
HISTORICAL_TAO = Path("/data2/user/TAO-Amodal")
SPILL_TAO = Path("/data3/user/TAO-Amodal")


def _first_existing(candidates: Iterable[Path | str]) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


@dataclass
class DatasetSpec:
    name: str
    tasks: tuple[str, ...]
    kind: str                       # mot | rmot | ovmot
    roots: list[Path]               # candidate roots, first existing wins
    splits: dict[str, str] = field(default_factory=dict)   # split -> role
    annotation: list[Path] = field(default_factory=list)
    protocol: Optional[Path] = None
    licence: str = ""
    version: str = ""
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    required_for_training: bool = True

    def resolved_root(self) -> Optional[Path]:
        return _first_existing(self.roots)

    def availability(self) -> dict[str, Any]:
        root = self.resolved_root()
        annotations = [str(p) for p in self.annotation if Path(p).exists()]
        return {
            "name": self.name,
            "tasks": list(self.tasks),
            "kind": self.kind,
            "root": str(root) if root else None,
            "roots_checked": [str(p) for p in self.roots],
            "available": bool(root),
            "annotation_present": annotations,
            "annotation_missing": [str(p) for p in self.annotation if not Path(p).exists()],
            "protocol": str(self.protocol) if self.protocol and Path(self.protocol).exists() else None,
            "licence": self.licence,
            "version": self.version,
            "notes": self.notes,
            "flags": self.flags,
        }


DATASETS: dict[str, DatasetSpec] = {
    # ---------------------------------------------------------------- MOT
    "mot17": DatasetSpec(
        name="mot17",
        tasks=("mot",),
        kind="mot",
        roots=[PROJECT_DATA / "MOT17" / "MOT17", PROJECT_DATA / "MOT17"],
        splits={"train": "train", "test": "test_without_gt"},
        protocol=U2_PROTOCOL_ROOT / "mot17_half.json",
        licence="CC BY-NC-SA 3.0 (MOTChallenge research licence)",
        version="MOT17 official release",
        notes=[
            "half-resolution images; frozen half protocol is the training boundary",
            "test split has no public GT and is never used for training",
        ],
    ),
    "mot20": DatasetSpec(
        name="mot20",
        tasks=("mot",),
        kind="mot",
        roots=[PROJECT_DATA / "MOT20" / "MOT20", PROJECT_DATA / "MOT20"],
        splits={"train": "train", "test": "test_without_gt"},
        protocol=U2_PROTOCOL_ROOT / "mot20_half.json",
        licence="CC BY-NC-SA 3.0 (MOTChallenge research licence)",
        version="MOT20 official release",
        notes=["half-resolution images; 4 train videos", "test split has no public GT"],
    ),
    "dancetrack": DatasetSpec(
        name="dancetrack",
        tasks=("mot",),
        kind="mot",
        roots=[
            PROJECT_DATA / "downloads" / "DanceTrack" / "extracted",
            PROJECT_DATA / "DanceTrack",
        ],
        splits={"train1": "train", "train2": "train", "val": "validation"},
        protocol=U2_PROTOCOL_ROOT / "dancetrack.json",
        licence="CC BY 4.0 annotations; non-commercial research video terms",
        version="a0ba42ac690c41e9850a20e76a0b9450a6fb6a47",
        notes=[
            "frozen protocol points at data/DanceTrack/train which is now an empty placeholder",
            "real frames live in data/downloads/DanceTrack/extracted/{train1,train2,val}",
            "val split holds the registered DT1203 development scope; must stay held out of training",
        ],
    ),
    "sportsmot": DatasetSpec(
        name="sportsmot",
        tasks=("mot",),
        kind="mot",
        roots=[
            DATA_V2_ROOT / "datasets" / "SportsMOT",
            DATA_V2_ROOT / "downloads" / "SportsMOT",
        ],
        splits={"train": "train", "val": "validation", "test": "test_without_gt"},
        licence="official SportsMOT research licence; see MCG-NJU/SportsMOT",
        version="MCG-NJU/SportsMOT official release",
        notes=[
            "45 official TRAIN videos, MOTChallenge style boxes and ids",
            "official test.tar is 23 GiB and is downloaded only when official submission is intended",
        ],
        flags=["MULTI_DATASET_TRAINING_NOT_SPORTSMOT_CHALLENGE_COMPLIANT"],
    ),
    "animaltrack": DatasetSpec(
        name="animaltrack",
        tasks=("mot",),
        kind="mot",
        roots=[
            DATA_V2_ROOT / "datasets" / "AnimalTrack",
            DATA_V2_ROOT / "downloads" / "AnimalTrack",
        ],
        splits={"train": "train", "test": "test_without_gt"},
        licence="non-commercial research use only (official AnimalTrack terms)",
        version="AnimalTrack 58 sequences (32 train / 26 test)",
        notes=[
            "box annotations with per-frame track ids, no referring-expression supervision",
            "official distribution is Google Drive / Baidu; availability is recorded honestly",
        ],
    ),
    # ---------------------------------------------------------------- RMOT
    "refer_kitti_v2": DatasetSpec(
        name="refer_kitti_v2",
        tasks=("rmot",),
        kind="rmot",
        roots=[PROJECT_DATA / "refer_kitti_v2"],
        splits={"fit": "train", "calibration": "calibration", "validation": "validation",
                "official_eval_reserved": "reserved"},
        annotation=[PROJECT_DATA / "refer_kitti_v2" / "expression"],
        licence="Refer-KITTI-V2 research terms",
        version="fit-v2 legal split registry (LEGAL_SPLITS v2)",
        notes=[
            "legal split registry in locatemot/unified/data/legal_scope.py is authoritative",
            "official evaluation videos 0005/0011/0013/0019 are never opened during development",
        ],
    ),
    "rmot26": DatasetSpec(
        name="rmot26",
        tasks=("rmot",),
        kind="rmot",
        roots=[DATA_V2_ROOT / "datasets" / "RMOT26", DATA_V2_ROOT / "downloads" / "RMOT26"],
        splits={"train": "train", "test": "test_without_gt"},
        annotation=[
            DATA_V2_ROOT / "downloads" / "RMOT26" / "train" / "annotations.json",
            DATA_V2_ROOT / "downloads" / "RMOT26" / "train" / "annotations_visionreasoner.json",
        ],
        licence="MIT (per dataset card)",
        version="GAASH-Lab/RMOT26",
        notes=[
            "query-driven tracking: only the annotated query targets carry reliable identities",
            "unannotated objects must be masked as UNKNOWN, never as background",
            "official metric is QTrack-style query tracking, not plain MOT HOTA",
        ],
    ),
    "aerialmind": DatasetSpec(
        name="aerialmind",
        tasks=("rmot",),
        kind="rmot",
        roots=[DATA_V2_ROOT / "datasets" / "AerialMind", DATA_V2_ROOT / "downloads" / "AerialMind"],
        splits={"train": "train", "validation": "validation", "test": "test_without_gt"},
        annotation=[
            DATA_V2_ROOT / "downloads" / "AerialMind" / "expression.zip",
            DATA_V2_ROOT / "downloads" / "AerialMind" / "labels_with_ids.zip",
        ],
        licence="official AerialMind terms (UAV RMOT)",
        version="shawnliang0420/AerialMind",
        notes=[
            "expression JSON filename without .json is the official query text and expression id",
            "the sentence field inside the JSON is source-annotation text and must not replace it",
            "VisDrone-domain train and UAVDT cross-domain evaluation protocols stay separate",
        ],
    ),
    # --------------------------------------------------------------- OVMOT
    "tao_amodal": DatasetSpec(
        name="tao_amodal",
        tasks=("ovmot",),
        kind="ovmot",
        roots=[
            HISTORICAL_TAO / "frames",
            SPILL_TAO / "frames",
            Path("/data0/user/TAO-Amodal/frames"),
            Path("/data1/user/TAO-Amodal/frames"),
        ],
        splits={"train": "train", "val": "validation", "test": "test_without_gt"},
        annotation=[
            HISTORICAL_TAO / "annotations" / "train_with_freeform.json",
            HISTORICAL_TAO / "annotations" / "validation.json",
        ],
        licence="TAO research licence (LVIS/OpenImages/YouTube sources)",
        version="TAO-Amodal release used by the historical index",
        notes=[
            "frames are split across /data2 and /data3 by source dataset; registry merges both",
            "input vocabulary stays the frozen 1203-category list; Base/Novel split is fixed",
        ],
    ),
    "burst": DatasetSpec(
        name="burst",
        tasks=("ovmot",),
        kind="ovmot",
        roots=[
            DATA_V2_ROOT / "datasets" / "BURST",
            HISTORICAL_TAO / "BURST_annotations",
            DATA_V2_ROOT / "downloads" / "BURST",
        ],
        splits={"train": "train", "val": "validation", "test": "test_without_gt"},
        annotation=[
            DATA_V2_ROOT / "downloads" / "BURST" / "annotations.zip",
            HISTORICAL_TAO / "BURST_annotations" / "train" / "all_classes.json",
            HISTORICAL_TAO / "BURST_annotations" / "train" / "train_visibility.json",
        ],
        licence="BURST research licence (TAO-derived)",
        version="WACV'23 release",
        notes=[
            "BURST TRAIN videos are a subset of TAO sources: never count them as new independent videos",
            "annotations are not class-exhaustive; automatic masks are verified but partial",
            "mask-derived boxes are not official BURST video-segmentation scores",
        ],
        flags=["SHARES_VIDEOS_WITH_TAO", "ANNOTATION_NOT_CLASS_EXHAUSTIVE"],
    ),
}


def dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:  # pragma: no cover - registry guard
        raise KeyError(f"unknown FullDataV2 dataset {name!r}; known: {sorted(DATASETS)}") from exc


def availability_table() -> list[dict[str, Any]]:
    return [spec.availability() for spec in DATASETS.values()]


def write_availability(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "root": str(DATA_V2_ROOT),
        "schema": "full-data-v2-registry",
        "datasets": availability_table(),
    }, indent=1, ensure_ascii=False))
    return out


def load_frozen_protocol(name: str) -> dict[str, Any]:
    """Load one frozen U2 MOT protocol and rewrite stale image/gt paths."""
    spec = dataset(name)
    if spec.protocol is None or not Path(spec.protocol).exists():
        raise FileNotFoundError(f"no frozen protocol for {name}")
    protocol = json.loads(Path(spec.protocol).read_text())
    if name == "dancetrack":
        extracted = PROJECT_DATA / "downloads" / "DanceTrack" / "extracted"
        for sequence in protocol["sequences"]:
            seq = sequence["sequence"]
            for split_dir in ("train1", "train2", "val"):
                candidate = extracted / split_dir / seq
                if (candidate / "img1").is_dir():
                    sequence["image_dir"] = str(candidate / "img1")
                    sequence["gt_path"] = str(candidate / "gt" / "gt.txt")
                    break
    return protocol
