"""RMOT adapters: Refer-KITTI-V2, RMOT26 (QTrack), AerialMind.

Three very different referring-expression datasets must not be flattened into a
single loose format.  Each adapter reports exactly which supervision it owns:

* Refer-KITTI-V2: per-video expression JSON with ``label`` = {frame: [ids]},
  legal split registry in ``locatemot/unified/data/legal_scope.py``.
* RMOT26: query-driven short clips; ``objects`` gives ``track_id`` and the
  initial box, ``expected_output`` gives the target trajectory.  Objects that
  are not query targets carry **no** identity supervision.
* AerialMind: one JSON per expression, where the file stem (minus ``.json``) is
  the official query text and expression id.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .registry import DatasetSpec
from .schema import AnnotationQuality, IdentityDecision, TrainingRecord

SOURCE_PREFIX_DATASETS = {
    "MOT17": "mot17",
    "MOT20": "mot20",
    "dancetrack": "dancetrack",
}


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmot26_source_sequence(sample_id: str) -> str:
    """``MOT17-10-SDP_000001_mot_multiple`` -> ``MOT17-10-SDP``."""
    parts = sample_id.split("_")
    return parts[0] if parts else sample_id


def rmot26_source_family(sample_id: str) -> str:
    sequence = rmot26_source_sequence(sample_id)
    for prefix, name in SOURCE_PREFIX_DATASETS.items():
        if sequence.startswith(prefix):
            return name
    if sequence.lower().startswith("dancetrack"):
        return "dancetrack"
    return "external"


@dataclass
class QueryInventory:
    dataset: str
    videos: int
    expressions: int
    target_identities: int
    positive_frames: int
    negative_frames: int
    frames_with_no_target: int
    sources: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    sample_structure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def inventory_refer_kitti(spec: DatasetSpec) -> QueryInventory:
    """Count the legal Refer-KITTI-V2 fit/calibration/validation scope only."""
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS

    key = "refer_kitti_v2" if "refer_kitti_v2" in LEGAL_SPLITS else "refer_kitti_v1"
    splits = LEGAL_SPLITS[key]
    root = spec.resolved_root()
    if root is None:
        raise FileNotFoundError("refer_kitti_v2 root missing")
    expression_root = root / "expression"
    videos = 0
    expressions = 0
    identities: set[tuple[str, int]] = set()
    positive = 0
    negative = 0
    empty_frames = 0
    notes: list[str] = []
    for part in ("fit", "calibration", "validation"):
        for video in splits[part]:
            directory = expression_root / video
            if not directory.is_dir():
                notes.append(f"missing expression directory for video {video}")
                continue
            videos += 1
            for path in sorted(directory.glob("*.json")):
                payload = json.loads(path.read_text())
                label = payload.get("label") or {}
                expressions += 1
                if not label:
                    empty_frames += 1
                for _frame, ids in label.items():
                    if ids:
                        positive += 1
                        for identity in ids:
                            identities.add((video, int(identity)))
                    else:
                        negative += 1
    return QueryInventory(
        dataset=spec.name,
        videos=videos,
        expressions=expressions,
        target_identities=len(identities),
        positive_frames=positive,
        negative_frames=negative,
        frames_with_no_target=empty_frames,
        sources={key: videos},
        notes=notes + ["official evaluation videos 0005/0011/0013/0019 stay closed"],
        sample_structure={"label": "dict frame->[ids]", "ignore": "list of frames",
                          "sentence": "expression text", "raw_sentence": "source text"},
    )


def load_rmot26(annotation_path: Path) -> list[dict[str, Any]]:
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    payload = json.loads(annotation_path.read_text())
    if not isinstance(payload, list):
        raise ValueError("unexpected RMOT26 annotation layout (expected a list)")
    return payload


def inventory_rmot26(annotation_path: Path) -> QueryInventory:
    samples = load_rmot26(annotation_path)
    sources: Counter[str] = Counter()
    identities: set[tuple[str, int]] = set()
    positive = 0
    empty = 0
    for sample in samples:
        sequence = rmot26_source_sequence(sample["id"])
        family = rmot26_source_family(sample["id"])
        sources[family] += 1
        objects = sample.get("objects") or []
        if not objects:
            empty += 1
        for obj in objects:
            identities.add((sequence, int(obj["track_id"])))
        try:
            expected = json.loads(sample.get("expected_output") or "{}")
        except json.JSONDecodeError:
            expected = {}
        entries: list[dict[str, Any]] = []
        if isinstance(expected, dict):
            entries = [v for v in expected.values() if isinstance(v, dict)]
        elif isinstance(expected, list):
            entries = [v for v in expected if isinstance(v, dict)]
            if not entries and expected:
                # some rows store a flat list of {frame,bbox} for one object
                entries = [{"trajectory": expected}]
        for entry in entries:
            if entry.get("trajectory"):
                positive += len(entry["trajectory"])
    return QueryInventory(
        dataset="rmot26",
        videos=len({rmot26_source_sequence(s["id"]) for s in samples}),
        expressions=len(samples),
        target_identities=len(identities),
        positive_frames=positive,
        negative_frames=0,
        frames_with_no_target=empty,
        sources=dict(sources),
        notes=[
            "query-driven clips only: non-target objects have no identity supervision",
            "source sequences are reused from MOT17/MOT20/DanceTrack/SportsMOT-family data",
            "these videos must NOT be counted as new independent videos",
        ],
        sample_structure={
            "id": "sourceSequence_frame_tasktype",
            "objects": "[{object_id, track_id, initial_bbox}]",
            "expected_output": "{object_i: {track_id, trajectory:[{frame,bbox}]}}",
            "question": "natural-language tracking query",
        },
    )


def inventory_aerialmind(spec: DatasetSpec) -> QueryInventory:
    root = spec.resolved_root()
    if root is None:
        raise FileNotFoundError("AerialMind root missing")
    expression_root = root / "expression"
    videos = 0
    expressions = 0
    identities: set[tuple[str, int]] = set()
    positive = 0
    negative = 0
    notes: list[str] = []
    if not expression_root.is_dir():
        return QueryInventory("aerialmind", 0, 0, 0, 0, 0, 0, notes=["expression/ not extracted yet"])
    for video_dir in sorted(p for p in expression_root.iterdir() if p.is_dir()):
        videos += 1
        for path in sorted(video_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            label = payload.get("label") or payload.get("labels") or {}
            expressions += 1
            for _frame, ids in label.items():
                if ids:
                    positive += 1
                    for identity in ids:
                        identities.add((video_dir.name, int(identity)))
                else:
                    negative += 1
    return QueryInventory(
        dataset=spec.name, videos=videos, expressions=expressions,
        target_identities=len(identities), positive_frames=positive,
        negative_frames=negative, frames_with_no_target=0, notes=notes + [
            "official query text and expression id come from the JSON filename stem",
            "the sentence field is source-annotation text and must not replace it",
        ],
        sample_structure={"official_query_text": "expression JSON filename stem",
                          "sentence_field": "source annotation text, not the official query"},
    )


def iter_refer_kitti_records(spec: DatasetSpec, video: str, expression_path: Path) -> Iterator[TrainingRecord]:
    payload = json.loads(Path(expression_path).read_text())
    query_text = payload.get("sentence") or Path(expression_path).stem
    query_id = f"{video}:{Path(expression_path).stem}"
    label = payload.get("label") or {}
    for frame, ids in sorted(label.items(), key=lambda kv: int(kv[0])):
        frame_index = int(frame)
        if ids:
            for identity in ids:
                yield TrainingRecord(
                    dataset_name=spec.name,
                    source_video_id=video,
                    canonical_video_uid=f"{spec.name}:{video}",
                    split="fit",
                    frame_index=frame_index,
                    timestamp=float(frame_index) / 10.0,
                    physical_identity=int(identity),
                    physical_identity_valid=True,
                    identity_decision=IdentityDecision.MATCH.value,
                    query_text=query_text,
                    query_id=query_id,
                    query_target_ids=[int(i) for i in ids],
                    query_supervision_valid=True,
                    annotation_quality=AnnotationQuality.HUMAN_EXHAUSTIVE.value,
                    annotation_exhaustive=True,
                    identity_match_label_valid=True,
                    source_annotation_path=str(expression_path),
                )
