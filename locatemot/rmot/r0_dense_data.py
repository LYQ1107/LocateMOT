"""R0 dense native-frame data contract.

Frame-specific supervision is read only from the R0A safe target artifact;
this module never invokes the historical full L49 query loader. It does not
invent a union target set: missing exact frame labels remain inactive or
present-uncovered according to the authoritative frame map.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from locatemot.rmot.r0_safe_target_source import (  # noqa: E402
    R0SafeQueryRecord,
    R0SafeSourceError,
    load_safe_query_records,
)


ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
L49_DATA = ASSET_ROOT / "outputs/l49/data"
L69_ROOT = ASSET_ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L82_SPLIT = ASSET_ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L49_SOURCE = ASSET_ROOT / "locatemot/rmot/l49_data.py"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
FIT_DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
EXPECTED_FIT_ROWS = 5314
FORBIDDEN_SCOPE_VIDEOS = {"0005", "0011", "0013", "0019"}


class R0DataContractError(RuntimeError):
    """Raised when a preregistered R0 data invariant is false."""


@dataclass(frozen=True)
class R0TrainQuery:
    dataset: str
    video: str
    query_id: int
    sentence: str


@dataclass(frozen=True)
class R0FrameSupervision:
    dataset: str
    video: str
    query_id: int
    frame_id: int
    target_ids: tuple[str, ...]
    covered_target_ids: tuple[str, ...]
    visible_target_count: int
    covered_target_count: int
    positive_row_count: int
    category: str
    target_present: bool
    candidate_present: bool
    present_uncovered: bool
    partially_covered: bool
    coverage_fraction: float


@dataclass(frozen=True)
class R0TrainVideoScope:
    dataset: str
    video: str
    queries: tuple[R0TrainQuery, ...]


@dataclass
class R0FrameBatch:
    dataset: str
    video: str
    query_id: int
    frame_id: int
    sentence: str
    bank_path: str
    row_offsets: list[int]
    row_keys: list[tuple[str, str, int, int, str, int]]
    candidate_indices: list[int]
    track_ids: list[int]
    pool_ids: list[int]
    boxes: torch.Tensor
    image_size: tuple[int, int]

    @property
    def candidate_count(self) -> int:
        return len(self.row_offsets)

    @property
    def image_hw(self) -> tuple[int, int]:
        """Return image dimensions as ``(height, width)`` for geometry code."""
        width, height = self.image_size
        return int(height), int(width)


class R0BankStore:
    """Streaming native L69 row reader used by cache/audit/index tools."""

    def __init__(self) -> None:
        self._video: str | None = None
        self._path: Path | None = None
        self._blob: dict[str, Any] | None = None
        self._track_rows: dict[int, list[int]] = {}
        self._candidate_gt: list[str | None] | None = None

    @property
    def tensors(self) -> dict[str, Any]:
        if self._blob is None:
            raise RuntimeError("no L69 video loaded")
        return self._blob["tensors"]

    @property
    def track_rows(self) -> dict[int, list[int]]:
        return self._track_rows

    def load_video(self, video: str) -> None:
        if str(video) == self._video:
            return
        path, blob = load_l69_bank(str(video))
        values = [int(x) for x in blob["tensors"]["track_id"].long().tolist()]
        frames = [int(x) for x in blob["tensors"]["frame"].long().tolist()]
        label_path = path.with_suffix(".labels.json")
        if not label_path.is_file():
            raise R0DataContractError(f"missing L69 candidate sidecar: {label_path}")
        label_payload = json.loads(label_path.read_text(encoding="utf-8"))
        candidate_gt = label_payload.get("candidate_gt")
        if not isinstance(candidate_gt, list) or len(candidate_gt) != len(values):
            raise R0DataContractError(f"candidate_gt/row mismatch: {label_path}")
        normalized_gt = [None if value is None else str(value) for value in candidate_gt]
        rows: dict[int, list[int]] = defaultdict(list)
        for offset, track in enumerate(values):
            rows[track].append(offset)
        for offsets in rows.values():
            offsets.sort(key=lambda x: (frames[x], x))
        self._video, self._path, self._blob, self._track_rows, self._candidate_gt = (
            str(video), path, blob, dict(rows), normalized_gt
        )

    def build_frame(self, dataset: str, video: str, query_id: int, frame_id: int, sentence: str) -> R0FrameBatch:
        self.load_video(str(video))
        tensors = self.tensors
        frame_ids = [int(x) for x in tensors["frame_ids"].long().tolist()]
        if int(frame_id) not in frame_ids:
            raise KeyError(f"frame {frame_id} missing from {video}")
        position = frame_ids.index(int(frame_id))
        _frame, begin, end = native_frame_slice(tensors, position)
        offsets = list(range(begin, end))
        candidate_indices = [int(x) for x in tensors["candidate_index"][begin:end].tolist()]
        track_ids = [int(x) for x in tensors["track_id"][begin:end].tolist()]
        pool_ids = [int(x) for x in tensors["pool_id"][begin:end].tolist()]
        boxes = tensors["box"][begin:end].float().clone()
        width, height = [int(x) for x in self._blob["metadata"]["image_size"][:2]]
        keys = [(str(dataset), str(video), int(query_id), int(frame_id), str(self._path), int(offset)) for offset in offsets]
        if len(keys) != len(offsets) or keys != sorted(keys, key=lambda value: value[-1]):
            raise R0DataContractError(f"row key/order drift at {dataset}|{video}|{query_id}|{frame_id}")
        return R0FrameBatch(str(dataset), str(video), int(query_id), int(frame_id), str(sentence), str(self._path),
                            offsets, keys, candidate_indices, track_ids, pool_ids, boxes, (width, height))

    def attach_frame_labels(
        self,
        batch: R0FrameBatch,
        target_ids: Iterable[object],
    ) -> dict[str, Any]:
        """Attach authoritative labels after a complete native frame is assembled."""
        label_path = Path(batch.bank_path).with_suffix(".labels.json")
        candidate_gt = self._candidate_gt
        if candidate_gt is None or max(batch.row_offsets, default=-1) >= len(candidate_gt):
            raise R0DataContractError(f"candidate_gt sidecar mismatch: {label_path}")
        targets = {str(value) for value in target_ids}
        values = [None if candidate_gt[offset] is None else str(candidate_gt[offset]) for offset in batch.row_offsets]
        labels = torch.tensor([value is not None and str(value) in targets for value in values], dtype=torch.bool)
        candidate_targets = {value for value in values if value is not None}
        covered_targets = targets & candidate_targets
        visible_target_count = len(targets)
        covered_target_count = len(covered_targets)
        positive_row_count = int(labels.sum())
        target_present = visible_target_count > 0
        candidate_present = covered_target_count > 0
        if visible_target_count == 0:
            category = "inactive"
        elif covered_target_count == 0:
            category = "present_uncovered"
        elif covered_target_count == 1:
            category = "positive"
        else:
            category = "multi_positive"
        partially_covered = 0 < covered_target_count < visible_target_count
        coverage_fraction = (
            float(covered_target_count) / float(visible_target_count)
            if visible_target_count > 0 else 1.0
        )
        return {
            "labels": labels, "candidate_gt": [None if value is None else str(value) for value in values],
            "target_ids": sorted(targets), "covered_target_ids": sorted(covered_targets),
            "visible_target_count": visible_target_count,
            "covered_target_count": covered_target_count,
            "positive_row_count": positive_row_count,
            "positive_count": positive_row_count,
            "category": category, "target_present": target_present,
            "candidate_present": candidate_present, "present_uncovered": bool(target_present and not candidate_present),
            "partially_covered": partially_covered,
            "coverage_fraction": coverage_fraction,
            "membership_mask": torch.full_like(labels, not (target_present and not candidate_present)),
            "labels_attached_after_feature_construction": True,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.exists() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_fit_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(L49_DATA / "train_units.jsonl")
    if len(rows) != EXPECTED_FIT_ROWS:
        raise R0DataContractError(f"expected {EXPECTED_FIT_ROWS} L49 fit rows, got {len(rows)}")
    illegal = [
        (str(row.get("unit_key")), row.get("split"), row.get("dataset"), row.get("video"))
        for row in rows
        if row.get("split") != "fit" or row.get("dataset") not in FIT_DATASETS
    ]
    if illegal:
        raise R0DataContractError(f"illegal non-fit/non-V1-V2 rows: {illegal[:5]}")
    if any(str(row.get("video")) in FORBIDDEN_SCOPE_VIDEOS for row in rows):
        raise R0DataContractError("official-eval video appeared in fit rows")
    return rows


def _target_tuple(row: dict[str, Any]) -> tuple[str, ...]:
    value = row.get("target_ids", [])
    if not isinstance(value, (list, tuple, set)):
        raise R0DataContractError(f"target_ids is not a sequence: {row.get('unit_key')}")
    return tuple(sorted({str(item) for item in value}))


def _sentence_value(row: dict[str, Any]) -> str:
    value = row.get("sentence") or row.get("expression") or ""
    return str(value)


def canonical_query_diagnostics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Audit stable query sentences while retaining frame target variation."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), str(row["video"]), int(row["query_id"]))
        grouped[key].append(row)
    sentence_issues: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    varying = 0
    static = 0
    variant_histogram: defaultdict[str, int] = defaultdict(int)
    for key in sorted(grouped):
        values = grouped[key]
        sentences = sorted({_sentence_value(row) for row in values})
        targets = sorted({_target_tuple(row) for row in values})
        if len(sentences) != 1 or not sentences[0]:
            sentence_issues.append({
                "dataset": key[0], "video": key[1], "query_id": key[2],
                "row_count": len(values), "sentence_values": sentences,
                "target_id_values": [list(item) for item in targets],
                "unit_keys": [str(row.get("unit_key")) for row in values[:8]],
                "root_cause": "query sentence is missing or unstable",
            })
            continue
        if len(targets) > 1:
            varying += 1
        else:
            static += 1
        variant_histogram[str(len(targets))] += 1
        canonical.append({
            "dataset": key[0], "video": key[1], "query_id": key[2],
            "sentence": sentences[0],
        })
    return {
        "group_count": len(grouped),
        "sentence_issue_count": len(sentence_issues),
        "frame_target_varying_query_count": varying,
        "frame_target_static_query_count": static,
        "target_variant_histogram": dict(sorted(variant_histogram.items(), key=lambda item: int(item[0]))),
        "issues": sentence_issues,
        "canonical_queries": canonical,
        "contract": "(dataset,video,query_id) identifies one stable non-empty sentence; target_ids are frame-specific supervision",
        "passed": not sentence_issues,
    }


def validate_canonical_query_contract(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result = canonical_query_diagnostics(rows)
    if result["sentence_issue_count"]:
        raise R0DataContractError(
            "R0 canonical query contract failed: query sentence is missing or unstable; "
            + json.dumps({"sentence_issue_count": result["sentence_issue_count"], "first": result["issues"][0]}, ensure_ascii=False)
        )
    return {
        "canonical_query_count": len(result["canonical_queries"]),
        "query_count_by_dataset": {
            dataset: sum(item["dataset"] == dataset for item in result["canonical_queries"])
            for dataset in FIT_DATASETS
        },
        "canonical_queries": result["canonical_queries"],
        "contract": result["contract"],
        "frame_target_varying_query_count": result["frame_target_varying_query_count"],
        "frame_target_static_query_count": result["frame_target_static_query_count"],
        "target_variant_histogram": result["target_variant_histogram"],
        "passed": True,
        "sentence_issue_count": 0,
    }


def load_canonical_train_queries(dataset: str) -> list[R0TrainQuery]:
    if dataset not in FIT_DATASETS:
        raise ValueError(dataset)
    allowed = train_video_pairs(dataset)
    rows = [
        row for row in load_fit_rows()
        if str(row["dataset"]) == dataset and (str(row["dataset"]), str(row["video"])) in allowed
    ]
    result = validate_canonical_query_contract(rows)
    return [R0TrainQuery(**item) for item in result["canonical_queries"]]


def load_l82_video_split() -> dict[str, Any]:
    """Load and validate the preregistered video-disjoint fit/dev split."""
    payload = json.loads(L82_SPLIT.read_text(encoding="utf-8"))
    if payload.get("format") != "locatemot-l82-video-disjoint-fit-dev-split-v1":
        raise R0DataContractError(f"unexpected L82 split format: {payload.get('format')}")
    if payload.get("status") != "complete":
        raise R0DataContractError(f"L82 split is not complete: {payload.get('status')}")
    train = payload.get("train_videos")
    dev = payload.get("dev_videos")
    if not isinstance(train, list) or not train or not isinstance(dev, list) or not dev:
        raise R0DataContractError("L82 split has empty or non-list train/dev videos")
    train_pairs = _parse_video_pairs(train, "train_videos")
    dev_pairs = _parse_video_pairs(dev, "dev_videos")
    overlap = sorted(train_pairs & dev_pairs)
    if overlap:
        raise R0DataContractError(f"L82 train/dev video overlap: {overlap}")
    return {
        **payload,
        "train_video_pair_count": len(train_pairs),
        "dev_video_pair_count": len(dev_pairs),
        "train_dev_overlap": overlap,
    }


def _parse_video_pairs(values: list[Any], field: str) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for value in values:
        if not isinstance(value, str) or value.count("|") != 1:
            raise R0DataContractError(f"invalid {field} entry: {value!r}")
        dataset, video = value.split("|", 1)
        if dataset not in FIT_DATASETS or not video:
            raise R0DataContractError(f"invalid {field} dataset/video: {value!r}")
        result.add((dataset, video))
    if len(result) != len(values):
        raise R0DataContractError(f"duplicate {field} entries")
    return result


def train_video_pairs(dataset: str | None = None) -> set[tuple[str, str]]:
    pairs = _parse_video_pairs(load_l82_video_split()["train_videos"], "train_videos")
    return {pair for pair in pairs if dataset is None or pair[0] == dataset}


def dev_video_pairs(dataset: str | None = None) -> set[tuple[str, str]]:
    pairs = _parse_video_pairs(load_l82_video_split()["dev_videos"], "dev_videos")
    return {pair for pair in pairs if dataset is None or pair[0] == dataset}


class R0FrameTargetSource:
    """Index already-isolated frame-specific records; never open raw sources."""

    _PURPOSES = {"train", "dev", "internal", "audit"}

    def __init__(
        self,
        *,
        purpose: str,
        allowed_video_pairs: set[tuple[str, str]],
        records: Iterable[R0SafeQueryRecord],
    ) -> None:
        if purpose not in self._PURPOSES:
            raise ValueError(f"unsupported target-source purpose: {purpose}")
        self.purpose = purpose
        self.allowed_video_pairs = set(allowed_video_pairs)
        if not self.allowed_video_pairs:
            raise R0DataContractError(f"empty allowed video scope for {purpose}")
        for dataset, video in self.allowed_video_pairs:
            if dataset not in FIT_DATASETS or not video:
                raise R0DataContractError(f"invalid allowed video pair: {(dataset, video)!r}")
        self._rows: dict[tuple[str, str, int], R0SafeQueryRecord] = {}
        for record in records:
            if not isinstance(record, R0SafeQueryRecord):
                raise R0DataContractError(f"non-safe target record supplied: {type(record).__name__}")
            pair = (record.dataset, record.video)
            if pair not in self.allowed_video_pairs:
                raise R0DataContractError(f"safe target record outside allow-list: {pair}")
            if not record.sentence:
                raise R0DataContractError(f"empty safe target sentence: {record}")
            key = (record.dataset, record.video, int(record.query_id))
            if key in self._rows:
                raise R0DataContractError(f"duplicate safe target query key: {key}")
            self._rows[key] = record
        if not self._rows:
            raise R0DataContractError(f"no authoritative queries for allowed {purpose} scope")

    def _assert_allowed(self, dataset: str, video: str) -> tuple[str, str]:
        pair = (str(dataset), str(video))
        if pair not in self.allowed_video_pairs:
            raise R0DataContractError(f"{self.purpose} target lookup outside allow-list: {pair}")
        return pair

    def _key(self, dataset: str, video: str, query_id: int) -> tuple[str, str, int]:
        pair = self._assert_allowed(dataset, video)
        key = (pair[0], pair[1], int(query_id))
        if key not in self._rows:
            raise KeyError(f"missing authoritative query: {key}")
        return key

    def query_sentence(self, dataset: str, video: str, query_id: int) -> str:
        return self._rows[self._key(dataset, video, query_id)].sentence

    def target_ids(self, dataset: str, video: str, query_id: int, frame_id: int) -> tuple[str, ...]:
        key = self._key(dataset, video, query_id)
        # Missing exact frame keys are legal empty/inactive frames; no temporal fallback.
        return self._rows[key].target.get(int(frame_id), ())

    def query_ids(self, dataset: str, video: str) -> tuple[int, ...]:
        pair = self._assert_allowed(dataset, video)
        return tuple(sorted(key[2] for key in self._rows if key[:2] == pair))

    def source_descriptor(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "allowed_video_pairs": [f"{dataset}|{video}" for dataset, video in sorted(self.allowed_video_pairs)],
            "query_count": len(self._rows),
            "safe_artifact_only": True,
            "frame_specific": True,
            "target_union_used": False,
            "nearest_frame_fallback": False,
            "forward_fill": False,
            "backward_fill": False,
            "pseudo_labels_used": False,
        }


def native_frame_ids(video: str) -> list[int]:
    path = (L69_ROOT / f"{str(video)}.pt").resolve()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    try:
        tensors = blob.get("tensors") if isinstance(blob, dict) else None
        if not isinstance(tensors, dict):
            raise R0DataContractError(f"invalid L69 bank container: {path}")
        frame_ids = tensors.get("frame_ids")
        frame_ptr = tensors.get("frame_ptr")
        if not torch.is_tensor(frame_ids) or not torch.is_tensor(frame_ptr):
            raise R0DataContractError(f"missing native frame pointers: {path}")
        values = [int(item) for item in frame_ids.tolist()]
        pointers = [int(item) for item in frame_ptr.tolist()]
        if len(pointers) != len(values) + 1 or any(right < left for left, right in zip(pointers, pointers[1:])):
            raise R0DataContractError(f"invalid native frame_ptr: {path}")
        if values != sorted(set(values)):
            raise R0DataContractError(f"native frame order drift: {path}")
        return values
    finally:
        del blob


def load_l69_bank(video: str) -> tuple[Path, dict[str, Any]]:
    """Load one immutable L69 video bank for a streaming R0 operation."""
    path = (L69_ROOT / f"{str(video)}.pt").resolve()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or not isinstance(blob.get("tensors"), dict):
        raise R0DataContractError(f"invalid L69 bank container: {path}")
    tensors = blob["tensors"]
    required = {"frame", "frame_ids", "frame_ptr", "candidate_index", "track_id", "pool_id", "box"}
    missing = sorted(required - set(tensors))
    if missing:
        raise R0DataContractError(f"{video}: missing L69 fields {missing}")
    frame_ids = tensors["frame_ids"].long()
    frame_ptr = tensors["frame_ptr"].long()
    total = int(tensors["track_id"].numel())
    if frame_ptr.numel() != frame_ids.numel() + 1 or int(frame_ptr[-1]) != total:
        raise R0DataContractError(f"{video}: frame pointer total mismatch")
    if not bool(torch.all(frame_ptr[1:] >= frame_ptr[:-1])):
        raise R0DataContractError(f"{video}: descending frame pointer")
    if [int(x) for x in frame_ids.tolist()] != sorted(set(int(x) for x in frame_ids.tolist())):
        raise R0DataContractError(f"{video}: frame order drift")
    metadata = blob.get("metadata")
    if not isinstance(metadata, dict):
        raise R0DataContractError(f"{video}: missing L69 metadata")
    width, height = metadata.get("image_size", [0, 0])[:2]
    if int(width) <= 0 or int(height) <= 0:
        raise R0DataContractError(f"{video}: invalid image_size {metadata.get('image_size')}")
    for name in ("frame", "frame_ids", "frame_ptr", "candidate_index", "track_id", "pool_id", "box"):
        if not torch.is_tensor(tensors[name]):
            raise R0DataContractError(f"{video}: non-tensor L69 field {name}")
    return path, blob


def native_frame_slice(tensors: dict[str, Any], frame_position: int) -> tuple[int, int, int]:
    frame_ids = tensors["frame_ids"].long()
    pointers = tensors["frame_ptr"].long()
    position = int(frame_position)
    if position < 0 or position >= int(frame_ids.numel()):
        raise IndexError(position)
    begin, end = int(pointers[position]), int(pointers[position + 1])
    frame = int(frame_ids[position])
    if end < begin:
        raise R0DataContractError(f"negative native frame slice {frame}")
    rows = tensors["frame"].long()[begin:end]
    if rows.numel() and not bool(torch.all(rows == frame)):
        raise R0DataContractError(f"native frame rows disagree at frame {frame}")
    return frame, begin, end


def contract_descriptor() -> dict[str, Any]:
    return {
        "format": "locatemot-r0-dense-data-contract-v1",
        "asset_root": str(ASSET_ROOT),
        "l49_fit": file_meta(L49_DATA / "train_units.jsonl"),
        "l69_root": str(L69_ROOT),
        "l82_split": file_meta(L82_SPLIT),
        "manifest": file_meta(MANIFEST),
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "fit_datasets": list(FIT_DATASETS),
        "expected_fit_rows": EXPECTED_FIT_ROWS,
        "official_evaluation_videos_rejected": sorted(FORBIDDEN_SCOPE_VIDEOS),
        "label_protocol": "stable sentence identity plus frame-specific targets from an R0A safe artifact",
        "safe_target_source_only": True,
        "no_screening_or_official_test_labels": True,
    }


__all__ = [
    "ASSET_ROOT", "EXPECTED_FIT_ROWS", "EXPECTED_MANIFEST_SHA", "FIT_DATASETS",
    "FORBIDDEN_SCOPE_VIDEOS", "L49_DATA", "L69_ROOT", "L82_SPLIT", "MANIFEST",
    "R0BankStore", "R0DataContractError", "R0FrameBatch", "R0FrameSupervision", "R0FrameTargetSource",
    "R0SafeQueryRecord", "R0SafeSourceError", "R0TrainQuery", "R0TrainVideoScope", "contract_descriptor",
    "file_meta", "load_canonical_train_queries", "load_fit_rows", "load_l69_bank", "native_frame_ids",
    "native_frame_slice",
    "read_jsonl", "sha256_file", "canonical_query_diagnostics", "validate_canonical_query_contract",
    "load_safe_query_records",
]
