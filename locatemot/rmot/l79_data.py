"""L79-only data/index contract for the frozen L69 budget-40 bank.

The module deliberately reconstructs every unit from the bank's native
``frame_ptr``/``frame_ids``.  L49 ``begin``/``end`` and
``positive_indices`` are never used to address L69 rows.  Label attachment is
an explicit operation that is called only after a complete label-free unit
has been constructed by the audit/training/evaluation caller.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_DATA = ROOT / "outputs/l49/data"
L48_TEXT = ROOT / "outputs/l48/data/text_cache.pt"
L62_ROWS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"

OBS_FIELDS = (
    "clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle",
    "objectness",
)
OBS_DIM = 512 + 512 + 384 + 7 + 8 + 8 + 1
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
EXPECTED_SHARED_CHECKPOINT_SHA = "f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343"
FIT_VIDEOS = (
    "0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009",
    "0010", "0012", "0014", "0015", "0020",
)
ALL_L69_VIDEOS = (
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0020",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path, include_hash: bool = True) -> dict[str, Any]:
    path = path.resolve()
    value: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
    }
    if include_hash and path.is_file():
        value["sha256"] = sha256_file(path)
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key_only_unit(unit: dict[str, Any], partition: str | None = None) -> dict[str, Any]:
    """Strip all supervision/category fields before pre-selection scoring."""
    sentence = str(unit.get("sentence") or unit.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty expression for {unit.get('unit_key')}")
    result = {
        "format": "locatemot-l79-key-only-unit-v1",
        "unit_key": str(unit["unit_key"]),
        "dataset": str(unit["dataset"]),
        "video": str(unit["video"]),
        "query_id": int(unit["query_id"]),
        "frame_id": int(unit["frame_id"]),
        "sentence": sentence,
        "expression": sentence,
    }
    if partition is not None:
        result["evaluation_partition"] = partition
    return result


def load_fit_units() -> list[dict[str, Any]]:
    rows = load_jsonl(L49_DATA / "train_units.jsonl")
    if len(rows) != 5314:
        raise AssertionError(f"L49 fit unit count changed: {len(rows)}")
    if any(x.get("split") != "fit" or x.get("dataset") not in {"refer_kitti_v1", "refer_kitti_v2"} for x in rows):
        raise AssertionError("fit loader saw a non-V1/V2 fit unit")
    if set(str(x["video"]) for x in rows) != set(FIT_VIDEOS):
        raise AssertionError("fit video set changed")
    return rows


def load_fixed_key_units() -> list[dict[str, Any]]:
    """Return the immutable 16+24 order with supervision fields removed."""
    source_rows = load_jsonl(L49_DATA / "calibration_units.jsonl") + load_jsonl(L49_DATA / "validation_units.jsonl")
    by_key = {str(x["unit_key"]): x for x in source_rows}
    if len(by_key) != len(source_rows):
        raise AssertionError("duplicate source unit keys")
    fixed = load_jsonl(L62_ROWS)
    if len(fixed) != 40 or len({str(x["unit_key"]) for x in fixed}) != 40:
        raise AssertionError("fixed L62 slice is not 40 unique keys")
    result = []
    for index, row in enumerate(fixed):
        key = str(row["unit_key"])
        if key not in by_key:
            raise AssertionError(f"fixed key missing in L49 metadata: {key}")
        result.append(key_only_unit(by_key[key], "calibration" if index < 16 else "validation"))
    return result


def load_full_unit_for_labels(unit_key: str) -> dict[str, Any]:
    for name in ("calibration_units.jsonl", "validation_units.jsonl", "train_units.jsonl"):
        for row in load_jsonl(L49_DATA / name):
            if str(row["unit_key"]) == str(unit_key):
                return row
    raise KeyError(unit_key)


@dataclass
class UnitBatch:
    unit_key: str
    dataset: str
    video: str
    query_id: int
    frame_id: int
    sentence: str
    bank_path: str
    image_path: str
    row_offsets: list[int]
    row_keys: list[tuple[Any, ...]]
    candidate_indices: list[int]
    track_ids: list[int]
    pool_ids: list[int]
    boxes: torch.Tensor
    boxes_norm: torch.Tensor
    observations: torch.Tensor
    history_observations: torch.Tensor
    history_mask: torch.Tensor
    history_frame_ids: torch.Tensor
    text_tokens: torch.Tensor
    text_mask: torch.Tensor
    image_size: tuple[int, int]

    @property
    def candidate_count(self) -> int:
        return len(self.row_offsets)


class L79BankStore:
    """Lazy one-video bank reader with causal history indexing."""

    def __init__(self, max_history: int = 16) -> None:
        self.max_history = int(max_history)
        self._video: str | None = None
        self._bank: dict[str, Any] | None = None
        self._bank_path: Path | None = None
        self._frame_to_index: dict[int, int] = {}
        self._track_rows: dict[int, list[int]] = {}
        self._text_cache: dict[str, Any] | None = None

    def _load_text(self) -> dict[str, Any]:
        if self._text_cache is None:
            cache = torch.load(L48_TEXT, map_location="cpu", weights_only=False)
            required = {"token_hidden", "attention_mask", "sentence_to_index"}
            if not required.issubset(cache):
                raise AssertionError(f"L48 text cache missing fields: {required - set(cache)}")
            self._text_cache = cache
        return self._text_cache

    def load_video(self, video: str) -> None:
        video = str(video)
        if self._video == video:
            return
        path = (L69_ROOT / f"{video}.pt").resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        bank = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(bank, dict) or "tensors" not in bank or "metadata" not in bank:
            raise AssertionError(f"invalid L69 bank container: {path}")
        tensors = bank["tensors"]
        required = set(OBS_FIELDS) | {"frame", "frame_ids", "frame_ptr", "candidate_index", "track_id", "pool_id", "box"}
        missing = required - set(tensors)
        if missing:
            raise AssertionError(f"{video} missing L69 fields: {sorted(missing)}")
        total = int(tensors["track_id"].numel())
        frame_ids = tensors["frame_ids"].long()
        frame_ptr = tensors["frame_ptr"].long()
        if frame_ptr.numel() != frame_ids.numel() + 1 or int(frame_ptr[-1]) != total:
            raise AssertionError(f"{video} frame pointer total mismatch")
        if not bool(torch.all(frame_ptr[1:] >= frame_ptr[:-1])):
            raise AssertionError(f"{video} descending frame pointer")
        frame_rows = tensors["frame"].long()
        for frame, start, end in zip(frame_ids.tolist(), frame_ptr[:-1].tolist(), frame_ptr[1:].tolist()):
            if end > start and not bool(torch.all(frame_rows[start:end] == int(frame))):
                raise AssertionError(f"{video} frame rows disagree at {frame}")
        for name in OBS_FIELDS + ("box",):
            if not bool(torch.isfinite(tensors[name].float()).all()):
                raise AssertionError(f"nonfinite L69 field {video}:{name}")
        track_rows: dict[int, list[int]] = defaultdict(list)
        frame_values = frame_rows.tolist()
        track_values = tensors["track_id"].tolist()
        for offset, track_id in enumerate(track_values):
            track_rows[int(track_id)].append(offset)
        for offsets in track_rows.values():
            offsets.sort(key=lambda x: (int(frame_values[x]), int(x)))
        self._video = video
        self._bank = bank
        self._bank_path = path
        self._frame_to_index = {int(value): i for i, value in enumerate(frame_ids.tolist())}
        self._track_rows = dict(track_rows)

    @property
    def bank(self) -> dict[str, Any]:
        if self._bank is None:
            raise RuntimeError("bank is not loaded")
        return self._bank

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return self.bank["tensors"]

    def _obs(self, offsets: Iterable[int]) -> torch.Tensor:
        offsets = list(offsets)
        tensors = self.tensors
        if not offsets:
            return torch.zeros((0, OBS_DIM), dtype=torch.float32)
        chunks = []
        for field in OBS_FIELDS:
            value = tensors[field][offsets].float()
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            chunks.append(value)
        result = torch.cat(chunks, dim=-1)
        if result.shape[-1] != OBS_DIM:
            raise AssertionError(f"observation dimension drift: {result.shape}")
        return result

    def build_unit(self, unit: dict[str, Any]) -> UnitBatch:
        """Construct all visual/history/text inputs without reading labels."""
        self.load_video(str(unit["video"]))
        if int(unit["frame_id"]) not in self._frame_to_index:
            raise KeyError(f"frame missing from L69: {unit['unit_key']}")
        frame_index = self._frame_to_index[int(unit["frame_id"])]
        start = int(self.tensors["frame_ptr"][frame_index])
        end = int(self.tensors["frame_ptr"][frame_index + 1])
        offsets = list(range(start, end))
        if len(offsets) != end - start:
            raise AssertionError("candidate range construction drift")
        frame = int(unit["frame_id"])
        frame_values = self.tensors["frame"].tolist()
        candidate_indices = [int(x) for x in self.tensors["candidate_index"][offsets].tolist()]
        track_ids = [int(x) for x in self.tensors["track_id"][offsets].tolist()]
        pool_ids = [int(x) for x in self.tensors["pool_id"][offsets].tolist()]
        boxes = self.tensors["box"][offsets].float().clone()
        width, height = self.bank["metadata"].get("image_size", [0, 0])
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise AssertionError(f"invalid image_size in L69 metadata: {width}x{height}")
        boxes_norm = boxes / torch.tensor([width, height, width, height], dtype=torch.float32)
        boxes_norm = boxes_norm.clamp(0.0, 1.0)
        observations = self._obs(offsets)
        history = torch.zeros((len(offsets), self.max_history, OBS_DIM), dtype=torch.float32)
        history_mask = torch.zeros((len(offsets), self.max_history), dtype=torch.bool)
        history_frame_ids = torch.full((len(offsets), self.max_history), -1, dtype=torch.int64)
        for row_index, offset in enumerate(offsets):
            track_id = int(self.tensors["track_id"][offset])
            past = [x for x in self._track_rows.get(track_id, []) if int(frame_values[x]) <= frame]
            past = past[-self.max_history:]
            if past:
                # Keep valid observations as a causal prefix. Right-padding
                # would make left padded query positions have no legal key
                # under the causal+padding mask, yielding NaNs.
                length = len(past)
                history[row_index, :length] = self._obs(past)
                history_mask[row_index, :length] = True
                history_frame_ids[row_index, :length] = torch.tensor([int(frame_values[x]) for x in past], dtype=torch.int64)
                if int(history_frame_ids[row_index].max()) > frame:
                    raise AssertionError(f"future history row for {unit['unit_key']}")
        text_cache = self._load_text()
        sentence = str(unit.get("sentence") or unit.get("expression") or "")
        text_index = text_cache["sentence_to_index"].get(sentence)
        if text_index is None:
            raise KeyError(f"expression absent from L48 cache: {sentence!r}")
        text_tokens = text_cache["token_hidden"][int(text_index)].float().clone()
        text_mask = text_cache["attention_mask"][int(text_index)].bool().clone()
        if not bool(torch.isfinite(text_tokens).all()) or not bool(text_mask.any()):
            raise AssertionError(f"invalid text cache row for {unit['unit_key']}")
        row_keys = [
            (str(unit["dataset"]), str(unit["video"]), int(unit["query_id"]), frame, str(self._bank_path), int(offset))
            for offset in offsets
        ]
        if row_keys != sorted(row_keys, key=lambda x: x[-1]):
            raise AssertionError(f"L69 row order changed for {unit['unit_key']}")
        image_path = IMAGE_ROOT / str(unit["video"]) / f"{frame:06d}.png"
        return UnitBatch(
            unit_key=str(unit["unit_key"]), dataset=str(unit["dataset"]), video=str(unit["video"]),
            query_id=int(unit["query_id"]), frame_id=frame, sentence=sentence,
            bank_path=str(self._bank_path), image_path=str(image_path), row_offsets=offsets,
            row_keys=row_keys, candidate_indices=candidate_indices, track_ids=track_ids,
            pool_ids=pool_ids, boxes=boxes, boxes_norm=boxes_norm, observations=observations,
            history_observations=history, history_mask=history_mask, history_frame_ids=history_frame_ids,
            text_tokens=text_tokens, text_mask=text_mask, image_size=(width, height),
        )

    @staticmethod
    def attach_labels(batch: UnitBatch, full_unit: dict[str, Any]) -> dict[str, Any]:
        """Attach expression-level labels only after ``build_unit`` completed."""
        label_path = Path(batch.bank_path).with_suffix(".labels.json")
        sidecar = json.loads(label_path.read_text())
        candidate_gt = sidecar.get("candidate_gt")
        if not isinstance(candidate_gt, list):
            raise AssertionError(f"missing candidate_gt sidecar: {label_path}")
        if max(batch.row_offsets, default=-1) >= len(candidate_gt):
            raise AssertionError(f"sidecar too short for {batch.unit_key}")
        targets = {str(x) for x in full_unit.get("target_ids", [])}
        labels = torch.tensor([
            candidate_gt[offset] is not None and str(candidate_gt[offset]) in targets
            for offset in batch.row_offsets
        ], dtype=torch.bool)
        target_present = bool(targets)
        candidate_present = bool(labels.any())
        present_uncovered = target_present and not candidate_present
        category = "inactive" if not target_present else ("present_uncovered" if present_uncovered else ("multi_positive" if int(labels.sum()) > 1 else "positive"))
        membership_mask = torch.full_like(labels, not present_uncovered, dtype=torch.bool)
        return {
            "labels": labels,
            "membership_mask": membership_mask,
            "target_present": target_present,
            "candidate_present": candidate_present,
            "present_uncovered": present_uncovered,
            "category": category,
            "declared_category": str(full_unit.get("category", "unknown")),
            "target_ids": sorted(targets),
            "candidate_gt": [None if candidate_gt[x] is None else str(candidate_gt[x]) for x in batch.row_offsets],
            "positive_count": int(labels.sum()),
            "row_count": batch.candidate_count,
            "history_mask_last": batch.history_mask.any(dim=1).clone(),
            "labels_attached_after_feature_construction": True,
        }


def source_file_manifest(videos: Iterable[str]) -> list[dict[str, Any]]:
    result = []
    for video in videos:
        path = (L69_ROOT / f"{video}.pt").resolve()
        result.append(file_meta(path))
    return result


__all__ = [
    "ALL_L69_VIDEOS", "EXPECTED_MANIFEST_SHA", "EXPECTED_SHARED_CHECKPOINT_SHA",
    "FIT_VIDEOS", "IMAGE_ROOT", "L48_TEXT", "L49_DATA", "L62_ROWS", "L69_ROOT",
    "MANIFEST", "OBS_DIM", "OBS_FIELDS", "L79BankStore", "UnitBatch",
    "file_meta", "key_only_unit", "load_fit_units", "load_fixed_key_units",
    "load_full_unit_for_labels", "sha256_file", "source_file_manifest",
]
