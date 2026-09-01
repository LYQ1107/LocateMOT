"""L75-only streaming data contract for candidate-marked VLM inputs.

This module deliberately rebuilds candidate rows from the native L69 frame
pointer.  The old L49 ``begin``/``end`` and ``positive_indices`` fields are
never used to address a candidate row.  Labels are joined only after the
current-frame row mapping has been constructed, so a present-but-uncovered
unit cannot silently become an inactive negative example.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
L69_FEATURE_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_DATA_ROOT = ROOT / "outputs/l49/data"
L62_RECORDS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
MANIFEST_PATH = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA256 = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
L69_VIDEOS = (
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0020",
)
OBS_FIELDS = ("clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness")
OBS_DIM = 1432


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 compatibility for read-only assets
        return torch.load(path, map_location="cpu")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def unit_key(unit: dict[str, Any]) -> str:
    if unit.get("unit_key"):
        return str(unit["unit_key"])
    return f"{unit['dataset']}|{unit['video']}|{int(unit['query_id'])}|{int(unit['frame_id'])}"


def load_splits() -> dict[str, list[dict[str, Any]]]:
    return {
        "fit": load_jsonl(L49_DATA_ROOT / "train_units.jsonl"),
        "calibration": load_jsonl(L49_DATA_ROOT / "calibration_units.jsonl"),
        "validation": load_jsonl(L49_DATA_ROOT / "validation_units.jsonl"),
    }


def load_fixed_order() -> list[dict[str, Any]]:
    rows = load_jsonl(L62_RECORDS)
    if len(rows) != 40:
        raise AssertionError(f"fixed L62 order must have 40 rows, got {len(rows)}")
    keys = [str(row["unit_key"]) for row in rows]
    if len(set(keys)) != 40:
        raise AssertionError("fixed L62 order contains duplicate unit keys")
    return rows


class L75Bank:
    """Lazy read-only view of a single materialized L69 feature bank."""

    REQUIRED = {
        "box", "frame", "frame_ids", "frame_ptr", "candidate_index", "track_id",
        "pool_id", "raw_rank", *OBS_FIELDS,
    }

    def __init__(self, video: str):
        self.video = str(video)
        if self.video not in L69_VIDEOS:
            raise ValueError(f"video outside L69 train pool: {self.video}")
        self.path = L69_FEATURE_ROOT / f"{self.video}.pt"
        self.label_path = self.path.with_suffix(".labels.json")
        if not self.path.exists() or not self.label_path.exists():
            raise FileNotFoundError(f"missing L69 bank/sidecar for {self.video}")
        self.blob = safe_torch_load(self.path)
        self.tensors: dict[str, torch.Tensor] = self.blob["tensors"]
        self.metadata = dict(self.blob.get("metadata", {}))
        self.sha256 = sha256_file(self.path)
        self.labels: list[Any] | None = None
        self._check_schema()
        self.frame_ranges: dict[int, tuple[int, int]] = {}
        frame_ids = self.tensors["frame_ids"].long().tolist()
        ptr = self.tensors["frame_ptr"].long().tolist()
        frame = self.tensors["frame"].long()
        if len(ptr) != len(frame_ids) + 1:
            raise AssertionError(f"{self.path}: frame pointer length mismatch")
        for index, frame_id in enumerate(frame_ids):
            begin, end = int(ptr[index]), int(ptr[index + 1])
            if not bool((frame[begin:end] == int(frame_id)).all()):
                raise AssertionError(f"{self.path}: frame pointer rows mismatch at {frame_id}")
            self.frame_ranges[int(frame_id)] = (begin, end)
        self.track_rows: dict[int, list[int]] = defaultdict(list)
        track = self.tensors["track_id"].long().tolist()
        for row, track_id in enumerate(track):
            self.track_rows[int(track_id)].append(int(row))

    def _check_schema(self) -> None:
        missing = self.REQUIRED.difference(self.tensors)
        if missing:
            raise KeyError(f"{self.path}: missing {sorted(missing)}")
        n = int(self.tensors["track_id"].numel())
        if int(self.tensors["frame_ptr"][-1]) != n:
            raise AssertionError(f"{self.path}: terminal frame pointer != row count")
        for name in ("box", *OBS_FIELDS):
            if int(self.tensors[name].shape[0]) != n:
                raise AssertionError(f"{self.path}: {name} row count drift")
            if not bool(torch.isfinite(self.tensors[name].float()).all()):
                raise AssertionError(f"{self.path}: nonfinite {name}")

    @property
    def count(self) -> int:
        return int(self.tensors["track_id"].numel())

    def rows_for(self, frame_id: int) -> list[int]:
        if int(frame_id) not in self.frame_ranges:
            raise KeyError(f"{self.video}: missing frame {frame_id}")
        begin, end = self.frame_ranges[int(frame_id)]
        return list(range(begin, end))

    def load_labels(self) -> list[Any]:
        if self.labels is None:
            payload = json.loads(self.label_path.read_text())
            values = payload["candidate_gt"]
            if len(values) != self.count:
                raise AssertionError(f"{self.path}: sidecar length mismatch")
            self.labels = values
        return self.labels

    def close(self) -> None:
        self.blob = None
        self.tensors = {}
        self.labels = None
        self.frame_ranges = {}
        self.track_rows = {}


def candidate_rows_before_labels(unit: dict[str, Any], bank: L75Bank) -> list[int]:
    if str(unit["video"]) != bank.video:
        raise AssertionError(f"unit/video mismatch for {unit_key(unit)}")
    rows = bank.rows_for(int(unit["frame_id"]))
    if not rows:
        raise AssertionError(f"empty L69 candidate row set for {unit_key(unit)}")
    # Row order is the native frame-pointer order.  Only query-independent
    # tensor/key checks occur here; labels are intentionally not read.
    boxes = bank.tensors["box"].index_select(0, torch.as_tensor(rows, dtype=torch.long))
    if boxes.shape[0] != len(rows) or not bool(torch.isfinite(boxes).all()):
        raise AssertionError(f"candidate box contract failed for {unit_key(unit)}")
    return rows


def make_record(unit: dict[str, Any], bank: L75Bank, include_labels: bool = True) -> dict[str, Any]:
    rows = candidate_rows_before_labels(unit, bank)
    dataset, video = str(unit["dataset"]), str(unit["video"])
    query_id, frame_id = int(unit["query_id"]), int(unit["frame_id"])
    candidate = bank.tensors["candidate_index"].long().tolist()
    track = bank.tensors["track_id"].long().tolist()
    pool = bank.tensors["pool_id"].long().tolist()
    raw_rank = bank.tensors["raw_rank"].long().tolist()
    row_keys = [[dataset, video, query_id, frame_id, str(bank.path), int(row)] for row in rows]
    if len({tuple(key) for key in row_keys}) != len(row_keys):
        raise AssertionError(f"duplicate immutable row key for {unit_key(unit)}")
    candidate_values = [int(candidate[row]) for row in rows]
    duplicate_candidate = sorted(v for v, c in Counter(candidate_values).items() if c > 1)
    result: dict[str, Any] = {
        "format": "locatemot-l75-unit-v1",
        "status": "complete",
        "dataset": dataset,
        "video": video,
        "query_id": query_id,
        "frame_id": frame_id,
        "unit_key": unit_key(unit),
        "sentence": str(unit.get("sentence") or unit.get("expression") or ""),
        "split": str(unit.get("split", "unknown")),
        "declared_category": str(unit.get("category", "unknown")) if include_labels else "unread_label_category",
        "row_offsets": rows,
        "row_keys": row_keys,
        "candidate_index_provenance": candidate_values,
        "track_id_provenance": [int(track[row]) for row in rows],
        "pool_id_provenance": [int(pool[row]) for row in rows],
        "raw_rank_provenance": [int(raw_rank[row]) for row in rows],
        "candidate_count": len(rows),
        "duplicate_candidate_index": duplicate_candidate,
        "bank_path": str(bank.path),
        "bank_sha256": bank.sha256,
        "image_path": str(IMAGE_ROOT / video / f"{frame_id:06d}.png"),
        "image_size_declared": [int(x) for x in unit.get("image_size", [])],
        "labels_joined_after_feature_contract": False,
        "present_uncovered_not_negative": True,
        "forbidden_semantic_ids_are_provenance_only": True,
    }
    if include_labels:
        target_ids = normalize_ids(unit.get("target_ids", []))
        result["target_ids"] = sorted(target_ids)
        sidecar = bank.load_labels()
        sidecar_rows = [None if sidecar[row] is None else str(sidecar[row]) for row in rows]
        labels = [value is not None and value in target_ids for value in sidecar_rows]
        positives = [index for index, value in enumerate(labels) if value]
        if len(positives) > 1:
            category = "multi_positive"
        elif positives:
            category = "positive"
        elif target_ids:
            category = "present_uncovered"
        else:
            category = "inactive"
        result.update({
            "labels": [bool(value) for value in labels],
            "sidecar_candidate_gt": sidecar_rows,
            "positive_indices": positives,
            "positive_count": len(positives),
            "category": category,
            "candidate_present": bool(positives),
            "coverage_mask": not (bool(target_ids) and not bool(positives)),
            "null_target": not bool(target_ids),
            "labels_joined_after_feature_contract": True,
            "label_path": str(bank.label_path),
        })
    else:
        result["labels_unread"] = True
    if result["candidate_count"] != len(result["row_offsets"]):
        raise AssertionError("candidate count/row offset mismatch")
    if result["row_keys"] != sorted(result["row_keys"], key=lambda key: key[-1]):
        raise AssertionError("immutable row order changed")
    return result


def fixed_eval_units(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    lookup = {unit_key(row): row for rows in splits.values() for row in rows}
    result = []
    for index, fixed in enumerate(load_fixed_order()):
        key = str(fixed["unit_key"])
        if key not in lookup:
            raise KeyError(f"fixed key absent from L49 data: {key}")
        row = dict(lookup[key])
        row["fixed_eval_order"] = index
        row["fixed_eval_split"] = "calibration" if index < 16 else "validation"
        result.append(row)
    return result


def fit_strata(splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    rows = [r for r in splits["fit"] if str(r["dataset"]) in DATASETS and str(r.get("split")) == "fit"]
    if len(rows) != 5314:
        raise AssertionError(f"expected 5314 L49 fit rows, got {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=unit_key):
        groups[f"{row['dataset']}::{row.get('category', 'unknown')}"] .append(row)
    return dict(groups)


def bank_observation_matrix(bank: L75Bank, rows: Iterable[int]) -> torch.Tensor:
    """Small numeric audit/control only; never used as L75 semantic input."""
    indices = torch.as_tensor([int(row) for row in rows], dtype=torch.long)
    values = []
    for name in OBS_FIELDS:
        values.append(bank.tensors[name].index_select(0, indices).float().reshape(len(indices), -1))
    result = torch.cat(values, dim=1)
    if result.shape[1] != OBS_DIM or not bool(torch.isfinite(result).all()):
        raise AssertionError("L69 observation audit failed")
    return result
