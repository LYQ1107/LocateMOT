"""L80 RMOT-only data contract for the immutable L69 budget-40 bank.

The wrapper deliberately does not use L49 ``begin/end`` or
``positive_indices`` to address L69 rows.  A complete label-free unit is
constructed first; the explicit ``attach_labels`` call is the only place that
loads the candidate sidecar and expression target ids.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from locatemot.rmot.l79_data import (
    ALL_L69_VIDEOS,
    FIT_VIDEOS,
    L48_TEXT,
    L49_DATA,
    L62_ROWS,
    L69_ROOT,
    OBS_DIM,
    OBS_FIELDS,
    L79BankStore,
    UnitBatch,
    load_fixed_key_units as _load_fixed_key_units,
    load_full_unit_for_labels as _load_full_unit_for_labels,
)


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
CATEGORIES = ("positive", "multi_positive", "inactive", "present_uncovered")
FORBIDDEN_LABEL_FIELDS = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "target_present", "candidate_gt", "sidecar_candidate_gt", "coverage_mask",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key_only(row: dict[str, Any], partition: str | None = None) -> dict[str, Any]:
    """Copy only identity/text/frame metadata used before label attachment."""
    allowed = ("unit_key", "dataset", "video", "query_id", "frame_id", "sentence", "expression")
    result = {key: row[key] for key in allowed if key in row}
    if "unit_key" not in result:
        result["unit_key"] = f"{result['dataset']}|{result['video']}|{int(result['query_id'])}|{int(result['frame_id'])}"
    sentence = str(result.get("sentence") or result.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty expression for {result['unit_key']}")
    result["sentence"] = sentence
    result["expression"] = sentence
    if partition is not None:
        result["evaluation_partition"] = str(partition)
    leaked = FORBIDDEN_LABEL_FIELDS.intersection(result)
    if leaked:
        raise AssertionError(f"label fields leaked into key-only row: {sorted(leaked)}")
    return result


def load_fit_units() -> list[dict[str, Any]]:
    rows = read_jsonl(L49_DATA / "train_units.jsonl")
    if len(rows) != 5314:
        raise AssertionError(f"expected 5314 L49 fit rows, got {len(rows)}")
    if any(row.get("split") != "fit" or row.get("dataset") not in DATASETS for row in rows):
        raise AssertionError("non-V1/V2 fit row encountered")
    if {str(row["video"]) for row in rows} != set(FIT_VIDEOS):
        raise AssertionError("fit video set drift")
    return rows


def load_fixed_key_units() -> list[dict[str, Any]]:
    result = _load_fixed_key_units()
    if len(result) != 40:
        raise AssertionError(f"fixed unit count drift: {len(result)}")
    for index, row in enumerate(result):
        expected = "calibration" if index < 16 else "validation"
        if row.get("evaluation_partition") != expected:
            raise AssertionError(f"fixed partition drift at {index}")
        result[index] = key_only(row, expected)
        result[index]["fixed_eval_order"] = index
    if len({str(row["unit_key"]) for row in result}) != 40:
        raise AssertionError("fixed unit keys are not unique")
    return result


def load_full_unit_for_labels(unit_key: str) -> dict[str, Any]:
    return _load_full_unit_for_labels(str(unit_key))


class L80BankStore:
    """Thin new-stage wrapper around the audited L69 native indexer."""

    def __init__(self, max_history: int = 8) -> None:
        self._store = L79BankStore(max_history=max_history)
        self.max_history = int(max_history)

    @property
    def video(self) -> str | None:
        return self._store._video

    @property
    def bank_path(self) -> str | None:
        value = self._store._bank_path
        return None if value is None else str(value)

    def build_unit(self, metadata: dict[str, Any]) -> UnitBatch:
        if FORBIDDEN_LABEL_FIELDS.intersection(metadata):
            raise AssertionError(f"build_unit received labels before feature construction: {metadata['unit_key']}")
        batch = self._store.build_unit(key_only(metadata, metadata.get("evaluation_partition")))
        if batch.candidate_count != len(batch.row_offsets):
            raise AssertionError("candidate count/row offset mismatch")
        if len(batch.row_keys) != batch.candidate_count:
            raise AssertionError("row key count mismatch")
        if [int(key[-1]) for key in batch.row_keys] != batch.row_offsets:
            raise AssertionError("native row order drift")
        if batch.history_frame_ids.numel() and bool((batch.history_frame_ids > int(batch.frame_id)).any()):
            raise AssertionError(f"future history in {batch.unit_key}")
        return batch

    @staticmethod
    def attach_labels(batch: UnitBatch, full_unit: dict[str, Any]) -> dict[str, Any]:
        """Load expression/candidate labels only after raw features are built."""
        label_path = Path(batch.bank_path).with_suffix(".labels.json")
        payload = json.loads(label_path.read_text())
        candidate_gt = payload.get("candidate_gt")
        if not isinstance(candidate_gt, list):
            raise AssertionError(f"missing candidate_gt in {label_path}")
        if max(batch.row_offsets, default=-1) >= len(candidate_gt):
            raise AssertionError(f"sidecar too short for {batch.unit_key}")
        targets = {str(x) for x in full_unit.get("target_ids", [])}
        values = [candidate_gt[offset] for offset in batch.row_offsets]
        labels = [bool(value is not None and str(value) in targets) for value in values]
        target_present = bool(targets)
        candidate_present = bool(any(labels))
        present_uncovered = target_present and not candidate_present
        if not target_present:
            category = "inactive"
        elif present_uncovered:
            category = "present_uncovered"
        elif sum(labels) > 1:
            category = "multi_positive"
        else:
            category = "positive"
        return {
            "labels": torch.tensor(labels, dtype=torch.bool),
            "membership_mask": torch.full((batch.candidate_count,), not present_uncovered, dtype=torch.bool),
            "target_ids": sorted(targets),
            "sidecar_candidate_gt": [None if value is None else str(value) for value in values],
            "positive_indices": [int(index) for index, value in enumerate(labels) if value],
            "positive_count": int(sum(labels)),
            "target_present": target_present,
            "candidate_present": candidate_present,
            "coverage_mask": not present_uncovered,
            "category": category,
            "declared_category": str(full_unit.get("category", "unknown")),
            "null_target": not target_present,
            "row_count": batch.candidate_count,
            "label_source": str(label_path.resolve()),
            "labels_attached_after_feature_construction": True,
        }


def source_manifest(videos: list[str] | tuple[str, ...] = ALL_L69_VIDEOS) -> list[dict[str, Any]]:
    result = []
    for video in videos:
        path = (L69_ROOT / f"{video}.pt").resolve()
        result.append({
            "video": str(video), "path": str(path), "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path),
        })
    return result


__all__ = [
    "ALL_L69_VIDEOS", "CATEGORIES", "DATASETS", "EXPECTED_MANIFEST_SHA", "FIT_VIDEOS",
    "FORBIDDEN_LABEL_FIELDS", "L48_TEXT", "L49_DATA", "L62_ROWS", "L69_ROOT", "MANIFEST",
    "OBS_DIM", "OBS_FIELDS", "L80BankStore", "UnitBatch", "key_only", "load_fit_units",
    "load_fixed_key_units", "load_full_unit_for_labels", "read_jsonl", "sha256_file",
    "source_manifest",
]
