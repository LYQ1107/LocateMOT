"""Independent L77 data/index helpers for the frozen L69 budget-40 bank.

The helper deliberately rebuilds every current-frame candidate set from the
L69 bank's native frame pointers.  L49 ``begin/end`` and old positive indices
are never used to address a row.  Candidate labels are attached only after a
complete label-free row/feature record has been constructed.  The helper
does not write a feature cache.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_ROOT = ROOT / "outputs/l49/data"
L62_RECORDS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
TEXT_CACHE = ROOT / "outputs/l48/data/text_cache.pt"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA256 = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L69_VIDEOS = (
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0020",
)
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
REGIONS = ("clip",)
FIXED_KEY_FIELDS = ("dataset", "video", "query_id", "frame_id", "sentence", "expression", "unit_key")
FIXED_FORBIDDEN_LABEL_FIELDS = (
    "target_ids", "positive_indices", "positive_count", "category", "labels", "target_present",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility for an older local torch only
        return torch.load(path, map_location="cpu")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def unit_key(unit: dict[str, Any]) -> str:
    if unit.get("unit_key"):
        return str(unit["unit_key"])
    return "{}|{}|{}|{}".format(
        unit["dataset"], unit["video"], int(unit["query_id"]), int(unit["frame_id"])
    )


def load_splits() -> dict[str, list[dict[str, Any]]]:
    return {
        "fit": read_jsonl(L49_ROOT / "train_units.jsonl"),
        "calibration": read_jsonl(L49_ROOT / "calibration_units.jsonl"),
        "validation": read_jsonl(L49_ROOT / "validation_units.jsonl"),
    }


def load_l62_order() -> list[dict[str, Any]]:
    rows = read_jsonl(L62_RECORDS)
    if len(rows) != 40:
        raise AssertionError(f"L62 fixed order must have 40 rows, got {len(rows)}")
    keys = [str(row["unit_key"]) for row in rows]
    if len(set(keys)) != 40:
        raise AssertionError("duplicate L62 unit key")
    return rows


def load_l62_key_order() -> list[dict[str, Any]]:
    """Read only fixed-evaluation keys; do not retain L62 labels or scores."""
    result: list[dict[str, Any]] = []
    for line in L62_RECORDS.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "unit_key" not in payload:
            raise KeyError("L62 fixed record has no unit_key")
        result.append({"unit_key": str(payload["unit_key"])})
    if len(result) != 40 or len({row["unit_key"] for row in result}) != 40:
        raise AssertionError(f"L62 fixed key order must contain 40 unique rows, got {len(result)}")
    return result


def _read_key_only_jsonl(path: Path) -> list[dict[str, Any]]:
    """Retain only fixed non-label metadata from a unit JSONL file."""
    result: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        result.append({key: payload[key] for key in FIXED_KEY_FIELDS if key in payload})
    if any(field in row for row in result for field in FIXED_FORBIDDEN_LABEL_FIELDS):
        raise AssertionError(f"forbidden label field retained from {path}")
    return result


def load_fixed_key_metadata() -> list[dict[str, Any]]:
    """Join the 40 fixed keys with only text/frame metadata, never labels."""
    lookup: dict[str, dict[str, Any]] = {}
    for path in (L49_ROOT / "calibration_units.jsonl", L49_ROOT / "validation_units.jsonl"):
        for row in _read_key_only_jsonl(path):
            key = unit_key(row)
            if key in lookup:
                raise AssertionError(f"duplicate fixed key metadata: {key}")
            lookup[key] = row
    result: list[dict[str, Any]] = []
    for order, fixed in enumerate(load_l62_key_order()):
        key = fixed["unit_key"]
        if key not in lookup:
            raise KeyError(f"fixed key missing from L49 metadata: {key}")
        row = dict(lookup[key])
        row["fixed_eval_order"] = order
        row["fixed_eval_split"] = "calibration" if order < 16 else "validation"
        if any(field in row for field in FIXED_FORBIDDEN_LABEL_FIELDS):
            raise AssertionError(f"forbidden pre-selection field: {key}")
        result.append(row)
    return result


def load_fixed_label_units(orders: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Load labels only for the explicitly authorized calibration or validation orders."""
    selected = sorted({int(order) for order in orders})
    if any(order < 0 or order >= 40 for order in selected):
        raise ValueError(f"fixed order outside 0..39: {selected}")
    key_order = load_l62_key_order()
    result: dict[int, dict[str, Any]] = {}
    for role, path in (
        ("calibration", L49_ROOT / "calibration_units.jsonl"),
        ("validation", L49_ROOT / "validation_units.jsonl"),
    ):
        role_orders = [order for order in selected
                       if (order < 16) == (role == "calibration")]
        if not role_orders:
            continue
        wanted = {key_order[order]["unit_key"] for order in role_orders}
        found: dict[str, dict[str, Any]] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            key = str(payload.get("unit_key", ""))
            if key in wanted:
                found[key] = payload
        if set(found) != wanted:
            raise KeyError(f"authorized {role} labels missing: {sorted(wanted - set(found))}")
        for order in role_orders:
            row = found[key_order[order]["unit_key"]]
            if str(row.get("split", role)) != role:
                raise AssertionError(f"fixed split mismatch at order {order}")
            result[order] = row
    if set(result) != set(selected):
        raise AssertionError("authorized fixed label rows incomplete")
    return result


def load_fixed_units(splits: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    splits = load_splits() if splits is None else splits
    lookup: dict[str, dict[str, Any]] = {}
    for role in ("calibration", "validation"):
        for row in splits[role]:
            key = unit_key(row)
            if key in lookup:
                raise AssertionError(f"duplicate fixed unit in L49 metadata: {key}")
            lookup[key] = row
    result: list[dict[str, Any]] = []
    for order, old in enumerate(load_l62_order()):
        key = str(old["unit_key"])
        if key not in lookup:
            raise KeyError(f"L62 unit missing from L49 cal/val metadata: {key}")
        row = dict(lookup[key])
        row["fixed_eval_order"] = order
        row["fixed_eval_split"] = "calibration" if order < 16 else "validation"
        result.append(row)
    return result


def load_text_cache() -> dict[str, Any]:
    cache = safe_torch_load(TEXT_CACHE)
    required = {"sentences", "token_hidden", "attention_mask", "sentence_to_index"}
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"text cache missing {sorted(missing)}")
    hidden = cache["token_hidden"]
    mask = cache["attention_mask"]
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (64, 768):
        raise AssertionError(f"unexpected text cache shape {tuple(hidden.shape)}")
    if mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise AssertionError("text cache mask contract failed")
    if not torch.isfinite(hidden.float()).all():
        raise AssertionError("nonfinite text cache")
    return cache


class L77Bank:
    """One-video read-only view of L69; labels are lazy for label-free audits."""

    def __init__(self, video: str):
        self.video = str(video)
        if self.video not in L69_VIDEOS:
            raise ValueError(f"video outside L69 train pool: {video}")
        self.path = L69_ROOT / f"{self.video}.pt"
        self.label_path = self.path.with_suffix(".labels.json")
        self.blob = safe_torch_load(self.path)
        if not isinstance(self.blob, dict) or "tensors" not in self.blob:
            raise AssertionError(f"invalid L69 bank wrapper: {self.path}")
        self.tensors: dict[str, torch.Tensor] = self.blob["tensors"]
        self.labels: list[str | None] | None = None
        self.frame_ranges: dict[int, tuple[int, int, int]] = {}
        self._check_schema()
        frame_ids = self.tensors["frame_ids"].long().tolist()
        frame_ptr = self.tensors["frame_ptr"].long().tolist()
        frame = self.tensors["frame"].long()
        for frame_index, frame_id in enumerate(frame_ids):
            begin, end = int(frame_ptr[frame_index]), int(frame_ptr[frame_index + 1])
            if begin < 0 or end < begin:
                raise AssertionError(f"invalid frame pointer in {self.path}")
            if not torch.equal(frame[begin:end], torch.full((end - begin,), int(frame_id), dtype=frame.dtype)):
                raise AssertionError(f"frame pointer/row mismatch {self.path} frame={frame_id}")
            self.frame_ranges[int(frame_id)] = (begin, end, frame_index)

    def _check_schema(self) -> None:
        required = {
            "clip", "frame", "frame_ids", "frame_ptr", "candidate_index", "track_id",
            "box", "pool_id", "raw_rank",
        }
        missing = required.difference(self.tensors)
        if missing:
            raise KeyError(f"{self.path}: missing {sorted(missing)}")
        count = int(self.tensors["track_id"].numel())
        if int(self.tensors["frame_ptr"][-1]) != count:
            raise AssertionError(f"{self.path}: frame_ptr terminal mismatch")
        for name in ("clip", "box", "objectness", "geometry", "motion", "lifecycle"):
            if name in self.tensors and int(self.tensors[name].shape[0]) != count:
                raise AssertionError(f"{self.path}: {name} row mismatch")
        if self.tensors["clip"].ndim != 2 or int(self.tensors["clip"].shape[1]) != 512:
            raise AssertionError(f"{self.path}: clip must be [rows,512]")
        if not torch.isfinite(self.tensors["clip"].float()).all():
            raise AssertionError(f"{self.path}: nonfinite clip")

    @property
    def count(self) -> int:
        return int(self.tensors["track_id"].numel())

    def load_labels(self) -> list[str | None]:
        if self.labels is None:
            if not self.label_path.exists():
                raise FileNotFoundError(self.label_path)
            raw = json.loads(self.label_path.read_text()).get("candidate_gt", [])
            if len(raw) != self.count:
                raise AssertionError(f"{self.label_path}: {len(raw)} labels != {self.count} rows")
            self.labels = [None if value is None else str(value) for value in raw]
        return self.labels

    def close(self) -> None:
        self.blob = None
        self.tensors = {}
        self.labels = None
        self.frame_ranges = {}


def _row_key(unit: dict[str, Any], row: int, path: Path) -> list[Any]:
    return [str(unit["dataset"]), str(unit["video"]), int(unit["query_id"]),
            int(unit["frame_id"]), str(path), int(row)]


def make_label_free_record(unit: dict[str, Any], bank: L77Bank,
                           include_declared_category: bool = True) -> dict[str, Any]:
    """Construct complete native rows and region features without sidecar GT."""
    if str(unit["video"]) != bank.video:
        raise AssertionError("unit/video mismatch")
    frame_id = int(unit["frame_id"])
    if frame_id not in bank.frame_ranges:
        raise KeyError(f"L69 frame missing: {unit_key(unit)}")
    begin, end, frame_index = bank.frame_ranges[frame_id]
    rows = list(range(begin, end))
    # Feature construction precedes any label access.  The returned tensor is
    # an in-memory unit view and is never written to disk.
    region = bank.tensors["clip"][begin:end].float().clone()
    if region.shape != (len(rows), 512) or not torch.isfinite(region).all():
        raise AssertionError(f"region feature contract failed: {unit_key(unit)}")
    tensors = bank.tensors
    row_keys = [_row_key(unit, row, bank.path) for row in rows]
    if [key[-1] for key in row_keys] != rows:
        raise AssertionError(f"native row order drift: {unit_key(unit)}")
    record = {
        "format": "locatemot-l77-label-free-unit-v1",
        "status": "complete",
        "unit_key": unit_key(unit),
        "dataset": str(unit["dataset"]), "video": str(unit["video"]),
        "query_id": int(unit["query_id"]), "frame_id": frame_id,
        "sentence": str(unit.get("sentence", unit.get("expression", ""))),
        "expression": str(unit.get("expression", "")),
        "frame_index": int(frame_index), "begin": begin, "end": end,
        "candidate_count": len(rows), "row_offsets": rows, "row_keys": row_keys,
        "candidate_index": tensors["candidate_index"][begin:end].long().tolist(),
        "track_id_provenance": tensors["track_id"][begin:end].long().tolist(),
        "pool_id_provenance": tensors["pool_id"][begin:end].long().tolist(),
        "raw_rank_provenance": tensors["raw_rank"][begin:end].long().tolist(),
        "region": region,
        "candidate_deletion": False, "candidate_truncation": False,
        "old_l49_begin_end_used": False, "old_l49_positive_indices_used": False,
        "source_pool_track_ids_are_provenance_only": True,
    }
    if include_declared_category:
        record["declared_category"] = str(unit.get("category", "unknown"))
    for key in ("schedule_position", "train_step"):
        if key in unit:
            record[key] = int(unit[key])
    return record


def attach_labels(record: dict[str, Any], unit: dict[str, Any], bank: L77Bank) -> dict[str, Any]:
    """Join expression-level labels after the label-free row/feature view exists."""
    labels = bank.load_labels()
    begin, end = int(record["begin"]), int(record["end"])
    targets = {str(value) for value in (unit.get("target_ids") or [])}
    sidecar = labels[begin:end]
    membership = [value is not None and str(value) in targets for value in sidecar]
    positive_indices = [index for index, value in enumerate(membership) if value]
    if len(positive_indices) > 1:
        category = "multi_positive"
    elif positive_indices:
        category = "positive"
    elif targets:
        category = "present_uncovered"
    else:
        category = "inactive"
    coverage_mask = not (bool(targets) and not bool(positive_indices))
    result = dict(record)
    result.update({
        "target_ids": sorted(targets), "sidecar_candidate_gt": sidecar,
        "labels": [int(value) for value in membership],
        "positive_indices": positive_indices, "positive_count": len(positive_indices),
        "target_present": bool(targets),
        "candidate_present": bool(positive_indices), "coverage_mask": bool(coverage_mask),
        "null_target": int(not bool(targets)), "category": category,
        "label_source": str(bank.label_path),
    })
    if len(result["labels"]) != int(record["candidate_count"]):
        raise AssertionError(f"label length drift: {record['unit_key']}")
    return result


def unit_tensors(record: dict[str, Any], text_cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    sentence = str(record["sentence"])
    if sentence not in text_cache["sentence_to_index"]:
        raise KeyError(f"sentence missing from L48 text cache: {sentence!r}")
    index = int(text_cache["sentence_to_index"][sentence])
    region = record["region"].float().clone()
    text = text_cache["token_hidden"][index].float().clone()
    mask = text_cache["attention_mask"][index].bool().clone()
    if region.ndim != 2 or region.shape[1] != 512:
        raise AssertionError("region tensor shape drift")
    if text.shape != (64, 768) or mask.shape != (64,) or not bool(mask.any()):
        raise AssertionError("text tensor contract failed")
    if not torch.isfinite(region).all() or not torch.isfinite(text).all():
        raise AssertionError("nonfinite L77 input")
    return {
        "region": region, "text": text, "text_mask": mask,
        "membership_target": torch.as_tensor(record.get("labels", []), dtype=torch.float32),
        "coverage_mask": torch.full((int(record["candidate_count"]),), bool(record.get("coverage_mask", True)), dtype=torch.bool),
        "null_target": torch.tensor(float(record.get("null_target", 0.0)), dtype=torch.float32),
    }


def make_schedule(records: list[dict[str, Any]], steps: int, seed: int = 20260829) -> list[dict[str, Any]]:
    """Deterministic 8-stratum schedule; uses fit metadata only."""
    fit = [row for row in records if str(row.get("split")) == "fit" and str(row.get("dataset")) in DATASETS]
    if len(fit) != 5314:
        raise AssertionError(f"expected 5314 V1/V2 fit units, got {len(fit)}")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fit:
        buckets[(str(row["dataset"]), str(row.get("category", "unknown")))].append(row)
    required = {(dataset, category) for dataset in DATASETS for category in (
        "positive", "multi_positive", "inactive", "present_uncovered")}
    if set(buckets) != required:
        raise AssertionError(f"fit strata mismatch: {sorted(set(buckets) ^ required)}")
    rng = np.random.default_rng(int(seed))
    keys = sorted(buckets)
    for key in keys:
        values = sorted(buckets[key], key=lambda row: (str(row["video"]), int(row["query_id"]), int(row["frame_id"]), unit_key(row)))
        order = rng.permutation(len(values)).tolist()
        buckets[key] = [values[int(index)] for index in order]
    selected: list[dict[str, Any]] = []
    for position in range(int(steps)):
        key = keys[position % len(keys)]
        row = dict(buckets[key][(position // len(keys)) % len(buckets[key])])
        row["schedule_position"] = position
        row["train_step"] = position + 1
        selected.append(row)
    if {str(row["dataset"]) for row in selected} != set(DATASETS):
        raise AssertionError("schedule missed one domain")
    if {str(row.get("category")) for row in selected} != {"positive", "multi_positive", "inactive", "present_uncovered"}:
        raise AssertionError("schedule missed one category")
    return selected


def load_fit_samples(schedule: list[dict[str, Any]], text_cache: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, torch.Tensor]]]:
    """Build a bounded in-memory sample list, loading each video serially."""
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in schedule:
        by_video[str(row["video"])].append(row)
    result: list[tuple[dict[str, Any], dict[str, torch.Tensor]]] = []
    for video in sorted(by_video):
        bank = L77Bank(video)
        try:
            for unit in by_video[video]:
                record = make_label_free_record(unit, bank)
                record = attach_labels(record, unit, bank)
                result.append((record, unit_tensors(record, text_cache)))
        finally:
            bank.close()
    result.sort(key=lambda pair: int(pair[0].get("train_step", 0)))
    if len(result) != len(schedule):
        raise AssertionError("fit sample count drift")
    return result


def load_fixed_label_free(text_cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct the 40 fixed rows without reading sidecar/unit labels."""
    units = load_fixed_key_metadata()
    by_video: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for order, unit in enumerate(units):
        by_video[str(unit["video"])].append((order, unit))
    result: list[dict[str, Any] | None] = [None] * len(units)
    for video in sorted(by_video):
        bank = L77Bank(video)
        try:
            for order, unit in by_video[video]:
                record = make_label_free_record(unit, bank, include_declared_category=False)
                record["fixed_eval_order"] = order
                record["fixed_eval_split"] = "calibration" if order < 16 else "validation"
                # Keep only non-label identity needed to reconstruct the row.
                # target_ids/category are intentionally not retained until the
                # calibration-only strategy has been selected by the evaluator.
                record["unit_metadata"] = {
                    key: unit[key] for key in FIXED_KEY_FIELDS
                    if key in unit
                }
                if any(field in record for field in FIXED_FORBIDDEN_LABEL_FIELDS):
                    raise AssertionError(f"forbidden pre-selection field in {record['unit_key']}")
                if any(field in record["unit_metadata"] for field in FIXED_FORBIDDEN_LABEL_FIELDS):
                    raise AssertionError(f"forbidden pre-selection metadata in {record['unit_key']}")
                record["unit_tensors"] = unit_tensors(record, text_cache)
                result[order] = record
        finally:
            bank.close()
    if any(row is None for row in result):
        raise AssertionError("fixed label-free row missing")
    return [row for row in result if row is not None]


def attach_record_labels(record: dict[str, Any], unit: dict[str, Any], bank: L77Bank) -> dict[str, Any]:
    return attach_labels(record, unit, bank)
