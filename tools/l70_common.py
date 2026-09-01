"""L70-only indexing and tensor construction for the L69 budget-40 bank.

This module deliberately does not import the old L49 feature-range helpers.
L49 ``begin/end`` and ``positive_indices`` refer to the old L19 bank and are
therefore never used to address an L69 row.  The L70 index is rebuilt from
each L69 bank's own frame pointer and sidecar.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
L69_FEATURE_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_DATA_ROOT = ROOT / "outputs/l49/data"
L62_RECORDS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
TEXT_CACHE_PATH = ROOT / "outputs/l48/data/text_cache.pt"
MANIFEST_PATH = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA256 = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"

L69_VIDEOS = (
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0020",
)
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
OBS_FIELDS = (
    "clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle",
    "objectness",
)
OBS_DIM = 1432
MAX_HISTORY = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def safe_torch_load(path: Path) -> Any:
    # L70 runs in the verified Torch 2.x environment.  The fallback keeps the
    # helper readable in older local interpreters without changing a source.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def unit_key(unit: dict[str, Any]) -> str:
    if unit.get("unit_key"):
        return str(unit["unit_key"])
    return "{}|{}|{}|{}".format(
        unit["dataset"], unit["video"], int(unit["query_id"]), int(unit["frame_id"])
    )


def parse_unit_key(value: str) -> tuple[str, str, int, int]:
    parts = str(value).split("|")
    if len(parts) != 4:
        raise ValueError(f"unexpected unit_key: {value!r}")
    return parts[0], parts[1], int(parts[2]), int(parts[3])


def load_l49_splits() -> dict[str, list[dict[str, Any]]]:
    return {
        "fit": load_jsonl(L49_DATA_ROOT / "train_units.jsonl"),
        "calibration": load_jsonl(L49_DATA_ROOT / "calibration_units.jsonl"),
        "validation": load_jsonl(L49_DATA_ROOT / "validation_units.jsonl"),
    }


def load_l62_order() -> list[dict[str, Any]]:
    rows = load_jsonl(L62_RECORDS)
    if len(rows) != 40:
        raise AssertionError(f"L62 fixed order must contain 40 rows, got {len(rows)}")
    keys = [str(row["unit_key"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise AssertionError("duplicate L62 fixed unit key")
    return rows


def load_text_cache() -> dict[str, Any]:
    cache = safe_torch_load(TEXT_CACHE_PATH)
    required = {"sentences", "token_hidden", "attention_mask", "sentence_to_index"}
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"text cache missing {sorted(missing)}")
    hidden = cache["token_hidden"]
    mask = cache["attention_mask"]
    if hidden.ndim != 3 or hidden.shape[-1] != 768:
        raise AssertionError(f"unexpected text hidden shape {tuple(hidden.shape)}")
    if mask.shape[:2] != hidden.shape[:2]:
        raise AssertionError("text hidden/mask shape mismatch")
    return cache


def feature_matrix(tensors: dict[str, torch.Tensor], rows: Iterable[int]) -> torch.Tensor:
    rows_list = [int(row) for row in rows]
    index = torch.as_tensor(rows_list, dtype=torch.long)
    parts: list[torch.Tensor] = []
    for name in OBS_FIELDS:
        value = tensors[name].index_select(0, index).float().reshape(len(rows_list), -1)
        parts.append(value)
    result = torch.cat(parts, dim=1)
    if result.shape[1] != OBS_DIM:
        raise AssertionError(f"observation dimension drift: {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise AssertionError("nonfinite L69 observation feature")
    return result


class L69Bank:
    """Read-only view of one materialized L69 feature bank."""

    def __init__(self, video: str):
        if str(video) not in L69_VIDEOS:
            raise ValueError(f"video outside L69 train pool: {video}")
        self.video = str(video)
        self.path = L69_FEATURE_ROOT / f"{self.video}.pt"
        self.label_path = self.path.with_suffix(".labels.json")
        self.blob = safe_torch_load(self.path)
        self.tensors: dict[str, torch.Tensor] = self.blob["tensors"]
        self.labels = json.loads(self.label_path.read_text())["candidate_gt"]
        self.sha256 = sha256_file(self.path)
        self._check_schema()
        self.frame_ranges: dict[int, tuple[int, int, int]] = {}
        frame_ids = self.tensors["frame_ids"].long().tolist()
        ptr = self.tensors["frame_ptr"].long().tolist()
        frame = self.tensors["frame"].long()
        for frame_index, frame_id in enumerate(frame_ids):
            begin, end = int(ptr[frame_index]), int(ptr[frame_index + 1])
            if not torch.equal(frame[begin:end], torch.full((end - begin,), int(frame_id), dtype=frame.dtype)):
                raise AssertionError(f"frame row mismatch in {self.path} frame {frame_id}")
            self.frame_ranges[int(frame_id)] = (begin, end, frame_index)
        self.track_rows: dict[int, list[int]] = defaultdict(list)
        for row, track_id in enumerate(self.tensors["track_id"].long().tolist()):
            self.track_rows[int(track_id)].append(int(row))

    def _check_schema(self) -> None:
        required = set(OBS_FIELDS) | {
            "frame", "frame_ids", "frame_ptr", "candidate_index", "track_id",
            "box", "pool_id", "raw_rank",
        }
        missing = required.difference(self.tensors)
        if missing:
            raise KeyError(f"{self.path}: missing {sorted(missing)}")
        count = int(self.tensors["track_id"].numel())
        if len(self.labels) != count:
            raise AssertionError(f"{self.path}: labels {len(self.labels)} != rows {count}")
        if int(self.tensors["frame_ptr"][-1]) != count:
            raise AssertionError(f"{self.path}: frame_ptr terminal mismatch")
        if any(int(self.tensors[name].shape[0]) != count for name in OBS_FIELDS):
            raise AssertionError(f"{self.path}: observation field row mismatch")

    @property
    def count(self) -> int:
        return int(self.tensors["track_id"].numel())

    @property
    def frame_count(self) -> int:
        return int(self.tensors["frame_ids"].numel())

    def close(self) -> None:
        # Explicitly release the large read-only object between videos.  No
        # feature tensors are written by L70.
        self.blob = None
        self.tensors = {}
        self.labels = []
        self.frame_ranges = {}
        self.track_rows = {}


def _row_key(dataset: str, video: str, frame_id: int, path: Path, row: int) -> list[Any]:
    return [str(dataset), str(video), int(frame_id), str(path), int(row)]


def make_unit_record(unit: dict[str, Any], bank: L69Bank) -> dict[str, Any]:
    """Construct an L70 unit/index record from the L69 frame pointer.

    The first operation is the candidate-row/history construction.  Only then
    are sidecar labels joined to derive supervision fields.
    """
    dataset = str(unit["dataset"])
    video = str(unit["video"])
    frame_id = int(unit["frame_id"])
    if video != bank.video:
        raise AssertionError(f"unit/video-bank mismatch: {video} vs {bank.video}")
    if frame_id not in bank.frame_ranges:
        raise KeyError(f"missing L69 frame {dataset}/{video}/{frame_id}")
    begin, end, frame_index = bank.frame_ranges[frame_id]
    rows = list(range(begin, end))
    tensors = bank.tensors
    # This reads feature fields before consulting the sidecar labels.  It is
    # intentionally not based on unit['begin'], unit['end'], or old positives.
    _ = feature_matrix(tensors, rows)
    target_ids = normalize_ids(unit.get("target_ids", []))
    sidecar = [None if x is None else str(x) for x in bank.labels[begin:end]]
    positive = [value is not None and value in target_ids for value in sidecar]
    positive_indices = [idx for idx, value in enumerate(positive) if value]
    if len(positive_indices) > 1:
        category = "multi_positive"
    elif positive_indices:
        category = "positive"
    elif target_ids:
        category = "present_uncovered"
    else:
        category = "inactive"
    coverage_mask = not (bool(target_ids) and not bool(positive_indices))

    frame_values = tensors["frame"].long().tolist()
    track_values = tensors["track_id"].long().tolist()
    candidate_values = tensors["candidate_index"].long().tolist()
    pool_values = tensors["pool_id"].long().tolist()
    rank_values = tensors["raw_rank"].long().tolist()
    history_rows: list[list[int]] = []
    history_frames: list[list[int]] = []
    history_positive: list[list[int]] = []
    continuation_target: list[int] = []
    for row in rows:
        track = int(track_values[row])
        eligible = [old for old in bank.track_rows.get(track, []) if frame_values[old] <= frame_id]
        chosen = eligible[-MAX_HISTORY:]
        if any(frame_values[old] > frame_id for old in chosen):
            raise AssertionError("future frame entered L70 history")
        hpos = [int(bank.labels[old] is not None and str(bank.labels[old]) in target_ids) for old in chosen]
        history_rows.append(chosen)
        history_frames.append([int(frame_values[old]) for old in chosen])
        history_positive.append(hpos)
        continuation_target.append(int(bool(positive[row - begin]) and any(hpos[:-1])))

    duplicate_candidates = [int(value) for value, count in Counter(candidate_values[row] for row in rows).items() if count > 1]
    main_count = sum(int(rank_values[row] == -1) for row in rows)
    reserve_count = len(rows) - main_count
    row_keys = [_row_key(dataset, video, frame_id, bank.path, row) for row in rows]
    if row_keys != sorted(row_keys, key=lambda value: value[-1]):
        raise AssertionError("L69 row key order drift")
    return {
        "format": "locatemot-l70-unit-index-v1",
        "status": "complete",
        "dataset": dataset,
        "video": video,
        "query_id": int(unit["query_id"]),
        "frame_id": frame_id,
        "unit_key": unit_key(unit),
        "sentence": str(unit.get("sentence", unit.get("expression", ""))),
        "expression": str(unit.get("expression", "")),
        "split": str(unit.get("split", "unknown")),
        "frame_index": frame_index,
        "frame_pointer": {"begin": begin, "end": end, "count": end - begin},
        "target_ids": sorted(target_ids),
        "candidate_present": bool(positive_indices),
        "coverage_mask": bool(coverage_mask),
        "null_target": int(not bool(target_ids)),
        "category": category,
        "source_category_from_l49": str(unit.get("category", "unknown")),
        "positive_indices": positive_indices,
        "positive_count": len(positive_indices),
        "candidate_count": len(rows),
        "labels": [int(value) for value in positive],
        "sidecar_candidate_gt": sidecar,
        "row_offsets": rows,
        "row_keys": row_keys,
        "candidate_index": [int(candidate_values[row]) for row in rows],
        "track_id_provenance": [int(track_values[row]) for row in rows],
        "pool_id_provenance": [int(pool_values[row]) for row in rows],
        "raw_rank_provenance": [int(rank_values[row]) for row in rows],
        "main_count": int(main_count),
        "reserve_count": int(reserve_count),
        "duplicate_candidate_index": sorted(duplicate_candidates),
        "history_row_offsets": history_rows,
        "history_frame_ids": history_frames,
        "history_positive": history_positive,
        "continuation_target": continuation_target,
        "bank_path": str(bank.path),
        "bank_sha256": bank.sha256,
        "label_path": str(bank.label_path),
        "label_count": len(bank.labels),
        "old_l49_range_ignored": True,
        "old_l49_positive_indices_ignored": True,
        "source_pool_group_ids_are_provenance_only": True,
    }


def unit_tensors(record: dict[str, Any], bank: L69Bank, text_cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Materialize one unit's small CPU tensors; no persistent feature cache."""
    rows = [int(row) for row in record["row_offsets"]]
    current = feature_matrix(bank.tensors, rows)
    n = len(rows)
    history = torch.zeros((n, MAX_HISTORY, OBS_DIM), dtype=torch.float32)
    history_mask = torch.zeros((n, MAX_HISTORY), dtype=torch.bool)
    history_time = torch.zeros((n, MAX_HISTORY), dtype=torch.float32)
    history_target = torch.zeros((n, MAX_HISTORY), dtype=torch.float32)
    flat: list[int] = []
    placements: list[tuple[int, int, int]] = []
    for i, selected in enumerate(record["history_row_offsets"]):
        length = len(selected)
        for j, row in enumerate(selected):
            flat.append(int(row))
            placements.append((i, j, int(row)))
    if flat:
        values = feature_matrix(bank.tensors, flat)
        by_row_occurrence: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(flat):
            by_row_occurrence[row].append(index)
        used: dict[int, int] = defaultdict(int)
        for i, slot, row in placements:
            take = used[row]
            source_index = by_row_occurrence[row][take]
            used[row] += 1
            history[i, slot] = values[source_index]
            history_mask[i, slot] = True
            history_target[i, slot] = float(
                record["history_positive"][i][slot]
            )
            frame = int(record["history_frame_ids"][i][slot])
            history_time[i, slot] = float(np.clip((frame - int(record["frame_id"])) / 8.0, -8.0, 0.0))
    sentence = record["sentence"]
    if sentence not in text_cache["sentence_to_index"]:
        raise KeyError(f"sentence missing from L48 text cache: {sentence!r}")
    text_index = int(text_cache["sentence_to_index"][sentence])
    text = text_cache["token_hidden"][text_index].float().clone()
    text_mask = text_cache["attention_mask"][text_index].bool().clone()
    return {
        "current": current,
        "history": history,
        "history_mask": history_mask,
        "history_time": history_time,
        "history_target": history_target,
        "membership_target": torch.as_tensor(record["labels"], dtype=torch.float32),
        "coverage_mask": torch.full((n,), bool(record["coverage_mask"]), dtype=torch.bool),
        "track_target": torch.as_tensor(record["labels"], dtype=torch.float32),
        "continuation_target": torch.as_tensor(record["continuation_target"], dtype=torch.float32),
        "null_target": torch.tensor(float(record["null_target"]), dtype=torch.float32),
        "text": text,
        "text_mask": text_mask,
    }


def fixed_eval_units(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    lookup = {unit_key(row): row for rows in splits.values() for row in rows}
    result: list[dict[str, Any]] = []
    for row in load_l62_order():
        key = str(row["unit_key"])
        if key not in lookup:
            raise KeyError(f"L62 fixed key missing from L49 units: {key}")
        result.append(lookup[key])
    return result


def dataset_video_counts(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        counts[str(row["dataset"])][str(row["video"])] += 1
    return {dataset: dict(counter) for dataset, counter in counts.items()}
