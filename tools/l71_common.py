"""Independent L71 indexing and streaming tensor helpers for the L69 bank.

L71 intentionally rebuilds frame ranges from the L69 budget-40 bank.  The
old L49 ``begin/end`` and ``positive_indices`` fields are never used to
address candidates.  IDs are retained only in provenance and for causal
history assembly.
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_torch_load(path: Path) -> Any:
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
    row_list = [int(row) for row in rows]
    index = torch.as_tensor(row_list, dtype=torch.long)
    parts: list[torch.Tensor] = []
    for name in OBS_FIELDS:
        value = tensors[name].index_select(0, index).float().reshape(len(row_list), -1)
        parts.append(value)
    result = torch.cat(parts, dim=1)
    if result.shape[1] != OBS_DIM:
        raise AssertionError(f"observation dimension drift: {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise AssertionError("nonfinite L69 observation feature")
    return result


class L71Bank:
    """Read-only, one-video view of the materialized L69 feature bank."""

    def __init__(self, video: str):
        self.video = str(video)
        if self.video not in L69_VIDEOS:
            raise ValueError(f"video outside L69 train pool: {video}")
        self.path = L69_FEATURE_ROOT / f"{self.video}.pt"
        self.label_path = self.path.with_suffix(".labels.json")
        self.blob = safe_torch_load(self.path)
        self.tensors: dict[str, torch.Tensor] = self.blob["tensors"]
        self.labels = json.loads(self.label_path.read_text())["candidate_gt"]
        self.sha256 = sha256_file(self.path)
        self.frame_ranges: dict[int, tuple[int, int, int]] = {}
        self.track_rows: dict[int, list[int]] = defaultdict(list)
        self._check_schema()
        frame_ids = self.tensors["frame_ids"].long().tolist()
        frame_ptr = self.tensors["frame_ptr"].long().tolist()
        frame = self.tensors["frame"].long()
        for frame_index, frame_id in enumerate(frame_ids):
            begin, end = int(frame_ptr[frame_index]), int(frame_ptr[frame_index + 1])
            expected = torch.full((end - begin,), int(frame_id), dtype=frame.dtype)
            if not torch.equal(frame[begin:end], expected):
                raise AssertionError(f"frame row mismatch in {self.path} frame {frame_id}")
            self.frame_ranges[int(frame_id)] = (begin, end, frame_index)
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
            raise AssertionError(f"{self.path}: sidecar {len(self.labels)} != rows {count}")
        if int(self.tensors["frame_ptr"][-1]) != count:
            raise AssertionError(f"{self.path}: frame_ptr terminal mismatch")
        for name in OBS_FIELDS:
            if int(self.tensors[name].shape[0]) != count:
                raise AssertionError(f"{self.path}: {name} row mismatch")
        if not all(torch.isfinite(self.tensors[name].float()).all() for name in OBS_FIELDS):
            raise AssertionError(f"{self.path}: nonfinite observation field")

    @property
    def count(self) -> int:
        return int(self.tensors["track_id"].numel())

    def close(self) -> None:
        self.blob = None
        self.tensors = {}
        self.labels = []
        self.frame_ranges = {}
        self.track_rows = {}


def _row_key(dataset: str, video: str, query_id: int, frame_id: int, path: Path, row: int) -> list[Any]:
    return [str(dataset), str(video), int(query_id), int(frame_id), str(path), int(row)]


def _history_for_row(bank: L71Bank, row: int, frame_id: int) -> list[int]:
    track = int(bank.tensors["track_id"][row])
    frame_values = bank.tensors["frame"].long()
    prior = [old for old in bank.track_rows.get(track, []) if int(frame_values[old]) < frame_id]
    prior = sorted(prior, key=lambda old: (int(frame_values[old]), old))
    return prior[-(MAX_HISTORY - 1):] + [int(row)]


def make_unit_record(unit: dict[str, Any], bank: L71Bank) -> dict[str, Any]:
    dataset = str(unit["dataset"])
    video = str(unit["video"])
    frame_id = int(unit["frame_id"])
    if video != bank.video:
        raise AssertionError(f"unit/video mismatch: {video} vs {bank.video}")
    if frame_id not in bank.frame_ranges:
        raise KeyError(f"missing L69 frame {dataset}/{video}/{frame_id}")
    begin, end, frame_index = bank.frame_ranges[frame_id]
    rows = list(range(begin, end))
    # Feature construction and frame/key checks occur before label joining.
    _ = feature_matrix(bank.tensors, rows)
    target_ids = normalize_ids(unit.get("target_ids", []))
    sidecar = [None if value is None else str(value) for value in bank.labels[begin:end]]
    positive = [value is not None and value in target_ids for value in sidecar]
    positive_rows = [idx for idx, value in enumerate(positive) if value]
    if len(positive_rows) > 1:
        category = "multi_positive"
    elif positive_rows:
        category = "positive"
    elif target_ids:
        category = "present_uncovered"
    else:
        category = "inactive"
    coverage_mask = not (bool(target_ids) and not bool(positive_rows))

    candidate_values = bank.tensors["candidate_index"].long().tolist()
    track_values = bank.tensors["track_id"].long().tolist()
    pool_values = bank.tensors["pool_id"].long().tolist()
    raw_rank_values = bank.tensors["raw_rank"].long().tolist()
    history_rows: list[list[int]] = []
    history_frames: list[list[int]] = []
    history_positive: list[list[int]] = []
    for row in rows:
        chosen = _history_for_row(bank, row, frame_id)
        if int(chosen[-1]) != int(row):
            raise AssertionError("current observation is not the final causal history slot")
        if any(int(bank.tensors["frame"][old]) > frame_id for old in chosen):
            raise AssertionError("future observation entered history")
        hframes = [int(bank.tensors["frame"][old]) for old in chosen]
        hpos = [int(bank.labels[old] is not None and str(bank.labels[old]) in target_ids) for old in chosen]
        history_rows.append(chosen)
        history_frames.append(hframes)
        history_positive.append(hpos)

    query_id = int(unit["query_id"])
    row_keys = [_row_key(dataset, video, query_id, frame_id, bank.path, row) for row in rows]
    if [key[2] for key in row_keys] != [query_id] * len(rows) or [key[-1] for key in row_keys] != rows:
        raise AssertionError("L69 row order drift")
    duplicate_candidates = sorted(
        int(value) for value, count in Counter(candidate_values[row] for row in rows).items() if count > 1
    )
    main_count = sum(int(raw_rank_values[row] == -1) for row in rows)
    return {
        "format": "locatemot-l71-unit-index-v1",
        "status": "complete",
        "dataset": dataset,
        "video": video,
        "query_id": query_id,
        "frame_id": frame_id,
        "unit_key": unit_key(unit),
        "sentence": str(unit.get("sentence", unit.get("expression", ""))),
        "expression": str(unit.get("expression", "")),
        "split": str(unit.get("split", "unknown")),
        "frame_index": int(frame_index),
        "frame_pointer": {"begin": begin, "end": end, "count": end - begin},
        "target_ids": sorted(target_ids),
        "candidate_present": bool(positive_rows),
        "coverage_mask": bool(coverage_mask),
        "null_target": int(not bool(target_ids)),
        "category": category,
        "source_category_from_l49": str(unit.get("category", "unknown")),
        "positive_row_indices": positive_rows,
        "positive_count": len(positive_rows),
        "candidate_count": len(rows),
        "labels": [int(value) for value in positive],
        "sidecar_candidate_gt": sidecar,
        "row_offsets": rows,
        "row_keys": row_keys,
        "candidate_index_provenance": [int(candidate_values[row]) for row in rows],
        "track_id_provenance": [int(track_values[row]) for row in rows],
        "pool_id_provenance": [int(pool_values[row]) for row in rows],
        "raw_rank_provenance": [int(raw_rank_values[row]) for row in rows],
        "main_count": int(main_count),
        "reserve_count": int(len(rows) - main_count),
        "duplicate_candidate_index": duplicate_candidates,
        "history_row_offsets": history_rows,
        "history_frame_ids": history_frames,
        "history_positive": history_positive,
        "bank_path": str(bank.path),
        "bank_sha256": bank.sha256,
        "label_path": str(bank.label_path),
        "label_count": len(bank.labels),
        "old_l49_begin_end_ignored": True,
        "old_l49_positive_indices_ignored": True,
        "source_pool_group_ids_are_provenance_only": True,
    }


def unit_tensors(record: dict[str, Any], bank: L71Bank, text_cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    rows = [int(row) for row in record["row_offsets"]]
    current = feature_matrix(bank.tensors, rows)
    n = len(rows)
    history = torch.zeros((n, MAX_HISTORY, OBS_DIM), dtype=torch.float32)
    history_mask = torch.zeros((n, MAX_HISTORY), dtype=torch.bool)
    history_time = torch.zeros((n, MAX_HISTORY), dtype=torch.float32)
    placements: list[tuple[int, int, int]] = []
    flat: list[int] = []
    for candidate, selected in enumerate(record["history_row_offsets"]):
        for slot, row in enumerate(selected):
            flat.append(int(row))
            placements.append((candidate, slot, int(row)))
    if flat:
        values = feature_matrix(bank.tensors, flat)
        for value_index, (candidate, slot, row) in enumerate(placements):
            history[candidate, slot] = values[value_index]
            history_mask[candidate, slot] = True
            frame = int(record["history_frame_ids"][candidate][slot])
            history_time[candidate, slot] = float(np.clip((frame - int(record["frame_id"])) / 8.0, -8.0, 0.0))
    sentence = record["sentence"]
    if sentence not in text_cache["sentence_to_index"]:
        raise KeyError(f"sentence missing from L48 cache: {sentence!r}")
    text_index = int(text_cache["sentence_to_index"][sentence])
    text = text_cache["token_hidden"][text_index].float().clone()
    text_mask = text_cache["attention_mask"][text_index].bool().clone()
    return {
        "current": current,
        "history": history,
        "history_mask": history_mask,
        "history_time": history_time,
        "membership_target": torch.as_tensor(record["labels"], dtype=torch.float32),
        "coverage_mask": torch.full((n,), bool(record["coverage_mask"]), dtype=torch.bool),
        "text": text,
        "text_mask": text_mask,
        "null_target": torch.tensor(float(record["null_target"]), dtype=torch.float32),
    }


def fixed_eval_units(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    lookup = {unit_key(row): row for values in splits.values() for row in values}
    result: list[dict[str, Any]] = []
    for row in load_l62_order():
        key = str(row["unit_key"])
        if key not in lookup:
            raise KeyError(f"fixed L62 key missing from L49 split files: {key}")
        result.append(lookup[key])
    return result


def dataset_video_counts(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        counts[str(row["dataset"])][str(row["video"])] += 1
    return {dataset: dict(counter) for dataset, counter in counts.items()}
