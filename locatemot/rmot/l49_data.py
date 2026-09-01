"""Auditable Refer-KITTI V1/V2 data helpers for the L49 RMOT branch.

Only the official train-pool videos are materialized here.  Official test
videos are represented as metadata in the contract and are deliberately not
loaded by these helpers.  Dataset/video/query/frame/candidate keys remain
explicit so the two versions cannot silently share a semantic sample.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
FAST_MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
TEXT_CACHE = ROOT / "outputs/l48/data/text_cache.pt"
V1_EXPR = ROOT / "outputs/l13/data/refer_kitti_v1/expression"
V1_META = ROOT / "outputs/l13/data/refer_kitti_v1/expressions.json"
V2_META_OLD = ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
V2_META_NEW = ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
KITTI_BANK = ROOT / "outputs/l19/dual_banks_features/kitti"
L29_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"

# These are all inside the official train pool.  The third split is reserved
# before any L49 training/selection and is disjoint by video from fit/val.
L49_SPLITS: dict[str, dict[str, list[str]]] = {
    "refer_kitti_v1": {
        "fit": ["0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0020"],
        "calibration": ["0016"],
        "validation": ["0004", "0018"],
        "official_eval": ["0005", "0011", "0013"],
    },
    "refer_kitti_v2": {
        "fit": ["0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014"],
        "calibration": ["0015"],
        "validation": ["0016", "0017", "0020"],
        "official_eval": ["0005", "0011", "0013", "0019"],
    },
}

REQUIRED_TENSORS = (
    "clip", "history_clip", "geometry", "motion", "context", "lifecycle",
    "objectness", "box", "track_id", "candidate_index", "frame",
    "frame_ids", "frame_ptr",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _targets(value: Any) -> dict[int, set[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, set[str]] = {}
    for frame, ids in value.items():
        if ids is None:
            result[int(frame)] = set()
        elif isinstance(ids, (list, tuple, set)):
            result[int(frame)] = {str(x) for x in ids}
        else:
            result[int(frame)] = {str(ids)}
    return result


def _query(dataset: str, video: str, item: dict[str, Any], source: str) -> dict[str, Any]:
    expression = str(item.get("expression", ""))
    return {
        "dataset": dataset,
        "video": str(video),
        "expression": expression,
        "sentence": str(item.get("sentence", expression)),
        "target": _targets(item.get("label", {})),
        "label_source": source,
    }


def load_l49_queries(dataset: str) -> list[dict[str, Any]]:
    """Load expression-level queries from the train pool only."""
    if dataset not in L49_SPLITS:
        raise KeyError(dataset)
    allowed = set(sum((L49_SPLITS[dataset][key] for key in
                      ("fit", "calibration", "validation")), []))
    rows: list[dict[str, Any]] = []
    if dataset == "refer_kitti_v1":
        for video in sorted(allowed):
            for path in sorted((V1_EXPR / video).glob("*.json")):
                rows.append(_query(dataset, video, json.loads(path.read_text()), str(path)))
    else:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for path in (V2_META_OLD, V2_META_NEW):
            data = json.loads(path.read_text())
            for video, items in data.items():
                if str(video) not in allowed:
                    continue
                for item in items:
                    key = (str(video), str(item["expression"]))
                    merged[key] = _query(dataset, str(video), item, str(path))
        rows = list(merged.values())
    rows.sort(key=lambda x: (x["video"], x["expression"], x["sentence"]))
    for query_id, row in enumerate(rows):
        row["query_id"] = int(query_id)
        row["split"] = next(
            split for split in ("fit", "calibration", "validation")
            if row["video"] in L49_SPLITS[dataset][split]
        )
    return rows


def bank_path(dataset: str, video: str) -> Path:
    if dataset not in L49_SPLITS:
        raise KeyError(dataset)
    return KITTI_BANK / f"{video}.pt"


def load_bank(dataset: str, video: str) -> dict[str, Any]:
    path = bank_path(dataset, video)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tensors = blob["tensors"]
    label_path = path.with_suffix(".labels.json")
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    labels = json.loads(label_path.read_text()).get("candidate_gt", [])
    count = int(tensors["track_id"].numel())
    if len(labels) != count:
        raise AssertionError(f"{path}: labels={len(labels)} rows={count}")
    track_rows: dict[int, list[int]] = {}
    for row, track_id in enumerate(tensors["track_id"].long().tolist()):
        track_rows.setdefault(int(track_id), []).append(int(row))
    return {
        "path": path,
        "metadata": blob.get("metadata", {}),
        "tensors": tensors,
        "labels": [None if x is None else str(x) for x in labels],
        "label_path": label_path,
        "track_rows": track_rows,
    }


def frame_descriptor(query: dict[str, Any], bank: dict[str, Any], frame_index: int) -> dict[str, Any]:
    tensors = bank["tensors"]
    begin = int(tensors["frame_ptr"][frame_index])
    end = int(tensors["frame_ptr"][frame_index + 1])
    frame = int(tensors["frame_ids"][frame_index])
    targets = query["target"].get(frame, set())
    labels = bank["labels"][begin:end]
    positive = np.asarray(
        [value is not None and str(value) in targets for value in labels], dtype=bool
    )
    if int(positive.sum()) > 1:
        category = "multi_positive"
    elif bool(positive.any()):
        category = "positive"
    elif targets:
        category = "present_uncovered"
    else:
        category = "inactive"
    return {
        "dataset": query["dataset"], "video": query["video"],
        "query_id": int(query["query_id"]), "expression": query["expression"],
        "sentence": query["sentence"], "split": query["split"],
        "frame_index": int(frame_index), "frame_id": frame,
        "begin": begin, "end": end, "candidate_count": int(end - begin),
        "target_ids": sorted(str(x) for x in targets),
        "positive_indices": np.flatnonzero(positive).astype(int).tolist(),
        "positive_count": int(positive.sum()), "category": category,
        "image_size": list(bank["metadata"].get("image_size", [])),
        "bank_path": str(bank["path"]), "label_path": str(bank["label_path"]),
        "unit_key": f"{query['dataset']}|{query['video']}|{query['query_id']}|{frame}",
    }


def unit_row_key(unit: dict[str, Any], candidate_index: int, track_id: int) -> tuple[Any, ...]:
    return (unit["dataset"], unit["video"], int(unit["query_id"]),
            int(unit["frame_id"]), int(candidate_index), int(track_id))


def relation_features(boxes: torch.Tensor, image_size: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Nearest-neighbour geometry, computed without labels/source metadata."""
    boxes = boxes.float()
    if len(boxes) == 0:
        return boxes.new_zeros((0, 4))
    width = float(image_size[0]) if len(image_size) >= 2 else max(1.0, float(boxes[:, 2].max()))
    height = float(image_size[1]) if len(image_size) >= 2 else max(1.0, float(boxes[:, 3].max()))
    scale = boxes.new_tensor([width, height, width, height])
    norm = boxes / scale
    centers = (norm[:, :2] + norm[:, 2:]) * 0.5
    if len(boxes) == 1:
        return boxes.new_zeros((1, 4))
    delta = centers[:, None, :] - centers[None, :, :]
    distance = delta.square().sum(-1)
    distance.fill_diagonal_(float("inf"))
    nearest = distance.argmin(-1)
    rows = torch.arange(len(boxes), device=boxes.device)
    other = norm[nearest]
    left = torch.maximum(norm[:, None, 0], other[None, :, 0])
    top = torch.maximum(norm[:, None, 1], other[None, :, 1])
    right = torch.minimum(norm[:, None, 2], other[None, :, 2])
    bottom = torch.minimum(norm[:, None, 3], other[None, :, 3])
    inter = (right - left).clamp_min(0) * (bottom - top).clamp_min(0)
    area = (norm[:, 2] - norm[:, 0]).clamp_min(0) * (norm[:, 3] - norm[:, 1]).clamp_min(0)
    other_area = (other[:, 2] - other[:, 0]).clamp_min(0) * (other[:, 3] - other[:, 1]).clamp_min(0)
    selected_inter = inter[rows, nearest]
    iou = selected_inter / (area + other_area[nearest] - selected_inter).clamp_min(1e-6)
    return torch.cat((delta[rows, nearest], iou[:, None], distance[rows, nearest, None].sqrt()), -1)


def history_sequence(bank: dict[str, Any], begin: int, end: int, length: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a causal, per-track history from the frozen bank rows.

    The current row is the last eligible observation.  This is an existing
    observation/history stream, not a new backbone or a GT-driven selection.
    """
    tensors = bank["tensors"]
    count = int(end - begin)
    values = torch.zeros((count, int(length), 512), dtype=torch.float32)
    mask = torch.zeros((count, int(length)), dtype=torch.bool)
    frames = tensors["frame"].long()
    history = tensors["history_clip"].float()
    for offset, row in enumerate(range(int(begin), int(end))):
        track = int(tensors["track_id"][row])
        candidates = bank["track_rows"].get(track, [row])
        frame = int(frames[row])
        eligible = [candidate for candidate in candidates if int(frames[candidate]) <= frame]
        chosen = eligible[-int(length):]
        start = int(length) - len(chosen)
        if chosen:
            index = torch.as_tensor(chosen, dtype=torch.long)
            values[offset, start:] = history[index]
            mask[offset, start:] = True
    return values, mask


def unit_features(unit: dict[str, Any], bank: dict[str, Any], text_payload: dict[str, Any], history: int = 8) -> dict[str, torch.Tensor]:
    """Materialize one complete current-frame candidate set on CPU."""
    tensors = bank["tensors"]
    begin, end = int(unit["begin"]), int(unit["end"])
    sl = slice(begin, end)
    text_index = text_payload["sentence_to_index"][unit["sentence"]]
    labels = torch.zeros(end - begin, dtype=torch.bool)
    if unit["positive_indices"]:
        labels[torch.as_tensor(unit["positive_indices"], dtype=torch.long)] = True
    sequence, sequence_mask = history_sequence(bank, begin, end, history)
    return {
        "clip": tensors["clip"][sl].float(),
        "history_clip": tensors["history_clip"][sl].float(),
        "geometry": tensors["geometry"][sl].float(),
        "motion": tensors["motion"][sl].float(),
        "context": tensors["context"][sl].float(),
        "lifecycle": tensors["lifecycle"][sl].float(),
        "objectness": tensors["objectness"][sl].float().reshape(-1),
        "relation": relation_features(tensors["box"][sl], unit.get("image_size", [])),
        "history_sequence": sequence,
        "history_mask": sequence_mask,
        "text": text_payload["token_hidden"][text_index].float(),
        "text_mask": text_payload["attention_mask"][text_index].bool(),
        "target": labels,
    }
