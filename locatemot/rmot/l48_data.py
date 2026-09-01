"""Shared, auditable data contract helpers for the Stage L48 RMOT branch.

The module deliberately exposes only expression-level labels and frozen
candidate features.  Dataset/source/pool identifiers are provenance fields and
are never assembled into model tensors by the L48 model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
SPLIT = ROOT / "outputs/l16/data/protocol/split_manifest.json"
FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
V1_EXPR = ROOT / "outputs/l13/data/refer_kitti_v1/expression"
V2_EXP_OLD = ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
V2_EXP_NEW = ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
DANCE_EXP = ROOT / "outputs/l16/data/protocol/refer_dance_expressions.json"
KITTI_BANK = ROOT / "outputs/l19/dual_banks_features/kitti"
DANCE_BANK = ROOT / "outputs/l16/track_banks/dance_train"
V5_TEXT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal/text_tokens.pt"
V5_TEXT_MANIFEST = ROOT / "outputs/l26/candidate_bank_v5_crossmodal/text_manifest.json"
L29_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"

DOMAIN_ORDER = ("refer_kitti_v1", "refer_kitti_v2", "refer_dance")
FEATURE_FIELDS = (
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


def _as_targets(value: Any) -> dict[int, set[str]]:
    if not isinstance(value, dict):
        return {}
    out: dict[int, set[str]] = {}
    for frame, ids in value.items():
        if ids is None:
            out[int(frame)] = set()
        elif isinstance(ids, (list, tuple, set)):
            out[int(frame)] = {str(x) for x in ids}
        else:
            out[int(frame)] = {str(ids)}
    return out


def _query(dataset: str, video: str, expression: str, sentence: str,
           label: Any, source: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "video": str(video),
        "expression": str(expression),
        "sentence": str(sentence or expression),
        "target": _as_targets(label),
        "label_source": str(source),
    }


def split_maps() -> dict[str, dict[str, list[str]]]:
    """Return internal fit/validation maps while leaving official eval unseen.

    The protocol's official evaluation video lists are read as split metadata,
    never as labels.  The internal validation choices are fixed before any
    model or threshold is evaluated.
    """
    protocol = json.loads(SPLIT.read_text())
    kitti = protocol["kitti_v2"]
    dance = protocol["refer_dance"]
    v1_eval = {str(x) for x in json.loads(
        (ROOT / "outputs/l13/data/refer_kitti_v1/build_manifest.json").read_text()
    )["eval_sequences"]}
    v1_all = sorted(p.name for p in V1_EXPR.iterdir() if p.is_dir())
    v1_train = sorted(set(v1_all) - v1_eval)
    v1_val = [x for x in ("0004", "0018") if x in v1_train]
    v2_train = [str(x) for x in kitti["train"]]
    # Keep this validation set inside the official V2 train pool, but disjoint
    # from fit.  It is deliberately different from the V1 internal holdout.
    v2_val = [x for x in ("0016", "0017", "0020") if x in v2_train]
    dance_train = [str(x) for x in dance["train"]]
    dance_val = [str(x) for x in dance["train_val"]]
    return {
        "refer_kitti_v1": {
            "fit": sorted(set(v1_train) - set(v1_val)),
            "validation": sorted(v1_val),
            "official_eval": sorted(v1_eval),
        },
        "refer_kitti_v2": {
            "fit": sorted(set(v2_train) - set(v2_val)),
            "validation": sorted(v2_val),
            "official_eval": [str(x) for x in kitti["official_eval"]],
        },
        "refer_dance": {
            "fit": sorted(dance_train),
            "validation": sorted(dance_val),
            "official_eval": [str(x) for x in dance["official_eval"]],
        },
    }


def load_queries(dataset: str, split: str | None = None) -> list[dict[str, Any]]:
    """Load V1/V2/Dance expression-level train/validation queries only."""
    maps = split_maps()[dataset]
    allowed = set(maps["fit"] + maps["validation"])
    rows: list[dict[str, Any]] = []
    if dataset == "refer_kitti_v1":
        for video in sorted(allowed):
            for path in sorted((V1_EXPR / video).glob("*.json")):
                item = json.loads(path.read_text())
                rows.append(_query(
                    dataset, video, path.stem,
                    item.get("sentence", path.stem), item.get("label", {}),
                    str(path),
                ))
    elif dataset == "refer_kitti_v2":
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for path in (V2_EXP_OLD, V2_EXP_NEW):
            data = json.loads(path.read_text())
            for video, values in data.items():
                if str(video) not in allowed:
                    continue
                for item in values:
                    key = (str(video), str(item["expression"]))
                    merged[key] = _query(
                        dataset, video, item["expression"],
                        item.get("sentence", item["expression"]),
                        item.get("label", {}), str(path),
                    )
        rows = [merged[key] for key in sorted(merged)]
    elif dataset == "refer_dance":
        data = json.loads(DANCE_EXP.read_text())
        for video in sorted(allowed):
            for item in data.get(video, []):
                rows.append(_query(
                    dataset, video, item["expression"],
                    item.get("sentence", item["expression"]),
                    item.get("label", {}), str(DANCE_EXP),
                ))
    else:
        raise KeyError(dataset)
    rows.sort(key=lambda x: (x["video"], x["expression"], x["sentence"]))
    for query_id, row in enumerate(rows):
        row["query_id"] = int(query_id)
        row["split"] = "fit" if row["video"] in maps["fit"] else "validation"
        if split is not None and row["split"] != split:
            continue
    if split is not None:
        rows = [x for x in rows if x["split"] == split]
    return rows


def bank_path(dataset: str, video: str) -> Path:
    if dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        return KITTI_BANK / f"{video}.pt"
    if dataset == "refer_dance":
        return DANCE_BANK / f"{video}.pt"
    raise KeyError(dataset)


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
    return {
        "path": path,
        "metadata": blob.get("metadata", {}),
        "tensors": tensors,
        "labels": [None if x is None else str(x) for x in labels],
        "label_path": label_path,
    }


def query_summary(query: dict[str, Any]) -> dict[str, Any]:
    target_frames = sum(bool(ids) for ids in query["target"].values())
    target_ids = sum(len(ids) for ids in query["target"].values())
    return {
        "dataset": query["dataset"], "video": query["video"],
        "query_id": int(query["query_id"]),
        "expression": query["expression"], "sentence": query["sentence"],
        "split": query["split"], "label_source": query["label_source"],
        "target_frame_count": int(target_frames),
        "target_id_count": int(target_ids),
        "text_word_count": len(query["sentence"].split()),
    }


def row_label_vector(bank: dict[str, Any], begin: int, end: int,
                     targets: set[str]) -> np.ndarray:
    labels = bank["labels"][begin:end]
    return np.asarray(
        [value is not None and str(value) in targets for value in labels],
        dtype=bool,
    )


def frame_descriptor(query: dict[str, Any], bank: dict[str, Any],
                     frame_index: int) -> dict[str, Any]:
    tensors = bank["tensors"]
    begin = int(tensors["frame_ptr"][frame_index])
    end = int(tensors["frame_ptr"][frame_index + 1])
    frame = int(tensors["frame_ids"][frame_index])
    targets = query["target"].get(frame, set())
    labels = row_label_vector(bank, begin, end, targets)
    if int(labels.sum()) > 1:
        category = "multi_positive"
    elif bool(labels.any()):
        category = "positive"
    elif targets:
        category = "present_uncovered"
    else:
        category = "inactive"
    return {
        "dataset": query["dataset"], "video": query["video"],
        "query_id": int(query["query_id"]),
        "expression": query["expression"], "sentence": query["sentence"],
        "split": query["split"], "frame_index": int(frame_index),
        "frame_id": frame, "begin": begin, "end": end,
        "candidate_count": int(end - begin),
        "target_ids": sorted(str(x) for x in targets),
        "positive_indices": np.flatnonzero(labels).astype(int).tolist(),
        "positive_count": int(labels.sum()), "category": category,
        "bank_path": str(bank["path"]),
        "label_path": str(bank["label_path"]),
        "image_size": list(bank["metadata"].get("image_size", [])),
    }
