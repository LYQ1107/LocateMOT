"""R1-specific data, provenance, and deterministic request helpers.

This file intentionally keeps the R1 sidecar separate from the production
LocateMOT loaders.  The R0A dense indexes are read-only, frame-specific
artifacts; their target payload is opened lazily by the training/evaluation
caller after the complete feature record exists.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
R0A_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0A").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
FIT_ROOTS = {
    "refer_kitti_v1": R0A_ROOT / "outputs/r0/data/v1_dense_train_index_retry2",
    "refer_kitti_v2": R0A_ROOT / "outputs/r0/data/v2_dense_train_index_retry2",
}
DEV_ROOTS = {
    "refer_kitti_v1": R0A_ROOT / "outputs/r0/data/v1_dev_index_retry1",
    "refer_kitti_v2": R0A_ROOT / "outputs/r0/data/v2_dev_index_retry1",
}
FORBIDDEN_VIDEOS = {"0005", "0011", "0013", "0019"}
CATEGORIES = ("positive", "multi_positive", "inactive", "present_uncovered")
QUOTA = {"multi_positive": 3, "positive": 2, "inactive": 2, "present_uncovered": 1}
ACCUMULATION = {1: 16, 2: 8, 3: 5, 4: 4}

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    path = Path(path).resolve()
    if not path.is_dir():
        return "missing"
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def check_manifest() -> str:
    observed = sha256_file(MANIFEST)
    if observed != MANIFEST_SHA:
        raise AssertionError(f"fixed manifest SHA drift: {observed}")
    return observed


def standard_flags(*, training_run: bool = False, hota_trackeval_run: bool = False) -> dict[str, Any]:
    return {
        "training_run": bool(training_run),
        "hota_trackeval_run": bool(hota_trackeval_run),
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "tracker_source_changed": False,
        "uidm_source_changed": False,
        "l69_source_changed": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED",
    }


def load_indexes(scope: str) -> dict[str, Any]:
    from tools.r0_common import DenseIndex
    roots = FIT_ROOTS if scope == "fit" else DEV_ROOTS if scope == "dev" else None
    if roots is None:
        raise ValueError(f"unsupported R1 scope: {scope}")
    result = {dataset: DenseIndex(root) for dataset, root in roots.items()}
    for dataset, dense in result.items():
        if any(str(record["video"]) in FORBIDDEN_VIDEOS for record in dense.label_records):
            raise AssertionError(f"forbidden official video in R1 {scope} index: {dataset}")
    return result


def load_fixed_l62_key_order() -> list[dict[str, Any]]:
    """Resolve the immutable 40-unit order without retaining L62 labels.

    L62 is consulted only for its already frozen order.  Sentence and split
    metadata are joined from the corresponding L49 calibration/validation
    files.  Target arrays, positive indices, categories, and scores are never
    present in the returned records.
    """
    l62_path = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
    cal_path = ROOT / "outputs/l49/data/calibration_units.jsonl"
    val_path = ROOT / "outputs/l49/data/validation_units.jsonl"
    if not l62_path.is_file() or not cal_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("R1 fixed-order input missing")
    metadata: dict[str, dict[str, Any]] = {}
    for path, split in ((cal_path, "calibration"), (val_path, "validation")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            key = str(raw["unit_key"])
            if key in metadata:
                raise AssertionError(f"duplicate L49 fixed unit key: {key}")
            metadata[key] = {
                "dataset": str(raw["dataset"]), "video": str(raw["video"]),
                "query_id": int(raw["query_id"]), "frame_id": int(raw["frame_id"]),
                "sentence": str(raw.get("sentence") or raw.get("expression") or ""),
                "unit_key": key, "split": split,
            }
    order: list[dict[str, Any]] = []
    for line in l62_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        key = str(raw["unit_key"])
        item = metadata.get(key)
        if item is None:
            raise AssertionError(f"fixed L62 key absent from L49 split files: {key}")
        order.append(dict(item))
    if len(order) != 40 or len({str(value["unit_key"]) for value in order}) != 40:
        raise AssertionError("R1 fixed 40-unit key/order contract failed")
    if [value["split"] for value in order[:16]] != ["calibration"] * 16 or \
       [value["split"] for value in order[16:]] != ["validation"] * 24:
        raise AssertionError("R1 fixed order is not 16 calibration followed by 24 validation")
    return order


def public_record(record: dict[str, Any], *, category: str | None = None) -> dict[str, Any]:
    """Return metadata safe to pass into feature construction.

    Target/positive arrays are deliberately not copied.  ``category`` is only
    included by the deterministic training sampler and is never serialized in
    a feature cache.
    """
    allowed = {
        "dataset": str(record["dataset"]), "video": str(record["video"]),
        "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]),
        "sentence": str(record["sentence"]), "candidate_count": int(record["candidate_count"]),
        "bank_path": str(record["bank_path"]),
        "unit_key": f"{record['dataset']}|{record['video']}|{int(record['query_id'])}|{int(record['frame_id'])}",
    }
    if category is not None:
        if str(category) not in CATEGORIES:
            raise ValueError(category)
        allowed["category"] = str(category)
    return allowed


def unit_key(record: dict[str, Any]) -> str:
    return f"{record['dataset']}|{record['video']}|{int(record['query_id'])}|{int(record['frame_id'])}"


def frame_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return str(record["dataset"]), str(record["video"]), int(record["frame_id"])


def record_digest(records: Iterable[dict[str, Any]]) -> str:
    values = [unit_key(record) for record in records]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def label_from_index(dense: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Read a frame label only at the explicit supervision boundary."""
    raw = dense.get_label(record)
    forbidden = {"screening", "official", "test"}
    if str(record["video"]) in FORBIDDEN_VIDEOS or any(x in str(raw).lower() for x in forbidden):
        raise AssertionError("R1 attempted forbidden label scope")
    return raw


def select_query_tile(frame_records: list[dict[str, Any]], rng: random.Random, max_queries: int = 8) -> list[dict[str, Any]]:
    if not frame_records:
        raise ValueError("empty R1 frame record group")
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    for record in frame_records:
        category = str(record["category"])
        if category in buckets:
            buckets[category].append(record)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for category in ("multi_positive", "positive", "inactive", "present_uncovered"):
        for record in buckets[category][: QUOTA[category]]:
            query_id = int(record["query_id"])
            if query_id not in used and len(selected) < int(max_queries):
                selected.append(record); used.add(query_id)
    if len(selected) < int(max_queries):
        remaining = [record for record in frame_records if int(record["query_id"]) not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[: int(max_queries) - len(selected)])
    selected.sort(key=lambda value: int(value["query_id"]))
    if not selected or len({int(value["query_id"]) for value in selected}) != len(selected):
        raise AssertionError("R1 tile query uniqueness failed")
    identity = frame_key(selected[0])
    if any(frame_key(value) != identity for value in selected):
        raise AssertionError("R1 tile mixes native frames")
    return selected


def build_epoch_frame_schedule(dense: Any, epoch: int, tiles: int, seed: int) -> list[tuple[str, str, int]]:
    if not dense.frame_keys:
        raise AssertionError("empty R1 dense frame index")
    result: list[tuple[str, str, int]] = []
    cycle = 0
    while len(result) < int(tiles):
        rng = random.Random(int(seed) + int(epoch) * 1009 + cycle * 1_000_003)
        values = list(dense.frame_keys); rng.shuffle(values)
        result.extend(values[: max(0, int(tiles) - len(result))])
        cycle += 1
    return result


def build_request_manifest(indexes: dict[str, Any], *, epochs: int = 6, tiles_per_epoch: int = 3000,
                           seed: int = SEED, max_queries: int = 8) -> dict[str, Any]:
    """Build deterministic frame/query requests before any cache extraction."""
    all_tiles: dict[str, list[dict[str, Any]]] = {}
    all_queries: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        dense = indexes[dataset]
        dataset_tiles: list[dict[str, Any]] = []
        query_map: dict[str, dict[str, Any]] = {}
        for epoch in range(1, int(epochs) + 1):
            schedule = build_epoch_frame_schedule(dense, epoch, tiles_per_epoch, seed)
            rng = random.Random(int(seed) + int(epoch) * 1009 + 31_000_019)
            for tile_index, key in enumerate(schedule):
                selected = select_query_tile(dense.by_frame[key], rng, max_queries=max_queries)
                public = [public_record(record, category=str(record["category"])) for record in selected]
                tile = {
                    "epoch": int(epoch), "tile_index": int(tile_index),
                    "dataset": dataset, "video": key[1], "frame_id": int(key[2]),
                    "query_unit_keys": [unit_key(record) for record in public],
                    "query_categories": [str(record["category"]) for record in selected],
                    "candidate_count": int(selected[0]["candidate_count"]),
                }
                dataset_tiles.append(tile)
                for record in public:
                    query_map[unit_key(record)] = record
        all_tiles[dataset] = dataset_tiles
        all_queries[dataset] = query_map
    category_counts = {dataset: dict(sorted(Counter(category for tile in tiles for category in tile["query_categories"]).items()))
                       for dataset, tiles in all_tiles.items()}
    return {
        "format": "locatemot-r1-request-manifest-v1", "status": "complete", "seed": int(seed),
        "epochs": int(epochs), "tiles_per_epoch": int(tiles_per_epoch), "max_queries": int(max_queries),
        "datasets": list(all_tiles), "tiles": all_tiles, "unique_query_records": all_queries,
        "category_counts": category_counts,
        "flags": standard_flags(training_run=True),
        "labels_in_cache": False, "cache_is_query_conditioned": True,
    }


def request_records_for_tile(manifest: dict[str, Any], dataset: str, epoch: int, tile_index: int,
                            indexes: dict[str, Any]) -> list[dict[str, Any]]:
    tiles = [value for value in manifest["tiles"][dataset]
             if int(value["epoch"]) == int(epoch) and int(value["tile_index"]) == int(tile_index)]
    if len(tiles) != 1:
        raise KeyError((dataset, epoch, tile_index))
    dense = indexes[dataset]
    by_key = {unit_key(record): record for record in dense.label_records}
    result = [public_record(by_key[key], category=str(by_key[key]["category"])) for key in tiles[0]["query_unit_keys"]]
    if [unit_key(value) for value in result] != list(tiles[0]["query_unit_keys"]):
        raise AssertionError("R1 request query order drift")
    return result


def provenance_inputs() -> dict[str, Any]:
    return {
        "thread": THREAD, "root": str(ROOT), "worktree": str(WORK_ROOT), "manifest": file_meta(MANIFEST),
        "fit_indexes": {key: file_meta(value / "summary.json") for key, value in FIT_ROOTS.items()},
        "dev_indexes": {key: file_meta(value / "summary.json") for key, value in DEV_ROOTS.items()},
        "fit_index_fingerprints": {key: directory_fingerprint(value) for key, value in FIT_ROOTS.items()},
        "dev_index_fingerprints": {key: directory_fingerprint(value) for key, value in DEV_ROOTS.items()},
        "forbidden_videos": sorted(FORBIDDEN_VIDEOS), "seed": SEED,
    }


__all__ = [
    "ACCUMULATION", "CATEGORIES", "DEV_ROOTS", "FIT_ROOTS", "FORBIDDEN_VIDEOS", "MANIFEST", "MANIFEST_SHA",
    "QUOTA", "ROOT", "SEED", "THREAD", "WORK_ROOT", "build_request_manifest", "check_manifest", "file_meta",
    "frame_key", "label_from_index", "load_fixed_l62_key_order", "load_indexes", "provenance_inputs", "public_record", "record_digest",
    "request_records_for_tile", "select_query_tile", "sha256_file", "standard_flags", "unit_key", "write_json",
]
