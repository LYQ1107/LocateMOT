#!/usr/bin/env python3
"""Shared contracts for the L89D true-native-timeline replay.

L89D is deliberately a deployment/evidence repair.  This module contains no
model or loss code.  It reconstructs the native L69 frame universe, resolves
the already-materialized compact Z1 cache (optionally with a label-free
supplement), and performs the row/timeline checks used by every L89D tool.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = WORK_ROOT / "tools"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
BASE_Z1_CACHE = ASSET_ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
DEFAULT_LANGUAGE_CACHE = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1"
).resolve()
L82_SPLIT = ASSET_ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
L69_ROOT = ASSET_ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_DATA = ASSET_ROOT / "outputs/l49/data"
L62_ROWS = ASSET_ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
INTERNAL_VIDEOS: dict[str, tuple[str, ...]] = {
    "refer_kitti_v1": ("0004", "0018"),
    "refer_kitti_v2": ("0016", "0017", "0020"),
}
RULES = ("B", "R", "P")
FORBIDDEN_CACHE_FIELDS = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "target_present", "candidate_gt", "candidate_scores", "candidate_index",
    "coverage_mask", "declared_category",
}

# The L89D worktree is based on L89C, while the audited L80/L79 data wrapper
# and generated L85 assets are intentionally external, read-only project
# assets.  ``locatemot`` has a regular package initializer in the asset root;
# append the worktree package directories so the L89 model in this worktree is
# importable without copying any old source or data.
for _path in (TOOL_ROOT, WORK_ROOT, ASSET_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import locatemot.models as _models_package  # noqa: E402
import locatemot.rmot as _rmot_package  # noqa: E402

for _package, _name in ((_models_package, "models"), (_rmot_package, "rmot")):
    _candidate = str(WORK_ROOT / "locatemot" / _name)
    if _candidate not in [str(value) for value in _package.__path__]:
        _package.__path__.append(_candidate)

from locatemot.rmot.l80_data import L80BankStore  # noqa: E402
from locatemot.rmot.l85_runtime import (  # noqa: E402
    capture_group_z1_batched,
    build_groups,
    digest_keys,
    load_fit_key_rows,
    load_internal_eval_groups,
    load_validation_key_rows,
)
from locatemot.rmot.l89_language_cache import L89LanguageTokenCache  # noqa: E402
from l86_infer_fullvideo import frame_groups, query_rows_for_video  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path, include_hash: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "bytes": path.stat().st_size if path.exists() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
    }
    if include_hash and path.is_file():
        result["sha256"] = sha256_file(path)
    return result


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def command_line(argv: list[str] | None = None) -> str:
    values = list(sys.argv if argv is None else argv)
    return " ".join(str(value) for value in values)


def standard_flags(*, hota_trackeval_run: bool = False) -> dict[str, Any]:
    return {
        "zero_training": True,
        "checkpoint_weights_changed": False,
        "new_checkpoint_created": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": bool(hota_trackeval_run),
        "no_screening_or_official_test": True,
    }


def manifest_assertion() -> dict[str, Any]:
    observed = sha256_file(MANIFEST)
    if observed != MANIFEST_SHA:
        raise AssertionError(f"fixed manifest SHA drift: {observed}")
    return {"path": str(MANIFEST), "sha256": observed, "expected_sha256": MANIFEST_SHA}


def scope_video_values(scope: str) -> list[tuple[str, str]]:
    if scope == "internal":
        return [(dataset, video) for dataset, videos in INTERNAL_VIDEOS.items() for video in videos]
    if scope != "dev":
        raise ValueError(f"unsupported scope: {scope}")
    split = json.loads(L82_SPLIT.read_text(encoding="utf-8"))
    values: list[tuple[str, str]] = []
    for raw in split.get("dev_videos", []):
        dataset, video = str(raw).split("|", 1)
        values.append((dataset, video))
    if len(values) != 6 or len(set(values)) != 6:
        raise AssertionError(f"L82 dev video contract drift: {values}")
    return sorted(values)


@dataclass(frozen=True)
class VideoScope:
    dataset: str
    video: str
    queries: tuple[dict[str, Any], ...]

    @property
    def scope_key(self) -> str:
        return f"{self.dataset}|{self.video}"


def _key_query(row: dict[str, Any]) -> dict[str, Any]:
    sentence = str(row.get("sentence") or row.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty legal sentence: {row.get('unit_key')}")
    forbidden = FORBIDDEN_CACHE_FIELDS.intersection(row)
    if forbidden:
        raise AssertionError(f"label field leaked into key-only query: {sorted(forbidden)}")
    return {
        "dataset": str(row["dataset"]), "video": str(row["video"]),
        "query_id": int(row["query_id"]), "sentence": sentence, "expression": sentence,
    }


def load_video_scopes(scope: str) -> list[VideoScope]:
    values = scope_video_values(scope)
    if scope == "dev":
        rows = load_fit_key_rows()
    else:
        rows = load_validation_key_rows()
    result: list[VideoScope] = []
    for dataset, video in values:
        queries = [_key_query(row) for row in query_rows_for_video(rows, dataset, video)]
        qids = [int(row["query_id"]) for row in queries]
        if qids != sorted(set(qids)) or not qids:
            raise AssertionError(f"legal query order drift: {dataset}|{video}")
        result.append(VideoScope(dataset, video, tuple(queries)))
    return result


def native_frame_ids(store: L80BankStore, video: str) -> list[int]:
    store._store.load_video(str(video))
    values = [int(value) for value in store._store.tensors["frame_ids"].tolist()]
    if not values or values != sorted(values) or len(values) != len(set(values)):
        raise AssertionError(f"native L69 frame order drift: {video}")
    return values


def native_groups(video_scope: VideoScope, store: L80BankStore) -> list[dict[str, Any]]:
    # This is the proven L86 helper, but the query list is supplied by the
    # L89D scope and the frame universe is reconstructed from native L69 IDs.
    groups = frame_groups(video_scope.dataset, video_scope.video, list(video_scope.queries), store)
    expected = native_frame_ids(store, video_scope.video)
    actual = [int(group["frame_id"]) for group in groups]
    if actual != expected:
        raise AssertionError(f"native group/frame contract drift: {video_scope.scope_key}")
    return groups


def native_frame_sha(frame_ids: Iterable[int]) -> str:
    payload = json.dumps([int(value) for value in frame_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_timeline_descriptor(scope: str, store: L80BankStore | None = None) -> dict[str, Any]:
    owns_store = store is None
    store = store or L80BankStore(max_history=8)
    videos: list[dict[str, Any]] = []
    try:
        for video_scope in load_video_scopes(scope):
            frames = native_frame_ids(store, video_scope.video)
            videos.append({
                "dataset": video_scope.dataset, "video": video_scope.video,
                "native_frame_count": len(frames), "native_frame_min": min(frames),
                "native_frame_max": max(frames), "native_frame_ids_sha256": native_frame_sha(frames),
                "legal_query_count": len(video_scope.queries),
                "query_ids": [int(row["query_id"]) for row in video_scope.queries],
                "expected_query_frame_pairs": len(frames) * len(video_scope.queries),
                "native_frame_ids": frames,
            })
    finally:
        if owns_store:
            store._store._bank = None; store._store._text_cache = None
            del store
            gc.collect()
    return {
        "scope": scope, "video_count": len(videos),
        "videos": videos,
        "expected_query_frame_pairs": int(sum(item["expected_query_frame_pairs"] for item in videos)),
    }


def native_row_key(batch: Any, query_id: int) -> list[list[Any]]:
    return [
        [str(batch.dataset), str(batch.video), int(query_id), int(batch.frame_id), str(batch.bank_path), int(offset)]
        for offset in batch.row_offsets
    ]


def validate_native_batch(batch: Any, query_id: int | None = None) -> dict[str, Any]:
    qid = int(batch.query_id if query_id is None else query_id)
    if len(batch.row_offsets) != int(batch.candidate_count):
        raise AssertionError(f"candidate count/offset drift: {batch.unit_key}")
    if [int(key[-1]) for key in batch.row_keys] != [int(value) for value in batch.row_offsets]:
        raise AssertionError(f"native row order drift: {batch.unit_key}")
    if int((batch.history_frame_ids > int(batch.frame_id)).sum()) != 0:
        raise AssertionError(f"future history: {batch.unit_key}")
    for value in (batch.boxes, batch.boxes_norm, batch.observations, batch.history_observations):
        if not bool(torch.isfinite(value.float()).all()):
            raise FloatingPointError(f"nonfinite native batch: {batch.unit_key}")
    if batch.text_tokens.ndim != 2 or batch.text_mask.ndim != 1 or batch.text_tokens.shape[0] != batch.text_mask.shape[0]:
        raise AssertionError(f"native text shape drift: {batch.unit_key}")
    if not bool(torch.isfinite(batch.text_tokens.float()).all()) or not bool(batch.text_mask.any()):
        raise FloatingPointError(f"nonfinite/empty native text: {batch.unit_key}")
    return {
        "unit_key": str(batch.unit_key), "query_id": qid, "frame_id": int(batch.frame_id),
        "candidate_count": int(batch.candidate_count), "row_offsets": [int(x) for x in batch.row_offsets],
        "row_keys": native_row_key(batch, qid), "row_key_digest": digest_keys(batch.row_keys),
        "candidate_indices": [int(x) for x in batch.candidate_indices],
        "track_ids": [int(x) for x in batch.track_ids], "pool_ids": [int(x) for x in batch.pool_ids],
        "duplicate_candidate_index_rows": int(len(batch.candidate_indices) - len(set(batch.candidate_indices))),
        "history_shape": list(batch.history_observations.shape), "history_mask_shape": list(batch.history_mask.shape),
        "history_frame_ids": batch.history_frame_ids.tolist(),
        "future_history_count": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
        "text_shape": list(batch.text_tokens.shape), "text_valid_tokens": int(batch.text_mask.sum()),
        "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
    }


def _cache_item_forbidden(item: dict[str, Any]) -> set[str]:
    return FORBIDDEN_CACHE_FIELDS.intersection(item)


class Z1CacheIndex:
    """Index compact L85/Z1 items, validating labels are absent."""

    def __init__(self, root: Path, require_complete_summary: bool = True) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.summary: dict[str, Any] = {}
        summary_path = self.root / "summary.json"
        if summary_path.is_file():
            self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if require_complete_summary and self.summary.get("status") != "complete":
                raise AssertionError(f"incomplete Z1 cache summary: {self.root}")
            if self.summary.get("labels_in_cache") or self.summary.get("candidate_deletion") or self.summary.get("candidate_truncation"):
                raise AssertionError(f"invalid Z1 cache summary: {self.root}")
        self.paths: dict[str, Path] = {}
        self.invalid: dict[str, str] = {}
        for path in sorted(self.root.rglob("*.pt")):
            try:
                item = torch.load(path, map_location="cpu", weights_only=False)
                key = str(item.get("group_key", ""))
                if not key:
                    raise AssertionError("missing group_key")
                if key in self.paths:
                    raise AssertionError(f"duplicate group_key {key}")
                forbidden = _cache_item_forbidden(item)
                if forbidden:
                    raise AssertionError(f"forbidden fields {sorted(forbidden)}")
                if item.get("labels_in_cache") or item.get("candidate_deletion") or item.get("candidate_truncation"):
                    raise AssertionError("invalid cache flags")
                self.paths[key] = path
                del item
            except Exception as exc:
                # Keep path-level invalidity visible to the coverage audit; a
                # corrupt item can never be silently substituted or zero-filled.
                self.invalid[str(path)] = f"{type(exc).__name__}: {exc}"
        if not self.paths and not self.invalid:
            raise AssertionError(f"empty Z1 cache root: {self.root}")

    def has(self, group_key: str) -> bool:
        return str(group_key) in self.paths

    def read(self, group_key: str) -> tuple[dict[str, Any], Path]:
        path = self.paths.get(str(group_key))
        if path is None:
            raise KeyError(str(group_key))
        item = torch.load(path, map_location="cpu", weights_only=False)
        forbidden = _cache_item_forbidden(item)
        if forbidden or item.get("labels_in_cache") or item.get("candidate_deletion") or item.get("candidate_truncation"):
            raise AssertionError(f"invalid Z1 item at read: {path}")
        return item, path

    def summary_descriptor(self) -> dict[str, Any]:
        return {
            "root": str(self.root), "summary": self.summary,
            "indexed_groups": len(self.paths), "invalid_files": self.invalid,
            "summary_sha256": sha256_file(self.root / "summary.json") if (self.root / "summary.json").is_file() else None,
        }


def _validate_z1_item(item: dict[str, Any], group: dict[str, Any], batch: Any) -> dict[str, Any]:
    group_key = str(group["group_key"])
    if str(item.get("group_key")) != group_key:
        raise AssertionError(f"Z1 group key mismatch: {group_key}")
    expected_queries = [dict(row) for row in group["queries"]]
    expected_qids = [int(row["query_id"]) for row in expected_queries]
    qids = [int(value) for value in item.get("query_ids", [])]
    if len(qids) != len(set(qids)) or not set(expected_qids).issubset(set(qids)):
        raise AssertionError(f"Z1 query set is not a superset of the native query set: {group_key}")
    query_indices = [qids.index(qid) for qid in expected_qids]
    sentences = [str(value) for value in item.get("sentences", [])]
    if len(sentences) != len(qids):
        raise AssertionError(f"Z1 sentence/query length mismatch: {group_key}")
    if [sentences[index] for index in query_indices] != [str(row["sentence"]) for row in expected_queries]:
        raise AssertionError(f"Z1 sentence order mismatch: {group_key}")
    if int(item.get("candidate_count", -1)) != int(batch.candidate_count):
        raise AssertionError(f"Z1 candidate count mismatch: {group_key}")
    offsets = [int(value) for value in item.get("row_offsets", [])]
    if offsets != [int(value) for value in batch.row_offsets]:
        raise AssertionError(f"Z1 row offset mismatch: {group_key}")
    z1 = item.get("z1"); text_global = item.get("text_global"); frame_global = item.get("frame_global")
    if not (torch.is_tensor(z1) and torch.is_tensor(text_global) and torch.is_tensor(frame_global)):
        raise AssertionError(f"Z1 tensor missing: {group_key}")
    if tuple(z1.shape)[0] != len(qids) or tuple(z1.shape[1:]) != (int(batch.candidate_count), 256):
        raise AssertionError(f"Z1 tensor shape mismatch: {group_key}: {tuple(z1.shape)}")
    if tuple(text_global.shape) != (len(qids), 256) or tuple(frame_global.shape) != (len(qids), 256):
        raise AssertionError(f"Z1 tensor shape mismatch: {group_key}: {tuple(z1.shape)}")
    if not all(bool(torch.isfinite(value.float()).all()) for value in (z1, text_global, frame_global)):
        raise FloatingPointError(f"nonfinite Z1 item: {group_key}")
    if not bool(item.get("all_rows_retained", False)) or item.get("candidate_deletion") or item.get("candidate_truncation"):
        raise AssertionError(f"Z1 row-retention flags invalid: {group_key}")
    return {
        "group_key": group_key, "query_ids": expected_qids, "query_indices": query_indices,
        "cached_query_ids": qids, "candidate_count": int(batch.candidate_count),
        "row_offsets": offsets, "z1_shape": list(z1.shape), "text_global_shape": list(text_global.shape),
        "frame_global_shape": list(frame_global.shape), "finite": True,
        "row_key_digest": str(item.get("row_keys_digest", "")),
    }


class DenseZ1Resolver:
    """Resolve complete Z1 groups from supplement first, then immutable base."""

    def __init__(self, base_root: Path, supplement_root: Path | None = None) -> None:
        self.base = Z1CacheIndex(base_root)
        self.supplement = Z1CacheIndex(supplement_root) if supplement_root is not None and Path(supplement_root).is_dir() else None

    def resolve(self, group: dict[str, Any], batch: Any) -> dict[str, Any]:
        key = str(group["group_key"])
        errors: list[str] = []
        # Supplement precedence is explicit; an invalid supplement is not
        # allowed to hide a valid base item, but the invalidity remains in
        # coverage/provenance and missing groups are never zero-filled.
        for source_name, index in (("supplement", self.supplement), ("base", self.base)):
            if index is None or not index.has(key):
                continue
            try:
                item, path = index.read(key)
                contract = _validate_z1_item(item, group, batch)
                return {
                    "group_key": key, "source": source_name, "path": str(path.resolve()),
                    "query_ids": [int(x) for x in contract["query_ids"]],
                    "z1": item["z1"].float().clone()[contract["query_indices"]],
                    "text_global": item["text_global"].float().clone()[contract["query_indices"]],
                    "frame_global": item["frame_global"].float().clone()[contract["query_indices"]],
                    "candidate_count": int(item["candidate_count"]),
                    "row_offsets": [int(x) for x in item["row_offsets"]],
                    "contract": contract,
                }
            except Exception as exc:
                errors.append(f"{source_name}:{type(exc).__name__}:{exc}")
        detail = "; ".join(errors) if errors else "group absent from both layers"
        raise KeyError(f"missing/incomplete dense Z1 group {key}: {detail}")


def clear_cpu_tensors(*values: Any) -> None:
    for value in values:
        del value
    gc.collect()


__all__ = [
    "ASSET_ROOT", "BASE_Z1_CACHE", "DEFAULT_LANGUAGE_CACHE", "INTERNAL_VIDEOS", "L49_DATA", "L62_ROWS",
    "L69_ROOT", "L82_SPLIT", "MANIFEST", "MANIFEST_SHA", "RULES", "SEED", "THREAD", "WORK_ROOT",
    "DenseZ1Resolver", "L80BankStore", "L89LanguageTokenCache", "VideoScope", "Z1CacheIndex",
    "capture_group_z1_batched", "command_line", "digest_keys", "expected_timeline_descriptor", "file_meta",
    "frame_groups", "load_fit_key_rows", "load_internal_eval_groups", "load_validation_key_rows",
    "load_video_scopes", "manifest_assertion", "native_frame_ids", "native_frame_sha", "native_groups",
    "native_row_key", "query_rows_for_video", "scope_video_values", "sha256_file", "standard_flags",
    "validate_native_batch", "write_json",
]
