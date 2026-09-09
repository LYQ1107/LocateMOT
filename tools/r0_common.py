#!/usr/bin/env python3
"""Shared streaming data/runtime utilities for the isolated R0 execution.

This module deliberately keeps the visual cache query-independent and keeps
label attachment at the last possible point.  It is used by R0 training and
legal dev/internal evaluation only; no screening or official-test source is
resolved here.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L69_ROOT = ASSET_ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_DATA = ASSET_ROOT / "outputs/l49/data"
TRAIN_DENSE = {
    "refer_kitti_v1": WORK_ROOT / "outputs/r0/data/v1_dense_train_index_retry2",
    "refer_kitti_v2": WORK_ROOT / "outputs/r0/data/v2_dense_train_index_retry2",
}
SAFE_TARGET_ROOT = WORK_ROOT / "outputs/r0/data/safe_targets_retry3"
DEFAULT_LANGUAGE_ROOTS = (
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1"),
    Path("/data2/usr_for_deadline/locatemot_r0a_language_tokens_retry1"),
)
FORBIDDEN_VIDEOS = {"0005", "0011", "0013", "0019"}

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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def command_line() -> str:
    return " ".join([str(value) for value in sys.argv])


class MergedLanguageCache:
    """Read-only union of the existing L89 cache and the R0 missing supplement."""

    def __init__(self, roots: Iterable[Path]) -> None:
        from locatemot.rmot.l89_language_cache import L89LanguageTokenCache, cache_key

        self._cache_key = cache_key
        self.caches = [L89LanguageTokenCache(Path(root).resolve()) for root in roots]
        self._key_sources: dict[str, int] = {}
        for index, cache in enumerate(self.caches):
            for key in cache._index:
                if key in self._key_sources:
                    raise AssertionError(f"duplicate language key across caches: {key}")
                self._key_sources[key] = index

    @property
    def entry_count(self) -> int:
        return len(self._key_sources)

    def get(self, dataset: str, video: str, query_id: int, sentence: str) -> Any:
        key = self._cache_key(dataset, video, query_id, sentence)
        index = self._key_sources.get(key)
        if index is None:
            raise KeyError(f"R0 merged language cache miss: {dataset}|{video}|{query_id}")
        return self.caches[index].get(dataset, video, query_id, sentence)


class VisualCacheIndex:
    """Index compact finalizer manifest; tensors are loaded one frame at a time."""

    def __init__(self, manifest_root: Path) -> None:
        self.root = Path(manifest_root).resolve()
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        self.entries: dict[tuple[str, str, int], dict[str, Any]] = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item["dataset"]), str(item["video"]), int(item["frame_id"]))
            if key in self.entries:
                raise AssertionError(f"duplicate R0 visual cache frame key: {key}")
            path = Path(str(item["path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            self.entries[key] = item | {"path": str(path)}

    def get(self, dataset: str, video: str, frame_id: int) -> dict[str, Any]:
        key = (str(dataset), str(video), int(frame_id))
        item = self.entries.get(key)
        if item is None:
            raise KeyError(f"R0 visual cache miss: {key}")
        payload = torch.load(Path(item["path"]), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise AssertionError(f"invalid visual cache item: {item['path']}")
        if (payload.get("dataset"), payload.get("video"), int(payload.get("frame_id", -1))) != key:
            raise AssertionError(f"visual cache item key drift: {item['path']}")
        if payload.get("group_key") != f"{dataset}|{video}|{int(frame_id)}":
            raise AssertionError(f"visual cache group key drift: {item['path']}")
        if payload.get("labels_in_cache") is not False or payload.get("query_independent") is not True:
            raise AssertionError(f"visual cache label/query contract drift: {item['path']}")
        for field in ("inner_tokens", "context_tokens", "boxes_normalized"):
            tensor = payload.get(field)
            if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor.float()).all()):
                raise FloatingPointError(f"invalid visual tensor {field}: {item['path']}")
        return payload


class DenseIndex:
    """Compact random-access index over one R0 dense native-frame artifact."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        summary_path = self.root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if self.summary.get("status") != "complete":
            raise AssertionError(f"incomplete dense R0 index: {self.root}")
        self.query_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        query_path = self.root / "queries.jsonl"
        for line in query_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["dataset"]), str(row["video"]), int(row["query_id"]))
            if key in self.query_by_key:
                raise AssertionError(f"duplicate R0 query key: {key}")
            self.query_by_key[key] = row
        self.label_path = self.root / "query_frame_labels.jsonl"
        if not self.label_path.is_file():
            raise FileNotFoundError(self.label_path)
        self.label_records: list[dict[str, Any]] = []
        self.label_offsets: list[int] = []
        self.by_category: dict[str, list[dict[str, Any]]] = {}
        self.by_domain_category: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.by_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        seen_keys: set[str] = set()
        with self.label_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                raw = json.loads(line)
                key = f"{raw['dataset']}|{raw['video']}|{int(raw['query_id'])}|{int(raw['frame_id'])}"
                if key in seen_keys:
                    raise AssertionError(f"duplicate R0 dense label key: {key}")
                seen_keys.add(key)
                record = {
                    "index": len(self.label_records),
                    "offset": int(offset),
                    "dataset": str(raw["dataset"]),
                    "video": str(raw["video"]),
                    "query_id": int(raw["query_id"]),
                    "frame_id": int(raw["frame_id"]),
                    "sentence": str(raw["sentence"]),
                    "category": str(raw["category"]),
                    "candidate_count": int(raw["candidate_count"]),
                    "bank_path": str(raw["bank_path"]),
                    "unit_key": key,
                }
                self.label_records.append(record)
                self.label_offsets.append(int(offset))
                self.by_category.setdefault(record["category"], []).append(record)
                self.by_domain_category.setdefault((record["dataset"], record["category"]), []).append(record)
                frame_key = (record["dataset"], record["video"], int(record["frame_id"]))
                self.by_frame.setdefault(frame_key, []).append(record)
        if not self.label_records:
            raise AssertionError(f"empty R0 dense labels: {self.root}")
        if int(self.summary.get("query_frame_count", -1)) != len(self.label_records):
            raise AssertionError(f"dense label count drift: {self.root}")
        for key, records in self.by_frame.items():
            records.sort(key=lambda value: (int(value["query_id"]), str(value["unit_key"])))
            if any((str(value["dataset"]), str(value["video"]), int(value["frame_id"])) != key for value in records):
                raise AssertionError(f"R0 frame-group identity drift: {key}")
        grouped_indices = [int(value["index"]) for records in self.by_frame.values() for value in records]
        if len(grouped_indices) != len(self.label_records) or set(grouped_indices) != set(range(len(self.label_records))):
            raise AssertionError("R0 frame grouping does not contain every label record exactly once")
        self.frame_keys = sorted(self.by_frame)

    def get_label(self, record: dict[str, Any]) -> dict[str, Any]:
        with self.label_path.open("rb") as handle:
            handle.seek(int(record["offset"]))
            raw = json.loads(handle.readline())
        if f"{raw['dataset']}|{raw['video']}|{int(raw['query_id'])}|{int(raw['frame_id'])}" != record["unit_key"]:
            raise AssertionError(f"random label lookup drift: {record['unit_key']}")
        return raw

    def query(self, record: dict[str, Any]) -> dict[str, Any]:
        key = (record["dataset"], record["video"], int(record["query_id"]))
        value = self.query_by_key.get(key)
        if value is None:
            raise KeyError(f"missing dense query metadata: {key}")
        if str(value["sentence"]) != record["sentence"]:
            raise AssertionError(f"dense sentence drift: {record['unit_key']}")
        return value


class EvalIndex:
    """Label-free query/frame index with lazy post-prediction labels.

    ``query_frame_labels.jsonl`` is only scanned as raw bytes to establish
    line cardinality and offsets.  Its JSON payload is deserialized by
    ``get_label`` only after a caller has completed prediction for the record.
    This keeps legal dev/internal GT out of model-input construction.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise AssertionError(f"incomplete R0 eval index: {self.root}")
        self.summary = summary
        self.query_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for line in (self.root / "queries.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["dataset"]), str(row["video"]), int(row["query_id"]))
            if key in self.query_by_key:
                raise AssertionError(f"duplicate R0 eval query key: {key}")
            self.query_by_key[key] = row
        frame_rows: list[dict[str, Any]] = []
        for line in (self.root / "frames.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            frame_rows.append(json.loads(line))
        self.records: list[dict[str, Any]] = []
        self.by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for frame in frame_rows:
            dataset = str(frame["dataset"]); video = str(frame["video"]); frame_id = int(frame["frame_id"])
            queries = [value for value in self.query_by_key.values()
                       if str(value["dataset"]) == dataset and str(value["video"]) == video]
            queries.sort(key=lambda value: int(value["query_id"]))
            group: list[dict[str, Any]] = []
            for query in queries:
                record = {
                    "index": len(self.records), "dataset": dataset, "video": video,
                    "query_id": int(query["query_id"]), "frame_id": frame_id,
                    "sentence": str(query["sentence"]), "candidate_count": int(frame["candidate_count"]),
                    "bank_path": str(frame["bank_path"]),
                    "unit_key": f"{dataset}|{video}|{int(query['query_id'])}|{frame_id}",
                }
                self.records.append(record); group.append(record)
            self.by_frame[(video, frame_id)] = group
        self.label_path = self.root / "query_frame_labels.jsonl"
        if not self.label_path.is_file():
            raise FileNotFoundError(self.label_path)
        self.label_offsets: list[int] = []
        with self.label_path.open("rb") as handle:
            while True:
                offset = handle.tell(); line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.label_offsets.append(int(offset))
        if len(self.label_offsets) != len(self.records):
            raise AssertionError(f"lazy eval label cardinality drift: {self.root} {len(self.label_offsets)} != {len(self.records)}")
        if int(summary.get("query_frame_count", -1)) != len(self.records):
            raise AssertionError(f"lazy eval query-frame count drift: {self.root}")
        self.label_records = self.records

    def query(self, record: dict[str, Any]) -> dict[str, Any]:
        key = (str(record["dataset"]), str(record["video"]), int(record["query_id"]))
        value = self.query_by_key.get(key)
        if value is None or str(value["sentence"]) != str(record["sentence"]):
            raise KeyError(f"missing or drifting R0 eval query: {key}")
        return value

    def get_label(self, record: dict[str, Any]) -> dict[str, Any]:
        index = int(record["index"])
        with self.label_path.open("rb") as handle:
            handle.seek(self.label_offsets[index]); raw = json.loads(handle.readline())
        observed = f"{raw['dataset']}|{raw['video']}|{int(raw['query_id'])}|{int(raw['frame_id'])}"
        if observed != str(record["unit_key"]):
            raise AssertionError(f"lazy eval label key drift: {observed} != {record['unit_key']}")
        return raw


class R0RuntimeData:
    """Materialize one complete frame/query input and attach labels last."""

    def __init__(self, dense: DenseIndex, visual: VisualCacheIndex, language: MergedLanguageCache, device: torch.device) -> None:
        from locatemot.rmot.r0_dense_data import R0BankStore

        self.dense = dense
        self.visual = visual
        self.language = language
        self.device = device
        self.store = R0BankStore()
        self._geometry = __import__("locatemot.rmot.r0_geometry", fromlist=["build_track_geometry"])

    def prepare(self, record: dict[str, Any], *, attach_labels: bool = True) -> dict[str, Any]:
        from locatemot.rmot.r0_dense_data import R0DataContractError

        dataset, video, query_id, frame_id = record["dataset"], record["video"], int(record["query_id"]), int(record["frame_id"])
        sentence = str(record["sentence"])
        query = self.dense.query(record)
        if str(query["sentence"]) != sentence:
            raise AssertionError(f"R0 query sentence mismatch: {record['unit_key']}")
        item = self.visual.get(dataset, video, frame_id)
        batch = self.store.build_frame(dataset, video, query_id, frame_id, sentence)
        expected_offsets = list(range(int(item["row_offsets"][0]), int(item["row_offsets"][-1]) + 1)) if item["row_offsets"] else []
        if list(item.get("row_offsets", [])) != expected_offsets or batch.row_offsets != expected_offsets:
            raise R0DataContractError(f"R0 visual/native row offset drift: {record['unit_key']}")
        if int(item.get("candidate_count", -1)) != batch.candidate_count or int(record["candidate_count"]) != batch.candidate_count:
            raise R0DataContractError(f"R0 candidate count drift: {record['unit_key']}")
        if [int(x) for x in item.get("candidate_indices", [])] != batch.candidate_indices:
            raise R0DataContractError(f"R0 candidate index/order drift: {record['unit_key']}")
        if str(batch.bank_path) != str(record["bank_path"]):
            raise R0DataContractError(f"R0 bank path drift: {record['unit_key']}")
        row_keys = [list(key) for key in batch.row_keys]
        if len(row_keys) != batch.candidate_count or len({tuple(key) for key in row_keys}) != len(row_keys):
            raise R0DataContractError(f"R0 native row key drift: {record['unit_key']}")
        geometry = self._geometry.build_track_geometry(self.store, batch, batch.image_hw, history_length=4)
        if not bool(torch.isfinite(geometry).all()):
            raise FloatingPointError(f"nonfinite R0 geometry: {record['unit_key']}")
        language_item = self.language.get(dataset, video, query_id, sentence)
        text_tokens = language_item.tokens.unsqueeze(0).to(device=self.device, dtype=torch.float32).clone()
        text_mask = language_item.mask.unsqueeze(0).to(device=self.device, dtype=torch.bool).clone()
        text_global = (text_tokens * text_mask.unsqueeze(-1).to(text_tokens.dtype)).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        if text_tokens.shape[-1] != 256 or text_mask.shape != text_tokens.shape[:2] or not bool(torch.isfinite(text_tokens).all()):
            raise FloatingPointError(f"invalid R0 language tensor: {record['unit_key']}")
        result = {
            "record": record,
            "item": item,
            "batch": batch,
            "row_keys": row_keys,
            "inner_tokens": item["inner_tokens"].float().to(self.device).clone(),
            "context_tokens": item["context_tokens"].float().to(self.device).clone(),
            "boxes_normalized": item["boxes_normalized"].float().to(self.device).clone(),
            "geometry": geometry.to(self.device).float().clone(),
            "text_tokens": text_tokens,
            "text_mask": text_mask,
            "text_global": text_global,
        }
        if attach_labels:
            raw_label = self.dense.get_label(record)
            supervision = self.store.attach_frame_labels(batch, raw_label["target_ids"])
            if str(supervision["category"]) != str(raw_label["category"]):
                raise AssertionError(f"R0 category drift: {record['unit_key']}")
            if int(supervision["positive_row_count"]) != int(raw_label["positive_row_count"]):
                raise AssertionError(f"R0 positive count drift: {record['unit_key']}")
            result["label"] = raw_label
            result["supervision"] = supervision
            result["candidate_gt"] = list(supervision["candidate_gt"])
        return result

    def prepare_group(self, records: list[dict[str, Any]], *, attach_labels: bool = True) -> dict[str, Any]:
        """Materialize one native frame for several expressions at once.

        The visual item, native rows, and causal geometry are shared across the
        query batch.  Text is still kept as a masked token sequence per query;
        labels are attached only after the shared feature contract is checked.
        """
        from locatemot.rmot.r0_dense_data import R0DataContractError

        if not records:
            raise ValueError("R0 query group cannot be empty")
        first = records[0]
        dataset = str(first["dataset"])
        video = str(first["video"])
        frame_id = int(first["frame_id"])
        if any((str(row["dataset"]), str(row["video"]), int(row["frame_id"])) != (dataset, video, frame_id)
               for row in records):
            raise R0DataContractError("R0 grouped query/frame identity drift")
        item = self.visual.get(dataset, video, frame_id)
        batch = self.store.build_frame(dataset, video, -1, frame_id, "__r0_group_features__")
        row_offsets = [int(value) for value in item.get("row_offsets", [])]
        expected_offsets = list(range(row_offsets[0], row_offsets[-1] + 1)) if row_offsets else []
        if row_offsets != expected_offsets or batch.row_offsets != expected_offsets:
            raise R0DataContractError(f"R0 grouped row offset drift: {dataset}|{video}|{frame_id}")
        if int(item.get("candidate_count", -1)) != batch.candidate_count:
            raise R0DataContractError(f"R0 grouped candidate count drift: {dataset}|{video}|{frame_id}")
        if [int(value) for value in item.get("candidate_indices", [])] != batch.candidate_indices:
            raise R0DataContractError(f"R0 grouped candidate order drift: {dataset}|{video}|{frame_id}")
        if str(batch.bank_path) != str(first["bank_path"]):
            raise R0DataContractError(f"R0 grouped bank path drift: {dataset}|{video}|{frame_id}")
        geometry = self._geometry.build_track_geometry(self.store, batch, batch.image_hw, history_length=4)
        if not bool(torch.isfinite(geometry).all()):
            raise FloatingPointError(f"nonfinite grouped R0 geometry: {dataset}|{video}|{frame_id}")

        tokens_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        for row in records:
            language_item = self.language.get(dataset, video, int(row["query_id"]), str(row["sentence"]))
            tokens_list.append(language_item.tokens.float().clone())
            masks_list.append(language_item.mask.bool().clone())
        max_length = max(int(value.shape[0]) for value in tokens_list)
        text_tokens = torch.zeros((len(records), max_length, 256), dtype=torch.float32, device=self.device)
        text_mask = torch.zeros((len(records), max_length), dtype=torch.bool, device=self.device)
        for index, (tokens, mask) in enumerate(zip(tokens_list, masks_list)):
            length = int(tokens.shape[0])
            text_tokens[index, :length] = tokens.to(device=self.device)
            text_mask[index, :length] = mask.to(device=self.device)
        text_tokens = text_tokens.clone()
        text_mask = text_mask.clone()
        if not bool(torch.isfinite(text_tokens).all()) or not bool(text_mask.any(dim=1).all()):
            raise FloatingPointError("invalid grouped R0 language tensor")
        text_global = (text_tokens * text_mask.unsqueeze(-1).to(text_tokens.dtype)).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        result: dict[str, Any] = {
            "records": records,
            "item": item,
            "batch": batch,
            "inner_tokens": item["inner_tokens"].float().to(self.device).clone(),
            "context_tokens": item["context_tokens"].float().to(self.device).clone(),
            "boxes_normalized": item["boxes_normalized"].float().to(self.device).clone(),
            "geometry": geometry.to(self.device).float().clone(),
            "text_tokens": text_tokens,
            "text_mask": text_mask,
            "text_global": text_global,
            "candidate_count": batch.candidate_count,
            "row_offsets": row_offsets,
            "candidate_indices": list(batch.candidate_indices),
        }
        if attach_labels:
            labels: list[dict[str, Any]] = []
            for row in records:
                raw_label = self.dense.get_label(row)
                query_batch = self.store.build_frame(dataset, video, int(row["query_id"]), frame_id, str(row["sentence"]))
                supervision = self.store.attach_frame_labels(query_batch, raw_label["target_ids"])
                if str(supervision["category"]) != str(raw_label["category"]):
                    raise AssertionError(f"R0 grouped category drift: {row['unit_key']}")
                labels.append({"label": raw_label, "supervision": supervision,
                               "row_keys": [list(value) for value in query_batch.row_keys]})
            result["labels"] = labels
        return result

    def close(self) -> None:
        self.store._blob = None
        self.store._candidate_gt = None
        gc.collect()


def model_forward(model: torch.nn.Module, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model(
        sample["inner_tokens"], sample["context_tokens"], sample["boxes_normalized"],
        sample["geometry"], sample["text_tokens"], sample["text_mask"], sample["text_global"],
    )


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
        "labels_in_visual_cache": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED",
    }


__all__ = [
    "ASSET_ROOT", "DEFAULT_LANGUAGE_ROOTS", "DenseIndex", "EvalIndex", "FORBIDDEN_VIDEOS", "L49_DATA", "L69_ROOT", "MANIFEST", "MANIFEST_SHA",
    "MergedLanguageCache", "R0RuntimeData", "SAFE_TARGET_ROOT", "SEED", "THREAD", "TRAIN_DENSE", "VisualCacheIndex", "WORK_ROOT",
    "check_manifest", "command_line", "file_meta", "model_forward", "sha256_file", "standard_flags", "write_json",
]
