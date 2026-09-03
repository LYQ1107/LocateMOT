"""Streaming L86 causal clip construction over the frozen L69/L85 view.

The existing L85 cache is label-free compact Z1/text state.  This module loads
that cache into RAM for the run and reconstructs observation/history tensors
from each video's native L69 frame pointers.  It never writes a new dense/raw
cache.  Labels are attached only after the complete current/history rows have
been built.
"""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from locatemot.rmot.l80_data import L80BankStore, load_fit_units
from locatemot.rmot.l85_runtime import (
    build_groups,
    load_fit_train_dev_groups,
)


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
OBS_DIM = 1432
HISTORY = 8
FORBIDDEN_INPUT_FIELDS = {"source", "source_id", "pool_id", "group_id", "track_id", "query_id", "state_key"}


@dataclass
class FrameExample:
    group_key: str
    dataset: str
    video: str
    frame_id: int
    query_ids: list[int]
    z1: torch.Tensor
    text_global: torch.Tensor
    frame_global: torch.Tensor
    current_observation: torch.Tensor
    history_observations: torch.Tensor
    history_mask: torch.Tensor
    history_frame_ids: torch.Tensor
    row_offsets: list[int]
    row_keys: list[tuple[Any, ...]]
    candidate_indices: list[int]
    track_ids: list[int]
    labels: list[dict[str, Any]]


def _clip_history(batch: Any, enabled: bool, length: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history = torch.zeros((batch.candidate_count, HISTORY, OBS_DIM), dtype=torch.float32)
    mask = torch.zeros((batch.candidate_count, HISTORY), dtype=torch.bool)
    frames = torch.full((batch.candidate_count, HISTORY), -1, dtype=torch.int64)
    if not enabled:
        return history, mask, frames
    for row in range(batch.candidate_count):
        valid = torch.nonzero(batch.history_mask[row], as_tuple=False).flatten().tolist()
        valid = valid[-int(length):]
        if valid:
            count = len(valid)
            history[row, :count] = batch.history_observations[row, valid]
            mask[row, :count] = True
            frames[row, :count] = batch.history_frame_ids[row, valid]
    if bool((frames[mask] > int(batch.frame_id)).any()):
        raise AssertionError(f"future frame entered L86 clip: {batch.unit_key}")
    return history, mask, frames


class L86ClipStore:
    def __init__(self, cache_root: Path, *, load_cache_into_ram: bool = True) -> None:
        self.cache_root = Path(cache_root).resolve()
        summary_path = self.cache_root / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("labels_in_cache"):
            raise AssertionError("L86 requires a complete label-free L85 cache")
        self.cache_paths = {
            str(value["group_key"]): self.cache_root / str(value["file"])
            for value in summary.get("groups", [])
            if "group_key" in value and "file" in value
        }
        if not self.cache_paths:
            for path in self.cache_root.rglob("*.pt"):
                item = torch.load(path, map_location="cpu", weights_only=False)
                self.cache_paths[str(item["group_key"])] = path
                del item
        self.cache_items: dict[str, dict[str, Any]] = {}
        if load_cache_into_ram:
            for key, path in self.cache_paths.items():
                self.cache_items[key] = torch.load(path, map_location="cpu", weights_only=False)
        self.groups, self.train_keys, self.dev_keys = load_fit_train_dev_groups()
        fit_rows = load_fit_units()
        self.labels_by_key = {str(row["unit_key"]): row for row in fit_rows}
        if len(self.labels_by_key) != 5314:
            raise AssertionError("L86 fit label count drift")
        self.frame_groups: dict[tuple[str, str, int], str] = {}
        self.video_frames: dict[tuple[str, str], list[int]] = {}
        for key, group in self.groups.items():
            identity = (str(group["dataset"]), str(group["video"]), int(group["frame_id"]))
            if identity in self.frame_groups and self.frame_groups[identity] != key:
                raise AssertionError(f"duplicate L86 frame group key: {identity}")
            self.frame_groups[identity] = key
            self.video_frames.setdefault((identity[0], identity[1]), []).append(identity[2])
        for key in list(self.video_frames):
            self.video_frames[key] = sorted(set(self.video_frames[key]))
        self.bank_store = L80BankStore(max_history=HISTORY)

    def cache_item(self, group_key: str) -> dict[str, Any]:
        key = str(group_key)
        if key not in self.cache_items:
            path = self.cache_paths.get(key)
            if path is None:
                raise KeyError(f"missing L86 label-free cache group: {key}")
            self.cache_items[key] = torch.load(path, map_location="cpu", weights_only=False)
        item = self.cache_items[key]
        if int(item.get("candidate_count", -1)) <= 0 or item.get("candidate_deletion") or item.get("candidate_truncation"):
            raise AssertionError(f"invalid L86 cache item: {key}")
        return item

    def _labels_for(self, batch: Any, query_row: dict[str, Any]) -> dict[str, Any]:
        # This is the explicit post-feature-construction label attachment point.
        result = self.bank_store.attach_labels(batch, self.labels_by_key[str(query_row["unit_key"])])
        result["candidate_gt"] = list(result["sidecar_candidate_gt"])
        result["query_id"] = int(query_row["query_id"])
        result["unit_key"] = str(query_row["unit_key"])
        result["row_keys"] = [list(key) for key in batch.row_keys]
        if len(result["labels"]) != batch.candidate_count:
            raise AssertionError(f"L86 label/candidate length drift: {query_row['unit_key']}")
        return result

    def build_frame(self, group_key: str, query_ids: Iterable[int] | None = None, *, temporal_enabled: bool) -> FrameExample:
        key = str(group_key)
        group = self.groups[key]
        item = self.cache_item(key)
        requested = None if query_ids is None else {int(value) for value in query_ids}
        query_rows = [row for row in group["queries"] if requested is None or int(row["query_id"]) in requested]
        if not query_rows:
            raise AssertionError(f"no requested queries in L86 frame: {key}")
        item_query_ids = [int(value) for value in item["query_ids"]]
        indices: list[int] = []
        for row in query_rows:
            qid = int(row["query_id"])
            if qid not in item_query_ids:
                raise AssertionError(f"cache query missing: {key}|{qid}")
            indices.append(item_query_ids.index(qid))
        first = self.bank_store.build_unit(query_rows[0])
        if int(item["candidate_count"]) != first.candidate_count:
            raise AssertionError(f"L86 cache/bank candidate count drift: {key}")
        if list(item["row_offsets"]) != [int(value) for value in first.row_offsets]:
            raise AssertionError(f"L86 cache/bank row offset drift: {key}")
        history, mask, frames = _clip_history(first, temporal_enabled)
        z1 = item["z1"][indices].float().clone()
        text_global = item["text_global"][indices].float().clone()
        frame_global = item["frame_global"][indices].float().clone()
        current = first.observations.float().clone()
        labels = [self._labels_for(first, row) for row in query_rows]
        row_keys = [tuple(value) for value in first.row_keys]
        if len(row_keys) != first.candidate_count:
            raise AssertionError(f"L86 row key drift: {key}")
        return FrameExample(
            group_key=key,
            dataset=str(group["dataset"]),
            video=str(group["video"]),
            frame_id=int(group["frame_id"]),
            query_ids=[int(row["query_id"]) for row in query_rows],
            z1=z1,
            text_global=text_global,
            frame_global=frame_global,
            current_observation=current,
            history_observations=history,
            history_mask=mask,
            history_frame_ids=frames,
            row_offsets=[int(value) for value in first.row_offsets],
            row_keys=row_keys,
            candidate_indices=[int(value) for value in first.candidate_indices],
            track_ids=[int(value) for value in first.track_ids],
            labels=labels,
        )

    def build_clip(self, anchor_key: str, *, temporal_enabled: bool, clip_length: int = 4) -> list[FrameExample]:
        anchor_key = str(anchor_key)
        current_group = self.groups[anchor_key]
        if not temporal_enabled:
            return [self.build_frame(anchor_key, temporal_enabled=False)]
        dataset, video, frame = str(current_group["dataset"]), str(current_group["video"]), int(current_group["frame_id"])
        current_qids = {int(row["query_id"]) for row in current_group["queries"]}
        prior_frames = [value for value in self.video_frames[(dataset, video)] if value < frame]
        selected: list[tuple[int, str, set[int]]] = []
        for previous_frame in reversed(prior_frames):
            previous_key = self.frame_groups.get((dataset, video, previous_frame))
            if previous_key is None:
                continue
            # The frozen L85 cache is intentionally limited to the registered
            # train/dev frame groups.  A legal temporal predecessor must have
            # a real cached Z1 state; never fabricate or reuse a neighboring
            # frame when that compact feature is unavailable.
            if previous_key not in self.cache_paths and previous_key not in self.cache_items:
                continue
            available = {int(row["query_id"]) for row in self.groups[previous_key]["queries"]}
            common = current_qids.intersection(available)
            if common:
                selected.append((previous_frame, previous_key, common))
            if len(selected) >= max(0, int(clip_length) - 1):
                break
        selected.sort(key=lambda value: value[0])
        frames = [self.build_frame(key, qids, temporal_enabled=True) for _, key, qids in selected]
        frames.append(self.build_frame(anchor_key, temporal_enabled=True))
        if any(value.frame_id > frame for value in frames):
            raise AssertionError(f"L86 clip future drift: {anchor_key}")
        return frames

    def close(self) -> None:
        self.cache_items.clear()
        self.bank_store._store._bank = None
        self.bank_store._store._text_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["FrameExample", "L86ClipStore", "OBS_DIM", "HISTORY"]
