#!/usr/bin/env python3
"""Shared label-isolated scoring helpers for the L88 evaluation stages.

The helper intentionally builds a complete L69 frame from key/text metadata
before attaching any expression or candidate labels.  It reuses one live
GroundingDINO instance while swapping only the compact L88 LoRA factors and
sidecar checkpoint between evaluation checkpoints.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L85_CACHE = ASSET_ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
L88_CACHE = WORK_ROOT / "outputs/l88/cache/encoder_inputs_v1"
L62_ROWS = ASSET_ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.l88_full_rmot import L86Config, L88FullRMOT  # noqa: E402
from locatemot.rmot.l88_clip_data import L88ClipStore  # noqa: E402
from locatemot.rmot.l88_grounding_runtime import (  # noqa: E402
    L88GroundingRuntime,
    forward_l88_z1,
    sha256_file,
)
from locatemot.rmot.l88_lora import inject_lora  # noqa: E402
from tools.l88_train_full_rmot import EncoderCacheReader, encode_z1  # noqa: E402
from locatemot.rmot.l80_data import load_full_unit_for_labels  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def sha256(path: Path) -> str:
    return sha256_file(path.resolve())


def load_package(path: Path) -> dict[str, Any]:
    package = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(package, dict) or package.get("format") != "locatemot-l88-full-rmot-checkpoint-v1":
        raise AssertionError(f"invalid L88 checkpoint package: {path}")
    if int(package.get("seed", -1)) != SEED:
        raise AssertionError(f"L88 checkpoint seed drift: {path}")
    if str(package.get("manifest_sha256")) != MANIFEST_SHA:
        raise AssertionError(f"L88 checkpoint manifest drift: {path}")
    return package


def checkpoint_info(path: Path, package: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "epoch": int(package.get("epoch", 0)),
        "optimizer_step": int(package.get("optimizer_step", 0)),
        "phase": str(package.get("phase", "unknown")),
        "model_config": package["model_config"],
        "strict_package": True,
    }


def make_runtime(device: torch.device) -> tuple[L88GroundingRuntime, Any, str]:
    runtime = L88GroundingRuntime(device)
    base_digest = ""
    injector = inject_lora(runtime.model)
    runtime.inject(injector)
    base_digest = injector.base_parameter_digest()
    if not all(not parameter.requires_grad for name, parameter in runtime.model.named_parameters()
               if ".parametrizations." not in name or not (name.endswith(".A") or name.endswith(".B"))):
        raise AssertionError("L88 base detector is not frozen during evaluation")
    return runtime, injector, base_digest


def load_checkpoint_into(runtime: L88GroundingRuntime, injector: Any, path: Path,
                         device: torch.device) -> tuple[L88FullRMOT, dict[str, Any]]:
    package = load_package(path)
    injector.load_adapter_state_dict(package["lora_state_dict"], strict=True)
    sidecar = L88FullRMOT(L86Config(**package["model_config"]))
    result = sidecar.load_state_dict(package["sidecar_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"L88 sidecar strict reload failed: {result}")
    sidecar.to(device=device, dtype=torch.float32).eval()
    for parameter in sidecar.parameters():
        parameter.requires_grad_(False)
    info = checkpoint_info(path, package)
    info["sidecar_strict_reload"] = True
    info["lora_strict_reload"] = True
    info["base_digest_after_load"] = injector.base_parameter_digest()
    return sidecar, info


@dataclass
class LabelFreeGroup:
    group_key: str
    dataset: str
    video: str
    frame_id: int
    query_ids: list[int]
    query_metadata: list[dict[str, Any]]
    batches: list[Any]
    frame: Any


def _key_metadata(row: dict[str, Any], *, partition: str | None = None) -> dict[str, Any]:
    allowed = ("unit_key", "dataset", "video", "query_id", "frame_id", "sentence", "expression")
    result = {key: row[key] for key in allowed if key in row}
    sentence = str(result.get("sentence") or result.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty L88 eval sentence: {row.get('unit_key')}")
    result["sentence"] = sentence
    result["expression"] = sentence
    if partition is not None:
        result["evaluation_partition"] = str(partition)
    forbidden = {"target_ids", "positive_indices", "positive_count", "category", "labels",
                 "target_present", "candidate_gt", "coverage_mask", "declared_category"}
    leaked = forbidden.intersection(result)
    if leaked:
        raise AssertionError(f"label fields leaked into L88 eval key row: {sorted(leaked)}")
    return result


def build_label_free_group(store: L88ClipStore, group_key: str,
                           *, query_ids: Iterable[int] | None = None,
                           temporal_enabled: bool = True) -> LabelFreeGroup:
    group = store.groups[str(group_key)]
    requested = None if query_ids is None else {int(value) for value in query_ids}
    rows = [_key_metadata(row) for row in group["queries"]
            if requested is None or int(row["query_id"]) in requested]
    if not rows:
        raise AssertionError(f"empty L88 eval query group: {group_key}")
    rows.sort(key=lambda row: (int(row["query_id"]), str(row["unit_key"])))
    batches = [store.bank_store.build_unit(row) for row in rows]
    first = batches[0]
    for batch in batches:
        if batch.candidate_count != first.candidate_count or batch.row_offsets != first.row_offsets:
            raise AssertionError(f"L88 eval candidate set drift: {group_key}")
        if batch.history_frame_ids.numel() and bool((batch.history_frame_ids > int(first.frame_id)).any()):
            raise AssertionError(f"future L88 eval history: {group_key}")
    cache_item = store._base.cache_item(str(group_key))
    item_qids = [int(value) for value in cache_item["query_ids"]]
    indices = []
    for row in rows:
        if int(row["query_id"]) not in item_qids:
            raise KeyError(f"L88 eval query absent from cache: {row['unit_key']}")
        indices.append(item_qids.index(int(row["query_id"])))
    text_global = cache_item["text_global"][indices].float().clone()
    frame_global = cache_item["frame_global"][indices].float().clone()
    # The cached z1 is deliberately ignored.  L88 recomputes adapted Z1 from
    # the query-independent encoder cache for every checkpoint.
    if tuple(text_global.shape) != (len(rows), 256) or tuple(frame_global.shape) != (len(rows), 256):
        raise AssertionError(f"L88 eval cache text/frame shape drift: {group_key}")
    frame = SimpleNamespace(
        group_key=str(group_key), dataset=str(group["dataset"]), video=str(group["video"]),
        frame_id=int(group["frame_id"]), query_ids=[int(row["query_id"]) for row in rows],
        text_global=text_global, frame_global=frame_global,
        current_observation=first.observations.float().clone(),
        history_observations=first.history_observations.float().clone(),
        history_mask=first.history_mask.clone(), history_frame_ids=first.history_frame_ids.clone(),
        row_offsets=[int(value) for value in first.row_offsets],
        row_keys=[tuple(value) for value in first.row_keys],
    )
    if not temporal_enabled:
        frame.history_mask.zero_()
        frame.history_frame_ids.fill_(-1)
        frame.history_observations.zero_()
    return LabelFreeGroup(str(group_key), str(group["dataset"]), str(group["video"]),
                          int(group["frame_id"]), frame.query_ids, rows, batches, frame)


def attach_labels(store: L88ClipStore, batch: Any) -> dict[str, Any]:
    # Fit/dev rows are already indexed by the inherited key-only store.  The
    # lookup is still deliberately made only by this explicit post-score
    # helper; no label field is present in ``build_label_free_group``.
    full = getattr(store._base, "labels_by_key", {}).get(str(batch.unit_key))
    if full is None:
        full = load_full_unit_for_labels(str(batch.unit_key))
    labels = store.bank_store.attach_labels(batch, full)
    labels["unit_key"] = str(batch.unit_key)
    labels["query_id"] = int(batch.query_id)
    return labels


def score_label_free_group(group: LabelFreeGroup, runtime: L88GroundingRuntime,
                           reader: EncoderCacheReader, store: L88ClipStore,
                           sidecar: L88FullRMOT, checkpoint: dict[str, Any],
                           device: torch.device, *, query_tile: int = 4,
                           attach_group_labels: bool = True) -> list[dict[str, Any]]:
    frame = group.frame
    with torch.inference_mode():
        z1 = encode_z1(runtime, reader, store, frame, device, query_tile=query_tile, bf16=False)
        output = sidecar(
            z1, frame.text_global.to(device), frame.frame_global.to(device),
            frame.current_observation.to(device), frame.history_observations.to(device),
            frame.history_mask.to(device), frame.history_frame_ids.to(device), frame.frame_id,
            temporal_enabled=True,
        )
    fields = {}
    for name in ("candidate_energy", "r_total", "candidate_prior", "presence_logit", "null_logit"):
        value = output[name].float().detach().cpu().numpy()
        if not np.isfinite(value).all():
            raise FloatingPointError(f"nonfinite L88 eval field {name}: {group.group_key}")
        fields[name] = value
    records: list[dict[str, Any]] = []
    for q, (row, batch) in enumerate(zip(group.query_metadata, group.batches)):
        labels = attach_labels(store, batch) if attach_group_labels else None
        score = fields["candidate_energy"][q]
        if score.shape != (batch.candidate_count,):
            raise AssertionError(f"L88 eval score/candidate count drift: {batch.unit_key}")
        record: dict[str, Any] = {
            "format": "locatemot-l88-score-record-v1",
            "checkpoint": checkpoint,
            "unit_key": str(batch.unit_key), "group_key": str(group.group_key),
            "dataset": str(batch.dataset), "video": str(batch.video),
            "query_id": int(batch.query_id), "frame_id": int(batch.frame_id),
            "candidate_count": int(batch.candidate_count),
            "row_offsets": [int(value) for value in batch.row_offsets],
            "row_keys": [list(value) for value in batch.row_keys],
            "candidate_indices": [int(value) for value in batch.candidate_indices],
            "track_ids": [int(value) for value in batch.track_ids],
            "pool_ids": [int(value) for value in batch.pool_ids],
            "score": score.astype(np.float64).tolist(),
            "r_total": fields["r_total"][q].astype(np.float64).tolist(),
            "candidate_prior": fields["candidate_prior"].astype(np.float64).tolist(),
            "presence_logit": float(fields["presence_logit"][q]),
            "null_logit": float(fields["null_logit"][q]),
            "future_history_count": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
            "candidate_rows_retained": True, "candidate_deletion": False,
            "candidate_truncation": False, "finite_scores": True,
            "labels_attached_after_feature_construction": bool(labels is not None),
        }
        if labels is not None:
            record.update({
                "labels": [bool(value) for value in labels["labels"].tolist()],
                "target_ids": [str(value) for value in labels["target_ids"]],
                "candidate_gt": [None if value is None else str(value)
                                 for value in labels["sidecar_candidate_gt"]],
                "positive_indices": [int(value) for value in labels["positive_indices"]],
                "positive_count": int(labels["positive_count"]),
                "target_present": bool(labels["target_present"]),
                "candidate_present": bool(labels["candidate_present"]),
                "coverage_mask": bool(labels["coverage_mask"]),
                "category": str(labels["category"]),
                "declared_category": str(labels.get("declared_category", "unknown")),
                "label_source": str(labels["label_source"]),
            })
        records.append(record)
    del output, z1
    return records


def release_group(store: L88ClipStore, group: LabelFreeGroup) -> None:
    for batch in group.batches:
        del batch
    del group
    store.release_loaded_cache_items()
    gc.collect()


__all__ = [
    "ASSET_ROOT", "L62_ROWS", "L85_CACHE", "L88_CACHE", "MANIFEST", "MANIFEST_SHA", "SEED", "THREAD",
    "L88ClipStore", "EncoderCacheReader", "LabelFreeGroup", "attach_labels", "build_label_free_group",
    "checkpoint_info", "load_checkpoint_into", "load_package", "make_runtime", "release_group",
    "score_label_free_group", "sha256", "write_json",
]
