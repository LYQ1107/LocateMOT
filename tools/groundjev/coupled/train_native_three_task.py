#!/usr/bin/env python3
"""Train a bounded native MOT/RMOT/OVMOT GroundJEV readout.

This is the replacement for the old independent OneJev source-manifest SFT
queue.  It loads one protected G2 Clean adapter, encodes each unique image
once, and trains only the typed JEV readouts, causal memory projector and the
small OVMOT category heads.  The G2 visual-language backbone is frozen in this
first joint phase; a later retention phase may unfreeze a registered LoRA.

The input is the audited three-task source manifest.  Rows are sampled in a
deterministic, balanced way and retain their original provenance.  Missing
history tokens are not replaced by a target identity, and invalid identity
targets are masked instead of being converted into NEW or NO_MATCH.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge  # noqa: E402
from locatemot.groundjev.coupled.groundjev_model import GroundJEVModel  # noqa: E402
from locatemot.groundjev.coupled.task_heads import GroundJEVTaskHeads  # noqa: E402


SCHEMA_VERSION = "groundjev-native-three-task-readout-v1"
MODEL_SHA_DEFAULT = "3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2"
_CANDIDATE_RE = re.compile(r"(?:_c|_membership_|_group_|_class_)(\d+)(?:_|$)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_rows(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _resolve_path(value: Any, *, manifest_path: Path, runtime_root: Path) -> Path | None:
    if not value or not isinstance(value, (str, Path)):
        return None
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        manifest_path.parent / raw,
        runtime_root / raw,
        PROJECT_ROOT / raw,
        Path.cwd() / raw,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    return None


def _task(row: Mapping[str, Any]) -> str:
    value = str(row.get("task", "")).lower()
    if value in {"mot", "identity", "identity_pair", "lifecycle", "occlusion_recovery", "memory"}:
        return "mot"
    if value in {"rmot", "query_membership", "membership", "empty_query_presence"}:
        return "rmot"
    if value in {"ovmot", "category_group", "category_leaf"}:
        return "ovmot"
    nested = row.get("state", {}).get("task", {}) if isinstance(row.get("state"), Mapping) else {}
    nested_value = str(nested.get("task", "")).lower()
    if nested_value in {"mot", "rmot", "ovmot"}:
        return nested_value
    raise ValueError(f"cannot infer task from row {row.get('id')!r}")


def _media_path(row: Mapping[str, Any], *, manifest_path: Path, runtime_root: Path) -> Path | None:
    values: list[Any] = []
    if row.get("current_image"):
        values.append(row["current_image"])
    for item in row.get("media", []) or []:
        if isinstance(item, Mapping) and item.get("path"):
            values.append(item["path"])
    for item in row.get("candidate_visual_references", []) or []:
        if isinstance(item, Mapping) and item.get("path"):
            values.append(item["path"])
        elif isinstance(item, str):
            values.append(item)
    for value in values:
        resolved = _resolve_path(value, manifest_path=manifest_path, runtime_root=runtime_root)
        if resolved is not None:
            return resolved
    return None


def _candidates(row: Mapping[str, Any]) -> tuple[list[list[float]], list[str]]:
    state = row.get("state") if isinstance(row.get("state"), Mapping) else {}
    candidates = row.get("native_candidates") or state.get("candidates") or []
    boxes: list[list[float]] = []
    labels: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or len(candidate.get("box", [])) != 4:
            continue
        boxes.append([float(value) for value in candidate["box"]])
        labels.append(str(candidate.get("label", "object")))
    return boxes, labels


def _candidate_index(row: Mapping[str, Any], count: int) -> int:
    explicit = row.get("candidate_index")
    if explicit is not None:
        index = int(explicit)
    else:
        match = _CANDIDATE_RE.search(str(row.get("id", "")))
        index = int(match.group(1)) if match else 0
    return max(0, min(index, max(count - 1, 0)))


def _query(row: Mapping[str, Any], task: str) -> str:
    if row.get("task_query"):
        return str(row["task_query"])
    question = row.get("question") if isinstance(row.get("question"), Mapping) else {}
    if question.get("instructions"):
        # Keep the native DLM prompt short; the full question remains in the
        # row provenance and is not silently discarded from the manifest.
        text = str(question["instructions"])
        if task == "rmot":
            return str((row.get("state", {}).get("task", {}) or {}).get("expression", "object"))
        return "object" if not text else text[:256]
    state_task = row.get("state", {}).get("task", {}) if isinstance(row.get("state"), Mapping) else {}
    vocabulary = state_task.get("vocabulary") if isinstance(state_task, Mapping) else None
    if vocabulary:
        return ", ".join(str(value) for value in vocabulary[:8])
    return "person" if task in {"mot", "rmot"} else "object"


def _frame(row: Mapping[str, Any]) -> float:
    for key in ("frame_id", "frame"):
        if row.get(key) is not None:
            return float(row[key])
    state = row.get("state") if isinstance(row.get("state"), Mapping) else {}
    for key in ("timestamp", "frame"):
        if state.get(key) is not None:
            return float(state[key])
    supervision = row.get("supervision") if isinstance(row.get("supervision"), Mapping) else {}
    return float(supervision.get("frame", 0))


def _source(row: Mapping[str, Any]) -> str:
    return str(row.get("video_id", row.get("source", row.get("id", "unknown"))))


def _history_ids(row: Mapping[str, Any]) -> list[int]:
    state = row.get("state") if isinstance(row.get("state"), Mapping) else {}
    tracks = row.get("past_track_memory", state.get("historical_tracks", [])) or []
    result: list[int] = []
    for track in tracks:
        if isinstance(track, Mapping) and track.get("id") is not None:
            value = int(track["id"])
        elif isinstance(track, (int, str)) and str(track).lstrip("-").isdigit():
            value = int(track)
        else:
            continue
        if value not in result:
            result.append(value)
    return result


def _target(row: Mapping[str, Any]) -> dict[str, float]:
    value = row.get("target")
    if isinstance(value, Mapping):
        return {str(key): float(number) for key, number in value.items()}
    identity = row.get("identity_choice_target")
    if identity is not None:
        return {str(identity): 1.0}
    yesno = row.get("identity_yesno_targets")
    if isinstance(yesno, Mapping):
        return {str(key): float(number) for key, number in yesno.items() if number is not None}
    return {}


def _mode(row: Mapping[str, Any], task: str) -> str:
    if task == "ovmot":
        return str(row.get("task", "ovmot"))
    if task == "rmot":
        return "yesno"
    raw_task = str(row.get("task", "")).lower()
    row_id = str(row.get("id", "")).lower()
    if "admit" in raw_task or "_admit" in row_id:
        return "admit"
    if "memory" in raw_task or "_memory" in row_id:
        return "memory"
    return "identity"


@dataclass
class PreparedRow:
    row: dict[str, Any]
    task: str
    mode: str
    source: str
    frame: float
    image_path: Path
    boxes: list[list[float]]
    labels: list[str]
    candidate_index: int
    query: str
    target: dict[str, float]
    history_ids: list[int]
    history_vectors: dict[int, torch.Tensor] = field(default_factory=dict)
    region_features: torch.Tensor | None = None
    context_features: torch.Tensor | None = None
    skipped_reason: str | None = None


def _iter_manifest_sources(manifest: Path) -> Iterator[tuple[Path, str]]:
    if manifest.suffix in {".jsonl", ".gz"}:
        yield manifest, manifest.stem
        return
    payload = json.loads(manifest.read_text())
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping) and item.get("path"):
                yield Path(str(item["path"])), str(item.get("source_name", item["path"]))
        return
    for item in payload.get("sources", []):
        if isinstance(item, Mapping) and item.get("path"):
            yield Path(str(item["path"])), str(item.get("source_name", item["path"]))


def _bound_native_boxes(
    boxes: list[list[float]], labels: list[str], index: int, limit: int,
) -> tuple[list[list[float]], list[str], int, bool]:
    """Keep a deterministic competition window for the native DLM prompt.

    MOT rows can contain over one hundred detector proposals.  Serializing all
    of them into the DLM's spatial answer is both unnecessary for the selected
    JEV candidate and outside the tested native context length.  The selected
    candidate and its unchanged box are always retained; the surrounding
    proposals are a deterministic local competition window.  The source row
    remains intact and the truncation is recorded in its provenance.
    """
    if limit < 1:
        raise ValueError("max_native_boxes must be positive")
    if len(boxes) <= limit:
        return boxes, labels, index, False
    selected_index = max(0, min(int(index), len(boxes) - 1))
    start = max(0, min(selected_index - limit // 2, len(boxes) - limit))
    selected = list(range(start, start + limit))
    remapped = selected.index(selected_index)
    return [boxes[i] for i in selected], [labels[i] for i in selected], remapped, True


def load_rows(
    manifest: Path, *, runtime_root: Path, rows_per_task: int, seed: int,
    max_native_boxes: int = 16,
) -> tuple[list[PreparedRow], dict[str, Any]]:
    rng = random.Random(int(seed))
    reservoirs: dict[str, list[dict[str, Any]]] = {"mot": [], "rmot": [], "ovmot": []}
    seen: dict[str, int] = {key: 0 for key in reservoirs}
    source_receipts: list[dict[str, Any]] = []
    reservoir_size = max(rows_per_task * 8, rows_per_task)
    for source_path, source_name in _iter_manifest_sources(manifest):
        if not source_path.is_absolute():
            source_path = (manifest.parent / source_path).resolve()
        source_count = 0
        with _open_rows(source_path) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("split", "train") != "train":
                    continue
                provenance = row.get("source_provenance") if isinstance(row.get("source_provenance"), Mapping) else {}
                supervision = row.get("supervision") if isinstance(row.get("supervision"), Mapping) else {}
                if provenance.get("future_evidence") or provenance.get("official_test_labels_read"):
                    raise ValueError(f"forbidden future/test evidence in {source_path}:{line_number}")
                if supervision.get("future_images_read") or supervision.get("official_test_labels_read"):
                    raise ValueError(f"forbidden future/test evidence in {source_path}:{line_number}")
                task = _task(row)
                seen[task] += 1
                source_count += 1
                item = dict(row)
                item["_source_path"] = str(source_path.resolve())
                item["_source_name"] = source_name
                item["_source_line"] = line_number
                bucket = reservoirs[task]
                if len(bucket) < reservoir_size:
                    bucket.append(item)
                else:
                    slot = rng.randrange(seen[task])
                    if slot < reservoir_size:
                        bucket[slot] = item
        source_receipts.append({"path": str(source_path.resolve()), "source_name": source_name, "rows_scanned": source_count})

    prepared: list[PreparedRow] = []
    skipped: dict[str, int] = {"missing_image": 0, "missing_candidate": 0, "invalid_target": 0}
    truncated_rows = 0
    max_boxes_before_truncation = 0
    for task, bucket in reservoirs.items():
        rng.shuffle(bucket)
        accepted = 0
        for row in bucket:
            if accepted >= rows_per_task:
                break
            source_path = Path(str(row["_source_path"]))
            image_path = _media_path(row, manifest_path=source_path, runtime_root=runtime_root)
            if image_path is None:
                skipped["missing_image"] += 1
                continue
            boxes, labels = _candidates(row)
            if not boxes:
                # OVMOT teacher rows contain an object crop sheet rather than
                # a full-frame candidate list.  A full crop is a legal visual
                # input for the category head; native TAO rows normally take
                # the candidate path above.
                try:
                    from PIL import Image
                    with Image.open(image_path) as image:
                        boxes, labels = [[0.0, 0.0, float(image.width), float(image.height)]], ["object"]
                except Exception:
                    skipped["missing_candidate"] += 1
                    continue
            index = _candidate_index(row, len(boxes))
            original_count = len(boxes)
            original_index = index
            max_boxes_before_truncation = max(max_boxes_before_truncation, original_count)
            boxes, labels, index, truncated = _bound_native_boxes(
                boxes, labels, index, max_native_boxes,
            )
            if truncated:
                truncated_rows += 1
                row["_groundjev_native_box_policy"] = {
                    "limit": max_native_boxes,
                    "original_count": original_count,
                    "selected_candidate_original_index": int(original_index),
                    "selection": "deterministic_local_window_with_target_preserved",
                }
            mode = _mode(row, task)
            target = _target(row)
            if not target:
                skipped["invalid_target"] += 1
                continue
            prepared.append(PreparedRow(
                row=row, task=task, mode=mode, source=_source(row), frame=_frame(row),
                image_path=image_path, boxes=boxes, labels=labels,
                candidate_index=index, query=_query(row, task), target=target,
                history_ids=_history_ids(row),
            ))
            accepted += 1
    prepared.sort(key=lambda item: (item.source, item.frame, str(item.row.get("id", ""))))
    return prepared, {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "scanned_rows_by_task": seen,
        "selected_rows_by_task": {task: sum(item.task == task for item in prepared) for task in reservoirs},
        "source_receipts": source_receipts,
        "skipped": skipped,
        "reservoir_size_per_task": reservoir_size,
        "native_dlm_box_limit": max_native_boxes,
        "rows_with_native_dlm_box_truncation": truncated_rows,
        "max_native_boxes_before_truncation": max_boxes_before_truncation,
    }


def encode_features(
    rows: list[PreparedRow], bridge: G2FeatureBridge, *,
    bridge_factory: Callable[[], G2FeatureBridge] | None = None,
    native_forward_chunk_size: int = 0,
) -> dict[str, Any]:
    if native_forward_chunk_size < 0:
        raise ValueError("native_forward_chunk_size must be non-negative")
    cache: dict[tuple[str, tuple[tuple[float, ...], ...], str], tuple[torch.Tensor, torch.Tensor]] = {}
    next_frame = 0
    total_forwards = 0
    provider_restarts = 0
    active_bridge = bridge
    del bridge
    for item in rows:
        key = (str(item.image_path), tuple(tuple(round(v, 4) for v in box) for box in item.boxes), item.query)
        if key not in cache:
            if (native_forward_chunk_size and next_frame > 0
                    and next_frame % native_forward_chunk_size == 0):
                if bridge_factory is None:
                    raise ValueError("chunked native encoding requires bridge_factory")
                if torch.cuda.is_available():
                    # Native DLM kernels are launched asynchronously.  Do not
                    # destroy/reload a provider while its last forward still
                    # owns CUDA work; that race surfaced as an illegal access
                    # exactly at the first chunk boundary.
                    torch.cuda.synchronize()
                del active_bridge
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                active_bridge = bridge_factory()
                provider_restarts += 1
            try:
                state = active_bridge.encode(
                    None, item.boxes, item.labels, frame_id=next_frame,
                    image_key=str(item.image_path), task_query=item.query,
                    metadata={"joint_task": item.task, "source_row_id": str(item.row.get("id"))},
                )
            except Exception as error:
                raise RuntimeError(
                    "native feature encoding failed at "
                    f"unique_input_index={next_frame}, row_id={item.row.get('id')!r}, "
                    f"image={str(item.image_path)!r}, task={item.task!r}"
                ) from error
            cache[key] = (
                state.vision_tokens.detach().to("cpu", dtype=torch.float32),
                (state.contextual_tokens.detach().to("cpu", dtype=torch.float32)
                 if state.contextual_tokens is not None else torch.empty((0, state.feature_dim), dtype=torch.float32)),
            )
            next_frame += 1
            total_forwards += 1
        item.region_features, item.context_features = cache[key]
    return {"unique_visual_inputs": len(cache), "bridge_forward_count": total_forwards,
            "native_forward_chunk_size": native_forward_chunk_size,
            "native_forward_provider_restarts": provider_restarts,
            "adapter_step": int(getattr(active_bridge.provider, "adapter_step", -1)),
            "feature_dim": int(rows[0].region_features.shape[-1]) if rows and rows[0].region_features is not None else 0}


def attach_causal_history(rows: list[PreparedRow]) -> dict[str, int]:
    """Attach only pre-frame candidate features to each row's memory slots."""
    groups: dict[tuple[str, float], list[PreparedRow]] = {}
    for item in rows:
        groups.setdefault((item.source, item.frame), []).append(item)
    bank: dict[str, dict[int, torch.Tensor]] = {}
    missing = 0
    attached = 0
    for key in sorted(groups, key=lambda value: (value[0], value[1])):
        source, _ = key
        source_bank = bank.setdefault(source, {})
        for item in groups[key]:
            item.history_vectors = {}
            for track_id in item.history_ids:
                vector = source_bank.get(track_id)
                if vector is None:
                    missing += 1
                else:
                    item.history_vectors[track_id] = vector
                    attached += 1
        for item in groups[key]:
            state = item.row.get("state") if isinstance(item.row.get("state"), Mapping) else {}
            supervision = item.row.get("supervision") if isinstance(item.row.get("supervision"), Mapping) else {}
            target_id = supervision.get("matched_gt")
            if target_id is None:
                identity_target = supervision.get("identity_target")
                if identity_target is not None and str(identity_target).lstrip("-").isdigit():
                    target_id = identity_target
            if target_id is None:
                target_value = item.target
                numeric = [key for key, value in target_value.items() if value > 0 and key.lstrip("-").isdigit()]
                target_id = numeric[0] if numeric else None
            if target_id is not None and item.region_features is not None:
                index = min(item.candidate_index, item.region_features.shape[0] - 1)
                source_bank[int(target_id)] = item.region_features[index].detach().clone()
    return {"history_vectors_attached": attached, "history_vectors_missing": missing}


def _forward(item: PreparedRow, model: GroundJEVModel, heads: GroundJEVTaskHeads, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if item.region_features is None or item.context_features is None:
        raise RuntimeError("row has no encoded G2 features")
    region = item.region_features[item.candidate_index:item.candidate_index + 1].to(device)
    context = item.context_features.to(device)
    context_bias = None if context.numel() == 0 else model.context_projector(context).mean(dim=0)
    ordered_ids = list(item.history_vectors)
    if ordered_ids:
        raw = torch.stack([item.history_vectors[track_id] for track_id in ordered_ids]).to(device)
        memory = model.memory_projector(raw)
    else:
        memory = region.new_empty((0, model.hidden_dim))
    slots = model.readout.make_slot_tokens(memory, ["NEW", "NO_MATCH", "REJECT"])
    outputs = model.readout(region, slots, action_start=len(ordered_ids), context_bias=context_bias)
    outputs.update(heads(outputs["candidate_hidden"]))
    return outputs, {"ordered_ids": ordered_ids, "action_start": len(ordered_ids)}


def _positive(target: Mapping[str, float]) -> float | None:
    for key in ("yes", "no"):
        if key in target:
            return 1.0 if key == "yes" else 0.0
    return None


def _loss_for_row(item: PreparedRow, outputs: Mapping[str, torch.Tensor], meta: Mapping[str, Any]) -> tuple[torch.Tensor | None, dict[str, Any]]:
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, Any] = {"task": item.task, "mode": item.mode, "identity_valid": False}
    if item.task == "mot" and item.mode == "identity":
        target_key = next((key for key, value in item.target.items() if value > 0), None)
        target_index: int | None = None
        if target_key == "NEW":
            target_index = int(meta["action_start"])
        elif target_key == "NO_MATCH":
            target_index = int(meta["action_start"]) + 1
        elif target_key == "REJECT":
            target_index = int(meta["action_start"]) + 2
        elif target_key is not None and target_key.lstrip("-").isdigit():
            try:
                target_index = list(meta["ordered_ids"]).index(int(target_key))
            except ValueError:
                target_index = None
        if target_index is not None and target_index < outputs["identity_logits"].shape[-1]:
            losses["identity_choice"] = F.cross_entropy(outputs["identity_logits"], torch.tensor([target_index], device=outputs["identity_logits"].device))
            metrics["identity_valid"] = True
            metrics["identity_correct_target"] = target_index
        else:
            metrics["identity_masked_reason"] = "target_identity_not_in_causal_slots"
    elif item.task == "mot" and item.mode == "admit":
        value = _positive(item.target)
        if value is not None:
            losses["admit"] = F.binary_cross_entropy_with_logits(outputs["admit_logits"], torch.tensor([value], device=outputs["admit_logits"].device))
            metrics["admit_target"] = value
    elif item.task == "mot" and item.mode == "memory":
        value = _positive(item.target)
        if value is not None:
            losses["memory"] = F.binary_cross_entropy_with_logits(outputs["memory_logits"], torch.tensor([value], device=outputs["memory_logits"].device))
            metrics["memory_target"] = value
    elif item.task == "rmot":
        value = _positive(item.target)
        if value is not None:
            losses["yesno"] = F.binary_cross_entropy_with_logits(outputs["yesno_logits"], torch.tensor([value], device=outputs["yesno_logits"].device))
            metrics["yesno_target"] = value
    elif item.task == "ovmot":
        target_key = next((key for key, value in item.target.items() if value > 0 and key.lstrip("-").isdigit()), None)
        if target_key is not None:
            target_index = int(target_key)
            head_key = "category_group_logits" if item.mode == "category_group" else "category_leaf_logits"
            logits = outputs[head_key]
            if 0 <= target_index < logits.shape[-1]:
                losses["category"] = F.cross_entropy(logits, torch.tensor([target_index], device=logits.device))
                metrics["category_target"] = target_index
    if not losses:
        return None, metrics
    total = sum(losses.values())
    metrics["loss_names"] = sorted(losses)
    metrics["total_loss"] = total
    return total, metrics


def evaluate(rows: Iterable[PreparedRow], model: GroundJEVModel, heads: GroundJEVTaskHeads, device: torch.device) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for item in rows:
            outputs, meta = _forward(item, model, heads, device)
            _, metrics = _loss_for_row(item, outputs, meta)
            bucket = totals.setdefault(item.task, {"valid": 0.0, "correct": 0.0, "loss_rows": 0.0})
            if item.task == "mot" and item.mode == "identity" and metrics.get("identity_valid"):
                target = int(metrics["identity_correct_target"])
                bucket["valid"] += 1
                bucket["correct"] += float(int(int(outputs["identity_logits"].argmax(-1).item()) == target))
            elif item.task == "mot" and item.mode == "admit" and "admit_target" in metrics:
                bucket["valid"] += 1
                bucket["correct"] += float(int(bool(torch.sigmoid(outputs["admit_logits"])[0].item() >= 0.5) == bool(metrics["admit_target"])))
            elif item.task == "mot" and item.mode == "memory" and "memory_target" in metrics:
                bucket["valid"] += 1
                bucket["correct"] += float(int(bool(torch.sigmoid(outputs["memory_logits"])[0].item() >= 0.5) == bool(metrics["memory_target"])))
            elif item.task == "rmot" and "yesno_target" in metrics:
                bucket["valid"] += 1
                bucket["correct"] += float(int(bool(torch.sigmoid(outputs["yesno_logits"])[0].item() >= 0.5) == bool(metrics["yesno_target"])))
            elif item.task == "ovmot" and "category_target" in metrics:
                key = "category_group_logits" if item.mode == "category_group" else "category_leaf_logits"
                bucket["valid"] += 1
                bucket["correct"] += float(int(int(outputs[key].argmax(-1).item()) == int(metrics["category_target"])))
            if metrics.get("loss_names"):
                bucket["loss_rows"] += 1
    for bucket in totals.values():
        bucket["accuracy"] = bucket["correct"] / max(bucket["valid"], 1.0)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", default=MODEL_SHA_DEFAULT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-task", type=int, default=24)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--max-native-boxes", type=int, default=16,
                        help="maximum competing boxes serialized into each native G2 DLM forward")
    parser.add_argument("--native-forward-chunk-size", type=int, default=64,
                        help="restart the frozen native G2 provider after this many unique inputs; 0 disables chunking")
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.rows_per_task < 1 or args.steps < 1 or args.max_native_boxes < 1 or args.native_forward_chunk_size < 0:
        raise ValueError("rows-per-task, steps and max-native-boxes must be positive; chunk size must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native three-task GroundJEV training requires CUDA")
    manifest = Path(args.manifest).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    source_root = Path(args.source_root).resolve()
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(source_root)
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    start = time.perf_counter()
    rows, audit = load_rows(
        manifest, runtime_root=runtime_root, rows_per_task=args.rows_per_task,
        seed=args.seed, max_native_boxes=args.max_native_boxes,
    )
    task_counts = {task: sum(item.task == task for item in rows) for task in ("mot", "rmot", "ovmot")}
    if any(value <= 0 for value in task_counts.values()):
        raise RuntimeError(f"joint manifest sampling did not retain all tasks: {task_counts}; audit={audit}")
    def make_bridge() -> G2FeatureBridge:
        return G2FeatureBridge.from_native_clean_dlm(
            args.checkpoint, args.adapter, runtime_root=runtime_root,
            device=str(device), model_sha256=args.model_sha256, task_query="person",
        )

    feature_audit = encode_features(
        rows, make_bridge(), bridge_factory=make_bridge,
        native_forward_chunk_size=args.native_forward_chunk_size,
    )
    history_audit = attach_causal_history(rows)
    feature_dim = int(feature_audit["feature_dim"])
    model = GroundJEVModel(feature_dim, hidden_dim=256).to(device)
    heads = GroundJEVTaskHeads(256).to(device)
    parameters = list(model.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
    before = evaluate(rows, model, heads, device)
    history: list[dict[str, Any]] = []
    update_counts: dict[str, int] = {"mot": 0, "rmot": 0, "ovmot": 0}
    masked_identity = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model.train(); heads.train()
    for step in range(1, args.steps + 1):
        item = rows[(step - 1) % len(rows)]
        outputs, meta = _forward(item, model, heads, device)
        total, metrics = _loss_for_row(item, outputs, meta)
        if total is None:
            masked_identity += int(metrics.get("identity_masked_reason") is not None)
            continue
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite joint loss at step={step}, row={item.row.get('id')}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError(f"non-finite joint gradient at step={step}")
        optimizer.step()
        update_counts[item.task] += 1
        history.append({"step": step, "row_id": item.row.get("id"), "task": item.task,
                        "mode": item.mode, "loss": float(total.detach().cpu()),
                        "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                        "loss_names": metrics.get("loss_names", [])})
    model.eval(); heads.eval()
    after = evaluate(rows, model, heads, device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"groundjev-native-three-task-step{args.steps:06d}.pt"
    torch.save({
        "format": SCHEMA_VERSION,
        "step": args.steps,
        "model_sha256": args.model_sha256,
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_step": int(feature_audit.get("adapter_step", -1)),
        "manifest": str(manifest),
        "manifest_sha256": audit["manifest_sha256"],
        "groundjev_model": model.state_dict(),
        "task_heads": heads.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "selected_row_ids": [str(item.row.get("id")) for item in rows],
        "task_counts": task_counts,
    }, checkpoint_path)
    receipt = {
        "status": "GROUNDJEV_NATIVE_THREE_TASK_COUPLED_READOUT_COMPLETE",
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": audit["manifest_sha256"],
        "checkpoint": str(checkpoint_path),
        "model_sha256": args.model_sha256,
        "g2_adapter": str(Path(args.adapter).resolve()),
        "g2_adapter_step": feature_audit["adapter_step"],
        "g2_frozen": True,
        "single_backbone": True,
        "independent_vlm_instances": 0,
        "unique_visual_inputs": feature_audit["unique_visual_inputs"],
        "visual_forward_count": feature_audit["bridge_forward_count"],
        "one_visual_encode_per_unique_input": True,
        "rows": len(rows),
        "task_counts": task_counts,
        "optimizer_steps_requested": args.steps,
        "optimizer_updates_by_task": update_counts,
        "masked_identity_targets": masked_identity,
        "history_audit": history_audit,
        "sampling_audit": audit,
        "loss_weights": {
            "identity_choice": 1.0, "admit": 1.0, "yesno": 1.0,
            "memory": 1.0, "category": 1.0,
        },
        "grounding_loss": "frozen-backbone phase; retained for the later G2 LoRA phase",
        "before": before,
        "after": after,
        "history": history,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "real GPU three-task shared-G2 readout training; no independent YesTrack/OneJev model and no benchmark generalization claim",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
