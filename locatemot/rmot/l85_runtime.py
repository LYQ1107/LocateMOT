"""Runtime and compact semantic-state view for the L85 full-RMOT model.

The only detector-derived semantic input is the selected L84 Z1 fixed-reference
state.  A group is built with one native GroundingDINO visual pass and one
expression replay per query; selected tensors are immediately copied to CPU.
The optional cache contains only compact Z1/text summaries and row provenance,
never pixels, detector weights, raw feature maps, labels, or candidate scores.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from locatemot.models.l84_grounding_states import capture_l84_states
from locatemot.rmot.l80_data import L80BankStore, load_full_unit_for_labels
from locatemot.rmot.l85_fullvideo_bank import L69_FEATURE_ROOT, file_meta

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
L49_DATA = ROOT / "outputs/l49/data"
L82_SPLIT = ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SELECTED_STAGE = "Z1"
OBS_DIM = 1432


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_keys(keys: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(json.dumps([list(value) for value in keys], sort_keys=False).encode()).hexdigest()


def group_key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}|{row['video']}|{int(row['frame_id'])}"


def key_only(row: dict[str, Any]) -> dict[str, Any]:
    sentence = str(row.get("sentence") or row.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty sentence for {row.get('unit_key')}")
    result = {key: row[key] for key in ("unit_key", "dataset", "video", "query_id", "frame_id")}
    result["sentence"] = sentence; result["expression"] = sentence
    if set(result) & {"target_ids", "positive_indices", "positive_count", "category", "labels", "target_present"}:
        raise AssertionError("label field leaked into key-only row")
    return result


def load_key_rows(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if split is not None and raw.get("split") != split:
                continue
            if raw.get("dataset") not in {"refer_kitti_v1", "refer_kitti_v2"}:
                continue
            result.append(key_only(raw))
    if len({str(x["unit_key"]) for x in result}) != len(result):
        raise AssertionError(f"duplicate key rows in {path}")
    return result


def load_fit_key_rows() -> list[dict[str, Any]]:
    result = load_key_rows(L49_DATA / "train_units.jsonl", "fit")
    if len(result) != 5314:
        raise AssertionError(f"fit unit count drift: {len(result)}")
    return result


def load_validation_key_rows() -> list[dict[str, Any]]:
    return load_key_rows(L49_DATA / "validation_units.jsonl", "validation")


def load_calibration_key_rows() -> list[dict[str, Any]]:
    return load_key_rows(L49_DATA / "calibration_units.jsonl", "calibration")


def build_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    result = {}
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda row: (int(row["query_id"]), str(row["unit_key"])))
        dataset, video, frame = key.split("|")
        result[key] = {"group_key": key, "dataset": dataset, "video": video, "frame_id": int(frame), "queries": ordered}
    return result


def load_fit_train_dev_groups() -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    groups = build_groups(load_fit_key_rows())
    split = json.loads(L82_SPLIT.read_text())
    train_keys = [str(x) for x in split["train_group_keys"]]
    dev_keys = [str(x) for x in split["dev_group_keys"]]
    if len(train_keys) != 524 or len(dev_keys) != 138:
        raise AssertionError("L82 train/dev group count drift")
    if any(key not in groups for key in train_keys + dev_keys):
        raise AssertionError("L82 group key missing from fit rows")
    return groups, train_keys, dev_keys


def load_internal_eval_groups() -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    calibration = build_groups(load_calibration_key_rows())
    validation = build_groups(load_validation_key_rows())
    return {**calibration, **validation}, sorted(calibration), sorted(validation)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 3 or mask.ndim != 2 or value.shape[:2] != mask.shape:
        raise AssertionError(f"{name} shape mismatch: {tuple(value.shape)} / {tuple(mask.shape)}")
    weights = mask.to(value.dtype).unsqueeze(-1)
    result = (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if not bool(torch.isfinite(result.float()).all()):
        raise FloatingPointError(f"nonfinite {name}")
    return result


def _row_audit(batch: Any) -> dict[str, Any]:
    return {"unit_key": str(batch.unit_key), "candidate_count": int(batch.candidate_count),
            "row_offsets": [int(x) for x in batch.row_offsets], "row_key_digest": digest_keys(batch.row_keys),
            "candidate_indices": [int(x) for x in batch.candidate_indices], "pool_ids": [int(x) for x in batch.pool_ids],
            "track_ids": [int(x) for x in batch.track_ids],
            "duplicate_candidate_index_rows": len(batch.candidate_indices) - len(set(batch.candidate_indices)),
            "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "future_history_count": int((batch.history_frame_ids > int(batch.frame_id)).sum())}


def capture_group_z1(group: dict[str, Any], device: torch.device, runtime: Any | None = None,
                     bank_store: L80BankStore | None = None) -> dict[str, Any]:
    """Capture one complete group before any labels are loaded."""
    from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime, install_clip_torchvision_compat
    install_clip_torchvision_compat()
    store = bank_store or L80BankStore(max_history=8)
    owns_store = bank_store is None
    batches = [store.build_unit(row) for row in group["queries"]]
    if not batches:
        raise AssertionError(f"empty group {group['group_key']}")
    first = batches[0]; n = first.candidate_count
    if n <= 0:
        raise AssertionError(f"empty candidate set {group['group_key']}")
    for batch in batches:
        if batch.candidate_count != n or batch.row_offsets != first.row_offsets:
            raise AssertionError(f"candidate set drift in {batch.unit_key}")
        if int((batch.history_frame_ids > int(batch.frame_id)).sum()) != 0:
            raise AssertionError(f"future history in {batch.unit_key}")
    owns_runtime = runtime is None
    if owns_runtime:
        runtime = GroundingCandidateReferenceRuntime(device)
    started = time.perf_counter()
    try:
        runtime.encoder_events.clear(); runtime.capture.clear()
        with torch.inference_mode():
            native = runtime.inference_detector(runtime.model, str(first.image_path), text_prompt=str(first.sentence), custom_entities=True)
        if len(runtime.encoder_events) != 1:
            raise AssertionError("native encoder event count drift")
        event = runtime.encoder_events[-1]
        visual_feats = runtime.capture.get("visual_feats")
        sample_template = runtime.capture.get("sample_template")
        if visual_feats is None or not isinstance(sample_template, (list, tuple)) or len(sample_template) != 1:
            raise AssertionError("reusable GroundingDINO visual contract missing")
        image_shape = tuple(int(x) for x in native.metainfo["img_shape"][:2])
        scale_factor = native.metainfo["scale_factor"]; sample_template = sample_template[0]
        z_values, text_values, frame_values, query_audits = [], [], [], []
        replay_seconds = 0.0
        for batch in batches:
            text_dict, caption, token_map = runtime.make_text_dict(runtime.model, str(batch.sentence), device, force_pad_to_max=True)
            sample = copy.deepcopy(sample_template); runtime.set_sample_text(sample, caption, token_map)
            before = len(runtime.encoder_events); replay_start = time.perf_counter()
            with torch.inference_mode():
                runtime.original_forward_transformer(visual_feats, text_dict, [sample])
            replay_seconds += time.perf_counter() - replay_start
            if len(runtime.encoder_events) != before + 1:
                raise AssertionError(f"replay encoder event drift: {batch.unit_key}")
            replay = runtime.encoder_events[-1]
            state = capture_l84_states(runtime.model, replay, batch.boxes.to(device), image_shape, scale_factor, selected_name=SELECTED_STAGE)
            z = state["states"][SELECTED_STAGE].float().detach().cpu().contiguous()
            text_memory = replay["memory_text"].float(); text_valid = replay["text_token_mask"]
            if text_valid is None:
                text_valid = torch.ones(text_memory.shape[:2], dtype=torch.bool, device=text_memory.device)
            text_global = _masked_mean(text_memory, text_valid.bool(), "memory_text").squeeze(0).detach().cpu().contiguous()
            visual_memory = replay["memory"].float(); memory_mask = replay["memory_mask"]
            visual_valid = torch.ones(visual_memory.shape[:2], dtype=torch.bool, device=visual_memory.device)
            if memory_mask is not None:
                visual_valid = ~memory_mask.bool()
            frame_global = _masked_mean(visual_memory, visual_valid, "encoder_memory").squeeze(0).detach().cpu().contiguous()
            for name, value, shape in (("Z1", z, (n, 256)), ("text_global", text_global, (256,)), ("frame_global", frame_global, (256,))):
                if tuple(value.shape) != shape or not bool(torch.isfinite(value.float()).all()):
                    raise AssertionError(f"{name} shape/finite drift: {batch.unit_key} {tuple(value.shape)}")
            z_values.append(z); text_values.append(text_global); frame_values.append(frame_global)
            query_audits.append(_row_audit(batch) | {"z1_shape": list(z.shape), "text_global_shape": list(text_global.shape),
                "frame_global_shape": list(frame_global.shape), "memory_shape": list(replay["memory"].shape),
                "memory_text_shape": list(replay["memory_text"].shape), "text_valid_tokens": int(text_valid.sum()),
                "visual_valid_tokens": int(visual_valid.sum())})
            runtime.encoder_events.clear(); del state, replay, sample, text_dict, z, text_global, frame_global
        return {"format": "locatemot-l85-z1-semantic-group-v1", "group_key": str(group["group_key"]),
                "dataset": str(group["dataset"]), "video": str(group["video"]), "frame_id": int(group["frame_id"]),
                "query_unit_keys": [str(batch.unit_key) for batch in batches], "query_ids": [int(batch.query_id) for batch in batches],
                "sentences": [str(batch.sentence) for batch in batches], "z1": torch.stack(z_values).half(),
                "text_global": torch.stack(text_values).half(), "frame_global": torch.stack(frame_values).half(),
                "candidate_count": int(first.candidate_count),
                "row_offsets": [int(x) for x in first.row_offsets], "row_keys_digest": digest_keys(first.row_keys),
                "candidate_indices": [int(x) for x in first.candidate_indices], "track_ids": [int(x) for x in first.track_ids],
                "pool_ids": [int(x) for x in first.pool_ids], "boxes": first.boxes.float().clone(),
                "boxes_norm": first.boxes_norm.float().clone(), "image_size": [int(first.image_size[0]), int(first.image_size[1])],
                "query_audits": query_audits, "native_image_shape": list(image_shape),
                "native_scale_factor": np.asarray(scale_factor).reshape(-1).tolist(),
                "native_seconds": float(time.perf_counter() - started - replay_seconds), "replay_seconds": float(replay_seconds),
                "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                "features_persistent": True, "persistent_payload": "compact Z1/text summaries only; no raw/dense map"}
    finally:
        if owns_runtime:
            runtime.close()
        del batches
        if owns_store:
            del store
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _slice_batched_encoder_event(event: dict[str, Any], index: int,
                                 batch_size: int) -> dict[str, Any]:
    """Select one sample from a batched encoder hook without changing tensors."""
    batch_keys = {"memory", "memory_text", "memory_mask", "valid_ratios",
                  "text_attention_mask", "text_token_mask"}
    result: dict[str, Any] = {}
    for key, value in event.items():
        if key in batch_keys and torch.is_tensor(value):
            if value.ndim == 0 or int(value.shape[0]) != int(batch_size):
                raise AssertionError(f"batched event leading dimension drift for {key}: {tuple(value.shape)}")
            result[key] = value[index:index + 1]
        else:
            result[key] = value
    return result


def _fixed_reference_decoder_batch(model: Any, seed: torch.Tensor,
                                   references: torch.Tensor,
                                   event: dict[str, Any]) -> torch.Tensor:
    """Run the audited fixed-reference decoder for a query batch.

    This is the same loop as ``tools.l82_audit_grounding_interface`` but keeps
    the leading query batch dimension.  L85 only needs the fixed-reference Z1
    state; the native iterative decoder is deliberately not rerun here.
    """
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding

    if seed.ndim != 3 or references.ndim != 3:
        raise ValueError("batched fixed-reference decoder expects [B,N,D]/[B,N,4]")
    if seed.shape[:2] != references.shape[:2] or references.shape[-1] != 4:
        raise ValueError(f"batched seed/reference shape mismatch: {tuple(seed.shape)} / {tuple(references.shape)}")
    if event["memory"].ndim != 3 or int(event["memory"].shape[0]) != int(seed.shape[0]):
        raise ValueError("batched event/memory shape mismatch")
    query = seed
    reference_batch = references
    intermediate = []
    # L85 selected Z1, which is the output of the first fixed-reference
    # decoder layer.  Computing later layers cannot affect that already
    # materialized state and would multiply the full-video cost.
    for layer in model.decoder.layers[:1]:
        reference_input = reference_batch[:, :, None] * torch.cat(
            [event["valid_ratios"], event["valid_ratios"]], dim=-1)[:, None]
        query_sine_embed = coordinate_to_encoding(reference_input[:, :, 0, :])
        query_pos = model.decoder.ref_point_head(query_sine_embed)
        query = layer(
            query,
            query_pos=query_pos,
            value=event["memory"],
            key_padding_mask=event["memory_mask"],
            self_attn_mask=None,
            spatial_shapes=event["spatial_shapes"],
            level_start_index=event["level_start_index"],
            valid_ratios=event["valid_ratios"],
            reference_points=reference_input,
            memory_text=event["memory_text"],
            text_attention_mask=event["text_attention_mask"])
        intermediate.append(model.decoder.norm(query))
    hidden = torch.stack(intermediate)
    if hidden.ndim != 4 or tuple(hidden.shape[1:3]) != tuple(seed.shape[:2]):
        raise AssertionError(f"unexpected batched fixed-reference output {tuple(hidden.shape)}")
    return hidden


def capture_group_z1_batched(group: dict[str, Any], device: torch.device,
                             runtime: Any | None = None,
                             bank_store: L80BankStore | None = None,
                             query_batch_size: int = 8) -> dict[str, Any]:
    """Capture L85 Z1 features with batched expression replays.

    The native image feature pass remains exactly one per frame.  Expression
    replays are grouped only for throughput; every query receives its own
    text, encoder memory and fixed-reference decoder state, and the returned
    row/order contract is identical to :func:`capture_group_z1`.
    """
    from locatemot.models.l82_grounding_reference import (
        boxes_to_reference_points, boxes_xyxy_to_normalized, candidate_seed_with_reference,
        pool_memory_by_box,
    )
    from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime, install_clip_torchvision_compat
    from tools.l82_audit_grounding_interface import make_text_batch_dict
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding

    if int(query_batch_size) < 1:
        raise ValueError("query_batch_size must be positive")
    install_clip_torchvision_compat()
    store = bank_store or L80BankStore(max_history=8)
    owns_store = bank_store is None
    query_rows = [dict(row) for row in group["queries"]]
    if not query_rows:
        raise AssertionError(f"empty group {group['group_key']}")
    first = store.build_unit(query_rows[0]); n = int(first.candidate_count)
    if n <= 0:
        raise AssertionError(f"empty candidate set {group['group_key']}")
    for row in query_rows:
        if (str(row["dataset"]), str(row["video"]), int(row["frame_id"])) != (
                str(first.dataset), str(first.video), int(first.frame_id)):
            raise AssertionError(f"query group identity drift in {row['unit_key']}")
    if int((first.history_frame_ids > int(first.frame_id)).sum()) != 0:
        raise AssertionError(f"future history in {first.unit_key}")

    def row_keys_for_query(row: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [(str(first.dataset), str(first.video), int(row["query_id"]), int(first.frame_id),
                 str(first.bank_path), int(offset)) for offset in first.row_offsets]
    owns_runtime = runtime is None
    if owns_runtime:
        runtime = GroundingCandidateReferenceRuntime(device)
    started = time.perf_counter()
    try:
        runtime.encoder_events.clear(); runtime.capture.clear()
        with torch.inference_mode():
            native = runtime.inference_detector(runtime.model, str(first.image_path),
                                                text_prompt=str(first.sentence), custom_entities=True)
        if len(runtime.encoder_events) != 1:
            raise AssertionError("native encoder event count drift")
        visual_feats = runtime.capture.get("visual_feats")
        sample_templates = runtime.capture.get("sample_template")
        if visual_feats is None or not isinstance(sample_templates, (list, tuple)) or len(sample_templates) != 1:
            raise AssertionError("reusable GroundingDINO visual contract missing")
        image_shape = tuple(int(x) for x in native.metainfo["img_shape"][:2])
        scale_factor = native.metainfo["scale_factor"]
        sample_template = sample_templates[0]
        boxes = first.boxes.to(device=device)
        boxes_norm = boxes_xyxy_to_normalized(boxes, image_shape, scale_factor)
        references = boxes_to_reference_points(boxes_norm)
        reference_encoding = coordinate_to_encoding(references.unsqueeze(0), num_feats=128)
        reference_position = runtime.model.decoder.ref_point_head(reference_encoding).squeeze(0)
        if reference_position.shape != (n, 256) or not bool(torch.isfinite(reference_position.float()).all()):
            raise AssertionError("reference position shape/finite drift")
        z_values: list[torch.Tensor] = []
        text_values: list[torch.Tensor] = []
        frame_values: list[torch.Tensor] = []
        query_audits: list[dict[str, Any]] = []
        replay_seconds = 0.0
        for begin in range(0, len(query_rows), int(query_batch_size)):
            chunk = query_rows[begin:begin + int(query_batch_size)]
            chunk_size = len(chunk)
            captions: list[str] = []
            token_maps: list[Any] = []
            old_pad_to_max = bool(runtime.model.language_model.pad_to_max)
            runtime.model.language_model.pad_to_max = True
            try:
                for batch in chunk:
                    token_map, caption, _positive_map, _entities = runtime.model.get_tokens_positive_and_prompts(
                        str(batch["sentence"]), True, None, None)
                    captions.append(caption)
                    token_maps.append(token_map)
                text_dict = make_text_batch_dict(runtime.model, captions, device, force_pad_to_max=True)
            finally:
                runtime.model.language_model.pad_to_max = old_pad_to_max
            samples = []
            for caption, token_map in zip(captions, token_maps):
                sample = copy.deepcopy(sample_template)
                runtime.set_sample_text(sample, caption, token_map)
                samples.append(sample)
            visual_batch = tuple(value.expand(chunk_size, *value.shape[1:]) for value in visual_feats)
            runtime.encoder_events.clear()
            replay_start = time.perf_counter()
            with torch.inference_mode():
                runtime.original_forward_transformer(visual_batch, text_dict, samples)
            replay_seconds += time.perf_counter() - replay_start
            if len(runtime.encoder_events) != 1:
                raise AssertionError(f"batched replay encoder event drift at {group['group_key']}:{begin}")
            event = runtime.encoder_events[-1]
            batch_memories: list[torch.Tensor] = []
            for index in range(chunk_size):
                replay = _slice_batched_encoder_event(event, index, chunk_size)
                visual_seed, _roi_audit = pool_memory_by_box(
                    replay["memory"], replay["spatial_shapes"], replay["level_start_index"],
                    boxes_norm, replay["memory_mask"], grid_size=4)
                if visual_seed.shape != (n, 256) or not bool(torch.isfinite(visual_seed.float()).all()):
                    raise AssertionError(f"visual seed shape/finite drift: {chunk[index].unit_key}")
                batch_memories.append(visual_seed + reference_position)
                text_memory = replay["memory_text"].float()
                text_valid = replay["text_token_mask"]
                if text_valid is None:
                    text_valid = torch.ones(text_memory.shape[:2], dtype=torch.bool, device=text_memory.device)
                text_global = _masked_mean(text_memory, text_valid.bool(), "memory_text").squeeze(0)
                visual_memory = replay["memory"].float()
                memory_mask = replay["memory_mask"]
                visual_valid = torch.ones(visual_memory.shape[:2], dtype=torch.bool, device=visual_memory.device)
                if memory_mask is not None:
                    visual_valid = ~memory_mask.bool()
                frame_global = _masked_mean(visual_memory, visual_valid, "encoder_memory").squeeze(0)
                text_values.append(text_global.detach().cpu().contiguous())
                frame_values.append(frame_global.detach().cpu().contiguous())
                query_audits.append(_row_audit(first) | {
                    "unit_key": str(chunk[index]["unit_key"]),
                    "query_id": int(chunk[index]["query_id"]),
                    "row_key_digest": digest_keys(row_keys_for_query(chunk[index])),
                    "z1_shape": [n, 256], "text_global_shape": list(text_global.shape),
                    "frame_global_shape": list(frame_global.shape),
                    "memory_shape": list(replay["memory"].shape),
                    "memory_text_shape": list(replay["memory_text"].shape),
                    "text_valid_tokens": int(text_valid.sum()),
                    "visual_valid_tokens": int(visual_valid.sum()),
                    "batched_replay_size": chunk_size,
                })
            seed_batch = torch.stack(batch_memories, dim=0)
            references_batch = references.unsqueeze(0).expand(chunk_size, -1, -1)
            event_batch = event
            hidden = _fixed_reference_decoder_batch(runtime.model, seed_batch, references_batch, event_batch)
            z_values.extend([hidden[0, index].float().detach().cpu().contiguous() for index in range(chunk_size)])
            runtime.encoder_events.clear()
            del event, text_dict, samples, visual_batch, seed_batch, references_batch, hidden, batch_memories
        for name, values, shape in (("Z1", z_values, (n, 256)), ("text_global", text_values, (256,)),
                                    ("frame_global", frame_values, (256,))):
            if len(values) != len(query_rows) or any(tuple(value.shape) != shape or not bool(torch.isfinite(value.float()).all())
                                                   for value in values):
                raise AssertionError(f"{name} batched output shape/finite drift")
        return {"format": "locatemot-l85-z1-semantic-group-v1", "group_key": str(group["group_key"]),
                "dataset": str(group["dataset"]), "video": str(group["video"]), "frame_id": int(group["frame_id"]),
                "query_unit_keys": [str(row["unit_key"]) for row in query_rows],
                "query_ids": [int(row["query_id"]) for row in query_rows],
                "sentences": [str(row["sentence"]) for row in query_rows], "z1": torch.stack(z_values).half(),
                "text_global": torch.stack(text_values).half(), "frame_global": torch.stack(frame_values).half(),
                "candidate_count": n, "row_offsets": [int(x) for x in first.row_offsets],
                "row_keys_digest": digest_keys(first.row_keys), "candidate_indices": [int(x) for x in first.candidate_indices],
                "track_ids": [int(x) for x in first.track_ids], "pool_ids": [int(x) for x in first.pool_ids],
                "boxes": first.boxes.float().clone(), "boxes_norm": boxes_norm.float().cpu().clone(),
                "image_size": [int(first.image_size[0]), int(first.image_size[1])], "query_audits": query_audits,
                "native_image_shape": list(image_shape), "native_scale_factor": np.asarray(scale_factor).reshape(-1).tolist(),
                "native_seconds": float(time.perf_counter() - started - replay_seconds), "replay_seconds": float(replay_seconds),
                "query_batch_size": int(query_batch_size), "all_rows_retained": True,
                "candidate_deletion": False, "candidate_truncation": False, "features_persistent": True,
                "persistent_payload": "compact Z1/text summaries only; no raw/dense map"}
    finally:
        if owns_runtime:
            runtime.close()
        del first, query_rows
        if owns_store:
            del store
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def attach_labels_after_cache(item: dict[str, Any], bank_store: L80BankStore) -> list[dict[str, Any]]:
    """Attach expression labels only after a cache item was constructed."""
    result = []
    for unit_key in item["query_unit_keys"]:
        full = load_full_unit_for_labels(unit_key)
        batch = bank_store.build_unit(key_only(full))
        if batch.row_offsets != item["row_offsets"] or digest_keys(batch.row_keys) != item["row_keys_digest"]:
            raise AssertionError(f"cached/native row contract drift: {unit_key}")
        result.append(bank_store.attach_labels(batch, full))
    return result


__all__ = ["EXPECTED_MANIFEST_SHA", "L49_DATA", "L48_TEXT", "L69_FEATURE_ROOT", "MANIFEST", "OBS_DIM", "SELECTED_STAGE", "THREAD", "attach_labels_after_cache", "build_groups", "capture_group_z1", "capture_group_z1_batched", "digest_keys", "file_meta", "group_key", "key_only", "load_calibration_key_rows", "load_fit_key_rows", "load_fit_train_dev_groups", "load_internal_eval_groups", "load_validation_key_rows", "sha256_file"]
