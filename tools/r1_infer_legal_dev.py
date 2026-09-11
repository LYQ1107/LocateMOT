#!/usr/bin/env python3
"""Replay the trained R1 sidecar on the complete legal-development videos.

The replay is deliberately separate from R1 fitting.  It scores every native
L69 candidate row with the frozen Stage-S anchor plus one R1 checkpoint per
epoch, writes Rule-B predictions before opening legal target records, and only
then materializes legal GT and runs the local TrackEval checkout.  It never
resolves screening or official-test labels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.r1_aligned_track_conditioning import (  # noqa: E402
    ANCHOR_SHA256,
    FrozenL89EAnchor,
    R1AlignedTrackConditioning,
    R1Config,
    load_l89e_rule,
)
from locatemot.rmot.r1_aligned_cache import (  # noqa: E402
    R0_VISUAL_MANIFEST,
    R1AlignedCacheIndex,
    R1BankStore,
    build_causal_history,
    build_geometry,
    row_digest,
    sha256_file,
    write_json,
)
from tools.r0_common import EvalIndex, MergedLanguageCache, VisualCacheIndex  # noqa: E402
from tools.r0_infer_true_fullvideo import (  # noqa: E402
    materialize_gt,
    prepare_paths,
    safe_internal_record,
    sequence_id,
)
from tools.r0_trackeval_matrix import run_dataset as run_trackeval_dataset  # noqa: E402
from tools.r1_common import (  # noqa: E402
    DEV_ROOTS,
    FORBIDDEN_VIDEOS,
    MANIFEST,
    MANIFEST_SHA,
    SEED,
    THREAD,
    check_manifest,
    file_meta,
    standard_flags,
    unit_key,
)


L89E_SELECTION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/dev/selection_attempt1/checkpoint_selection.json"
)
L89E_ANCHOR = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch004.pt"
)
ALIGNED_CACHE = WORK_ROOT / "outputs/r1/cache/eval_attempt2"
SAFE_TARGET_ROOT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0A/outputs/r0/data/safe_targets_retry3"
)
LEGAL_DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
LEGAL_VIDEOS = {
    "refer_kitti_v1": ("0008", "0010", "0020"),
    "refer_kitti_v2": ("0000", "0008", "0009"),
}
EPOCHS = (1, 2, 4, 6)
ELIGIBLE_EPOCHS = (2, 4, 6)
RULE_NAME = "B"
RULE = {"candidate_threshold": 1.0, "presence_threshold": 0.5, "null_margin": 0.0}


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.float()).all())


def _record_digest(keys: Iterable[tuple[Any, ...]]) -> str:
    return row_digest(keys)


def _load_sidecar(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    package = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(package, dict) or package.get("format") != "locatemot-r1-aligned-track-conditioning-checkpoint-v1":
        raise AssertionError(f"invalid R1 checkpoint package: {path}")
    config_payload = package.get("model_config")
    if not isinstance(config_payload, dict):
        raise AssertionError(f"R1 checkpoint lacks model_config: {path}")
    config = R1Config(**config_payload)
    if bool(config.presence_residual):
        raise AssertionError("legal-dev replay received a checkpoint with presence residual enabled")
    model = R1AlignedTrackConditioning(config).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package.get("model_state_dict", {}), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"R1 sidecar strict reload failed: {path}: {result}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("R1 legal-dev model unexpectedly has trainable parameters")
    info = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "epoch": int(package.get("epoch", -1)),
        "optimizer_step": int(package.get("optimizer_step", -1)),
        "model_config": config_payload,
        "format": package["format"],
        "strict_reload": True,
        "model_parameter_report": model.parameter_report(),
        "anchor_checkpoint_sha256": str(package.get("anchor_checkpoint_sha256")),
    }
    if info["anchor_checkpoint_sha256"] != ANCHOR_SHA256:
        raise AssertionError(f"R1 checkpoint anchor SHA provenance drift: {path}")
    return model, info


def _language_for_video(language: MergedLanguageCache, records: list[dict[str, Any]], dataset: str, video: str) -> dict[int, Any]:
    sentences: dict[int, str] = {}
    for record in records:
        if str(record["video"]) != str(video):
            continue
        query_id = int(record["query_id"])
        sentence = str(record["sentence"])
        if query_id in sentences and sentences[query_id] != sentence:
            raise AssertionError(f"legal query sentence drift: {dataset}|{video}|{query_id}")
        sentences[query_id] = sentence
    result: dict[int, Any] = {}
    for query_id in sorted(sentences):
        item = language.get(dataset, video, query_id, sentences[query_id])
        if tuple(item.tokens.shape) != (256, 256) or tuple(item.mask.shape) != (256,) or not _finite(item.tokens):
            raise AssertionError(f"legal language cache shape/finite drift: {dataset}|{video}|{query_id}")
        result[query_id] = item
    return result


def _stack_language(items: list[Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("empty legal language batch")
    length = max(int(item.tokens.shape[0]) for item in items)
    dim = int(items[0].tokens.shape[1])
    if dim != 256:
        raise AssertionError(f"R1 language hidden dimension drift: {dim}")
    tokens = torch.zeros((len(items), length, dim), dtype=torch.float32, device=device)
    masks = torch.zeros((len(items), length), dtype=torch.bool, device=device)
    for index, item in enumerate(items):
        value = item.tokens.float()
        mask = item.mask.bool()
        tokens[index, : value.shape[0]] = value.to(device=device)
        masks[index, : mask.shape[0]] = mask.to(device=device)
    if not _finite(tokens) or not bool(masks.any(dim=1).all()):
        raise FloatingPointError("legal language batch is nonfinite/empty")
    return tokens.clone(), masks.clone()


def _prepare_frame(
    store: R1BankStore,
    visual_index: VisualCacheIndex,
    aligned_payload: dict[str, Any],
    dataset: str,
    video: str,
    frame_id: int,
) -> dict[str, Any]:
    """Construct one complete, label-free native frame representation."""
    store.load_video(video)
    batch = store.build_frame(dataset, video, -1, frame_id, "__r1_legal_dev_replay__")
    visual = visual_index.get(dataset, video, frame_id)
    offsets = [int(value) for value in batch.row_offsets]
    if offsets != list(aligned_payload["row_offsets"]):
        raise AssertionError(f"legal aligned/native row offset drift: {dataset}|{video}|{frame_id}")
    if offsets != list(visual.get("row_offsets", [])):
        raise AssertionError(f"legal visual/native row offset drift: {dataset}|{video}|{frame_id}")
    count = len(offsets)
    if int(aligned_payload["candidate_count"]) != count or int(visual["candidate_count"]) != count:
        raise AssertionError(f"legal candidate count drift: {dataset}|{video}|{frame_id}")
    candidate_indices = [int(value) for value in batch.candidate_indices]
    if candidate_indices != [int(value) for value in aligned_payload["candidate_indices"]] or \
            candidate_indices != [int(value) for value in visual["candidate_indices"]]:
        raise AssertionError(f"legal candidate-index order drift: {dataset}|{video}|{frame_id}")
    if not torch.is_tensor(aligned_payload["boxes"]) or tuple(aligned_payload["boxes"].shape) != (count, 4):
        raise AssertionError("legal aligned box shape drift")
    if not torch.allclose(batch.boxes.float(), aligned_payload["boxes"].float(), atol=1e-4, rtol=0.0):
        raise AssertionError(f"legal aligned box drift: {dataset}|{video}|{frame_id}")
    if not torch.allclose(torch.as_tensor(visual["boxes_normalized"]).float(), aligned_payload["boxes_norm"].float(), atol=2e-3, rtol=0.0):
        raise AssertionError(f"legal normalized-box drift: {dataset}|{video}|{frame_id}")
    row_keys = [
        (str(dataset), str(video), int(query_id), int(frame_id), str(batch.bank_path), int(offset))
        for query_id in [int(value) for value in aligned_payload["query_ids"][:1]]
        for offset in offsets
    ]
    # The group-level digest is tied to its first query.  It is checked here;
    # per-query keys are constructed in the scoring loop below.
    if len(row_keys) != count:
        raise AssertionError("legal row-key count drift")
    expected_raw = torch.cat((visual["inner_tokens"].float(), visual["context_tokens"].float()), dim=1)
    if tuple(expected_raw.shape) != (count, 72, 256) or not _finite(expected_raw):
        raise AssertionError(f"legal raw visual shape/finite drift: {dataset}|{video}|{frame_id}")
    if list(aligned_payload["candidate_indices"]) != candidate_indices:
        raise AssertionError("legal aligned candidate order drift")
    history, history_mask, history_frames = build_causal_history(store, batch)
    geometry = build_geometry(store, batch)
    if tuple(history.shape) != (count, 8, 1432) or tuple(history_mask.shape) != (count, 8) or \
            tuple(history_frames.shape) != (count, 8) or tuple(geometry.shape) != (count, 5, 10):
        raise AssertionError(f"legal causal feature shape drift: {dataset}|{video}|{frame_id}")
    if bool((history_frames[history_mask] > int(frame_id)).any()) or not _finite(history) or not _finite(geometry):
        raise AssertionError(f"legal causal history/geometry drift: {dataset}|{video}|{frame_id}")
    if not bool(history_mask[:, -1].all()):
        raise AssertionError(f"legal current observation missing: {dataset}|{video}|{frame_id}")
    return {
        "batch": batch,
        "visual": visual,
        "raw": expected_raw,
        "history": history,
        "history_mask": history_mask,
        "history_frames": history_frames,
        "geometry": geometry,
        "boxes_norm": aligned_payload["boxes_norm"].float().clone(),
        "row_key_digest_template": _record_digest(row_keys),
    }


def _score_group(
    anchor: FrozenL89EAnchor,
    model: torch.nn.Module,
    payload: dict[str, Any],
    frame: dict[str, Any],
    query_ids: list[int],
    language_by_query: dict[int, Any],
    device: torch.device,
    query_batch_size: int,
) -> list[dict[str, Any]]:
    raw = frame["raw"].to(device=device).float().clone()
    history = frame["history"].to(device=device).float().clone()
    history_mask = frame["history_mask"].to(device=device).clone()
    history_frames = frame["history_frames"].to(device=device).clone()
    geometry = frame["geometry"].to(device=device).float().clone()
    boxes_norm = frame["boxes_norm"].to(device=device).float().clone()
    count = int(raw.shape[0])
    if len(query_ids) != len(payload["query_ids"]):
        raise AssertionError("legal group query count drift")
    cache_positions = {int(value): index for index, value in enumerate(payload["query_ids"])}
    if set(query_ids) != set(cache_positions):
        raise AssertionError("legal aligned query set drift")
    result: list[dict[str, Any]] = []
    for start in range(0, len(query_ids), int(query_batch_size)):
        chunk = query_ids[start : start + int(query_batch_size)]
        positions = [cache_positions[qid] for qid in chunk]
        z0 = payload["z0"][positions].float().to(device=device).clone()
        z1 = payload["z1"][positions].float().to(device=device).clone()
        z4 = payload["z4"][positions].float().to(device=device).clone()
        text_items = [language_by_query[qid] for qid in chunk]
        text_tokens, text_mask = _stack_language(text_items, device)
        text_global = payload["text_global"][positions].float().to(device=device).clone()
        frame_global = payload["frame_global"][positions].float().to(device=device).clone()
        with torch.inference_mode():
            base = anchor(
                z1, text_tokens, text_mask, text_global, frame_global,
                history[:, -1], history, history_mask, history_frames, int(frame["batch"].frame_id),
            )
            raw_batch = raw.unsqueeze(0).expand(len(chunk), -1, -1, -1)
            geometry_mask = torch.ones(geometry.shape[:2], dtype=torch.bool, device=device)
            side = model(
                z0, z1, z4, text_tokens, text_mask, text_global,
                raw_batch, geometry, geometry_mask, boxes_norm,
                base["candidate_energy"], base["presence_logit"], base["null_logit"],
            )
        if tuple(side["final_energy"].shape) != (len(chunk), count):
            raise AssertionError("legal R1 candidate-energy shape drift")
        if not all(_finite(value) for value in side.values()):
            raise FloatingPointError("legal R1 sidecar nonfinite output")
        energy = side["final_energy"].float().cpu()
        presence = side["final_presence"].float().cpu()
        null = side["final_null"].float().cpu()
        for index, query_id in enumerate(chunk):
            scores = [float(value) for value in energy[index].tolist()]
            present = float(presence[index].item())
            null_value = float(null[index].item())
            if not all(math.isfinite(value) for value in scores + [present, null_value]):
                raise FloatingPointError("legal R1 score nonfinite")
            selected = [
                local for local, value in enumerate(scores)
                if value >= RULE["candidate_threshold"] and
                present >= RULE["presence_threshold"] and
                value - null_value >= RULE["null_margin"]
            ]
            keys = [
                [str(frame["batch"].dataset), str(frame["batch"].video), int(query_id), int(frame["batch"].frame_id),
                 str(frame["batch"].bank_path), int(offset)]
                for offset in frame["batch"].row_offsets
            ]
            if len(keys) != count or len({tuple(value) for value in keys}) != count:
                raise AssertionError("legal R1 row-key completeness drift")
            result.append({
                "dataset": str(frame["batch"].dataset), "video": str(frame["batch"].video),
                "query_id": int(query_id), "frame_id": int(frame["batch"].frame_id),
                "unit_key": f"{frame['batch'].dataset}|{frame['batch'].video}|{int(query_id)}|{int(frame['batch'].frame_id)}",
                "candidate_count": count, "candidate_rows_scored": count,
                "selected_rows": len(selected), "scores": scores, "presence_logit": present,
                "null_logit": null_value, "selected_indices": selected, "row_keys": keys,
                "row_key_digest": _record_digest(tuple(tuple(key) for key in keys)),
                "candidate_rows_retained": True, "candidate_deletion": False,
                "candidate_truncation": False, "labels_used_for_prediction": False,
            })
        del base, side, raw_batch, z0, z1, z4, text_tokens, text_mask, text_global, frame_global
    del raw, history, history_mask, history_frames, geometry, boxes_norm
    return result


def _prediction_line(frame_id: int, track_id: int, box: Iterable[float], score: float) -> str:
    x1, y1, x2, y2 = [float(value) for value in box]
    confidence = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(score)))))
    return f"{int(frame_id) + 1},{int(track_id)},{x1:.6f},{y1:.6f},{x2 - x1:.6f},{y2 - y1:.6f},{confidence:.8f},1,1,1\n"


def _write_prediction_files(
    records: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    handles: dict[int, Any],
    boxes: list[list[float]],
    track_ids: list[int],
) -> None:
    if len(scores) != len(records):
        raise AssertionError("legal score/record count drift")
    if len(boxes) != len(track_ids):
        raise AssertionError("legal box/track count drift")
    if len(track_ids) != len(set(track_ids)):
        raise AssertionError("duplicate native track IDs would make TrackEval identity ambiguous")
    for record, scored in zip(records, scores):
        if str(record["unit_key"]) != str(scored["unit_key"]):
            raise AssertionError("legal score unit order drift")
        for local in scored["selected_indices"]:
            handles[int(record["query_id"])].write(
                _prediction_line(int(record["frame_id"]), track_ids[int(local)], boxes[int(local)], float(scored["scores"][int(local)]))
            )


def _load_group_records(index: EvalIndex) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in index.records:
        if str(record["video"]) not in LEGAL_VIDEOS[str(record["dataset"])] or str(record["video"]) in FORBIDDEN_VIDEOS:
            raise AssertionError(f"unexpected legal-dev video: {record['unit_key']}")
        grouped[(str(record["video"]), int(record["frame_id"]))].append(record)
    for key, values in grouped.items():
        values.sort(key=lambda value: int(value["query_id"]))
        if len({unit_key(value) for value in values}) != len(values):
            raise AssertionError(f"duplicate legal-dev unit: {key}")
    return dict(sorted(grouped.items()))


def _init_prediction_layout(large_root: Path, dataset: str, query_ids_by_video: dict[str, list[int]]) -> dict[int, dict[int, Path]]:
    paths: dict[int, dict[int, Path]] = {}
    for epoch in EPOCHS:
        root = large_root / f"candidate_epoch{epoch:03d}_zero"
        dataset_root = root / dataset
        (dataset_root / "trackers" / "r0" / "data").mkdir(parents=True, exist_ok=True)
        paths[epoch] = {}
        for video, query_ids in query_ids_by_video.items():
            for query_id in query_ids:
                path = dataset_root / "trackers" / "r0" / "data" / f"{sequence_id(video, query_id)}.txt"
                if path.exists():
                    raise FileExistsError(f"prediction collision: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
                paths[epoch][(video, query_id)] = path
    return paths


def _legal_descriptor(
    dataset: str,
    epoch_root: Path,
    queries_by_video: dict[str, list[dict[str, Any]]],
    frame_map: dict[str, list[int]],
    safe_root: Path,
) -> dict[str, Any]:
    """Compute legal-dev descriptors only after all predictions exist."""
    from locatemot.rmot.r0_safe_target_source import load_safe_query_records

    safe_records = load_safe_query_records(safe_root.resolve())
    safe_by_key = {(str(value.dataset), str(value.video), int(value.query_id)): value for value in safe_records}
    inactive_units = inactive_accept = target_total = target_hit = missing_gt = 0
    query_count = 0
    selected_rows = 0
    for video in sorted(queries_by_video):
        record_path, raw_record = safe_internal_record(video)
        frame_records = {int(value["frame"]): value for value in raw_record["frames"]}
        for query in queries_by_video[video]:
            query_id = int(query["query_id"])
            safe = safe_by_key.get((dataset, video, query_id))
            if safe is None or str(safe.sentence) != str(query["sentence"]):
                raise AssertionError(f"legal descriptor safe query mismatch: {dataset}|{video}|{query_id}")
            path = epoch_root / dataset / "trackers" / "r0" / "data" / f"{sequence_id(video, query_id)}.txt"
            predictions: dict[int, list[list[float]]] = defaultdict(list)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                values = line.split(",")
                if len(values) < 7:
                    raise AssertionError(f"malformed legal prediction line: {path}")
                frame = int(values[0]) - 1
                x, y, w, h = [float(values[index]) for index in (2, 3, 4, 5)]
                predictions[frame].append([x, y, x + w, y + h])
                selected_rows += 1
            query_count += 1
            for frame in frame_map[video]:
                boxes = predictions.get(int(frame), [])
                targets = tuple(str(value) for value in safe.target.get(int(frame), ()))
                if not targets:
                    inactive_units += 1
                    inactive_accept += int(bool(boxes))
                    continue
                gt_boxes = frame_records[int(frame)].get("gt_boxes", {})
                for target in sorted(set(targets)):
                    value = gt_boxes.get(target)
                    if value is None and target.isdigit():
                        value = gt_boxes.get(int(target))
                    if value is None:
                        missing_gt += 1
                        continue
                    gt = [float(item) for item in value]
                    if len(gt) != 4 or gt[2] <= gt[0] or gt[3] <= gt[1]:
                        raise AssertionError(f"invalid legal descriptor GT box: {dataset}|{video}|{query_id}|{frame}")
                    target_total += 1
                    hit = False
                    for pred in boxes:
                        ix1, iy1 = max(gt[0], pred[0]), max(gt[1], pred[1])
                        ix2, iy2 = min(gt[2], pred[2]), min(gt[3], pred[3])
                        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        union = (gt[2] - gt[0]) * (gt[3] - gt[1]) + (pred[2] - pred[0]) * (pred[3] - pred[1]) - intersection
                        if union > 0.0 and intersection / union >= 0.5:
                            hit = True
                            break
                    target_hit += int(hit)
    return {
        "dataset": dataset, "query_count": query_count,
        "target_frame_total": target_total, "target_frame_hit": target_hit,
        "distinct_target_recall": float(target_hit / max(1, target_total)),
        "missing_gt_box_count": missing_gt,
        "inactive_units": inactive_units, "inactive_accept_units": inactive_accept,
        "inactive_false_acceptance": float(inactive_accept / max(1, inactive_units)),
        "selected_rows": selected_rows,
        "descriptor_evidence": "legal safe target records opened after prediction completion; IoU>=0.5 target-frame hit",
    }


def _run_dataset_replay(
    dataset: str,
    out: Path,
    large_root: Path,
    aligned: R1AlignedCacheIndex,
    visual_index: VisualCacheIndex,
    language: MergedLanguageCache,
    device: torch.device,
    query_batch_size: int,
    max_groups: int,
    video_filter: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    index = EvalIndex(DEV_ROOTS[dataset])
    grouped = _load_group_records(index)
    if video_filter is not None:
        allowed_videos = {str(value) for value in video_filter}
        grouped = {key: value for key, value in grouped.items() if str(key[0]) in allowed_videos}
    if not grouped:
        raise AssertionError(f"empty legal-dev index: {dataset}")
    grouped_by_video: dict[str, list[tuple[int, list[dict[str, Any]]]]] = defaultdict(list)
    for (video, frame_id), records in grouped.items():
        grouped_by_video[video].append((frame_id, records))
    for video in grouped_by_video:
        grouped_by_video[video].sort(key=lambda value: value[0])
    query_ids_by_video = {video: sorted({int(value["query_id"]) for _frame, rows in values for value in rows}) for video, values in grouped_by_video.items()}
    prediction_paths = _init_prediction_layout(large_root, dataset, query_ids_by_video)
    checkpoints = {
        epoch: WORK_ROOT / "outputs/r1/train/formal_fit_attempt1" / f"checkpoint_r1_epoch{epoch:03d}_{dataset}.pt"
        for epoch in EPOCHS
    }
    models: dict[int, torch.nn.Module] = {}
    model_infos: dict[int, dict[str, Any]] = {}
    for epoch in EPOCHS:
        if not checkpoints[epoch].is_file():
            raise FileNotFoundError(checkpoints[epoch])
        models[epoch], model_infos[epoch] = _load_sidecar(checkpoints[epoch], device)
    store = R1BankStore()
    video_audits: list[dict[str, Any]] = []
    total_groups = sum(len(value) for value in grouped_by_video.values())
    all_group_keys = [(video, frame_id) for video in sorted(grouped_by_video)
                      for frame_id, _records in grouped_by_video[video]]
    allowed_group_keys = set(all_group_keys if max_groups <= 0 else all_group_keys[:max_groups])
    visited_group_keys: set[tuple[str, int]] = set()
    language_by_video = {
        video: _language_for_video(language, index.records, dataset, video)
        for video in sorted(grouped_by_video)
    }
    try:
        for video in sorted(grouped_by_video):
            store.load_video(video)
            video_started = time.perf_counter()
            video_groups = 0
            for epoch in EPOCHS:
                model = models[epoch]
                audit_path = large_root / f"candidate_epoch{epoch:03d}_zero" / dataset / "prediction_audits.jsonl"
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as audit_handle:
                    for frame_id, records in grouped_by_video[video]:
                        group_key = (video, frame_id)
                        if group_key not in allowed_group_keys:
                            break
                        payload = aligned.read_group(dataset, video, frame_id)
                        expected_qids = [int(value["query_id"]) for value in records]
                        if [int(value) for value in payload["query_ids"]] != expected_qids:
                            raise AssertionError(f"legal aligned query order drift: {dataset}|{video}|{frame_id}")
                        if [str(value) for value in payload["query_unit_keys"]] != [unit_key(value) for value in records]:
                            raise AssertionError(f"legal aligned unit-key drift: {dataset}|{video}|{frame_id}")
                        frame = _prepare_frame(store, visual_index, payload, dataset, video, frame_id)
                        scored = _score_group(anchor=ANCHOR_MODEL, model=model, payload=payload, frame=frame,
                                              query_ids=expected_qids, language_by_query=language_by_video[video],
                                              device=device, query_batch_size=query_batch_size)
                        handles: dict[int, Any] = {}
                        try:
                            for record in records:
                                handles[int(record["query_id"])] = prediction_paths[epoch][(video, int(record["query_id"]))].open("a", encoding="utf-8")
                            boxes = [[float(value) for value in row] for row in frame["batch"].boxes.float().tolist()]
                            track_ids = [int(value) for value in frame["batch"].track_ids]
                            _write_prediction_files(records, scored, handles, boxes, track_ids)
                        finally:
                            for handle in handles.values():
                                handle.close()
                        for item in scored:
                            audit_handle.write(json.dumps({
                                key: item[key] for key in (
                                    "dataset", "video", "query_id", "frame_id", "unit_key", "candidate_count",
                                    "candidate_rows_scored", "selected_rows", "presence_logit", "null_logit",
                                    "row_key_digest", "candidate_rows_retained", "candidate_deletion", "candidate_truncation",
                                    "labels_used_for_prediction",
                                )
                            }, ensure_ascii=False, sort_keys=True) + "\n")
                        del payload, frame, scored
                        visited_group_keys.add(group_key)
                        if epoch == EPOCHS[-1]:
                            video_groups += 1
                        if max_groups > 0 and len(visited_group_keys) >= max_groups and epoch == EPOCHS[-1]:
                            break
                    if max_groups > 0 and len(visited_group_keys) >= max_groups and epoch == EPOCHS[-1]:
                        break
                if max_groups > 0 and len(visited_group_keys) >= max_groups and epoch == EPOCHS[-1]:
                    break
            video_audits.append({"dataset": dataset, "video": video, "groups_expected": len(grouped_by_video[video]),
                                 "groups_visited": video_groups, "query_count": len(query_ids_by_video[video]),
                                 "elapsed_seconds": time.perf_counter() - video_started,
                                 "candidate_deletion": False, "candidate_truncation": False,
                                 "labels_used_for_prediction": False})
            if max_groups > 0 and len(visited_group_keys) >= max_groups:
                break
    finally:
        store._blob = None
        gc.collect()
        for model in models.values():
            del model
        models.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    visited_groups = len(visited_group_keys) if max_groups > 0 else total_groups
    if max_groups <= 0 and visited_groups != total_groups:
        raise AssertionError(f"legal group count incomplete: {dataset} {visited_groups}/{total_groups}")
    return {
        "dataset": dataset, "status": "targeted_complete" if max_groups > 0 else "prediction_complete",
        "legal_video_list": sorted(grouped_by_video), "index_record_count": len(index.records),
        "group_count_expected": total_groups, "group_count_visited": visited_groups,
        "query_count": len(index.query_by_key), "video_audits": video_audits,
        "prediction_roots": {str(epoch): str((large_root / f"candidate_epoch{epoch:03d}_zero").resolve()) for epoch in EPOCHS},
        "model_infos": model_infos, "candidate_deletion": False, "candidate_truncation": False,
    }


# Set only while the script owns one frozen anchor.  Keeping the argument list
# of _run_dataset_replay compact makes it harder to accidentally instantiate a
# second detector/anchor for each video.
ANCHOR_MODEL: FrozenL89EAnchor


def _materialize_and_trackeval(
    dataset: str,
    large_root: Path,
    query_records: list[dict[str, Any]],
    frame_map: dict[str, list[int]],
    trackeval_root: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    queries_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in query_records:
        queries_by_video[str(record["video"])].append({
            "dataset": dataset, "video": str(record["video"]), "query_id": int(record["query_id"]),
            "sentence": str(record["sentence"]),
        })
    for video in queries_by_video:
        unique = {int(value["query_id"]): value for value in queries_by_video[video]}
        queries_by_video[video] = [unique[key] for key in sorted(unique)]
    results: list[dict[str, Any]] = []
    descriptors: dict[int, dict[str, Any]] = {}
    safe_root = SAFE_TARGET_ROOT / ("v1_dev" if dataset == "refer_kitti_v1" else "v2_dev")
    for epoch in EPOCHS:
        epoch_root = large_root / f"candidate_epoch{epoch:03d}_zero"
        paths = prepare_paths(epoch_root, dataset)
        gt_audit = materialize_gt(dataset, queries_by_video, safe_root, paths, frame_map)
        descriptor = _legal_descriptor(dataset, epoch_root, queries_by_video, frame_map, safe_root)
        descriptors[epoch] = descriptor
        trackeval_destination = trackeval_root / f"candidate_epoch{epoch:03d}" / dataset
        trackeval = run_trackeval_dataset(epoch_root / dataset, trackeval_destination, dataset, tracker_name="r0")
        results.append({
            "dataset": dataset, "epoch": epoch, "rule": RULE_NAME,
            "checkpoint": str((WORK_ROOT / "outputs/r1/train/formal_fit_attempt1" / f"checkpoint_r1_epoch{epoch:03d}_{dataset}.pt").resolve()),
            "checkpoint_sha256": sha256_file(WORK_ROOT / "outputs/r1/train/formal_fit_attempt1" / f"checkpoint_r1_epoch{epoch:03d}_{dataset}.pt"),
            "gt_audit": gt_audit, "descriptor": descriptor, "trackeval": trackeval,
            "prediction_before_gt": True, "labels_used_for_prediction": False,
        })
    return results, descriptors


def _selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    raw = result["trackeval"]["metrics_raw"]
    descriptor = result["descriptor"]
    return (
        float(raw["HOTA___AUC"]), float(raw["DetA___AUC"]), float(raw["AssA___AUC"]),
        float(descriptor["distinct_target_recall"]), -float(descriptor["inactive_false_acceptance"]),
        -int(result["epoch"]),
    )


def run(args: argparse.Namespace) -> int:
    global ANCHOR_MODEL
    selected_datasets = [args.dataset] if args.dataset else list(LEGAL_DATASETS)
    if args.video is not None:
        if args.dataset is None:
            raise ValueError("--video requires --dataset")
        if args.video not in LEGAL_VIDEOS[args.dataset]:
            raise ValueError(f"video {args.video} is outside the legal scope for {args.dataset}")
    selected_video_filter = (args.video,) if args.video is not None else None
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    large_root = args.large_root.resolve()
    trackeval_root = args.trackeval_root.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 legal-dev output: {out}")
    if large_root.exists() and any(large_root.iterdir()):
        raise FileExistsError(f"refusing nonempty large prediction root: {large_root}")
    if trackeval_root.exists() and any(trackeval_root.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 TrackEval root: {trackeval_root}")
    out.mkdir(parents=True, exist_ok=True)
    large_root.mkdir(parents=True, exist_ok=True)
    trackeval_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([str(sys.executable), *sys.argv])
    base = {
        "format": "locatemot-r1-legal-dev-replay-v1", "status": "incomplete", "command": command,
        "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
        "manifest_sha256": None, "anchor": {"path": str(L89E_ANCHOR), "sha256": None},
        "rule": {"name": RULE_NAME, **RULE}, "scope": "legal_dev_only",
        "legal_video_scope": {key: list(value) for key, value in LEGAL_VIDEOS.items()},
        "aligned_cache": str(args.aligned_cache.resolve()), "visual_cache": str(R0_VISUAL_MANIFEST),
        "large_prediction_root": str(large_root), "trackeval_root": str(trackeval_root),
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "failure_root_cause": None, "next_action": "select legal-dev checkpoint, then run fixed semantic diagnostic",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 legal-dev cwd: {Path.cwd()}")
        if args.max_groups < 0:
            raise ValueError("max-groups cannot be negative")
        manifest_sha = check_manifest()
        base["manifest_sha256"] = manifest_sha
        if sha256_file(L89E_ANCHOR) != ANCHOR_SHA256:
            raise AssertionError("R1 anchor SHA drift")
        base["anchor"]["sha256"] = ANCHOR_SHA256
        rule = load_l89e_rule(L89E_SELECTION)
        if rule["rule"] != RULE_NAME or any(float(rule[key]) != float(RULE[key]) for key in RULE):
            raise AssertionError(f"R1 Rule-B drift: {rule}")
        aligned_cache = args.aligned_cache.resolve()
        if not aligned_cache.is_dir():
            raise FileNotFoundError(aligned_cache)
        aligned = R1AlignedCacheIndex(aligned_cache)
        visual_index = VisualCacheIndex(R0_VISUAL_MANIFEST)
        language = MergedLanguageCache(tuple(Path(value).resolve() for value in (
            "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1",
            "/data2/usr_for_deadline/locatemot_r0a_language_tokens_retry1",
        )))
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("R1 legal-dev requested CUDA but CUDA is unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        ANCHOR_MODEL = FrozenL89EAnchor(L89E_ANCHOR).to(device)
        ANCHOR_MODEL.eval()
        if any(parameter.requires_grad for parameter in ANCHOR_MODEL.parameters()):
            raise AssertionError("R1 legal-dev anchor is not frozen")
        indexes = {dataset: EvalIndex(DEV_ROOTS[dataset]) for dataset in selected_datasets}
        prediction_summaries: list[dict[str, Any]] = []
        for dataset in selected_datasets:
            prediction_summaries.append(_run_dataset_replay(
                dataset, out, large_root, aligned, visual_index, language, device,
                int(args.query_batch_size), int(args.max_groups), selected_video_filter,
            ))
            if args.max_groups > 0:
                break
        if args.max_groups > 0 or args.prediction_only:
            payload = {
                **base, "status": "targeted_complete" if args.max_groups > 0 else "prediction_complete",
                "targeted_regression": bool(args.max_groups > 0),
                "prediction_only": bool(args.prediction_only),
                "selected_scope": {"datasets": selected_datasets, "video": args.video},
                "prediction_summaries": prediction_summaries,
                "labels_attached_after_predictions": False, "hota_trackeval_run": False,
                "no_hota_or_trackeval": True, "wall_seconds": time.perf_counter() - started,
                "next_action": "aggregate complete per-video predictions, then attach legal-dev labels and run TrackEval" if args.prediction_only else "run the complete legal-dev replay in a fresh attempt",
                "inputs": {"manifest": file_meta(MANIFEST), "anchor": file_meta(L89E_ANCHOR),
                           "selection": file_meta(L89E_SELECTION), "aligned_cache": file_meta(aligned_cache / "status.json"),
                           "visual_manifest": file_meta(R0_VISUAL_MANIFEST / "manifest.jsonl")},
            }
            write_json(out / "provenance.json", payload)
            write_json(out / "status.json", {"format": base["format"], "status": payload["status"],
                                              "command": command, "output": str(out), "failure_root_cause": None,
                                              "next_action": payload["next_action"], "screening_gt_used": False,
                                              "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False})
            return 0
        all_trackeval: list[dict[str, Any]] = []
        selections: dict[str, Any] = {}
        for dataset in selected_datasets:
            index = indexes[dataset]
            grouped = _load_group_records(index)
            frame_map = {video: sorted({int(frame) for (found_video, frame) in grouped if found_video == video})
                         for video in LEGAL_VIDEOS[dataset] if any(found_video == video for found_video, _frame in grouped)}
            results, _descriptors = _materialize_and_trackeval(
                dataset, large_root, index.records, frame_map, trackeval_root,
            )
            all_trackeval.extend(results)
            eligible = [item for item in results if int(item["epoch"]) in ELIGIBLE_EPOCHS]
            if len(eligible) != len(ELIGIBLE_EPOCHS):
                raise AssertionError(f"R1 legal-dev eligible epoch count drift: {dataset}")
            chosen = max(eligible, key=_selection_key)
            selections[dataset] = {
                "selection_key": list(_selection_key(chosen)),
                "selection_tuple": "(HOTA, DetA, AssA, distinct_target_recall, -inactive_false_acceptance, -epoch)",
                "selected_epoch": int(chosen["epoch"]),
                "selected_checkpoint": chosen["checkpoint"],
                "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
                "rule": RULE_NAME, "eligible_epochs": list(ELIGIBLE_EPOCHS),
                "legal_dev_only": True,
            }
        payload = {
            **base, "status": "complete", "targeted_regression": False,
            "prediction_summaries": prediction_summaries, "trackeval_results": all_trackeval,
            "selection": selections, "labels_attached_after_predictions": True,
            "hota_trackeval_run": True, "no_hota_or_trackeval": False,
            "anchor_parameter_count": sum(int(value.numel()) for value in ANCHOR_MODEL.parameters()),
            "aligned_cache_group_count": len(aligned.group_paths), "visual_cache_entry_count": len(visual_index.entries),
            "language_cache_entry_count": language.entry_count,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started,
            "inputs": {"manifest": file_meta(MANIFEST), "anchor": file_meta(L89E_ANCHOR),
                       "selection": file_meta(L89E_SELECTION), "aligned_cache": file_meta(aligned_cache / "status.json"),
                       "visual_manifest": file_meta(R0_VISUAL_MANIFEST / "manifest.jsonl"),
                       "dev_indexes": {dataset: file_meta(DEV_ROOTS[dataset] / "summary.json") for dataset in selected_datasets},
                       "safe_targets": {dataset: file_meta(SAFE_TARGET_ROOT / ("v1_dev" if dataset == "refer_kitti_v1" else "v2_dev") / "manifest.json") for dataset in selected_datasets}},
            "next_action": "freeze the separately selected V1/V2 epochs and run fixed 16-calibration/24-validation semantic diagnostics",
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-r1-legal-dev-provenance-v1"})
        write_json(out / "status.json", {"format": base["format"], "status": "complete", "output": str(out),
                                          "selection": selections, "trackeval_result_count": len(all_trackeval),
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
                                          "candidate_deletion": False, "candidate_truncation": False,
                                          "failure_root_cause": None, "next_action": payload["next_action"]})
        return 0
    except BaseException as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R1 legal-dev replay — INCOMPLETE\n\n```text\n" + trace + "```\n", encoding="utf-8")
        write_json(out / "provenance.json", {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                             "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                                             "wall_seconds": time.perf_counter() - started})
        write_json(out / "status.json", {"format": base["format"], "status": "incomplete", "command": command,
                                          "output": str(out), "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                          "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "next_action": "repair only the first legal-dev replay contract error in a new attempt"})
        return 2
    finally:
        if 'ANCHOR_MODEL' in globals() and ANCHOR_MODEL is not None:
            del ANCHOR_MODEL
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-cache", type=Path, default=ALIGNED_CACHE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--large-root", type=Path, required=True)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=LEGAL_DATASETS, default=None,
                        help="optional single-dataset targeted replay; default runs both legal datasets")
    parser.add_argument("--video", default=None,
                        help="optional one legal-dev video; use with --dataset and --prediction-only for recoverable replay")
    parser.add_argument("--prediction-only", action="store_true",
                        help="write complete predictions without opening legal-dev GT or running TrackEval")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--max-groups", type=int, default=0,
                        help="targeted regression limit; zero means all legal groups")
    args = parser.parse_args()
    if args.query_batch_size <= 0:
        raise ValueError("query-batch-size must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
