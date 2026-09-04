#!/usr/bin/env python3
"""Score the frozen L88 shortlist on legal full-video internal scopes.

The dev pass is used only for the preregistered checkpoint/rule selection.  A
later ``--scope internal`` pass is used for the fixed internal V1/V2
TrackEval evidence.  Both scopes score every native L69 row.  The
query-independent L88 encoder cache is read when present; a missing frame is
computed transiently in RAM and is never serialized as a new cache.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from l88_eval_common import (
    ASSET_ROOT, L88_CACHE, L85_CACHE, MANIFEST, MANIFEST_SHA, SEED, THREAD,
    EncoderCacheReader, L88ClipStore, load_checkpoint_into, make_runtime,
    sha256, write_json,
)


WORK_ROOT = Path(__file__).resolve().parents[1]
SPLIT = ASSET_ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
FIT_UNITS = ASSET_ROOT / "outputs/l49/data/train_units.jsonl"
INTERNAL_DATASETS = {
    "refer_kitti_v1": ("0004", "0018"),
    "refer_kitti_v2": ("0016", "0017", "0020"),
}


def _asset_rmot_path() -> None:
    import locatemot.rmot as rmot_package
    asset_path = str(ASSET_ROOT / "locatemot" / "rmot")
    if asset_path not in [str(value) for value in rmot_package.__path__]:
        rmot_package.__path__.append(asset_path)


_asset_rmot_path()
from locatemot.rmot.l49_data import load_l49_queries  # noqa: E402
from locatemot.rmot.l85_runtime import load_validation_key_rows  # noqa: E402
from locatemot.rmot.l80_data import load_fixed_key_units  # noqa: E402


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key_only(row: dict[str, Any], frame_id: int | None = None) -> dict[str, Any]:
    sentence = str(row.get("sentence") or row.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty L88 full-video sentence: {row.get('unit_key')}")
    frame = int(row["frame_id"] if frame_id is None else frame_id)
    return {
        "unit_key": f"{row['dataset']}|{row['video']}|{int(row['query_id'])}|{frame}",
        "dataset": str(row["dataset"]), "video": str(row["video"]),
        "query_id": int(row["query_id"]), "frame_id": frame,
        "sentence": sentence, "expression": sentence,
    }


def unique_query_rows(rows: list[dict[str, Any]], dataset: str, video: str) -> list[dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("dataset")) != str(dataset) or str(row.get("video")) != str(video):
            continue
        qid = int(row["query_id"])
        sentence = str(row.get("sentence") or row.get("expression") or "")
        if qid in found and found[qid]["sentence"] != sentence:
            raise AssertionError(f"full-video query sentence drift: {dataset}|{video}|{qid}")
        found.setdefault(qid, {"dataset": dataset, "video": video, "query_id": qid,
                               "sentence": sentence, "expression": sentence})
    if not found:
        raise AssertionError(f"no legal query rows for {dataset}|{video}")
    return [found[qid] for qid in sorted(found)]


def scope_queries(store: L88ClipStore, scope: str, dataset: str, video: str,
                  split: dict[str, Any]) -> list[dict[str, Any]]:
    if scope == "dev":
        rows = list(store._base.labels_by_key.values())
        return unique_query_rows(rows, dataset, video)
    if scope == "internal":
        return unique_query_rows(load_validation_key_rows(), dataset, video)
    raise ValueError(scope)


def scope_videos(scope: str, datasets: list[str], split: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    if scope == "internal":
        return {dataset: tuple(video for video in INTERNAL_DATASETS[dataset]) for dataset in datasets}
    if scope == "dev":
        result: dict[str, list[str]] = defaultdict(list)
        for value in split.get("dev_videos", []):
            dataset, video = str(value).split("|", 1)
            if dataset in datasets:
                result[dataset].append(video)
        if not result:
            raise AssertionError("L82 dev video split is empty")
        return {dataset: tuple(sorted(set(values))) for dataset, values in sorted(result.items())}
    raise ValueError(scope)


def frame_ids(store: L88ClipStore, video: str) -> list[int]:
    store.bank_store._store.load_video(str(video))
    values = [int(value) for value in store.bank_store._store.tensors["frame_ids"].tolist()]
    if values != sorted(values) or len(values) != len(set(values)):
        raise AssertionError(f"L69 frame order drift: {video}")
    return values


def make_frame(store: L88ClipStore, dataset: str, video: str, frame_id: int,
               query: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    row = key_only(query, frame_id)
    batch = store.bank_store.build_unit(row)
    if batch.dataset != dataset or batch.video != video or int(batch.frame_id) != int(frame_id):
        raise AssertionError(f"full-video bank identity drift: {row['unit_key']}")
    if batch.candidate_count != len(batch.row_offsets) or len(batch.row_keys) != batch.candidate_count:
        raise AssertionError(f"full-video candidate count/key drift: {row['unit_key']}")
    if [int(value[-1]) for value in batch.row_keys] != batch.row_offsets:
        raise AssertionError(f"full-video native row order drift: {row['unit_key']}")
    if int((batch.history_frame_ids > int(frame_id)).sum()) != 0:
        raise AssertionError(f"full-video future history: {row['unit_key']}")
    queries = [key_only(query, frame_id)]
    return batch, queries


def valid_mean(value: torch.Tensor, mask: torch.Tensor | None, name: str) -> torch.Tensor:
    if value.ndim != 3:
        raise AssertionError(f"{name} rank drift: {tuple(value.shape)}")
    if mask is None:
        valid = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
    else:
        valid = mask.bool()
        if valid.shape != value.shape[:2]:
            raise AssertionError(f"{name} mask shape drift: {tuple(value.shape)} / {tuple(valid.shape)}")
    weights = valid.to(value.dtype).unsqueeze(-1)
    result = (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if not bool(torch.isfinite(result.float()).all()):
        raise FloatingPointError(f"nonfinite {name}")
    return result


def cache_for_frame(reader: EncoderCacheReader, runtime: Any, first: Any,
                    device: torch.device) -> tuple[dict[str, Any], str]:
    cache_key = f"{first.dataset}|{first.video}|{int(first.frame_id):06d}"
    if cache_key in reader.entries:
        return reader.read(cache_key, device), "persistent_query_independent_cache_read"
    # The L88 cache intentionally contains all internal frames but not every
    # video-disjoint dev frame.  Missing dev frames are computed transiently;
    # no new feature/cache file is written.
    with torch.inference_mode():
        item = runtime.cache_frame(Path(first.image_path))
    item.update({"cache_key": cache_key, "dataset": str(first.dataset),
                 "video": str(first.video), "frame": int(first.frame_id)})
    item["query_independent"] = True
    item["labels_in_cache"] = False
    item["candidate_deletion"] = False
    item["candidate_truncation"] = False
    return item, "transient_query_independent_frame_rebuild"


def encode_query(runtime: Any, reader: EncoderCacheReader, item: dict[str, Any],
                 first: Any, sentence: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        replay = __import__("locatemot.rmot.l88_grounding_runtime", fromlist=["forward_l88_z1"]).forward_l88_z1(
            runtime.model, item, first.boxes, [str(sentence)], device,
            query_tile=1, autocast_bf16=False)
    if bool(replay.get("candidate_deletion")) or bool(replay.get("candidate_truncation")):
        raise AssertionError(f"L88 full-video replay deleted candidates: {first.unit_key}")
    z1 = replay["z1"].float()
    text_global = valid_mean(replay["memory_text"].float(), replay.get("text_token_mask"), "text_global")
    memory_mask = replay.get("memory_mask")
    frame_global = valid_mean(replay["memory"].float(), None if memory_mask is None else ~memory_mask.bool(), "frame_global")
    expected = (1, int(first.candidate_count), 256)
    if tuple(z1.shape) != expected or tuple(text_global.shape) != (1, 256) or tuple(frame_global.shape) != (1, 256):
        raise AssertionError(f"full-video replay shape drift: {first.unit_key}: {tuple(z1.shape)}")
    if not all(bool(torch.isfinite(value.float()).all()) for value in (z1, text_global, frame_global)):
        raise FloatingPointError(f"full-video replay nonfinite: {first.unit_key}")
    del replay
    return z1, text_global, frame_global


def score_frame_queries(runtime: Any, reader: EncoderCacheReader, sidecar: Any,
                        store: L88ClipStore, first: Any, queries: list[dict[str, Any]],
                        device: torch.device, query_tile: int) -> list[dict[str, Any]]:
    item, _source = cache_for_frame(reader, runtime, first, device)
    z1_parts: list[torch.Tensor] = []
    text_parts: list[torch.Tensor] = []
    frame_parts: list[torch.Tensor] = []
    try:
        for query in queries:
            z1, text_global, frame_global = encode_query(runtime, reader, item, first,
                                                          str(query["sentence"]), device)
            z1_parts.append(z1); text_parts.append(text_global); frame_parts.append(frame_global)
        records: list[dict[str, Any]] = []
        current = first.observations.float().to(device)
        history = first.history_observations.float().to(device)
        history_mask = first.history_mask.clone().to(device)
        history_frames = first.history_frame_ids.clone().to(device)
        for start in range(0, len(queries), max(1, int(query_tile))):
            stop = min(len(queries), start + max(1, int(query_tile)))
            with torch.inference_mode():
                output = sidecar(
                    torch.cat(z1_parts[start:stop], dim=0),
                    torch.cat(text_parts[start:stop], dim=0),
                    torch.cat(frame_parts[start:stop], dim=0),
                    current.unsqueeze(0).expand(stop - start, -1, -1),
                    history.unsqueeze(0).expand(stop - start, -1, -1, -1),
                    history_mask.unsqueeze(0).expand(stop - start, -1, -1),
                    history_frames.unsqueeze(0).expand(stop - start, -1, -1),
                    int(first.frame_id), temporal_enabled=True,
                )
            scores = output["candidate_energy"].float().detach().cpu().numpy()
            presence = output["presence_logit"].float().detach().cpu().numpy()
            null = output["null_logit"].float().detach().cpu().numpy()
            if not (np.isfinite(scores).all() and np.isfinite(presence).all() and np.isfinite(null).all()):
                raise FloatingPointError(f"nonfinite full-video sidecar output: {first.unit_key}")
            for local, query in enumerate(queries[start:stop]):
                values = scores[local]
                qid = int(query["query_id"])
                row_keys = [(str(first.dataset), str(first.video), qid, int(first.frame_id),
                             str(first.bank_path), int(offset)) for offset in first.row_offsets]
                if values.shape != (first.candidate_count,) or row_keys != [
                        (str(key[0]), str(key[1]), qid, int(key[3]), str(key[4]), int(key[5]))
                        for key in first.row_keys]:
                    raise AssertionError(f"full-video score/key drift: {query['unit_key']}")
                records.append({
                    "format": "locatemot-l88-fullvideo-score-v1",
                    "unit_key": f"{first.dataset}|{first.video}|{qid}|{int(first.frame_id)}",
                    "group_key": f"{first.dataset}|{first.video}|{int(first.frame_id)}",
                    "dataset": str(first.dataset), "video": str(first.video),
                    "query_id": qid, "frame_id": int(first.frame_id),
                    "candidate_count": int(first.candidate_count),
                    "row_offsets": [int(value) for value in first.row_offsets],
                    "row_keys": [list(value) for value in row_keys],
                    "candidate_indices": [int(value) for value in first.candidate_indices],
                    "track_ids": [int(value) for value in first.track_ids],
                    "pool_ids": [int(value) for value in first.pool_ids],
                    "score": values.astype(np.float64).tolist(),
                    "presence_logit": float(presence[local]), "null_logit": float(null[local]),
                    "future_history_count": int((first.history_frame_ids > int(first.frame_id)).sum()),
                    "candidate_rows_retained": True, "candidate_deletion": False,
                    "candidate_truncation": False, "finite_scores": True,
                    "labels_attached": False,
                })
            del output, scores, presence, null
        return records
    finally:
        del item, z1_parts, text_parts, frame_parts
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def strategy_root(out: Path, candidate: dict[str, Any], rule_name: str) -> Path:
    epoch = int(candidate["checkpoint_info"]["epoch"])
    return out / f"candidate_epoch{epoch:03d}" / str(rule_name)


def prepare_strategy(root: Path, datasets: list[str], video_queries: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for dataset in datasets:
        root_ds = root / dataset
        tracker = root_ds / "trackers" / "l88" / "data"
        paths[dataset] = {
            "root": root_ds, "gt": root_ds / "gt", "tracker_data": tracker,
            "seqmap": root_ds / "seqmap.txt",
        }
        for path in paths[dataset].values():
            if path.suffix != ".txt":
                path.mkdir(parents=True, exist_ok=True)
        tracker.mkdir(parents=True, exist_ok=True)
    return paths


def sequence_id(video: str, query_id: int) -> str:
    return f"{str(video)}__q{int(query_id):05d}"


def sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def load_record(video: str) -> tuple[Path, dict[str, Any]]:
    candidates = [ASSET_ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl",
                 ASSET_ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"]
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no legal internal record for {video}")
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with path.open("rb") as handle:
        return path, pickle.load(handle)


def materialize_gt(scope: str, dataset: str, video_queries: dict[str, list[dict[str, Any]]],
                   paths: dict[str, Path], store: L88ClipStore) -> dict[str, Any]:
    """Read fit/dev or internal labels only after this strategy's predictions are frozen."""
    metadata = {int(row["query_id"]): row for row in load_l49_queries(dataset)
                if str(row.get("video")) in set(video_queries)}
    seqs: list[str] = []
    query_audits: list[dict[str, Any]] = []
    record_audits: list[dict[str, Any]] = []
    for video, queries in sorted(video_queries.items()):
        record_path, record = load_record(video)
        frames = {int(value["frame"]): value for value in record["frames"]}
        native_frames = frame_ids(store, video)
        if set(native_frames) != set(frames):
            raise AssertionError(f"full-video GT/bank frame drift: {dataset}|{video}")
        record_audits.append({"video": video, "record_path": str(record_path.resolve()),
                              "record_sha256": file_sha(record_path), "frame_count": len(frames),
                              "bank_frame_count": len(native_frames), "scope": scope})
        image_size = [int(value) for value in record.get("image_size", [0, 0])]
        if image_size[0] <= 0 or image_size[1] <= 0:
            image_size = [int(value) for value in store.bank_store._store.bank["metadata"]["image_size"]]
        for query in queries:
            qid = int(query["query_id"])
            if qid not in metadata:
                raise KeyError(f"full-video label query missing: {dataset}|{video}|{qid}")
            entry = metadata[qid]
            if str(entry["sentence"]) != str(query["sentence"]):
                raise AssertionError(f"full-video sentence drift: {dataset}|{video}|{qid}")
            target = entry.get("target", {})
            seq = sequence_id(video, qid); seqs.append(seq)
            gt_dir = paths["gt"] / seq; gt_dir.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []; gt_rows = 0; target_frames = 0
            for frame in native_frames:
                ids = target.get(int(frame), target.get(str(frame), set()))
                ids = set(ids or [])
                if ids:
                    target_frames += 1
                boxes = frames[int(frame)].get("gt_boxes", {})
                for raw_id in sorted(str(value) for value in ids):
                    box = boxes.get(raw_id, boxes.get(int(raw_id)) if raw_id.isdigit() else None)
                    if box is None:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box]
                    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
                        raise AssertionError(f"invalid GT box: {dataset}|{video}|{qid}|{frame}|{raw_id}")
                    lines.append(f"{int(frame)+1},{int(raw_id)},{x1:.6f},{y1:.6f},"
                                 f"{x2-x1:.6f},{y2-y1:.6f},1,1,1\n")
                    gt_rows += 1
            (gt_dir / "gt.txt").write_text("".join(lines))
            (gt_dir / "seqinfo.ini").write_text(
                "[Sequence]\n" f"name={seq}\n" "imDir=img1\n" "frameRate=10\n"
                f"seqLength={len(native_frames)}\n" f"imWidth={image_size[0]}\n"
                f"imHeight={image_size[1]}\n" "imExt=.png\n")
            query_audits.append({"dataset": dataset, "video": video, "query_id": qid,
                                 "sequence": seq, "gt_rows": gt_rows,
                                 "target_present_frames": target_frames,
                                 "labels_attached_after_predictions": True,
                                 "label_source": str(entry.get("label_source", "L49 fit/dev/internal"))})
        paths["seqmap"].write_text("name\n" + "\n".join(seqs) + "\n")
    return {"sequence_count": len(seqs), "sequences": seqs,
            "query_gt_audits": query_audits, "record_audits": record_audits,
            "labels_attached_after_predictions": True,
            "label_scope": "fit/dev only" if scope == "dev" else "internal validation only"}


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 full-video output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    runtime = store = reader = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if args.scope not in {"dev", "internal"}:
            raise ValueError(args.scope)
        if args.dataset == "all":
            datasets = ["refer_kitti_v1", "refer_kitti_v2"]
        else:
            datasets = [str(args.dataset)]
        split = json.loads(SPLIT.read_text())
        videos_by_dataset = scope_videos(args.scope, datasets, split)
        shortlist = json.loads(args.shortlist.resolve().read_text())
        if shortlist.get("status") != "complete" or not shortlist.get("shortlist"):
            raise AssertionError("L88 shortlist is not complete")
        candidates = shortlist["shortlist"]
        if args.max_candidates:
            candidates = candidates[: int(args.max_candidates)]
        if args.max_frames < 0 or args.max_queries < 0:
            raise ValueError("targeted limits must be nonnegative")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        reader = EncoderCacheReader(args.cache)
        store = L88ClipStore(L85_CACHE, load_cache_into_ram=False)
        runtime, injector, base_digest = make_runtime(device)
        video_queries: dict[str, dict[str, list[dict[str, Any]]]] = {dataset: {} for dataset in datasets}
        for dataset, videos in videos_by_dataset.items():
            for video in videos:
                queries = scope_queries(store, args.scope, dataset, video, split)
                if args.max_queries:
                    queries = queries[: int(args.max_queries)]
                if not queries:
                    raise AssertionError(f"empty full-video query selection: {dataset}|{video}")
                video_queries[dataset][video] = queries

        summaries: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            checkpoint_info = dict(candidate["checkpoint_info"])
            checkpoint_path = Path(str(checkpoint_info["path"])).resolve()
            if sha256(checkpoint_path) != str(checkpoint_info["sha256"]):
                raise AssertionError(f"shortlist checkpoint SHA drift: {checkpoint_path}")
            sidecar, loaded_info = load_checkpoint_into(runtime, injector, checkpoint_path, device)
            rule_names = ("B", "R", "P")
            strategy_paths: dict[str, dict[str, dict[str, Path]]] = {}
            for rule_name in rule_names:
                strategy_paths[rule_name] = prepare_strategy(
                    strategy_root(out, candidate, rule_name), datasets, video_queries)
            audit_handles = {rule_name: (strategy_root(out, candidate, rule_name) / "prediction_audits.jsonl").open("w")
                             for rule_name in rule_names}
            counters = {rule_name: {"frames": 0, "queries": 0, "candidate_rows": 0,
                                    "selected_rows": 0, "key_digest": hashlib.sha256()}
                        for rule_name in rule_names}
            try:
                for dataset in datasets:
                    for video in videos_by_dataset[dataset]:
                        queries = video_queries[dataset][video]
                        frames = frame_ids(store, video)
                        if args.max_frames:
                            frames = frames[: int(args.max_frames)]
                        for frame_index, frame_id in enumerate(frames):
                            first, _ = make_frame(store, dataset, video, frame_id, queries[0])
                            records = score_frame_queries(runtime, reader, sidecar, store, first, queries, device,
                                                          int(args.query_tile))
                            if len(records) != len(queries):
                                raise AssertionError(f"full-video query count drift: {dataset}|{video}|{frame_id}")
                            boxes = first.boxes.detach().cpu().numpy()
                            for record in records:
                                values = np.asarray(record["score"], dtype=np.float64)
                                tracks = [int(value) for value in record["track_ids"]]
                                if values.shape != (int(record["candidate_count"]),) or len(tracks) != len(set(tracks)):
                                    raise AssertionError(f"full-video row/track drift: {record['unit_key']}")
                                key_digest_value = json.dumps({"unit_key": record["unit_key"],
                                                               "row_keys": record["row_keys"]}, sort_keys=False).encode()
                                for rule_name in rule_names:
                                    rule = candidate["rule_fits"][rule_name]
                                    gate = (float(record["presence_logit"]) >= float(rule["presence_threshold"]) and
                                            float(record["presence_logit"]) - float(record["null_logit"]) >= float(rule["null_margin"]))
                                    selected = np.flatnonzero((values >= float(rule["candidate_threshold"])) & bool(gate))
                                    path = strategy_paths[rule_name][dataset]["tracker_data"] / f"{sequence_id(video, int(record['query_id']))}.txt"
                                    with path.open("a") as handle:
                                        for local in selected.tolist():
                                            x1, y1, x2, y2 = [float(value) for value in boxes[local]]
                                            handle.write(f"{int(frame_id)+1},{tracks[local]},{x1:.6f},{y1:.6f},"
                                                         f"{x2-x1:.6f},{y2-y1:.6f},{sigmoid(values[local]):.8f},1,1,1\n")
                                    counter = counters[rule_name]
                                    counter["frames"] += 1; counter["queries"] += 1
                                    counter["candidate_rows"] += int(record["candidate_count"])
                                    counter["selected_rows"] += int(selected.size)
                                    counter["key_digest"].update(key_digest_value)
                                    audit_handles[rule_name].write(json.dumps({
                                        "dataset": dataset, "video": video, "query_id": int(record["query_id"]),
                                        "frame_id": int(frame_id), "unit_key": record["unit_key"],
                                        "candidate_rows_scored": int(record["candidate_count"]),
                                        "selected_rows": int(selected.size), "presence_gate": bool(gate),
                                        "candidate_rows_retained": True, "candidate_deletion": False,
                                        "candidate_truncation": False, "labels_attached": False,
                                    }, sort_keys=True) + "\n")
                            del first, records
                            store.release_loaded_cache_items(); gc.collect()
                            if device.type == "cuda" and frame_index % 10 == 0:
                                torch.cuda.empty_cache()
                        print(f"[l88-fullvideo] scope={args.scope} epoch={checkpoint_info['epoch']} "
                              f"{dataset} video={video} frames={len(frames)} queries={len(queries)} "
                              f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
            finally:
                for handle in audit_handles.values():
                    handle.close()
            # Every query has a tracker file, including a legal all-empty file.
            for rule_name in rule_names:
                for dataset in datasets:
                    for video, queries in video_queries[dataset].items():
                        tracker_dir = strategy_paths[rule_name][dataset]["tracker_data"]
                        for query in queries:
                            (tracker_dir / f"{sequence_id(video, int(query['query_id']))}.txt").touch(exist_ok=True)
            # Labels are attached only after all this candidate's predictions
            # are written.  They are fit/dev labels in the dev scope and
            # internal validation labels in the final internal scope.
            eval_summary = {}
            for rule_name in rule_names:
                eval_summary[rule_name] = {}
                for dataset in datasets:
                    eval_summary[rule_name][dataset] = materialize_gt(
                        args.scope, dataset, video_queries[dataset],
                        {**strategy_paths[rule_name][dataset], "seqmap": strategy_paths[rule_name][dataset]["seqmap"]}, store)
            candidate_summary = {
                "candidate_index": candidate_index, "checkpoint": loaded_info,
                "rules": {rule: {**counters[rule], "key_digest": counters[rule]["key_digest"].hexdigest(),
                                  "candidate_rows_retained": True, "candidate_deletion": False,
                                  "candidate_truncation": False}
                           for rule in rule_names},
                "strategy_paths": {rule: {dataset: {name: str(path.resolve()) for name, path in paths.items()}
                                           for dataset, paths in dataset_paths.items()}
                                   for rule, dataset_paths in strategy_paths.items()},
                "eval_summary": eval_summary, "prediction_strategy_frozen_before_gt": True,
            }
            summaries.append(candidate_summary)
            del sidecar
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        full = not args.max_frames and not args.max_queries
        payload = {
            "format": "locatemot-l88-fullvideo-matrix-v1", "status": "complete",
            "scope": "internal full-video dev selection" if args.scope == "dev" else "internal V1/V2 full-video validation",
            "full_video": bool(full), "command": command, "cwd": str(WORK_ROOT),
            "luna_thread": THREAD, "seed": SEED, "datasets": datasets,
            "videos": videos_by_dataset, "shortlist_source": str(args.shortlist.resolve()),
            "shortlist_sha256": sha256(args.shortlist.resolve()), "candidates": summaries,
            "cache": str(args.cache.resolve()), "cache_summary_sha256": reader.summary_sha256,
            "base_detector_digest": base_digest, "query_tile": int(args.query_tile),
            "all_candidate_rows_scored": True, "candidate_deletion": False,
            "candidate_truncation": False, "model_selection_used_validation": False,
            "labels_attached_after_predictions": True,
            "fit_dev_labels_only": args.scope == "dev", "internal_validation_labels_only": args.scope == "internal",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "groundingdino_lora_used": True,
            "groundingdino_backbone_trainable": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "manifest_sha256": MANIFEST_SHA,
            "transient_missing_cache_frames_allowed": True,
            "persistent_dense_cache_written": False, "wall_seconds": time.perf_counter()-started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "failure_root_cause": None,
            "next_action": "run L88 TrackEval matrix on the frozen strategy outputs",
        }
        write_json(out / "summary.json", payload); write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": "locatemot-l88-fullvideo-matrix-status-v1", "status": "complete",
                                          "scope": args.scope, "full_video": bool(full),
                                          "candidate_count": len(summaries), "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": False, "no_hota_or_trackeval": True})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 full-video matrix — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l88-fullvideo-matrix-status-v1", "status": "incomplete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if store is not None:
            store.release_loaded_cache_items(); store.close()
        if reader is not None:
            del reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=L88_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=("all", "refer_kitti_v1", "refer_kitti_v2"), default="all")
    parser.add_argument("--query-tile", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
