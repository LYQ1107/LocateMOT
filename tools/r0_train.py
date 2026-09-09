#!/usr/bin/env python3
"""Train the registered R0 track-centric head with grouped native-frame tiles.

The scientific model and loss are intentionally unchanged.  The optimization
unit in this driver is a native frame containing a deterministic tile of
same-frame queries, not an individual query.  The visual cache is frozen and
query-independent; labels are attached by :class:`R0RuntimeData` only for the
fit index selected by this benchmark-specific run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r0_common import (  # noqa: E402
    DEFAULT_LANGUAGE_ROOTS,
    DenseIndex,
    MergedLanguageCache,
    R0RuntimeData,
    SEED,
    THREAD,
    VisualCacheIndex,
    check_manifest,
    model_forward,
    sha256_file,
    standard_flags,
    write_json,
)
from locatemot.models.r0_track_grounding import R0Config, R0TrackGroundingHead  # noqa: E402
from locatemot.rmot.r0_losses import r0_total_loss  # noqa: E402


CATEGORY_QUOTA = (
    ("multi_positive", 3),
    ("positive", 2),
    ("inactive", 2),
    ("present_uncovered", 1),
)
CATEGORIES = tuple(value[0] for value in CATEGORY_QUOTA)
CHECKPOINT_EPOCHS = (2, 4, 6, 8, 10, 12)
ACCUMULATION = {1: 16, 2: 8, 3: 5, 4: 4}
VIDEO_BLOCK_SIZE = 16
CHECKPOINT_FORMAT = "locatemot-r0-tcgh-checkpoint-v2"


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def init_dist(requested_device: str) -> tuple[int, int, int, torch.device]:
    """Initialize the registered local-rank-aware R0 process group."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 4:
        raise RuntimeError("R0 max world size is 4")
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("R0 DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("R0 training requested CUDA but it is unavailable")
            torch.cuda.set_device(device)
    return world, rank, local_rank, device


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    """Detailed report used only at the first optimizer step of epoch one."""
    total = 0.0
    finite = True
    nonzero = 0
    names: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        norm = float(value.norm())
        names[name] = norm
        total += norm
        nonzero += int(norm > 0.0)
        finite = finite and bool(torch.isfinite(value).all())
    return {
        "total_norm": total,
        "nonzero_parameter_grads": nonzero,
        "finite": finite,
        "by_name": names,
    }


def select_query_tile(
    frame_records: list[dict[str, Any]],
    rng: random.Random,
    max_queries: int = 8,
) -> list[dict[str, Any]]:
    """Select a same-frame query tile with the registered category quotas."""
    if not frame_records:
        raise ValueError("cannot select a query tile from an empty frame")
    if int(max_queries) < 1:
        raise ValueError("max_queries must be positive")
    buckets: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for record in frame_records:
        category = str(record["category"])
        if category in buckets:
            buckets[category].append(record)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for category, quota in CATEGORY_QUOTA:
        for record in buckets[category][: int(quota)]:
            query_id = int(record["query_id"])
            if query_id not in used and len(selected) < int(max_queries):
                selected.append(record)
                used.add(query_id)
    for category, _quota in CATEGORY_QUOTA:
        if len(selected) >= int(max_queries):
            break
        for record in buckets[category]:
            query_id = int(record["query_id"])
            if query_id not in used:
                selected.append(record)
                used.add(query_id)
                if len(selected) >= int(max_queries):
                    break
    if len(selected) < int(max_queries):
        remaining = [record for record in frame_records if int(record["query_id"]) not in used]
        rng.shuffle(remaining)
        for record in remaining:
            selected.append(record)
            used.add(int(record["query_id"]))
            if len(selected) >= int(max_queries):
                break
    rng.shuffle(selected)
    if not selected:
        raise AssertionError("R0 query tile is empty")
    if len({int(record["query_id"]) for record in selected}) != len(selected):
        raise AssertionError("duplicate query in R0 tile")
    frame_key = (str(selected[0]["dataset"]), str(selected[0]["video"]), int(selected[0]["frame_id"]))
    if any((str(record["dataset"]), str(record["video"]), int(record["frame_id"])) != frame_key for record in selected):
        raise AssertionError("R0 tile mixes native frames")
    return selected


def build_epoch_frame_schedule(
    dense: DenseIndex,
    epoch: int,
    global_tiles: int,
    seed: int,
    world_size: int,
) -> list[tuple[str, str, int]]:
    """Build the deterministic frame-key schedule before block sharding."""
    if int(world_size) not in ACCUMULATION:
        raise ValueError(f"unsupported R0 world size: {world_size}")
    if int(global_tiles) < 1 or not dense.frame_keys:
        raise ValueError("R0 frame schedule is empty")
    base = list(dense.frame_keys)
    rng = random.Random(int(seed) + int(epoch) * 1009)
    rng.shuffle(base)
    result: list[tuple[str, str, int]] = []
    cycle = 0
    while len(result) < int(global_tiles):
        cycle_rng = random.Random(int(seed) + int(epoch) * 1009 + cycle * 1_000_003)
        cycle_keys = list(dense.frame_keys)
        cycle_rng.shuffle(cycle_keys)
        result.extend(cycle_keys[: max(0, int(global_tiles) - len(result))])
        cycle += 1
    return result


def block_shard(
    frame_keys: list[tuple[str, str, int]],
    epoch: int,
    seed: int,
    world_size: int,
    rank: int,
    accumulation: int,
) -> tuple[list[tuple[str, str, int]], dict[str, Any]]:
    """Shuffle video-local blocks, then pad rank lengths for DDP safety."""
    by_video: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
    for key in frame_keys:
        by_video[(str(key[0]), str(key[1]))].append(key)
    block_rng = random.Random(int(seed) + int(epoch) * 1009 + 7_000_003)
    blocks: list[list[tuple[str, str, int]]] = []
    for video_key in sorted(by_video):
        values = list(by_video[video_key])
        block_rng.shuffle(values)
        for start in range(0, len(values), VIDEO_BLOCK_SIZE):
            blocks.append(values[start:start + VIDEO_BLOCK_SIZE])
    block_rng.shuffle(blocks)
    rank_values: list[list[tuple[str, str, int]]] = [[] for _ in range(int(world_size))]
    for block_index, block in enumerate(blocks):
        rank_values[block_index % int(world_size)].extend(block)
    max_rank_tiles = max(len(value) for value in rank_values)
    rank_pad_counts = [max_rank_tiles - len(value) for value in rank_values]
    for rank_index, values in enumerate(rank_values):
        for pad_index in range(rank_pad_counts[rank_index]):
            values.append(frame_keys[(rank_index + pad_index) % len(frame_keys)])
    local_target = max(len(value) for value in rank_values)
    remainder = local_target % int(accumulation)
    if remainder:
        extra = int(accumulation) - remainder
        for rank_index, values in enumerate(rank_values):
            for pad_index in range(extra):
                values.append(frame_keys[(rank_index + len(values) + pad_index) % len(frame_keys)])
        local_target += extra
    if any(len(value) != local_target for value in rank_values):
        raise AssertionError("R0 DDP local tile count mismatch")
    actual_global = int(local_target) * int(world_size)
    metadata = {
        "video_block_size": VIDEO_BLOCK_SIZE,
        "block_count": len(blocks),
        "rank_tile_counts_before_padding": [int(len(value) - rank_pad_counts[i] - (local_target - max_rank_tiles)) for i, value in enumerate(rank_values)],
        "rank_padding_tiles": int(sum(rank_pad_counts) + int(world_size) * (local_target - max_rank_tiles)),
        "local_tiles_per_rank": int(local_target),
        "actual_global_tiles_after_padding": actual_global,
        "rank": int(rank),
        "world_size": int(world_size),
    }
    return rank_values[int(rank)], metadata


def build_epoch_tiles(
    dense: DenseIndex,
    epoch: int,
    requested_global_tiles: int,
    seed: int,
    world_size: int,
    rank: int,
    max_queries: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    frame_schedule = build_epoch_frame_schedule(dense, epoch, requested_global_tiles, seed, world_size)
    accumulation = ACCUMULATION[int(world_size)]
    local_keys, shard_info = block_shard(frame_schedule, epoch, seed, world_size, rank, accumulation)
    rng = random.Random(int(seed) + int(epoch) * 1009 + 31_000_019 * (int(rank) + 1))
    tiles: list[list[dict[str, Any]]] = []
    category_counts: Counter[str] = Counter()
    query_exposures = 0
    candidate_sum = 0
    q_values: list[int] = []
    for frame_key in local_keys:
        records = dense.by_frame[frame_key]
        tile = select_query_tile(records, rng, max_queries=max_queries)
        tiles.append(tile)
        q_values.append(len(tile))
        query_exposures += len(tile)
        candidate_sum += int(tile[0]["candidate_count"])
        category_counts.update(str(record["category"]) for record in tile)
    if not tiles:
        raise AssertionError("R0 rank received no frame tiles")
    schedule_info = {
        "epoch": int(epoch),
        "requested_global_tiles": int(requested_global_tiles),
        "global_tiles_after_divisibility_padding": int(
            math.ceil(int(requested_global_tiles) / float(world_size * accumulation)) * int(world_size * accumulation)
        ),
        "actual_global_tiles_after_padding": int(shard_info["actual_global_tiles_after_padding"]),
        "padding_tiles": int(shard_info["actual_global_tiles_after_padding"] - int(requested_global_tiles)),
        "accumulation": int(accumulation),
        "world_size": int(world_size),
        "rank": int(rank),
        "local_microtiles": len(tiles),
        "mean_Q": float(sum(q_values) / len(q_values)),
        "median_Q": float(np.median(np.asarray(q_values, dtype=np.float64))),
        "min_Q": int(min(q_values)),
        "max_Q": int(max(q_values)),
        "query_exposures": int(query_exposures),
        "category_counts": dict(sorted(category_counts.items())),
        "candidate_count_sum": int(candidate_sum),
        "frame_tile_count": len(tiles),
        "shard": shard_info,
        "deterministic": True,
        "category_quota": [[name, int(quota)] for name, quota in CATEGORY_QUOTA],
    }
    return tiles, schedule_info


def directory_fingerprint(root: Path) -> str:
    """Cheap immutable cache fingerprint without reading tensor payloads."""
    root = Path(root).resolve()
    digest = hashlib.sha256()
    if not root.exists():
        return "missing"
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def input_fingerprints(args: argparse.Namespace) -> dict[str, Any]:
    dense_root = args.dense_root.resolve()
    visual_root = args.visual_cache.resolve()
    return {
        "manifest_sha256": check_manifest(),
        "dense_index_summary_sha256": sha256_file(dense_root / "summary.json"),
        "dense_index_fingerprint": directory_fingerprint(dense_root),
        "visual_manifest_sha256": sha256_file(visual_root / "manifest.jsonl"),
        "language_cache_fingerprints": [directory_fingerprint(Path(value)) for value in args.language_root],
        "language_cache_roots": [str(Path(value).resolve()) for value in args.language_root],
    }


def cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return value


def checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    out: Path,
    epoch: int,
    optimizer_step: int,
    global_microtiles_seen: int,
    query_exposures_seen: int,
    dataset: str,
    model_config: dict[str, Any],
    input_hashes: dict[str, Any],
    args: argparse.Namespace,
    world_size: int,
    accumulation: int,
    optimizer_steps_per_epoch: int,
    preview_only: bool,
) -> dict[str, Any]:
    suffix = "_preview" if int(epoch) == 1 else ""
    path = out / "checkpoints" / f"checkpoint_r0_{dataset}_epoch{int(epoch):02d}{suffix}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in unwrap(model).state_dict().items()}
    package = {
        "format": CHECKPOINT_FORMAT,
        "driver": "grouped_ddp_v2",
        "grouped_query_training": True,
        "max_queries_per_tile": int(args.max_queries_per_tile),
        "dataset": str(dataset),
        "epoch": int(epoch),
        "completed_epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "global_step": int(optimizer_step),
        "global_microtiles_seen": int(global_microtiles_seen),
        "query_exposures_seen": int(query_exposures_seen),
        "world_size": int(world_size),
        "accumulation": int(accumulation),
        "global_tiles_per_epoch": int(args.global_tiles_per_epoch),
        "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
        "model_config": model_config,
        "model_state_dict": state,
        "optimizer_state_dict": cpu_clone(optimizer.state_dict()),
        "scheduler_state_dict": cpu_clone(scheduler.state_dict()),
        "seed": int(SEED),
        "training_seed": int(SEED),
        "input_fingerprints": input_hashes,
        "manifest_sha256": input_hashes["manifest_sha256"],
        "visual_cache_sha256": input_hashes["visual_manifest_sha256"],
        "dense_index_sha256": input_hashes["dense_index_summary_sha256"],
        "language_cache_shas": input_hashes["language_cache_fingerprints"],
        "detector_state_included": False,
        "tracker_state_included": False,
        "ordinary_mot_ovmot_touched": False,
        "preview_only": bool(preview_only),
        "primary_selection_eligible": bool(not preview_only and int(epoch) in CHECKPOINT_EPOCHS),
    }
    torch.save(package, path)
    reload_model = R0TrackGroundingHead(R0Config(**model_config))
    result = reload_model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict R0 v2 model reload failed: {result}")
    reload_optimizer = torch.optim.AdamW(reload_model.parameters(), lr=1e-4, weight_decay=1e-2)
    reload_optimizer.load_state_dict(package["optimizer_state_dict"])
    reload_scheduler = torch.optim.lr_scheduler.LambdaLR(reload_optimizer, lambda _step: 1.0)
    reload_scheduler.load_state_dict(package["scheduler_state_dict"])
    reload_diff = 0.0
    for key, value in reload_model.state_dict().items():
        reload_diff = max(reload_diff, float((value - state[key]).abs().max()))
    if reload_diff != 0.0:
        raise AssertionError(f"R0 v2 reload drift: {reload_diff}")
    del reload_model, reload_optimizer, reload_scheduler, state
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "global_microtiles_seen": int(global_microtiles_seen),
        "query_exposures_seen": int(query_exposures_seen),
        "strict_reload": True,
        "reload_max_abs_diff": reload_diff,
        "bytes": int(path.stat().st_size),
        "format": CHECKPOINT_FORMAT,
        "preview_only": bool(preview_only),
        "primary_selection_eligible": bool(not preview_only and int(epoch) in CHECKPOINT_EPOCHS),
    }


def validate_resume(
    package: dict[str, Any],
    args: argparse.Namespace,
    model_config: dict[str, Any],
    input_hashes: dict[str, Any],
    world_size: int,
    accumulation: int,
) -> None:
    if package.get("format") != CHECKPOINT_FORMAT:
        raise AssertionError("R0 resume requires checkpoint format v2")
    if package.get("driver") != "grouped_ddp_v2" or package.get("grouped_query_training") is not True:
        raise AssertionError("R0 resume checkpoint is not grouped training")
    checks = {
        "dataset": (str(package.get("dataset")), str(args.dataset)),
        "model_config": (package.get("model_config"), model_config),
        "seed": (int(package.get("seed", -1)), int(SEED)),
        "world_size": (int(package.get("world_size", -1)), int(world_size)),
        "accumulation": (int(package.get("accumulation", -1)), int(accumulation)),
        "global_tiles_per_epoch": (int(package.get("global_tiles_per_epoch", -1)), int(args.global_tiles_per_epoch)),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise AssertionError(f"R0 resume {name} mismatch: {observed!r} != {expected!r}")
    if package.get("input_fingerprints") != input_hashes:
        raise AssertionError("R0 resume frozen input fingerprint mismatch")
    if int(package.get("completed_epoch", package.get("epoch", 0))) >= int(args.epochs):
        raise AssertionError("R0 resume checkpoint already reaches requested epochs")


def reduce_window(
    device: torch.device,
    world_size: int,
    loss_sum: torch.Tensor,
    tile_count: int,
    query_count: int,
    candidate_sum: int,
    grad_sum: torch.Tensor,
) -> dict[str, float]:
    values = torch.stack((
        loss_sum.detach().float(),
        torch.tensor(float(tile_count), device=device),
        torch.tensor(float(query_count), device=device),
        torch.tensor(float(candidate_sum), device=device),
        grad_sum.detach().float(),
    ))
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "loss_sum": float(values[0].item()),
        "tile_count": float(values[1].item()),
        "query_count": float(values[2].item()),
        "candidate_sum": float(values[3].item()),
        "grad_sum": float(values[4].item()),
    }


def gpu_utilization(local_rank: int) -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-i", str(local_rank), "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def run(args: argparse.Namespace) -> int:
    world_size, rank, local_rank, device = init_dist(args.device)
    is_main = rank == 0
    out = args.out.resolve()
    if is_main:
        if args.resume_checkpoint is None and out.exists() and any(out.iterdir()):
            raise FileExistsError(f"refusing nonempty R0 grouped output: {out}")
        out.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    else:
        out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable, *sys.argv])
    base = {
        "format": "locatemot-r0-grouped-ddp-training-v2",
        "status": "incomplete",
        "command": command,
        "cwd": str(Path.cwd().resolve()),
        "worktree": str(WORK_ROOT),
        "luna_thread": THREAD,
        "seed": int(SEED),
        "dataset": str(args.dataset),
        "epochs_requested": int(args.epochs),
        "stop_after_epoch": int(args.stop_after_epoch),
        "global_tiles_per_epoch_requested": int(args.global_tiles_per_epoch),
        "max_queries_per_tile": int(args.max_queries_per_tile),
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "accumulation": int(ACCUMULATION[world_size]),
        "bf16_requested": bool(args.bf16),
        "device": str(device),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "inputs": {
            "dense_root": str(args.dense_root.resolve()),
            "visual_cache": str(args.visual_cache.resolve()),
            "language_roots": [str(Path(value).resolve()) for value in args.language_root],
        },
        "outputs": {"root": str(out)},
        "candidate_deletion": False,
        "candidate_truncation": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "failure_root_cause": None,
        "next_action": "resume formal grouped R0 to epoch12, then legal dev selection",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 training cwd: {Path.cwd()}")
        if world_size > 4:
            raise RuntimeError("R0 max world size is 4")
        set_seed(SEED)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        dense = DenseIndex(args.dense_root)
        visual = VisualCacheIndex(args.visual_cache)
        language = MergedLanguageCache([Path(value) for value in args.language_root])
        input_hashes = input_fingerprints(args)
        model_core = R0TrackGroundingHead().to(device=device, dtype=torch.float32)
        model_config = model_core.config_dict()
        trainable = [(name, parameter) for name, parameter in model_core.named_parameters() if parameter.requires_grad]
        if len(trainable) != len(list(model_core.parameters())) or not trainable:
            raise AssertionError("R0 trainable parameter contract drift")
        if world_size > 1:
            model: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
                model_core, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False, find_unused_parameters=False,
            )
        else:
            model = model_core
        optimizer = torch.optim.AdamW([parameter for _name, parameter in trainable], lr=1e-4, weight_decay=1e-2)
        accumulation = ACCUMULATION[world_size]
        # The number of optimizer steps is derived from the padded frame-tile
        # schedule, not from the historical query-centric 4000 value.
        nominal_global = math.ceil(int(args.global_tiles_per_epoch) / float(world_size * accumulation)) * int(world_size * accumulation)
        nominal_steps_per_epoch = nominal_global // (world_size * accumulation)
        total_optimizer_steps = nominal_steps_per_epoch * int(args.epochs)
        warmup_steps = max(1, int(round(total_optimizer_steps * 0.05)))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps - 1))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        start_epoch = 1
        optimizer_step = 0
        global_microtiles_seen = 0
        query_exposures_seen = 0
        resume_info = None
        if args.resume_checkpoint is not None:
            resume_path = args.resume_checkpoint.resolve()
            package = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_resume(package, args, model_config, input_hashes, world_size, accumulation)
            loaded = unwrap(model).load_state_dict(package["model_state_dict"], strict=True)
            if loaded.missing_keys or loaded.unexpected_keys:
                raise AssertionError(f"R0 resume model reload failed: {loaded}")
            optimizer.load_state_dict(package["optimizer_state_dict"])
            scheduler.load_state_dict(package["scheduler_state_dict"])
            completed_epoch = int(package.get("completed_epoch", package["epoch"]))
            start_epoch = completed_epoch + 1
            optimizer_step = int(package.get("optimizer_step", package.get("global_step", 0)))
            global_microtiles_seen = int(package.get("global_microtiles_seen", 0))
            query_exposures_seen = int(package.get("query_exposures_seen", 0))
            resume_info = {
                "path": str(resume_path), "sha256": sha256_file(resume_path),
                "completed_epoch": completed_epoch, "strict_reload": True,
                "resume_start_epoch": start_epoch,
            }
            del package
        if is_main and args.resume_checkpoint is None:
            config_payload = {
                **base,
                "status": "initialized",
                "model_config": model_config,
                "model_parameter_count": sum(parameter.numel() for parameter in model_core.parameters()),
                "trainable_parameter_count": sum(parameter.numel() for _name, parameter in trainable),
                "trainable_parameter_names": [name for name, _parameter in trainable],
                "input_fingerprints": input_hashes,
                "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-2, "warmup_fraction": 0.05, "schedule": "cosine", "grad_clip": 1.0},
                "nominal_global_tiles_after_divisibility_padding": nominal_global,
                "nominal_optimizer_steps_per_epoch": nominal_steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "accumulation_mapping": ACCUMULATION,
                "category_quota": [[name, int(quota)] for name, quota in CATEGORY_QUOTA],
                "video_block_size": VIDEO_BLOCK_SIZE,
                "resume_supported": True,
                "no_persistent_raw_dense_cache_created": True,
                "token_span_region_alignment": "UNALIGNED",
                "static_motion_alignment": "UNALIGNED",
            }
            write_json(out / "config.json", config_payload)
        if world_size > 1:
            dist.barrier()
        runtime = R0RuntimeData(dense, visual, language, device)
        if int(args.contract_smoke_microtiles) > 0:
            # This mode exercises only local-rank device selection and DDP
            # backward synchronization.  It never produces a formal
            # checkpoint or enters the grouped epoch scheduler.
            smoke_rng = random.Random(SEED + 91_000_003 * (rank + 1))
            smoke_losses: list[float] = []
            smoke_reports: list[dict[str, Any]] = []
            for smoke_index in range(int(args.contract_smoke_microtiles)):
                frame_key = dense.frame_keys[(rank * int(args.contract_smoke_microtiles) + smoke_index) % len(dense.frame_keys)]
                tile = select_query_tile(dense.by_frame[frame_key], smoke_rng, max_queries=int(args.max_queries_per_tile))
                sample = runtime.prepare_group(tile, attach_labels=True)
                sups = [item["supervision"] for item in sample["labels"]]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=bool(args.bf16 and device.type == "cuda")):
                    output = model_forward(model, sample)
                    if output["membership_logit"].shape != (len(tile), int(sample["candidate_count"])):
                        raise AssertionError("R0 DDP smoke grouped output shape drift")
                    loss, _info = r0_total_loss(
                        output["membership_logit"], output["coverage_presence_logit"],
                        [str(sup["category"]) for sup in sups],
                        [sup["target_ids"] for sup in sups],
                        [sup["candidate_gt"] for sup in sups],
                        collect_info=False,
                    )
                if not bool(torch.isfinite(loss.float()).all()):
                    raise FloatingPointError("R0 DDP smoke nonfinite loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
                if not bool(torch.isfinite(grad_norm).all()) or not bool((grad_norm > 0).all()):
                    raise FloatingPointError("R0 DDP smoke nonfinite/zero gradient")
                smoke_losses.append(float(loss.detach()))
                smoke_reports.append({"frame_key": list(frame_key), "Q": len(tile),
                                      "candidate_count": int(sample["candidate_count"]),
                                      "loss": smoke_losses[-1], "grad_norm": float(grad_norm.detach()),
                                      "candidate_rows_retained": True})
                del sample, sups, output, loss, _info
            if world_size > 1:
                dist.barrier()
            if is_main:
                smoke_payload = {
                    **base, "format": "locatemot-r0-ddp-contract-smoke-v2", "status": "complete",
                    "contract_smoke": True, "local_microtiles_per_rank": int(args.contract_smoke_microtiles),
                    "rank_reports": smoke_reports, "world_size": int(world_size),
                    "local_rank_devices": "LOCAL_RANK owns one CUDA device", "finite": True,
                    "ddp_backward_synchronized": bool(world_size == 1 or dist.is_initialized()),
                    "candidate_rows_retained": True, "candidate_deletion": False,
                    "candidate_truncation": False, "input_fingerprints": input_hashes,
                    "wall_seconds": time.perf_counter() - started,
                    "next_action": "run the grouped formal R0 driver",
                }
                write_json(out / "contract.json", smoke_payload)
                write_json(out / "provenance.json", smoke_payload | {"format": "locatemot-r0-ddp-contract-provenance-v2"})
                write_json(out / "status.json", smoke_payload | standard_flags(training_run=False))
            runtime.close()
            return 0
        loss_path = out / "loss_trace.jsonl"
        summary_path = out / "epoch_summary.jsonl"
        loss_handle = loss_path.open("a" if args.resume_checkpoint is not None else "w", encoding="utf-8") if is_main else None
        summary_handle = summary_path.open("a" if args.resume_checkpoint is not None else "w", encoding="utf-8") if is_main else None
        epoch_summaries: list[dict[str, Any]] = []
        checkpoints: list[dict[str, Any]] = []
        if args.resume_checkpoint is not None and is_main:
            prior_metrics = out / "metrics.json"
            if prior_metrics.is_file():
                prior_payload = json.loads(prior_metrics.read_text(encoding="utf-8"))
                if isinstance(prior_payload.get("checkpoints"), list):
                    checkpoints.extend(prior_payload["checkpoints"])
        if args.resume_checkpoint is not None and is_main and summary_path.is_file():
            for line in summary_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    epoch_summaries.append(json.loads(line))
        sampling_summaries: list[dict[str, Any]] = []
        if args.resume_checkpoint is not None and is_main:
            prior_sampling = out / "sampling_trace.json"
            if prior_sampling.is_file():
                prior_payload = json.loads(prior_sampling.read_text(encoding="utf-8"))
                if isinstance(prior_payload.get("schedule"), list):
                    sampling_summaries.extend(prior_payload["schedule"])
        first50 = None
        final_epoch = start_epoch - 1
        run_stop = int(args.stop_after_epoch) if int(args.stop_after_epoch) > 0 else int(args.epochs)
        if run_stop > int(args.epochs):
            raise ValueError("stop-after-epoch cannot exceed epochs")
        if start_epoch > run_stop:
            raise AssertionError(f"R0 no epochs remain: start={start_epoch} stop={run_stop}")
        for epoch in range(start_epoch, run_stop + 1):
            tiles, schedule_info = build_epoch_tiles(
                dense, epoch, int(args.global_tiles_per_epoch), SEED,
                world_size, rank, int(args.max_queries_per_tile),
            )
            # Check the grouped composition globally before the first backward.
            local_tile_stats = torch.tensor(
                [float(len(tiles)), float(schedule_info["query_exposures"])],
                dtype=torch.float64, device=device,
            )
            if world_size > 1:
                dist.all_reduce(local_tile_stats, op=dist.ReduceOp.SUM)
            mean_q_global = float((local_tile_stats[1] / local_tile_stats[0]).item())
            if mean_q_global < 2.0:
                raise AssertionError(f"R0 grouped mean Q below 2: {mean_q_global}")
            schedule_info["global_mean_Q"] = mean_q_global
            category_values = torch.tensor(
                [float(schedule_info["category_counts"].get(category, 0)) for category in CATEGORIES],
                dtype=torch.float64, device=device,
            )
            if world_size > 1:
                dist.all_reduce(category_values, op=dist.ReduceOp.SUM)
            schedule_info["global_frame_tile_count"] = int(round(float(local_tile_stats[0].item())))
            schedule_info["global_query_exposures"] = int(round(float(local_tile_stats[1].item())))
            schedule_info["global_category_counts"] = {
                category: int(round(float(category_values[index].item())))
                for index, category in enumerate(CATEGORIES)
            }
            if is_main:
                sampling_summaries.append(schedule_info)
            epoch_started = time.perf_counter()
            epoch_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
            epoch_grad_sum = torch.zeros((), dtype=torch.float32, device=device)
            epoch_tiles = 0
            epoch_queries = 0
            epoch_candidates = 0
            epoch_categories: Counter[str] = Counter()
            window_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
            window_grad_sum = torch.zeros((), dtype=torch.float32, device=device)
            window_tiles = 0
            window_queries = 0
            window_candidates = 0
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps_epoch = len(tiles) // accumulation
            optimizer_step_at_epoch_start = optimizer_step
            for local_index, tile_records in enumerate(tiles):
                micro_in_window = local_index % accumulation
                sync_now = (micro_in_window + 1) % accumulation == 0
                optimizer_index_epoch = local_index // accumulation + 1
                collect_diagnostics = bool(
                    (epoch == 1 and optimizer_index_epoch == 1)
                    or optimizer_index_epoch % 25 == 0
                    or optimizer_index_epoch == optimizer_steps_epoch
                )
                sample = runtime.prepare_group(tile_records, attach_labels=True)
                if int(sample["candidate_count"]) != int(tile_records[0]["candidate_count"]):
                    raise AssertionError("R0 grouped candidate count drift")
                supervisions = [item["supervision"] for item in sample["labels"]]
                if len(supervisions) != len(tile_records):
                    raise AssertionError("R0 grouped supervision count drift")
                sync_context = nullcontext() if sync_now or world_size == 1 else model.no_sync()  # type: ignore[attr-defined]
                with sync_context:
                    autocast_enabled = bool(args.bf16 and device.type == "cuda")
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                        output = model_forward(model, sample)
                        if output["membership_logit"].shape != (len(tile_records), int(sample["candidate_count"])):
                            raise AssertionError("R0 grouped membership output shape drift")
                        loss, info = r0_total_loss(
                            output["membership_logit"], output["coverage_presence_logit"],
                            [str(sup["category"]) for sup in supervisions],
                            [sup["target_ids"] for sup in supervisions],
                            [sup["candidate_gt"] for sup in supervisions],
                            collect_info=collect_diagnostics,
                        )
                        if not bool(torch.isfinite(loss.float()).all()):
                            raise FloatingPointError(f"nonfinite grouped R0 loss at epoch {epoch} microtile {local_index + 1}")
                        (loss / float(accumulation)).backward()
                epoch_loss_sum = epoch_loss_sum + loss.detach().float()
                window_loss_sum = window_loss_sum + loss.detach().float()
                epoch_tiles += 1
                epoch_queries += len(tile_records)
                epoch_candidates += int(sample["candidate_count"])
                window_tiles += 1
                window_queries += len(tile_records)
                window_candidates += int(sample["candidate_count"])
                epoch_categories.update(str(record["category"]) for record in tile_records)
                global_microtiles_seen += world_size
                query_exposures_seen += len(tile_records) * world_size
                if sync_now:
                    grad_norm = torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
                    if not bool(torch.isfinite(grad_norm).all()) or not bool((grad_norm > 0).all()):
                        raise FloatingPointError(f"R0 grouped gradient contract failed at epoch {epoch} optimizer {optimizer_step + 1}")
                    if epoch == 1 and optimizer_index_epoch == 1 and rank == 0:
                        detailed = gradient_report(unwrap(model))
                        if not detailed["finite"] or int(detailed["nonzero_parameter_grads"]) <= 0:
                            raise FloatingPointError(f"R0 detailed first-step gradient contract failed: {detailed}")
                        write_json(out / "first_optimizer_gradient_report.json", detailed)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    epoch_grad_sum = epoch_grad_sum + grad_norm.detach().float()
                    window_grad_sum = window_grad_sum + grad_norm.detach().float()
                    should_log = optimizer_step <= 5 or optimizer_step % 10 == 0 or optimizer_index_epoch == optimizer_steps_epoch
                    if should_log:
                        reduced = reduce_window(device, world_size, window_loss_sum, window_tiles, window_queries, window_candidates, window_grad_sum)
                        if is_main:
                            elapsed = max(1e-9, time.perf_counter() - epoch_started)
                            row = {
                                "format": "locatemot-r0-grouped-loss-trace-v2",
                                "optimizer_step": int(optimizer_step),
                                "epoch": int(epoch),
                                "global_microtiles_seen": int(global_microtiles_seen),
                                "local_microtiles_seen": int(local_index + 1),
                                "mean_Q_in_accumulation_window": reduced["query_count"] / max(1.0, reduced["tile_count"]),
                                "query_exposures": int(reduced["query_count"]),
                                "category_exposures": dict(sorted(epoch_categories.items())),
                                "mean_candidate_count": reduced["candidate_sum"] / max(1.0, reduced["tile_count"]),
                                "loss_mean": reduced["loss_sum"] / max(1.0, reduced["tile_count"]),
                                "grad_norm_mean": reduced["grad_sum"] / max(1.0, float(world_size)),
                                "lr": float(optimizer.param_groups[0]["lr"]),
                                "tiles_per_second_global": reduced["tile_count"] / max(1e-9, elapsed),
                                "queries_per_second_global": reduced["query_count"] / max(1e-9, elapsed),
                                "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                                "finite": True,
                                "diagnostics_collected": bool(collect_diagnostics),
                                "candidate_rows_retained": True,
                                "candidate_deletion": False,
                                "candidate_truncation": False,
                            }
                            assert loss_handle is not None
                            loss_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                            loss_handle.flush()
                        window_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                        window_grad_sum = torch.zeros((), dtype=torch.float32, device=device)
                        window_tiles = 0
                        window_queries = 0
                        window_candidates = 0
                    else:
                        # A non-logged window still needs a fresh accumulation
                        # counter, while avoiding any scalar Python conversion.
                        window_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                        window_grad_sum = torch.zeros((), dtype=torch.float32, device=device)
                        window_tiles = 0
                        window_queries = 0
                        window_candidates = 0
                del sample, supervisions, output, loss, info, tile_records
            if optimizer_step - optimizer_step_at_epoch_start != optimizer_steps_epoch:
                raise AssertionError("R0 optimizer step count drift")
            # Epoch scalar reduction is intentionally small; no candidate rows
            # or per-query objects are gathered across ranks.
            epoch_values = torch.tensor(
                [float(epoch_loss_sum.detach()), float(epoch_tiles), float(epoch_queries), float(epoch_candidates), float(epoch_grad_sum.detach())],
                dtype=torch.float64, device=device,
            )
            category_values = torch.tensor([float(epoch_categories.get(category, 0)) for category in CATEGORIES], dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(epoch_values, op=dist.ReduceOp.SUM)
                dist.all_reduce(category_values, op=dist.ReduceOp.SUM)
            elapsed = max(1e-9, time.perf_counter() - epoch_started)
            global_epoch_tiles = int(round(float(epoch_values[1].item())))
            global_epoch_queries = int(round(float(epoch_values[2].item())))
            global_epoch_candidates = int(round(float(epoch_values[3].item())))
            summary = {
                "format": "locatemot-r0-grouped-epoch-summary-v2",
                "epoch": int(epoch),
                "optimizer_steps": int(optimizer_steps_epoch),
                "global_tiles": global_epoch_tiles,
                "query_exposures": global_epoch_queries,
                "mean_Q": float(global_epoch_queries / max(1, global_epoch_tiles)),
                "category_counts": {category: int(round(float(category_values[index].item()))) for index, category in enumerate(CATEGORIES)},
                "mean_loss": float(epoch_values[0].item() / max(1, global_epoch_tiles)),
                "last_loss": float(epoch_loss_sum.detach().item() / max(1, epoch_tiles)),
                "mean_grad_norm": float(epoch_values[4].item() / max(1, world_size * optimizer_steps_epoch)),
                "wall_seconds": elapsed,
                "tiles_per_second_global": float(global_epoch_tiles / elapsed),
                "queries_per_second_global": float(global_epoch_queries / elapsed),
                "peak_memory_per_rank": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "requested_global_tiles": int(args.global_tiles_per_epoch),
                "actual_global_tiles_after_padding": int(schedule_info["actual_global_tiles_after_padding"]),
                "padding_tiles": int(schedule_info["padding_tiles"]),
                "local_tiles_per_rank": int(len(tiles)),
                "candidate_rows_retained": True,
                "candidate_deletion": False,
                "candidate_truncation": False,
                "finite": True,
            }
            if is_main:
                assert summary_handle is not None
                summary_handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
                summary_handle.flush()
                epoch_summaries.append(summary)
            if optimizer_step >= 50 and first50 is None:
                first50 = {
                    "world_size": int(world_size),
                    "accumulation": int(accumulation),
                    "optimizer_steps_seen": int(optimizer_step),
                    "global_microtiles_seen": int(global_microtiles_seen),
                    "query_exposures_seen": int(query_exposures_seen),
                    "mean_Q": float(global_epoch_queries / max(1, global_epoch_tiles)),
                    "tiles_per_second_global": float(summary["tiles_per_second_global"]),
                    "queries_per_second_global": float(summary["queries_per_second_global"]),
                    "peak_memory_per_rank": summary["peak_memory_per_rank"],
                    "gpu_utilization_sample": gpu_utilization(local_rank) if is_main else None,
                    "loss_finite": True,
                    "grad_norm_finite": True,
                }
            final_epoch = epoch
            if epoch in CHECKPOINT_EPOCHS or epoch == 1:
                if world_size > 1:
                    dist.barrier()
                if is_main:
                    checkpoints.append(checkpoint(
                        model, optimizer, scheduler, out, epoch, optimizer_step,
                        global_microtiles_seen, query_exposures_seen, str(args.dataset),
                        model_config, input_hashes, args, world_size, accumulation,
                        optimizer_steps_epoch,
                        preview_only=(epoch == 1),
                    ))
                if world_size > 1:
                    dist.barrier()
            if is_main:
                gc.collect()
            if world_size > 1:
                dist.barrier()
        if loss_handle is not None:
            loss_handle.close()
        if summary_handle is not None:
            summary_handle.close()
        runtime.close()
        if is_main:
            existing_checkpoint_paths = []
            for path in sorted((out / "checkpoints").glob("checkpoint_r0_*.pt")):
                existing_checkpoint_paths.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
            complete = final_epoch >= int(args.epochs)
            payload = {
                **base,
                "status": "complete" if complete else "paused_after_epoch",
                "completed_epoch": int(final_epoch),
                "optimizer_step": int(optimizer_step),
                "global_microtiles_seen": int(global_microtiles_seen),
                "query_exposures_seen": int(query_exposures_seen),
                "model_config": model_config,
                "model_parameter_count": sum(parameter.numel() for parameter in model_core.parameters()),
                "trainable_parameter_count": sum(parameter.numel() for _name, parameter in trainable),
                "trainable_parameter_names": [name for name, _parameter in trainable],
                "input_fingerprints": input_hashes,
                "schedule": sampling_summaries,
                "mean_queries_per_tile": float(sum(item["global_query_exposures"] for item in sampling_summaries) / max(1, sum(item["global_frame_tile_count"] for item in sampling_summaries))),
                "query_category_exposure": dict(sorted({key: sum(item["global_category_counts"].get(key, 0) for item in sampling_summaries) for key in CATEGORIES}.items())),
                "frame_tile_count": int(sum(item["global_frame_tile_count"] for item in sampling_summaries)),
                "query_exposure_count": int(sum(item["global_query_exposures"] for item in sampling_summaries)),
                "checkpoints": checkpoints,
                "checkpoint_files_present": existing_checkpoint_paths,
                "resume": resume_info,
                "epoch_summaries": str(summary_path.resolve()),
                "loss_trace": str(loss_path.resolve()),
                "finite_steps": int(optimizer_step),
                "nonzero_gradient_steps": int(optimizer_step),
                "first_50_optimizer_steps": first50,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "wall_seconds": time.perf_counter() - started,
                "no_persistent_raw_dense_cache_created": True,
                "same_class_hard_negative_metadata": "unavailable; all-negative target-bag fallback",
                "token_span_region_alignment": "UNALIGNED",
                "static_motion_alignment": "UNALIGNED",
                "primary_selection_eligible_epochs": list(CHECKPOINT_EPOCHS),
                "next_action": "run fast legal dev preview after epoch2, then resume to epoch12" if not complete else "score all six legal dev checkpoints and perform registered selection",
            }
            write_json(out / "sampling_trace.json", {
                "format": "locatemot-r0-grouped-sampling-trace-v2",
                "status": payload["status"],
                "schedule": sampling_summaries,
                "category_quota": [[name, int(quota)] for name, quota in CATEGORY_QUOTA],
                "candidate_rows_retained": True,
                "candidate_deletion": False,
                "candidate_truncation": False,
            })
            write_json(out / "metrics.json", payload)
            write_json(out / "provenance.json", payload | {"format": "locatemot-r0-grouped-ddp-provenance-v2"})
            write_json(out / "status.json", payload | standard_flags(training_run=True))
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        if is_main:
            (out / "INCOMPLETE.md").write_text("# R0 grouped training — INCOMPLETE\n\n" + trace, encoding="utf-8")
            payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                       "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                       "wall_seconds": time.perf_counter() - started}
            write_json(out / "provenance.json", payload)
            write_json(out / "status.json", payload)
        else:
            try:
                (out / f"rank_{rank}_INCOMPLETE.md").write_text(trace, encoding="utf-8")
            except Exception:
                pass
        return 2
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2"), required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--language-root", type=Path, action="append", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--global-tiles-per-epoch", type=int, default=4000)
    parser.add_argument("--max-queries-per-tile", type=int, default=8)
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--contract-smoke-microtiles", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.language_root is None:
        args.language_root = [Path(value) for value in DEFAULT_LANGUAGE_ROOTS]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
