#!/usr/bin/env python3
"""Train the L85 factorized RMOT sidecar on the compact label-free cache.

The cache is constructed before this process starts.  Fit labels are attached
only after a complete cached row group has been checked.  The script supports
the registered one-GPU smoke and the four-GPU full curriculum; it never loads
the detector or writes a full detector checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l85_full_rmot import L85Config, L85FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore, key_only, load_fit_units, load_full_unit_for_labels  # noqa: E402
from locatemot.rmot.l85_fullvideo_bank import EXPECTED_MANIFEST_SHA, MANIFEST, sha256_file  # noqa: E402
from locatemot.rmot.l85_losses import l85_loss  # noqa: E402

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
CACHE_FORMAT = "locatemot-l85-z1-semantic-cache-v1"
GROUP_FORMAT = "locatemot-l85-z1-semantic-group-v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def set_seed(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def setup_distributed(device_arg: str) -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("L85 DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_arg)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("requested CUDA but CUDA is unavailable")
            torch.cuda.set_device(device)
    return world, rank, local_rank, device


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def reduce_scalar(value: float, device: torch.device, world: int) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.cpu())


def load_cache_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.joinpath("manifest.jsonl").read_text().splitlines() if line.strip()]
    fit = [row for row in rows if row.get("partition") == "fit_group"]
    if len(fit) != 524 or len({str(row["group_key"]) for row in fit}) != len(fit):
        raise AssertionError(f"L85 fit group manifest drift: {len(fit)}")
    if any(row.get("candidate_deletion") or row.get("candidate_truncation") for row in fit):
        raise AssertionError("cache manifest contains deletion/truncation")
    for row in fit:
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    return fit


def balanced_order(rows: list[dict[str, Any]], epoch: int, seed: int) -> list[dict[str, Any]]:
    """Shuffle whole video blocks while interleaving V1/V2 at video granularity."""
    by_dataset_video: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset_video[(str(row["dataset"]), str(row["video"]))].append(row)
    rng = np.random.default_rng(int(seed) + int(epoch) * 104729)
    blocks: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for dataset, video in by_dataset_video:
        blocks[dataset].append((dataset, video))
    for dataset in blocks:
        rng.shuffle(blocks[dataset])
        for key in blocks[dataset]:
            by_dataset_video[key] = sorted(by_dataset_video[key], key=lambda x: (int(x["frame_id"]), str(x["group_key"])))
    order: list[dict[str, Any]] = []
    position = 0
    while blocks.get("refer_kitti_v1") or blocks.get("refer_kitti_v2"):
        for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
            if blocks.get(dataset):
                key = blocks[dataset].pop(0)
                for row in by_dataset_video[key]:
                    copy_row = dict(row)
                    copy_row["schedule_position"] = position
                    copy_row["epoch"] = int(epoch)
                    copy_row["schedule_kind"] = "domain_balanced_video_block"
                    order.append(copy_row)
                    position += 1
    if len(order) != len(rows) or {str(x["group_key"]) for x in order} != {str(x["group_key"]) for x in rows}:
        raise AssertionError("balanced group order is not one-to-one")
    return order


def shard_order(order: list[dict[str, Any]], rank: int, world: int) -> list[dict[str, Any]]:
    if world == 1:
        return order
    size = (len(order) + world - 1) // world
    start = min(rank * size, len(order))
    values = order[start:start + size]
    while len(values) < size:
        values.append(order[(len(values) + rank * size) % len(order)])
    return values


def history_for_stage(batch: Any, stage: str, max_length: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep a causal prefix, optionally limiting the visible history to four."""
    history = torch.zeros_like(batch.history_observations)
    mask = torch.zeros_like(batch.history_mask)
    frames = torch.full_like(batch.history_frame_ids, -1)
    limit = 1 if stage == "S" else 4
    for row in range(batch.candidate_count):
        valid = torch.nonzero(batch.history_mask[row], as_tuple=False).flatten().tolist()
        valid = valid[-limit:]
        if valid:
            length = len(valid)
            history[row, :length] = batch.history_observations[row, valid]
            mask[row, :length] = True
            frames[row, :length] = batch.history_frame_ids[row, valid]
    if bool((frames[mask] > int(batch.frame_id)).any()):
        raise AssertionError(f"future history selected for {batch.unit_key}")
    return history, mask, frames


def row_digest(row_keys: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(json.dumps([list(x) for x in row_keys], sort_keys=False).encode()).hexdigest()


def load_group_inputs(item: dict[str, Any], labels_by_key: dict[str, dict[str, Any]],
                      store: L80BankStore) -> tuple[Any, list[dict[str, Any]]]:
    query_keys = [str(x) for x in item["query_unit_keys"]]
    if len(query_keys) != int(item["z1"].shape[0]):
        raise AssertionError(f"cache query count drift: {item['group_key']}")
    batches = []
    labels = []
    for query_index, query_key in enumerate(query_keys):
        if query_key not in labels_by_key:
            raise KeyError(f"fit label missing for cached unit {query_key}")
        full = labels_by_key[query_key]
        batch = store.build_unit(key_only(full))
        if batch.row_offsets != [int(x) for x in item["row_offsets"]]:
            raise AssertionError(f"cache/bank row offsets drift: {query_key}")
        expected_digest = str(item["row_keys_digest"])
        audits = item.get("query_audits", [])
        if query_index < len(audits):
            expected_digest = str(audits[query_index]["row_key_digest"])
        if row_digest(batch.row_keys) != expected_digest:
            raise AssertionError(f"cache/bank key digest drift: {query_key}")
        if batch.candidate_count != int(item["candidate_count"]):
            raise AssertionError(f"cache/bank candidate count drift: {query_key}")
        batches.append(batch)
        labels.append(store.attach_labels(batch, full))
    first = batches[0]
    first_row_structure = [(key[0], key[1], key[3], key[4], key[5]) for key in first.row_keys]
    for batch in batches[1:]:
        row_structure = [(key[0], key[1], key[3], key[4], key[5]) for key in batch.row_keys]
        if row_structure != first_row_structure:
            raise AssertionError(f"query candidate row order drift in group {item['group_key']}")
    return first, labels


def checkpoint_payload(model: L85FullRMOT, optimizer: torch.optim.Optimizer, scheduler: Any,
                       epoch: int, step: int, args: argparse.Namespace, rank: int) -> dict[str, Any]:
    state = model.state_dict()
    return {
        "format": "locatemot-l85-full-rmot-checkpoint-v1", "stage": str(args.stage),
        "dataset": "refer_kitti_v1+refer_kitti_v2", "epoch": int(epoch), "step": int(step),
        "seed": int(args.seed), "rank": int(rank), "model_config": model.config.__dict__,
        "model_state_dict": {key: value.detach().cpu() for key, value in state.items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "detector_checkpoint_included": False, "raw_dense_cache_included": False,
        "manifest_sha256": sha256_file(MANIFEST), "selected_representation": "L84 Z1",
        "sampler_seed": int(args.seed), "curriculum": {"S_epochs": 8, "T_epochs": 12, "J_epochs": 20},
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
    }


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def strict_reload(path: Path, device: torch.device) -> dict[str, Any]:
    package = torch.load(path, map_location=device, weights_only=False)
    model = L85FullRMOT(L85Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict L85 reload mismatch {result}")
    model.eval()
    return {"strict": True, "missing_keys": [], "unexpected_keys": [],
            "step": int(package.get("step", -1)), "epoch": int(package.get("epoch", -1)),
            "parameter_count": int(sum(x.numel() for x in model.parameters()))}


def train(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    world, rank, local_rank, device = setup_distributed(args.device)
    is_main = rank == 0
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if world > 1 and world != 4:
            raise AssertionError(f"registered full run requires world_size=4, got {world}")
        if is_main:
            if out.exists() and any(out.iterdir()):
                raise FileExistsError(f"refusing nonempty L85 output {out}")
            out.mkdir(parents=True, exist_ok=True)
        barrier(world)
        out.mkdir(parents=True, exist_ok=True)
        set_seed(args.seed, rank)
        cache_root = (args.cache if args.cache.is_absolute() else ROOT / args.cache).resolve()
        cache_summary = json.loads((cache_root / "summary.json").read_text())
        if cache_summary.get("format") != CACHE_FORMAT or cache_summary.get("status") != "complete":
            raise AssertionError("L85 cache is not complete")
        rows = load_cache_manifest(cache_root)
        fit_rows = load_fit_units()
        labels_by_key = {str(row["unit_key"]): row for row in fit_rows}
        if len(labels_by_key) != 5314:
            raise AssertionError("fit label key count drift")
        config = L85Config(hidden=int(args.hidden), history_length=8)
        base_model = L85FullRMOT(config).to(device=device, dtype=torch.float32)
        for parameter in base_model.parameters():
            parameter.requires_grad_(True)
        model: torch.nn.Module = base_model
        if world > 1:
            model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        total_epochs = int(args.epochs) if args.epochs > 0 else 1
        total_steps_target = int(args.steps) if args.steps > 0 else None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=float(args.lr) * 0.1)
        store = L80BankStore(max_history=8)
        loss_trace: list[dict[str, Any]] = []
        sampling_trace: list[dict[str, Any]] = []
        checkpoint_paths: dict[str, str] = {}
        global_step = 0
        finite_steps = 0
        nonzero_steps = 0
        domain_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        group_count = len(rows)
        start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        try:
            model.train()
            for epoch_zero in range(total_epochs):
                epoch_number = epoch_zero + 1
                stage = "S" if epoch_number <= 8 else ("T" if epoch_number <= 20 else "J")
                order = balanced_order(rows, epoch_zero, args.seed)
                local_order = shard_order(order, rank, world)
                for item in local_order:
                    if total_steps_target is not None and global_step >= total_steps_target:
                        break
                    cache_item = torch.load(item["path"], map_location="cpu", weights_only=False)
                    if cache_item.get("format") != GROUP_FORMAT or cache_item.get("labels_in_cache", False):
                        raise AssertionError(f"invalid/labelful cache item {item['group_key']}")
                    first, label_list = load_group_inputs(cache_item, labels_by_key, store)
                    if int(cache_item["candidate_count"]) != first.candidate_count:
                        raise AssertionError(f"candidate count drift {item['group_key']}")
                    history, history_mask, history_frames = history_for_stage(first, stage)
                    z1 = cache_item["z1"].float().clone().to(device=device)
                    presence = torch.cat((cache_item["text_global"].float(), cache_item["frame_global"].float()), dim=-1).clone().to(device=device)
                    current = first.observations.float().clone().to(device=device)
                    history = history.float().clone().to(device=device)
                    history_mask = history_mask.clone().to(device=device)
                    history_frames = history_frames.clone().to(device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.bf16)):
                        output = model(z1, presence, current, history, history_mask, history_frames,
                                       int(first.frame_id), temporal_enabled=(stage != "S"))
                        labels = [entry["labels"] for entry in label_list]
                        masks = [entry["membership_mask"] for entry in label_list]
                        categories = [str(entry["category"]) for entry in label_list]
                        loss, parts = l85_loss(output, labels, masks, categories, current, history_mask,
                                               temporal_enabled=(stage != "S"))
                    if not bool(torch.isfinite(loss.float())):
                        raise FloatingPointError(f"nonfinite L85 loss at {item['group_key']}")
                    loss.backward()
                    finite_grad = True
                    any_nonzero = False
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            finite_grad = finite_grad and bool(torch.isfinite(parameter.grad.float()).all())
                            any_nonzero = any_nonzero or bool((parameter.grad.float().abs() > 0).any())
                    if not finite_grad or not any_nonzero:
                        raise FloatingPointError(f"invalid L85 gradient at {item['group_key']}")
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)))
                    optimizer.step()
                    global_step += 1
                    finite_steps += 1
                    nonzero_steps += int(any_nonzero)
                    for entry in label_list:
                        domain_counts[str(first.dataset)] += 1
                        category_counts[str(entry["category"])] += 1
                    if is_main:
                        loss_trace.append({"step": global_step, "epoch": epoch_number, "curriculum_stage": stage,
                                           **parts, "gradient_norm": grad_norm, "loss_finite": True,
                                           "gradient_finite": True, "gradient_nonzero": True,
                                           "candidate_count": first.candidate_count,
                                           "candidate_key_digest": row_digest(first.row_keys),
                                           "candidate_deletion": False, "candidate_truncation": False})
                        sampling_trace.append({"step": global_step, "epoch": epoch_number,
                                               "curriculum_stage": stage, "group_key": item["group_key"],
                                               "query_unit_keys": cache_item["query_unit_keys"],
                                               "dataset": first.dataset, "video": first.video,
                                               "frame_id": int(first.frame_id), "candidate_count": first.candidate_count,
                                               "positive_count": int(sum(x["positive_count"] for x in label_list)),
                                               "categories": categories, "schedule_position": int(item["schedule_position"]),
                                               "schedule_kind": item["schedule_kind"],
                                               "all_rows_retained": True, "candidate_deletion": False,
                                               "candidate_truncation": False, "labels_attached_after_cache": True})
                    del output, cache_item, label_list, first, z1, presence, current, history, history_mask, history_frames
                    if global_step % 64 == 0:
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                scheduler.step()
                if is_main and (args.epochs > 0 or global_step in {100}):
                    module = model.module if isinstance(model, DDP) else model
                    path = out / (f"checkpoint_l85_epoch{epoch_number:02d}.pt" if args.epochs > 0 else f"checkpoint_l85_step{global_step}.pt")
                    atomic_save(path, checkpoint_payload(module, optimizer, scheduler, epoch_number, global_step, args, rank))
                    checkpoint_paths[str(epoch_number if args.epochs > 0 else global_step)] = str(path)
                if total_steps_target is not None and global_step >= total_steps_target:
                    break
            if total_steps_target is not None and global_step != total_steps_target:
                raise AssertionError(f"step count mismatch {global_step} != {total_steps_target}")
        except Exception:
            if is_main:
                (out / "INCOMPLETE.md").write_text("# L85 training — INCOMPLETE\n\n" + traceback.format_exc() +
                                                     "\nThe partial output is retained; no screening/test/TrackEval action was run.\n")
            raise
        finally:
            store._store._bank = None
            store._store._text_cache = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        barrier(world)
        if is_main:
            module = model.module if isinstance(model, DDP) else model
            reload_audit = {key: strict_reload(Path(value), device) for key, value in checkpoint_paths.items()}
            elapsed = time.perf_counter() - start_time
            metrics = {
                "format": "locatemot-l85-training-metrics-v1", "status": "complete", "stage": str(args.stage),
                "dataset": "refer_kitti_v1+refer_kitti_v2", "command": " ".join([sys.executable] + sys.argv),
                "cwd": str(ROOT), "luna_thread": THREAD, "seed": int(args.seed), "world_size": world,
                "local_rank": local_rank, "steps": int(global_step), "epochs": int(args.epochs),
                "finite_steps": int(finite_steps), "nonzero_gradient_steps": int(nonzero_steps),
                "fit_units_available": 5314, "fit_groups_available": len(rows),
                "domains_seen": dict(domain_counts), "categories_seen": dict(category_counts),
                "model": module.parameter_report(), "optimizer": {"name": "AdamW", "lr": args.lr,
                    "weight_decay": args.weight_decay, "grad_clip": args.grad_clip, "scheduler": "CosineAnnealingLR"},
                "curriculum": {"S": "epochs 1-8, current observation only, temporal disabled",
                               "T": "epochs 9-20, causal last four observations",
                               "J": "epochs 21-40, causal last four observations plus temporal auxiliary"},
                "candidate_set": "complete L69 rows from native frame pointers; no top-k/NMS/deletion",
                "history_length": 8, "candidate_key_drift": 0, "candidate_deletion": False,
                "candidate_truncation": False, "future_history_rows": 0,
                "labels_source": "fit-only expression-level L49/L69 labels attached after label-free cache construction",
                "same_class_hard_negative_metadata": "unavailable; all-negative objectness fallback",
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
                "cache": str(cache_root), "cache_sha256": sha256_file(cache_root / "summary.json"),
                "manifest_sha256": sha256_file(MANIFEST), "checkpoint_paths": checkpoint_paths,
                "checkpoint_reload": reload_audit, "elapsed_sec": elapsed,
                "throughput_groups_per_sec": float(global_step / max(elapsed, 1e-9)),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            }
            write_json(out / ("metrics_l85_step100.json" if args.steps > 0 else "metrics_l85_training.json"), metrics)
            write_json(out / "config.json", {"format": "locatemot-l85-training-config-v1", "stage": args.stage,
                "dataset": "refer_kitti_v1+refer_kitti_v2", "seed": args.seed, "steps": global_step,
                "epochs": args.epochs, "world_size": world, "model_config": module.config.__dict__,
                "cache": str(cache_root), "query_tile": 32, "bf16": bool(args.bf16),
                "curriculum": {"S_epochs": 8, "T_epochs": 12, "J_epochs": 20},
                "loss_weights": {"semantic_rank_r_total": 1.0, "semantic_rank_r_static": 0.30,
                    "membership_s": 1.0, "presence_b": 0.50, "null_rank": 0.50, "temporal": 0.10},
                "candidate_set": "complete; no top-k/NMS/deletion", "screening_gt_used": False,
                "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                "hota_trackeval_run": False})
            (out / "loss_trace.json").write_text(json.dumps(loss_trace, indent=2, ensure_ascii=False) + "\n")
            (out / "sampling_trace.json").write_text(json.dumps(sampling_trace, indent=2, ensure_ascii=False) + "\n")
            write_json(out / "reload_audit.json", {"format": "locatemot-l85-reload-audit-v1", "status": "complete",
                "strict": True, "checkpoints": reload_audit, "missing_keys": [], "unexpected_keys": []})
            write_json(out / "provenance.json", {"format": "locatemot-l85-training-provenance-v1", "status": "complete",
                "command": " ".join([sys.executable] + sys.argv), "cwd": str(ROOT), "luna_thread": THREAD,
                "inputs": {"cache": str(cache_root), "cache_summary_sha256": sha256_file(cache_root / "summary.json"),
                           "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
                           "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST)},
                "outputs": list(checkpoint_paths.values()), "label_attachment": "after complete label-free cache item",
                "fit_only": True, "detector_loaded": False, "detector_checkpoint_copied": False,
                "raw_dense_cache_written": False, "candidate_deletion": False, "candidate_truncation": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED"})
            write_json(out / "status.json", {"format": "locatemot-l85-status-v1", "status": "complete",
                "stage": str(args.stage), "outputs": list(checkpoint_paths.values()),
                "failure_root_cause": None, "next_action": "run calibration-only dev selection and legal full-video validation",
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        barrier(world)
        return metrics if is_main else {"status": "complete", "rank": rank, "steps": global_step}
    except Exception:
        if is_main and out.exists() and not (out / "INCOMPLETE.md").exists():
            (out / "INCOMPLETE.md").write_text("# L85 training — INCOMPLETE\n\n" + traceback.format_exc() + "\n")
        raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", default="full-joint40")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.steps > 0 and args.epochs not in (0, 1):
        args.epochs = 1
    result = train(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
