#!/usr/bin/env python3
"""Registered 40-epoch L86 V1/V2 full-RMOT training run.

The script trains only the new L86 factorized head.  L69 observations and the
label-free L85 Z1 cache are read-only; all supervision is attached by the
L86ClipStore after a complete causal frame/clip has been constructed.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DEFAULT_CACHE = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l86_full_rmot import L86Config, L86FullRMOT  # noqa: E402
from locatemot.rmot.l86_clip_data import L86ClipStore  # noqa: E402
from locatemot.rmot.l86_losses import l86_loss  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def init_dist() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested without CUDA")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        requested = os.environ.get("L86_DEVICE", "cuda:0")
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(device)
    return world, rank, local, device


def unwrap(model: torch.nn.Module) -> L86FullRMOT:
    return model.module if hasattr(model, "module") else model  # type: ignore[return-value]


def all_reduce_mean(value: float, device: torch.device, world: int) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(world)
    return float(tensor.cpu())


def all_reduce_sum(value: int, device: torch.device, world: int) -> int:
    tensor = torch.tensor(int(value), dtype=torch.int64, device=device)
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.cpu())


def phase_for_epoch(epoch: int) -> tuple[str, bool]:
    if epoch <= 8:
        return "S", False
    if epoch <= 20:
        return "T", True
    return "J", True


def grad_norms(model: torch.nn.Module) -> tuple[float, int, bool]:
    total = 0.0; nonzero = 0; finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        norm = float(value.norm())
        finite = finite and bool(torch.isfinite(value).all())
        total += norm
        nonzero += int(norm > 0.0)
    return total, nonzero, finite


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, step: int,
                    args: argparse.Namespace, world: int, phase: str) -> dict[str, Any]:
    core = unwrap(model)
    package = {
        "format": "locatemot-l86-checkpoint-v1", "model_config": asdict(core.config),
        "model_state_dict": {key: value.detach().cpu() for key, value in core.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch), "step": int(step), "phase": phase,
        "seed": SEED, "world_size": int(world), "args": vars(args),
        "manifest_sha256": MANIFEST_SHA, "z1_representation_changed": False,
        "groundingdino_lora_used": False, "candidate_deletion": False,
        "candidate_truncation": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(package, temporary)
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "epoch": int(epoch), "step": int(step), "phase": phase}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l86/train/joint40")
    parser.add_argument("--effective-clip-batch", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L86 train output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    world = rank = local = 0; device = torch.device("cpu")
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if int(args.epochs) != 40:
            raise AssertionError("L86 registered run requires exactly 40 epochs")
        world, rank, local, device = init_dist(); set_seed(int(args.seed), rank)
        if rank == 0:
            write_json(out / "config.json", {
                "format": "locatemot-l86-training-config-v1", "status": "running",
                "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                "epochs": 40, "seed": int(args.seed), "world_size": world,
                "effective_clip_batch_requested": int(args.effective_clip_batch),
                "accumulation_rule": {"world4": 2, "world3": 3, "world2": 4, "world1": 8},
                "bf16": bool(args.bf16), "cache": str(args.cache.resolve()),
                "cache_sha256": sha256_file(args.cache / "summary.json"),
                "manifest_sha256": MANIFEST_SHA, "curriculum": {"S": [1, 8], "T": [9, 20], "J": [21, 40]},
                "labels_in_optimization": "L49 fit only; no calibration/validation/screening/official-test labels",
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "candidate_deletion": False, "candidate_truncation": False,
                "z1_representation_changed": False, "groundingdino_lora_used": False,
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            })
        store = L86ClipStore(args.cache.resolve(), load_cache_into_ram=True)
        keys = [str(value) for value in store.train_keys]
        if len(keys) != 524:
            raise AssertionError(f"L82 training group count drift: {len(keys)}")
        accumulation = {1: 8, 2: 4, 3: 3, 4: 2}.get(world, max(1, round(8 / world)))
        local_count = (len(keys) + world - 1) // world
        steps_per_epoch = math.ceil(local_count / accumulation)
        total_steps = steps_per_epoch * int(args.epochs)
        model_core = L86FullRMOT(L86Config()).to(device=device, dtype=torch.float32)
        if world > 1:
            model: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
                model_core, device_ids=[local], output_device=local, broadcast_buffers=False,
                find_unused_parameters=False)
        else:
            model = model_core
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2, betas=(.9, .999))
        warmup_steps = max(1, int(round(total_steps * .05)))
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(1e-8, float(step + 1) / float(warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return .5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        trace: list[dict[str, Any]] = []
        sampling_trace: list[dict[str, Any]] = []
        global_step = 0; optimizer_step = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, int(args.epochs) + 1):
            phase, temporal_enabled = phase_for_epoch(epoch)
            schedule = list(keys)
            random.Random(int(args.seed) + epoch).shuffle(schedule)
            pad = (-len(schedule)) % world
            if pad:
                schedule.extend(schedule[:pad])
            local_keys = schedule[rank::world]
            epoch_total = 0.0; epoch_steps = 0; epoch_finite = True
            category_counts = {"positive": 0, "multi_positive": 0, "inactive": 0, "present_uncovered": 0}
            domain_counts = {"refer_kitti_v1": 0, "refer_kitti_v2": 0}
            positive_count = negative_count = temporal_pairs = masked_missing = 0
            nonzero_grad_steps = 0
            for local_index, anchor in enumerate(local_keys):
                clip = store.build_clip(anchor, temporal_enabled=temporal_enabled, clip_length=4)
                current = clip[-1]
                current_outputs: dict[str, torch.Tensor]
                previous_outputs: list[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]] = []
                amp_enabled = bool(args.bf16 and device.type == "cuda")
                amp_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled)
                with amp_context:
                    current_outputs = model(
                        current.z1.to(device), current.text_global.to(device), current.frame_global.to(device),
                        current.current_observation.to(device), current.history_observations.to(device),
                        current.history_mask.to(device), current.history_frame_ids.to(device), current.frame_id,
                        temporal_enabled=temporal_enabled)
                    for previous in clip[:-1]:
                        previous_outputs.append((model(
                            previous.z1.to(device), previous.text_global.to(device), previous.frame_global.to(device),
                            previous.current_observation.to(device), previous.history_observations.to(device),
                            previous.history_mask.to(device), previous.history_frame_ids.to(device), previous.frame_id,
                            temporal_enabled=temporal_enabled), previous.labels))
                    loss, parts = l86_loss(current_outputs, current.labels, current.current_observation.to(device),
                                           previous_outputs, temporal_enabled=temporal_enabled)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"nonfinite loss at epoch={epoch} group={anchor}")
                    (loss / float(accumulation)).backward()
                epoch_total += float(loss.detach()); epoch_steps += 1; global_step += 1
                epoch_finite = epoch_finite and bool(torch.isfinite(loss.detach()))
                positive_count += int(parts["positive_count"]); negative_count += int(parts["negative_target_bags"])
                temporal_pairs += int(parts["positive_pairs"]); masked_missing += int(parts["masked_missing_count"])
                domain_counts[str(current.dataset)] = domain_counts.get(str(current.dataset), 0) + 1
                for label in current.labels:
                    category_counts[str(label["category"])] = category_counts.get(str(label["category"]), 0) + 1
                should_step = ((local_index + 1) % accumulation == 0) or (local_index + 1 == len(local_keys))
                if should_step:
                    grad_norm, nonzero, grad_finite = grad_norms(model)
                    if not grad_finite or not math.isfinite(grad_norm):
                        raise FloatingPointError(f"nonfinite gradient at epoch={epoch}, group={anchor}")
                    if nonzero == 0:
                        raise FloatingPointError(f"zero gradient at epoch={epoch}, group={anchor}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                    optimizer_step += 1; nonzero_grad_steps += 1
                del current_outputs, previous_outputs, clip
                gc.collect()
                if device.type == "cuda" and local_index % 16 == 0:
                    torch.cuda.empty_cache()
            reduced_loss = all_reduce_mean(epoch_total / max(1, epoch_steps), device, world)
            reduced_steps = all_reduce_sum(epoch_steps, device, world)
            reduced_pos = all_reduce_sum(positive_count, device, world)
            reduced_neg = all_reduce_sum(negative_count, device, world)
            entry = {
                "epoch": epoch, "phase": phase, "temporal_enabled": temporal_enabled,
                "loss_mean": reduced_loss, "local_groups": epoch_steps, "global_group_updates": reduced_steps,
                "positive_rows": reduced_pos, "negative_target_bags": reduced_neg,
                "temporal_identity_pairs": all_reduce_sum(temporal_pairs, device, world),
                "masked_missing_count": all_reduce_sum(masked_missing, device, world),
                "finite": epoch_finite, "optimizer_steps": optimizer_step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "category_counts_local": category_counts, "domain_counts_local": domain_counts,
                "world_size": world, "effective_clip_batch": world * accumulation,
            }
            if rank == 0:
                trace.append(entry)
                sampling_trace.append({"epoch": epoch, "phase": phase, "category_counts": category_counts,
                                       "domain_counts": domain_counts, "groups": len(keys),
                                       "all_candidate_rows": True, "candidate_deletion": False,
                                       "candidate_truncation": False})
                print(json.dumps(entry, sort_keys=True), flush=True)
                if epoch % 2 == 0 or epoch in (8, 20, 40):
                    save_checkpoint(out / f"checkpoint_l86_epoch{epoch:03d}.pt", model, optimizer, scheduler,
                                    epoch, optimizer_step, args, world, phase)
            if world > 1:
                dist.barrier()
        if rank == 0:
            final_ckpt = out / "checkpoint_l86_step40epoch.pt"
            selected_info = save_checkpoint(final_ckpt, model, optimizer, scheduler, 40, optimizer_step, args, world, "J")
            write_json(out / "loss_trace.json", trace); write_json(out / "sampling_trace.json", sampling_trace)
            write_json(out / "config.json", json.loads((out / "config.json").read_text()) | {
                "status": "complete", "wall_seconds": time.perf_counter() - started,
                "checkpoint_count": len(list(out.glob("checkpoint_l86_epoch*.pt"))) + 1,
                "final_checkpoint": selected_info, "optimizer_steps": optimizer_step,
                "actual_world_size": world, "effective_clip_batch": world * accumulation,
            })
            write_json(out / "provenance.json", {
                "format": "locatemot-l86-training-provenance-v1", "status": "complete",
                "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                "manifest_sha256": MANIFEST_SHA, "cache": str(args.cache.resolve()),
                "cache_summary_sha256": sha256_file(args.cache / "summary.json"),
                "model_parameters": unwrap(model).parameter_report(), "epochs": 40,
                "curriculum": {"S": [1, 8], "T": [9, 20], "J": [21, 40]},
                "seed": int(args.seed), "world_size": world, "effective_clip_batch": world * accumulation,
                "all_candidate_rows": True, "candidate_deletion": False, "candidate_truncation": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "z1_representation_changed": False, "groundingdino_lora_used": False,
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
                "wall_seconds": time.perf_counter() - started, "final_checkpoint": selected_info,
            })
            write_json(out / "status.json", {"format": "locatemot-l86-training-v1", "status": "complete",
                                             "final_checkpoint": selected_info, "epochs": 40,
                                             "world_size": world, "screening_gt_used": False,
                                             "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                             "hota_trackeval_run": False})
        if world > 1:
            dist.barrier(); dist.destroy_process_group()
        return 0
    except Exception:
        trace = traceback.format_exc()
        if rank == 0:
            (out / "INCOMPLETE.md").write_text("# L86 full training — INCOMPLETE\n\n" + trace)
            write_json(out / "status.json", {"format": "locatemot-l86-training-v1", "status": "incomplete",
                                             "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                                             "failure_root_cause": "first traceback in INCOMPLETE.md",
                                             "screening_gt_used": False, "official_test_labels_read": False,
                                             "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
