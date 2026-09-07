#!/usr/bin/env python3
"""Registered L89 QSC-D full-RMOT training run.

The only newly trainable representation path is the QSC-D candidate-set
decoder in :class:`L89FullRMOT`.  L85 Z1 states, L69 observations, and the
pure language cache are read-only.  The L87-A objective is imported unchanged
so the temporal/prior/presence/NULL contract remains attributable to the
L89 set decoder.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
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
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DEFAULT_Z1 = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
DEFAULT_LANG = WORK_ROOT / "outputs/l89/cache/language_tokens_retry1"

for path in (WORK_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import locatemot.models as _models_package  # noqa: E402
import locatemot.rmot as _rmot_package  # noqa: E402

for package, name in ((_models_package, "models"), (_rmot_package, "rmot")):
    package_path = str(WORK_ROOT / "locatemot" / name)
    if package_path not in [str(value) for value in package.__path__]:
        package.__path__.append(package_path)

from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT  # noqa: E402
from locatemot.rmot.l86_clip_data import L86ClipStore  # noqa: E402
from locatemot.rmot.l87a_losses import l87a_loss  # noqa: E402
from locatemot.rmot.l89_language_cache import L89LanguageTokenCache  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def init_dist() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("L89 DDP requires CUDA")
        torch.cuda.set_device(local)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local)
    else:
        requested = os.environ.get("L89_DEVICE", "cuda:0")
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(device)
    return world, rank, local, device


def unwrap(module: torch.nn.Module) -> L89FullRMOT:
    return module.module if hasattr(module, "module") else module  # type: ignore[return-value]


def reduce_float(value: float, device: torch.device, world: int) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(world)
    return float(tensor.cpu())


def reduce_int(value: int, device: torch.device, world: int) -> int:
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


def language_for(cache: L89LanguageTokenCache, store: L86ClipStore, frame: Any,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    query_rows = store.groups[str(frame.group_key)]["queries"]
    by_qid = {int(row["query_id"]): str(row["sentence"]) for row in query_rows}
    sentences = [by_qid[int(qid)] for qid in frame.query_ids]
    return cache.get_batch(frame.dataset, frame.video, frame.query_ids, sentences, device)


def grad_stats(module: torch.nn.Module) -> tuple[float, int, bool]:
    total = 0.0
    nonzero = 0
    finite = True
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        norm = float(value.norm())
        total += norm
        nonzero += int(norm > 0.0)
        finite = finite and bool(torch.isfinite(value).all())
    return total, nonzero, finite


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int,
                    optimizer_step: int, args: argparse.Namespace, world: int,
                    phase: str, z1_cache: Path, language_cache: Path) -> dict[str, Any]:
    core = unwrap(model)
    package = {
        "format": "locatemot-l89-checkpoint-v1",
        "model_config": asdict(core.config),
        "model_state_dict": {key: value.detach().cpu() for key, value in core.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch), "optimizer_step": int(optimizer_step), "phase": str(phase),
        "seed": int(args.seed), "world_size": int(world), "args": vars(args),
        "manifest_sha256": MANIFEST_SHA,
        "z1_cache_summary_sha256": sha256_file(z1_cache / "summary.json"),
        "language_cache_summary_sha256": sha256_file(language_cache / "summary.json"),
        "z1_representation_changed": False, "groundingdino_lora_used": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "source_pool_group_query_ids_as_inputs": False,
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(package, temporary)
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "epoch": int(epoch),
            "optimizer_step": int(optimizer_step), "phase": str(phase)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--z1-cache", type=Path, default=DEFAULT_Z1)
    parser.add_argument("--language-cache", type=Path, default=DEFAULT_LANG)
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "outputs/l89/train/joint40")
    parser.add_argument("--effective-frame-batch", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 training output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    world = rank = local = 0
    device = torch.device("cpu")
    store = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89 worktree cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if int(args.seed) != SEED or int(args.epochs) != 40:
            raise AssertionError("L89 registered seed/epoch contract changed")
        if not (args.z1_cache.resolve() / "summary.json").is_file():
            raise FileNotFoundError(args.z1_cache)
        if not (args.language_cache.resolve() / "summary.json").is_file():
            raise FileNotFoundError(args.language_cache)
        world, rank, local, device = init_dist()
        set_seed(int(args.seed), rank)
        if rank == 0:
            write_json(out / "config.json", {
                "format": "locatemot-l89-training-config-v1", "status": "running",
                "command": command, "cwd": str(WORK_ROOT), "asset_root": str(ROOT),
                "luna_thread": THREAD, "epochs": 40, "seed": int(args.seed),
                "world_size": world, "effective_frame_batch_requested": int(args.effective_frame_batch),
                "accumulation_rule": {"world4": 2, "world3": 3, "world2": 4, "world1": 8},
                "bf16": bool(args.bf16), "z1_cache": str(args.z1_cache.resolve()),
                "language_cache": str(args.language_cache.resolve()),
                "z1_cache_summary_sha256": sha256_file(args.z1_cache.resolve() / "summary.json"),
                "language_cache_summary_sha256": sha256_file(args.language_cache.resolve() / "summary.json"),
                "manifest_sha256": MANIFEST_SHA,
                "qsc_d": {"dim": 256, "heads": 8, "layers": 2, "ffn_dim": 1024, "dropout": 0.10},
                "curriculum": {"S": [1, 8], "T": [9, 20], "J": [21, 40]},
                "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-2, "betas": [0.9, 0.999], "clip_norm": 1.0},
                "labels_in_optimization": "L49 fit groups only; no calibration/validation/screening/official-test labels",
                "same_class_hard_negative_metadata": "unavailable; unchanged L87-A all-negative target-bag fallback",
                "all_candidate_rows": True, "candidate_deletion": False, "candidate_truncation": False,
                "z1_representation_changed": False, "groundingdino_lora_used": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            })
        store = L86ClipStore(args.z1_cache.resolve(), load_cache_into_ram=True)
        language = L89LanguageTokenCache(args.language_cache.resolve())
        keys = [str(value) for value in store.train_keys]
        if len(keys) != 524:
            raise AssertionError(f"L89 train group count drift: {len(keys)}")
        accumulation = {1: 8, 2: 4, 3: 3, 4: 2}.get(world, max(1, round(8 / max(world, 1))))
        local_count = (len(keys) + world - 1) // world
        steps_per_epoch = math.ceil(local_count / accumulation)
        total_steps = steps_per_epoch * int(args.epochs)
        model_core = L89FullRMOT(L89Config()).to(device=device, dtype=torch.float32)
        parameter_report = model_core.parameter_report()
        if world > 1:
            model: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
                model_core, device_ids=[local], output_device=local,
                broadcast_buffers=False, find_unused_parameters=False,
            )
        else:
            model = model_core
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2, betas=(0.9, 0.999))
        warmup = max(1, int(round(total_steps * 0.05)))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return max(1e-8, float(step + 1) / float(warmup))
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        trace: list[dict[str, Any]] = []
        sampling_trace: list[dict[str, Any]] = []
        optimizer_step = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, int(args.epochs) + 1):
            phase, temporal_enabled = phase_for_epoch(epoch)
            schedule = list(keys)
            random.Random(int(args.seed) + epoch).shuffle(schedule)
            padding = (-len(schedule)) % max(world, 1)
            if padding:
                schedule.extend(schedule[:padding])
            local_keys = schedule[rank::max(world, 1)]
            epoch_loss = 0.0
            epoch_groups = 0
            local_finite = True
            category_counts = {"positive": 0, "multi_positive": 0, "inactive": 0, "present_uncovered": 0}
            domain_counts = {"refer_kitti_v1": 0, "refer_kitti_v2": 0}
            positive_rows = negative_bags = masked_missing = temporal_pairs = 0
            nonzero_updates = 0
            for local_index, anchor in enumerate(local_keys):
                clip = store.build_clip(anchor, temporal_enabled=temporal_enabled, clip_length=4)
                current = clip[-1]
                current_tokens, current_mask = language_for(language, store, current, device)
                previous_outputs: list[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]] = []
                amp_enabled = bool(args.bf16 and device.type == "cuda")
                amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled)
                with amp:
                    current_output = model(
                        current.z1.to(device), current_tokens, current_mask,
                        current.text_global.to(device), current.frame_global.to(device),
                        current.current_observation.to(device), current.history_observations.to(device),
                        current.history_mask.to(device), current.history_frame_ids.to(device), current.frame_id,
                        temporal_enabled=temporal_enabled,
                    )
                    for previous in clip[:-1]:
                        previous_tokens, previous_mask = language_for(language, store, previous, device)
                        previous_output = model(
                            previous.z1.to(device), previous_tokens, previous_mask,
                            previous.text_global.to(device), previous.frame_global.to(device),
                            previous.current_observation.to(device), previous.history_observations.to(device),
                            previous.history_mask.to(device), previous.history_frame_ids.to(device), previous.frame_id,
                            temporal_enabled=temporal_enabled,
                        )
                        previous_outputs.append((previous_output, previous.labels))
                    loss, parts = l87a_loss(
                        current_output, current.labels, current.current_observation.to(device),
                        previous_outputs, temporal_enabled=temporal_enabled,
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"nonfinite loss epoch={epoch} group={anchor}")
                    (loss / float(accumulation)).backward()
                epoch_loss += float(loss.detach()); epoch_groups += 1
                local_finite = local_finite and bool(torch.isfinite(loss.detach()))
                positive_rows += int(parts["positive_count"])
                negative_bags += int(parts["negative_target_bags"])
                masked_missing += int(parts["masked_missing_count"])
                temporal_pairs += int(parts["positive_pairs"])
                domain_counts[str(current.dataset)] = domain_counts.get(str(current.dataset), 0) + 1
                for label in current.labels:
                    category = str(label["category"])
                    category_counts[category] = category_counts.get(category, 0) + 1
                should_step = (local_index + 1) % accumulation == 0 or local_index + 1 == len(local_keys)
                if should_step:
                    norm, nonzero, grad_finite = grad_stats(model)
                    if not grad_finite or not math.isfinite(norm):
                        raise FloatingPointError(f"nonfinite gradient epoch={epoch} group={anchor}")
                    if nonzero <= 0:
                        raise FloatingPointError(f"zero gradient epoch={epoch} group={anchor}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                    optimizer_step += 1; nonzero_updates += 1
                del current_output, previous_outputs, current_tokens, current_mask, clip
                gc.collect()
                if device.type == "cuda" and local_index % 16 == 0:
                    torch.cuda.empty_cache()
            entry = {
                "epoch": epoch, "phase": phase, "temporal_enabled": temporal_enabled,
                "loss_mean": reduce_float(epoch_loss / max(1, epoch_groups), device, world),
                "local_groups": epoch_groups, "global_group_updates": reduce_int(epoch_groups, device, world),
                "positive_rows": reduce_int(positive_rows, device, world),
                "negative_target_bags": reduce_int(negative_bags, device, world),
                "masked_missing_count": reduce_int(masked_missing, device, world),
                "temporal_identity_pairs": reduce_int(temporal_pairs, device, world),
                "finite": local_finite, "optimizer_steps": optimizer_step,
                "nonzero_gradient_updates": nonzero_updates, "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "category_counts_local": category_counts, "domain_counts_local": domain_counts,
                "world_size": world, "effective_frame_batch": world * accumulation,
            }
            if rank == 0:
                trace.append(entry)
                sampling_trace.append({"epoch": epoch, "phase": phase, "groups": len(keys),
                                       "category_counts_local": category_counts, "domain_counts_local": domain_counts,
                                       "all_candidate_rows": True, "candidate_deletion": False,
                                       "candidate_truncation": False, "fit_only": True})
                print(json.dumps(entry, sort_keys=True), flush=True)
                if epoch % 2 == 0 or epoch in (8, 20, 40):
                    save_checkpoint(out / f"checkpoint_l89_epoch{epoch:03d}.pt", model, optimizer, scheduler,
                                    epoch, optimizer_step, args, world, phase, args.z1_cache.resolve(), args.language_cache.resolve())
            if world > 1:
                dist.barrier()
        if rank == 0:
            final_info = save_checkpoint(out / "checkpoint_l89_step40epoch.pt", model, optimizer, scheduler,
                                         40, optimizer_step, args, world, "J", args.z1_cache.resolve(), args.language_cache.resolve())
            write_json(out / "loss_trace.json", trace)
            write_json(out / "sampling_trace.json", sampling_trace)
            write_json(out / "config.json", json.loads((out / "config.json").read_text()) | {
                "status": "complete", "wall_seconds": time.perf_counter() - started,
                "checkpoint_count": len(list(out.glob("checkpoint_l89_epoch*.pt"))) + 1,
                "final_checkpoint": final_info, "optimizer_steps": optimizer_step,
                "actual_world_size": world, "effective_frame_batch": world * accumulation,
                "model_parameters": parameter_report,
            })
            provenance = {
                "format": "locatemot-l89-training-provenance-v1", "status": "complete",
                "command": command, "cwd": str(WORK_ROOT), "asset_root": str(ROOT), "luna_thread": THREAD,
                "manifest_sha256": MANIFEST_SHA, "z1_cache": str(args.z1_cache.resolve()),
                "z1_cache_summary_sha256": sha256_file(args.z1_cache.resolve() / "summary.json"),
                "language_cache": str(args.language_cache.resolve()),
                "language_cache_summary_sha256": sha256_file(args.language_cache.resolve() / "summary.json"),
                "model_parameters": parameter_report, "epochs": 40,
                "curriculum": {"S": [1, 8], "T": [9, 20], "J": [21, 40]},
                "seed": int(args.seed), "world_size": world, "effective_frame_batch": world * accumulation,
                "all_candidate_rows": True, "candidate_deletion": False, "candidate_truncation": False,
                "z1_representation_changed": False, "groundingdino_lora_used": False,
                "same_class_hard_negative_metadata": "unavailable; unchanged L87-A all-negative target-bag fallback",
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
                "wall_seconds": time.perf_counter() - started, "final_checkpoint": final_info,
            }
            write_json(out / "provenance.json", provenance)
            write_json(out / "status.json", {"format": "locatemot-l89-training-v1", "status": "complete",
                                             "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                             "epochs": 40, "world_size": world, "final_checkpoint": final_info,
                                             "failure_root_cause": None, "next_action": "score legal fit/dev checkpoints",
                                             "screening_gt_used": False, "official_test_labels_read": False,
                                             "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        if world > 1:
            dist.barrier(); dist.destroy_process_group()
        return 0
    except Exception:
        trace = traceback.format_exc()
        if rank == 0:
            (out / "INCOMPLETE.md").write_text("# L89 full training — INCOMPLETE\n\n" + trace, encoding="utf-8")
            write_json(out / "status.json", {"format": "locatemot-l89-training-v1", "status": "incomplete",
                                             "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                             "failure_root_cause": "first traceback in INCOMPLETE.md",
                                             "screening_gt_used": False, "official_test_labels_read": False,
                                             "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        if store is not None:
            store.close()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
