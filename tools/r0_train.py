#!/usr/bin/env python3
"""Train one benchmark-specific R0 TCGH head on legal fit data.

The detector and all visual/language artifacts are frozen.  This script only
loads the already-built native-frame index, attaches its safe fit labels, and
optimizes the compact R0 head.  V1 and V2 are intentionally separate runs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    file_meta,
    model_forward,
    sha256_file,
    standard_flags,
    write_json,
)
from locatemot.models.r0_track_grounding import R0TrackGroundingHead  # noqa: E402
from locatemot.rmot.r0_losses import r0_total_loss  # noqa: E402


CATEGORIES = ("inactive", "positive", "multi_positive", "present_uncovered")
CHECKPOINT_EPOCHS = (2, 4, 6, 8, 10, 12)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_report(model: torch.nn.Module) -> dict[str, Any]:
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
    return {"total_norm": total, "nonzero_parameter_grads": nonzero,
            "finite": finite, "by_name": names}


def checkpoint(model: torch.nn.Module, out: Path, epoch: int, global_step: int,
               dataset: str, config: dict[str, Any]) -> dict[str, Any]:
    path = out / "checkpoints" / f"checkpoint_r0_{dataset}_epoch{epoch:02d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    package = {
        "format": "locatemot-r0-tcgh-checkpoint-v1",
        "dataset": dataset,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_config": config,
        "model_state_dict": state,
        "detector_state_included": False,
        "tracker_state_included": False,
        "training_seed": SEED,
    }
    torch.save(package, path)
    reloaded = R0TrackGroundingHead()
    result = reloaded.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict R0 checkpoint reload failed: {result}")
    reload_diff = 0.0
    for key, value in reloaded.state_dict().items():
        reload_diff = max(reload_diff, float((value - state[key]).abs().max()))
    if reload_diff != 0.0:
        raise AssertionError(f"R0 checkpoint state reload drift: {reload_diff}")
    del reloaded, state
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "epoch": int(epoch), "global_step": int(global_step),
            "strict_reload": True, "reload_max_abs_diff": reload_diff,
            "bytes": int(path.stat().st_size)}


def build_schedule(index: DenseIndex, epochs: int, steps_per_epoch: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets = {category: list(index.by_category.get(category, [])) for category in CATEGORIES}
    if any(not values for values in buckets.values()):
        raise AssertionError(f"R0 fit index lacks required strata: { {key: len(value) for key, value in buckets.items()} }")
    schedule: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for epoch in range(1, int(epochs) + 1):
        rng = random.Random(int(seed) + epoch * 1009)
        for values in buckets.values():
            rng.shuffle(values)
        cursors = {category: 0 for category in CATEGORIES}
        for local_step in range(int(steps_per_epoch)):
            category = CATEGORIES[local_step % len(CATEGORIES)]
            values = buckets[category]
            cursor = cursors[category]
            row = values[cursor % len(values)]
            cursors[category] += 1
            selected = dict(row)
            selected["epoch"] = epoch
            selected["step_in_epoch"] = local_step + 1
            schedule.append(selected)
            counts[category] += 1
    return schedule, {"category_counts": dict(sorted(counts.items())),
                      "schedule_seed": int(seed), "deterministic": True,
                      "category_order": list(CATEGORIES), "total_steps": len(schedule)}


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 train output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable, *sys.argv])
    base = {
        "format": "locatemot-r0-tcgh-training-v1", "status": "incomplete",
        "command": command, "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD,
        "seed": SEED, "dataset": str(args.dataset), "epochs": int(args.epochs),
        "steps_per_epoch": int(args.steps_per_epoch),
        "registered_total_steps": int(args.epochs) * int(args.steps_per_epoch),
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-2,
                       "warmup_fraction": 0.05, "schedule": "cosine", "grad_clip": 1.0},
        "bf16_requested": bool(args.bf16), "device": str(args.device),
        "inputs": {"dense_root": str(args.dense_root.resolve()),
                    "visual_cache": str(args.visual_cache.resolve()),
                    "language_roots": [str(Path(value).resolve()) for value in args.language_root],
                    "manifest_sha256": check_manifest()},
        "outputs": {"root": str(out)},
        "candidate_deletion": False, "candidate_truncation": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "failure_root_cause": None,
        "next_action": "score all legal dev rows and perform dev-only checkpoint shortlist",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 training cwd: {Path.cwd()}")
        set_seed(SEED)
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("R0 training requested CUDA but it is unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        dense = DenseIndex(args.dense_root)
        visual = VisualCacheIndex(args.visual_cache)
        language = MergedLanguageCache([Path(value) for value in args.language_root])
        schedule, schedule_info = build_schedule(dense, args.epochs, args.steps_per_epoch, SEED)
        model = R0TrackGroundingHead().to(device=device, dtype=torch.float32)
        config = model.config_dict()
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if len(trainable) != len(list(model.parameters())) or not trainable:
            raise AssertionError("R0 trainable parameter contract drift")
        optimizer = torch.optim.AdamW([parameter for _name, parameter in trainable], lr=1e-4, weight_decay=1e-2)
        total_steps = len(schedule)
        warmup_steps = max(1, int(round(total_steps * 0.05)))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        runtime = R0RuntimeData(dense, visual, language, device)
        loss_path = out / "loss_trace.jsonl"
        loss_handle = loss_path.open("w", encoding="utf-8")
        finite_steps = 0
        nonzero_steps = 0
        sampling_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        checkpoints: list[dict[str, Any]] = []
        last_loss = None
        try:
            for global_index, selected in enumerate(schedule):
                record = selected
                sampling_counts[str(record["category"])] += 1
                domain_counts[str(record["dataset"])] += 1
                sample = runtime.prepare(record, attach_labels=True)
                supervision = sample["supervision"]
                optimizer.zero_grad(set_to_none=True)
                autocast_enabled = bool(args.bf16 and device.type == "cuda")
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    output = model_forward(model, sample)
                    loss, info = r0_total_loss(
                        output["membership_logit"], output["coverage_presence_logit"],
                        [str(supervision["category"])], [supervision["target_ids"]],
                        [supervision["candidate_gt"]],
                    )
                if not bool(torch.isfinite(loss.float()).all()):
                    raise FloatingPointError(f"nonfinite R0 training loss at step {global_index + 1}")
                loss.backward()
                gradients = gradient_report(model)
                if not gradients["finite"] or gradients["nonzero_parameter_grads"] <= 0:
                    raise FloatingPointError(f"R0 gradient contract failed at step {global_index + 1}: {gradients}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                finite_steps += 1
                nonzero_steps += 1
                last_loss = float(loss.detach())
                loss_handle.write(json.dumps({
                    "step": global_index + 1, "epoch": int(record["epoch"]),
                    "step_in_epoch": int(record["step_in_epoch"]),
                    "dataset": record["dataset"], "video": record["video"],
                    "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]),
                    "category": record["category"], "candidate_count": int(record["candidate_count"]),
                    "positive_row_count": int(supervision["positive_row_count"]),
                    "covered_target_count": int(supervision["covered_target_count"]),
                    "loss": last_loss, "loss_info": info,
                    "gradient_total_norm": gradients["total_norm"],
                    "gradient_nonzero_parameter_count": gradients["nonzero_parameter_grads"],
                    "lr": float(optimizer.param_groups[0]["lr"]), "finite": True,
                    "candidate_rows_retained": True, "candidate_deletion": False,
                    "candidate_truncation": False,
                }, ensure_ascii=False, sort_keys=True) + "\n")
                if (global_index + 1) % 100 == 0:
                    loss_handle.flush()
                if int(record["step_in_epoch"]) == int(args.steps_per_epoch) and int(record["epoch"]) in CHECKPOINT_EPOCHS:
                    checkpoints.append(checkpoint(model, out, int(record["epoch"]), global_index + 1,
                                                   str(args.dataset), config))
                del sample, supervision, output, loss, info, gradients
                if device.type == "cuda" and (global_index + 1) % 32 == 0:
                    torch.cuda.empty_cache()
                if (global_index + 1) % 64 == 0:
                    gc.collect()
        finally:
            loss_handle.close()
            runtime.close()
        if finite_steps != total_steps or nonzero_steps != total_steps:
            raise AssertionError(f"R0 training step coverage drift: {finite_steps}/{total_steps}, {nonzero_steps}/{total_steps}")
        if len(checkpoints) != len(CHECKPOINT_EPOCHS):
            raise AssertionError(f"R0 checkpoint count drift: {len(checkpoints)}")
        model_norm = math.sqrt(sum(float(parameter.detach().float().pow(2).sum()) for parameter in model.parameters()))
        payload = {
            **base, "status": "complete", "model_config": config,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(parameter.numel() for _name, parameter in trainable),
            "trainable_parameter_names": [name for name, _parameter in trainable],
            "schedule": schedule_info, "sampling_counts": dict(sorted(sampling_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "finite_steps": finite_steps, "nonzero_gradient_steps": nonzero_steps,
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS), "checkpoints": checkpoints,
            "loss_trace": str(loss_path.resolve()), "final_loss": last_loss,
            "final_model_parameter_l2": model_norm, "language_cache_entries": language.entry_count,
            "visual_cache_frame_entries": len(visual.entries),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started,
            "no_persistent_raw_dense_cache_created": True,
            "same_class_hard_negative_metadata": "unavailable; all-negative target-bag fallback",
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "next_action": "score all legal dev rows and perform dev-only checkpoint shortlist",
        }
        write_json(out / "config.json", {"format": "locatemot-r0-train-config-v1", **base,
                                           "model_config": config, "schedule": schedule_info,
                                           "optimizer": base["optimizer"]})
        write_json(out / "sampling_trace.json", {"format": "locatemot-r0-sampling-trace-v1",
                                                   "status": "complete", "schedule": schedule_info,
                                                   "sampling_counts": dict(sorted(sampling_counts.items())),
                                                   "domain_counts": dict(sorted(domain_counts.items())),
                                                   "candidate_rows_retained": True})
        write_json(out / "metrics.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-r0-train-provenance-v1",
                                                         "input_metadata": {
                                                             "dense": file_meta(args.dense_root / "summary.json"),
                                                             "visual_manifest": file_meta(args.visual_cache / "manifest.jsonl"),
                                                         }})
        write_json(out / "status.json", payload | standard_flags(training_run=True))
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 training — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2"), required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--language-root", type=Path, action="append", default=[Path(value) for value in DEFAULT_LANGUAGE_ROOTS])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
