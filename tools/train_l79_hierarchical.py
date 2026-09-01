#!/usr/bin/env python3
"""L79 RMOT-only fit smoke and the pre-registered bounded fit schedule.

This trainer deliberately builds each unit's complete L69 row set before
attaching its expression-level labels.  CLIP features are either ephemeral or
held in a small process-local frame cache; no raw/dense feature tensor is
serialized.
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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l79_hierarchical_correspondence import L79Config, L79HierarchicalCorrespondence  # noqa: E402
from locatemot.rmot.l79_data import (  # noqa: E402
    EXPECTED_MANIFEST_SHA,
    FIT_VIDEOS,
    L79BankStore,
    MANIFEST,
    UnitBatch,
    file_meta,
    key_only_unit,
    load_fit_units,
    sha256_file,
)
from locatemot.rmot.l79_runtime import (  # noqa: E402
    CLIP_SHA256,
    CLIP_WEIGHT,
    MemoryFrameCache,
    lora_parameters,
    load_clip_visual,
    lora_state_dict,
    preprocess_full_frame,
    set_lora_enabled,
    visual_pyramid,
)
from locatemot.rmot.l79_train_utils import (  # noqa: E402
    compute_l79_loss,
    deterministic_fit_order,
    smoke_stratified_order,
)


SEED = 20260829
FORBIDDEN_LABEL_FIELDS = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "target_present", "candidate_gt", "begin", "end",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trainable_grad_stats(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    finite = True
    nonzero = 0
    total = 0
    l2_sq = 0.0
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        total += 1
        grad = parameter.grad
        if grad is None:
            continue
        if not bool(torch.isfinite(grad.float()).all()):
            finite = False
        if bool((grad.detach().float().abs() > 0).any()):
            nonzero += 1
        l2_sq += float(grad.detach().float().pow(2).sum().cpu())
    return {"parameter_tensors": total, "nonzero_parameter_tensors": nonzero,
            "finite": finite, "l2": math.sqrt(max(0.0, l2_sq))}


def base_gradient_nonzero(clip_model: torch.nn.Module) -> int:
    count = 0
    for name, parameter in clip_model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            continue
        if parameter.grad is not None and bool((parameter.grad.detach().float().abs() > 0).any()):
            count += 1
    return count


def unit_features(
    store: L79BankStore,
    unit: dict[str, Any],
    clip_model: torch.nn.Module,
    device: torch.device,
    cache: MemoryFrameCache,
    lora_enabled: bool,
) -> tuple[UnitBatch, torch.Tensor]:
    # No label fields are passed to build_unit.  This is the hard data boundary
    # used by both the audit and all training modes.
    key_unit = key_only_unit(unit)
    if FORBIDDEN_LABEL_FIELDS.intersection(key_unit):
        raise AssertionError(f"label field leaked before feature construction: {sorted(FORBIDDEN_LABEL_FIELDS.intersection(key_unit))}")
    batch = store.build_unit(key_unit)
    if not Path(batch.image_path).is_file():
        raise FileNotFoundError(batch.image_path)
    cache_key = (batch.video, int(batch.frame_id))
    pyramid = None if lora_enabled else cache.get(cache_key)
    if pyramid is None:
        image = preprocess_full_frame(
            batch.image_path, device, clip_model.visual.conv1.weight.dtype,
        )
        pyramid = visual_pyramid(clip_model, image, with_grad=lora_enabled)
        if not lora_enabled:
            cache.put(cache_key, pyramid)
        del image
    if tuple(pyramid.shape) != (3, 1, 196, 768):
        raise AssertionError(f"L79 visual pyramid drift: {tuple(pyramid.shape)}")
    if not bool(torch.isfinite(pyramid.float()).all()):
        raise FloatingPointError(f"nonfinite visual pyramid {batch.unit_key}")
    return batch, pyramid


def tensor_inputs(batch: UnitBatch, pyramid: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch.observations.to(device=device),
        batch.history_observations.to(device=device),
        batch.history_mask.to(device=device),
        batch.text_tokens.to(device=device),
        batch.text_mask.to(device=device),
        batch.boxes_norm.to(device=device),
        pyramid,
    )


def checkpoint_payload(
    model: L79HierarchicalCorrespondence,
    clip_model: torch.nn.Module,
    step: int,
    epoch: int,
    lora_enabled: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format": "locatemot-l79-hierarchical-checkpoint-v1",
        "stage": str(args.stage), "step": int(step), "epoch": int(epoch),
        "seed": SEED, "model_config": model.config_dict(),
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "lora_state_dict": lora_state_dict(clip_model),
        "lora_enabled": bool(lora_enabled),
        "lora_contract": {"blocks": [8, 9, 10, 11], "rank": 32, "alpha": 16.0, "dtype": "float32", "merged": False},
        "optimizer_scope": "L79 decoder plus private visual LoRA only after fixed epoch 3 schedule",
        "no_clip_full_weights": True,
    }


def save_checkpoint(
    path: Path,
    model: L79HierarchicalCorrespondence,
    clip_model: torch.nn.Module,
    step: int,
    epoch: int,
    lora_enabled: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = checkpoint_payload(model, clip_model, step, epoch, lora_enabled, args)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size,
            "step": int(step), "epoch": int(epoch), "lora_enabled": bool(lora_enabled)}


def strict_reload(
    checkpoint_path: Path,
    model: L79HierarchicalCorrespondence,
    clip_model: torch.nn.Module,
    snapshot: tuple[torch.Tensor, ...] | None,
    device: torch.device,
) -> dict[str, Any]:
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = L79Config(**package["model_config"])
    reference = L79HierarchicalCorrespondence(config).to(device=device, dtype=torch.float32)
    restored = L79HierarchicalCorrespondence(config).to(device=device, dtype=torch.float32)
    for candidate, role in ((reference, "reference"), (restored, "restored")):
        result = candidate.load_state_dict(package["model_state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise AssertionError(f"strict L79 {role} reload mismatch: {result}")
    lora_result = "not_loaded"
    if bool(package.get("lora_state_dict")):
        from locatemot.rmot.l79_runtime import load_lora_state_dict
        load_lora_state_dict(clip_model, package["lora_state_dict"])
        lora_result = "A/B loaded strict"
    output_shape: dict[str, list[int]] = {}
    max_difference = None
    if snapshot is not None:
        reference.eval(); restored.eval()
        with torch.inference_mode():
            original = reference(*snapshot)
            reloaded_output = restored(*snapshot)
        for name in original:
            if original[name].shape != reloaded_output[name].shape:
                raise AssertionError(f"reload shape drift for {name}")
            output_shape[name] = list(reloaded_output[name].shape)
        max_difference = max(float((original[name] - reloaded_output[name]).abs().max().cpu()) for name in original)
        if max_difference > 1e-5:
            raise AssertionError(f"strict reload output difference {max_difference}")
        model.train()
    del reference, restored, package
    return {"strict_model_state": True, "lora_state": lora_result,
            "output_shapes": output_shape, "max_output_difference": max_difference}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("p1_smoke", "p2_fit"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    command = " ".join([sys.executable] + sys.argv)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status_path = out / "status.json"
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA changed")
        if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
            raise AssertionError("CLIP SHA changed")
        set_seed(SEED)
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("L79 training requires GPU0")
        torch.cuda.set_device(args.gpu)
        fit_units = load_fit_units()
        if args.stage == "p1_smoke":
            if args.steps != 100:
                raise ValueError("P1 is exactly 100 optimizer steps")
            schedule = smoke_stratified_order(fit_units, SEED, 100)
            grad_accum = 1 if args.grad_accum is None else int(args.grad_accum)
        else:
            schedule = deterministic_fit_order(fit_units, SEED)
            grad_accum = 4 if args.grad_accum is None else int(args.grad_accum)
            if args.epochs < 1:
                raise ValueError("P2 requires at least one full epoch")
        if grad_accum < 1:
            raise ValueError("grad accumulation must be positive")

        config = {
            "format": "locatemot-l79-training-config-v1", "stage": args.stage,
            "seed": SEED, "gpu": args.gpu, "device": str(device),
            "steps": int(args.steps) if args.stage == "p1_smoke" else None,
            "epochs": int(args.epochs) if args.stage == "p2_fit" else None,
            "fit_unit_count": len(fit_units), "schedule_unit_count": len(schedule),
            "grad_accumulation": grad_accum, "history_length": 16,
            "model_config": L79Config().__dict__,
            "optimizer": {"decoder_lr": 2e-4, "lora_lr": 2e-5, "weight_decay": 1e-4, "clip_norm": 1.0, "phase2_lora_after_epoch": 3},
            "loss_weights": {"frame_membership": 1.0, "track_set": 1.0, "same_frame_hard_pair": 1.0, "all_positive_min_margin": 0.75, "null_inactive": 0.75, "fragment_continuation": 0.50, "temporal_consistency": 0.50, "observation_quality": 0.25, "teacher_identity_stability": 0.15, "source_cross_fragment_consistency": 0.10},
            "same_class_hard_negative_metadata": "unavailable",
            "hard_negative_fallback": "all current-frame negatives",
            "present_uncovered": "membership loss masked; not converted to inactive",
            "candidate_set": "complete native L69 frame rows; no deletion/top-k/NMS/truncation",
            "text_alignment": "UNALIGNED (L48 word-token cache; no verified token/span labels)",
            "no_raw_or_dense_cache": True,
        }
        write_json(out / "config.json", config)

        clip_model = load_clip_visual(device, enable_lora=False)
        model = L79HierarchicalCorrespondence().to(device=device, dtype=torch.float32)
        if args.init_checkpoint is not None:
            package = torch.load(args.init_checkpoint.resolve(), map_location="cpu", weights_only=False)
            result = model.load_state_dict(package["model_state_dict"], strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise AssertionError(f"initial checkpoint mismatch: {result}")
            if bool(package.get("lora_state_dict")):
                from locatemot.rmot.l79_runtime import load_lora_state_dict
                load_lora_state_dict(clip_model, package["lora_state_dict"])
            del package
        model.train()
        for parameter in clip_model.parameters():
            if not (".lora_A" in str(parameter) or ".lora_B" in str(parameter)):
                parameter.requires_grad_(False)
        decoder_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(decoder_parameters, lr=2e-4, weight_decay=1e-4)
        lora_added = False
        lora_parameter_list = list(lora_parameters(clip_model))
        cache = MemoryFrameCache(max_items=64)
        store = L79BankStore(max_history=16)
        loss_trace: list[dict[str, Any]] = []
        sampling_trace: list[dict[str, Any]] = []
        domain_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        stratum_counts: Counter[str] = Counter()
        finite_steps = 0
        nonzero_steps = 0
        base_nonzero_max = 0
        lora_nonzero_steps = 0
        candidate_drift = 0
        truncation_count = 0
        key_drift = 0
        tested_positive_grad = 0
        tested_negative_grad = 0
        tested_min_positive_grad = 0
        optimizer_steps = 0
        current_epoch = 1
        last_snapshot: tuple[torch.Tensor, ...] | None = None
        checkpoint_records: list[dict[str, Any]] = []
        optimizer.zero_grad(set_to_none=True)

        def run_one(unit: dict[str, Any], step_number: int, epoch_number: int) -> dict[str, Any]:
            nonlocal last_snapshot, tested_positive_grad, tested_negative_grad, tested_min_positive_grad
            nonlocal candidate_drift, truncation_count, key_drift, finite_steps, nonzero_steps
            nonlocal base_nonzero_max, lora_nonzero_steps
            batch, pyramid = unit_features(store, unit, clip_model, device, cache, lora_added)
            expected_count = batch.candidate_count
            expected_keys = list(batch.row_keys)
            inputs = tensor_inputs(batch, pyramid, device)
            # Labels are attached after all candidate/image/text feature
            # construction, never before it.
            labels = store.attach_labels(batch, unit)
            if labels["row_count"] != expected_count or len(batch.row_keys) != expected_count:
                candidate_drift += 1
            if list(batch.row_keys) != expected_keys:
                key_drift += 1
            if labels["row_count"] != expected_count:
                candidate_drift += 1
            if labels["present_uncovered"] and labels["membership_mask"].any():
                raise AssertionError("present-uncovered membership mask drift")
            outputs = model(*inputs)
            total, details = compute_l79_loss(outputs, labels)
            if not bool(torch.isfinite(total).all()):
                raise FloatingPointError(f"nonfinite loss at {batch.unit_key}")
            row_grad = torch.autograd.grad(total, outputs["frame_membership_logits"], retain_graph=True, allow_unused=False)[0]
            supervised = labels["membership_mask"].to(device=device)
            target = labels["labels"].to(device=device)
            positive_grad = bool(((row_grad[target & supervised].abs() > 0).any())) if bool((target & supervised).any()) else True
            negative_grad = bool(((row_grad[(~target) & supervised].abs() > 0).any())) if bool(((~target) & supervised).any()) else True
            min_positive_grad = positive_grad if int(target.sum()) > 1 and bool(supervised.any()) else True
            tested_positive_grad += int(bool((target & supervised).any()))
            tested_negative_grad += int(bool(((~target) & supervised).any()))
            tested_min_positive_grad += int(int(target.sum()) > 1 and bool(supervised.any()))
            (total / float(grad_accum)).backward()
            grad_stats = trainable_grad_stats(decoder_parameters + (lora_parameter_list if lora_added else []))
            if not grad_stats["finite"]:
                raise FloatingPointError(f"nonfinite gradient at {batch.unit_key}")
            finite_steps += 1
            if grad_stats["nonzero_parameter_tensors"] > 0:
                nonzero_steps += 1
            base_nonzero_max = max(base_nonzero_max, base_gradient_nonzero(clip_model))
            lora_grad_stats = trainable_grad_stats(lora_parameter_list) if lora_added else {"nonzero_parameter_tensors": 0, "finite": True, "parameter_tensors": len(lora_parameter_list), "l2": 0.0}
            if lora_added and lora_grad_stats["nonzero_parameter_tensors"] > 0:
                lora_nonzero_steps += 1
            domain_counts[str(batch.dataset)] += 1
            category_counts[str(labels["category"])] += 1
            stratum_counts[str(labels["category"])] += 1
            if labels["declared_category"] != labels["category"]:
                stratum_counts["declared/derived_mismatch"] += 1
            truncation_count += int(False)
            sampling_trace.append({"step": int(step_number), "epoch": int(epoch_number), "unit_key": batch.unit_key,
                                   "dataset": batch.dataset, "video": batch.video, "frame_id": batch.frame_id,
                                   "category": labels["category"], "declared_category": labels["declared_category"],
                                   "candidate_count": expected_count, "positive_count": labels["positive_count"],
                                   "present_uncovered": labels["present_uncovered"], "positive_grad_nonzero": positive_grad,
                                   "negative_grad_nonzero": negative_grad, "minimum_positive_grad_nonzero": min_positive_grad,
                                   "history_future_rows": int((batch.history_frame_ids > batch.frame_id).sum()),
                                   "row_key_count": len(expected_keys), "candidate_deletion": False, "candidate_truncation": False})
            loss_trace.append({"step": int(step_number), "epoch": int(epoch_number), "unit_key": batch.unit_key,
                               **details, "gradient": grad_stats, "lora_gradient": lora_grad_stats,
                               "positive_grad_nonzero": positive_grad, "negative_grad_nonzero": negative_grad,
                               "minimum_positive_grad_nonzero": min_positive_grad, "candidate_count": expected_count,
                               "finite": True})
            # Keep only one small in-memory reload fixture, never a serialized
            # visual feature cache.
            last_snapshot = tuple(value.detach().clone() if isinstance(value, torch.Tensor) else value for value in inputs)
            del outputs, total, row_grad, inputs, pyramid, batch, labels
            return {"details": details, "grad": grad_stats}

        total_units = 100 if args.stage == "p1_smoke" else len(schedule) * int(args.epochs)
        processed_units = 0
        for epoch in range(1, (1 if args.stage == "p1_smoke" else int(args.epochs)) + 1):
            current_epoch = epoch
            if args.stage == "p2_fit" and epoch == 3 and not lora_added:
                set_lora_enabled(clip_model, True)
                optimizer.add_param_group({"params": lora_parameter_list, "lr": 2e-5, "weight_decay": 0.0})
                lora_added = True
                cache.clear()
            epoch_schedule = schedule if args.stage == "p2_fit" else schedule[:100]
            for unit in epoch_schedule:
                processed_units += 1
                run_one(unit, processed_units, epoch)
                should_step = processed_units % grad_accum == 0 or processed_units == total_units
                if should_step:
                    all_parameters = decoder_parameters + (lora_parameter_list if lora_added else [])
                    grad_stats = trainable_grad_stats(all_parameters)
                    if not grad_stats["finite"] or grad_stats["nonzero_parameter_tensors"] == 0:
                        raise FloatingPointError(f"optimizer gradient contract failed at unit {processed_units}: {grad_stats}")
                    torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if args.stage == "p1_smoke" and optimizer_steps != processed_units:
                        raise AssertionError("P1 step accounting drift")
            if args.stage == "p2_fit" and epoch in {1, 3, 5, 10}:
                checkpoint_path = out / f"checkpoint_l79_epoch{epoch:02d}.pt"
                checkpoint_records.append(save_checkpoint(checkpoint_path, model, clip_model, optimizer_steps, epoch, lora_added, args))
                reload_audit = strict_reload(checkpoint_path, model, clip_model, last_snapshot, device)
                write_json(out / f"reload_audit_epoch{epoch:02d}.json", reload_audit)

        if args.stage == "p1_smoke":
            checkpoint_path = out / "checkpoint_l79_step100.pt"
            checkpoint_records.append(save_checkpoint(checkpoint_path, model, clip_model, optimizer_steps, 1, lora_added, args))
            reload_audit = strict_reload(checkpoint_path, model, clip_model, last_snapshot, device)
            write_json(out / "reload_audit.json", reload_audit)
        else:
            reload_audit = {"milestone_files": [x["path"] for x in checkpoint_records]}

        if processed_units != total_units:
            raise AssertionError(f"processed unit count drift: {processed_units} != {total_units}")
        expected_domains = {"refer_kitti_v1", "refer_kitti_v2"}
        if set(domain_counts) != expected_domains:
            raise AssertionError(f"domain coverage drift: {domain_counts}")
        required_categories = {"positive", "multi_positive", "inactive", "present_uncovered"}
        if not required_categories.issubset(set(category_counts)):
            raise AssertionError(f"category coverage drift: {category_counts}")
        if args.stage == "p1_smoke" and (finite_steps != 100 or nonzero_steps != 100):
            raise AssertionError(f"P1 finite/nonzero drift: {finite_steps}/{nonzero_steps}")
        base_hash = sha256_file(MANIFEST)
        metrics = {
            "format": "locatemot-l79-training-metrics-v1", "status": "complete", "stage": args.stage,
            "steps": int(optimizer_steps), "processed_units": int(processed_units), "epochs": int(args.epochs) if args.stage == "p2_fit" else 1,
            "finite_steps": int(finite_steps), "nonzero_gradient_steps": int(nonzero_steps),
            "lora_nonzero_gradient_steps": int(lora_nonzero_steps), "base_nonzero_gradient_max": int(base_nonzero_max),
            "candidate_key_drift": int(key_drift), "candidate_count_drift": int(candidate_drift),
            "candidate_truncation": bool(truncation_count), "candidate_deletion": False,
            "tested_positive_grad_units": int(tested_positive_grad), "tested_negative_grad_units": int(tested_negative_grad),
            "tested_minimum_positive_grad_units": int(tested_min_positive_grad),
            "domain_counts": dict(domain_counts), "category_counts": dict(category_counts),
            "required_fifth_diagnostic_stratum": "occlusion_reappear_or_cross_fragment_diagnostic unavailable in L49 fit metadata; no fabricated sampling field",
            "decoder_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "decoder_trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "lora_parameter_count": int(sum(p.numel() for p in lora_parameter_list)),
            "lora_enabled_at_end": bool(lora_added), "checkpoint_records": checkpoint_records,
            "reload_audit": reload_audit, "wall_time_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "manifest_sha256_at_end": base_hash, "manifest_unchanged": base_hash == EXPECTED_MANIFEST_SHA,
            "no_persistent_raw_dense_cache": True, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
        }
        write_json(out / ("metrics_l79_step100.json" if args.stage == "p1_smoke" else "metrics_l79_fit.json"), metrics)
        write_json(out / "sampling_trace.json", {"format": "locatemot-l79-sampling-trace-v1", "status": "complete", "records": sampling_trace,
                                                   "fit_unit_count": len(fit_units), "seed": SEED, "domains": dict(domain_counts), "categories": dict(category_counts),
                                                   "five_strata_note": "four registered categories sampled; occlusion/reappear diagnostic unavailable in source metadata", "screening_gt_used": False})
        write_json(out / "loss_trace.json", {"format": "locatemot-l79-loss-trace-v1", "status": "complete", "records": loss_trace,
                                               "finite_all": all(bool(x["finite"]) for x in loss_trace), "screening_gt_used": False})
        provenance = {
            "format": "locatemot-l79-training-provenance-v1", "status": "complete", "project_root": str(ROOT),
            "cwd": str(Path.cwd().resolve()), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "command": command,
            "python": sys.executable, "python_version": sys.version, "torch_version": torch.__version__, "cuda": torch.version.cuda,
            "device": str(device), "visible_cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "seed": SEED,
            "inputs": {"manifest": file_meta(MANIFEST), "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                       "l48_text_cache": file_meta(ROOT / "outputs/l48/data/text_cache.pt"), "clip_weight": {"path": str(CLIP_WEIGHT), "sha256": sha256_file(CLIP_WEIGHT), "expected": CLIP_SHA256},
                       "fit_unit_source": str(ROOT / "outputs/l49/data/train_units.jsonl"), "fit_unit_count": len(fit_units), "fit_videos": list(FIT_VIDEOS)},
            "outputs": [str(x["path"]) for x in checkpoint_records], "candidate_index": "L69 native frame_ptr/row offset; duplicate candidate_index retained",
            "labels": "expression-level membership attached after complete feature construction; present-uncovered masked",
            "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
            "token_span_alignment": "UNALIGNED", "no_raw_or_dense_cache_written": True,
            "model_input_forbidden": ["source_id", "pool_id", "group_id", "query_id", "track_id", "state_key", "old scores", "GT identity"],
            "ordinary_mot_ovmot_touched": False, "screening_gt_used": False, "official_test_labels_read": False,
            "hota_trackeval_run": False,
        }
        write_json(out / "provenance.json", provenance)
        write_json(status_path, {"format": "locatemot-l79-training-status-v1", "status": "complete", "stage": args.stage,
                                 "command": command, "failure_root_cause": None,
                                 "next_action": "run fixed calibration/validation evaluator after P2 fit only; no screening before semantic gate",
                                 "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        # Release all process-local feature/model objects before returning.
        cache.clear(); store._bank = None; store._text_cache = None
        del optimizer, model, clip_model
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"status": "complete", "stage": args.stage, "steps": optimizer_steps, "out": str(out)}, indent=2), flush=True)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        write_json(status_path, {"format": "locatemot-l79-training-status-v1", "status": "incomplete", "stage": args.stage,
                                 "command": command, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": tb,
                                 "next_action": "preserve this attempt; fix only the first actionable training-contract error and rerun in a new output directory",
                                 "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        (out / "INCOMPLETE.md").write_text("# L79 training incomplete\n\nFirst actionable error:\n\n```text\n" + tb + "```\n")
        print(tb, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
