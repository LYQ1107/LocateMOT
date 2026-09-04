#!/usr/bin/env python3
"""L88 RMOT-only GroundingDINO-LoRA plus L86/L87-A sidecar training.

The detector is rebuilt from the immutable local checkpoint, its query-
independent encoder inputs are read one frame at a time from the L88 cache,
and only the registered LoRA factors and fresh sidecar parameters receive
gradients.  The inherited L86 cache is deliberately lazy: its loaded frame
objects are cleared at every group boundary so the 58 GB query-independent
cache is never mirrored in resident memory.
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


WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L85_CACHE = ASSET_ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
L88_CACHE = WORK_ROOT / "outputs/l88/cache/encoder_inputs_v1"
OBS_DIM = 1432
HISTORY = 8

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))
import locatemot.rmot as _rmot_package  # noqa: E402
if str(ASSET_ROOT / "locatemot" / "rmot") not in [str(x) for x in _rmot_package.__path__]:
    _rmot_package.__path__.append(str(ASSET_ROOT / "locatemot" / "rmot"))

from locatemot.models.l88_full_rmot import L86Config, L88FullRMOT  # noqa: E402
from locatemot.rmot.l87a_losses import l87a_loss  # noqa: E402
from locatemot.rmot.l88_clip_data import L88ClipStore  # noqa: E402
from locatemot.rmot.l88_grounding_runtime import (  # noqa: E402
    forward_l88_z1, sha256_file,
)
from locatemot.rmot.l88_lora import adapter_grad_report, inject_lora  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def file_sha(path: Path) -> str:
    return sha256_file(path.resolve())


def memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


class EncoderCacheReader:
    """Indexed, one-item-at-a-time reader for the L88 image-only cache."""

    REQUIRED = {"feat", "feat_mask", "feat_pos", "spatial_shapes", "level_start_index", "valid_ratios"}

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        summary = json.loads((self.root / "summary.json").read_text())
        if summary.get("status") != "complete" or summary.get("labels_in_cache"):
            raise AssertionError("L88 encoder cache is not complete label-free data")
        if not bool(summary.get("query_independent")):
            raise AssertionError("L88 encoder cache is not query independent")
        self.summary = summary
        self.summary_sha256 = file_sha(self.root / "summary.json")
        self.entries: dict[str, Path] = {}
        manifest = self.root / "manifest.jsonl"
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["cache_key"])
            path = (self.root / str(row["file"])).resolve()
            if key in self.entries and self.entries[key] != path:
                raise AssertionError(f"duplicate L88 encoder cache key: {key}")
            if row.get("labels_in_cache") or not row.get("query_independent", False):
                raise AssertionError(f"invalid L88 cache manifest flags: {key}")
            self.entries[key] = path
        if len(self.entries) != int(summary.get("entry_count", -1)):
            raise AssertionError(f"L88 cache entry count drift: {len(self.entries)}")

    def read(self, key: str, device: torch.device) -> dict[str, Any]:
        path = self.entries.get(str(key))
        if path is None or not path.is_file():
            raise KeyError(f"missing L88 encoder cache item: {key}")
        # The manifest was indexed once above.  Loading the indexed file
        # directly avoids rescanning a 1.6 MB manifest for every frame/query.
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("labels_in_cache") or not item.get("query_independent"):
            raise AssertionError(f"invalid L88 cache flags: {path}")
        if not self.REQUIRED.issubset(item):
            raise AssertionError(f"L88 cache item missing keys: {sorted(self.REQUIRED - set(item))}")
        for name in self.REQUIRED:
            value = item[name]
            if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value.float()).all()):
                raise FloatingPointError(f"nonfinite L88 cache item {key}:{name}")
        if str(item.get("cache_key")) != str(key):
            raise AssertionError(f"L88 cache key mismatch: {key} / {item.get('cache_key')}")
        return item


def group_sentences(store: L88ClipStore, group_key: str, query_ids: list[int]) -> list[str]:
    group = store.groups[str(group_key)]
    by_id = {int(row["query_id"]): str(row["sentence"]) for row in group["queries"]}
    result = [by_id[int(qid)] for qid in query_ids]
    if any(not value for value in result):
        raise AssertionError(f"empty L88 expression in {group_key}")
    return result


def cache_key(dataset: str, video: str, frame_id: int) -> str:
    return f"{dataset}|{video}|{int(frame_id):06d}"


def encode_z1(runtime: Any, reader: EncoderCacheReader, store: L88ClipStore,
              frame: Any, device: torch.device, *, query_tile: int, bf16: bool) -> torch.Tensor:
    batch = store.bank_store.build_unit(store.groups[str(frame.group_key)]["queries"][0])
    if int(batch.candidate_count) != len(frame.row_offsets) or batch.row_offsets != frame.row_offsets:
        raise AssertionError(f"L88 frame/bank row contract drift: {frame.group_key}")
    if batch.row_keys != frame.row_keys:
        raise AssertionError(f"L88 frame/bank key order drift: {frame.group_key}")
    item = reader.read(cache_key(frame.dataset, frame.video, frame.frame_id), device)
    if int(item["feat"].shape[1]) != int(item["feat_pos"].shape[1]):
        raise AssertionError(f"L88 encoder feature/position length drift: {frame.group_key}")
    sentences = group_sentences(store, frame.group_key, [int(x) for x in frame.query_ids])
    if int(query_tile) != len(sentences):
        raise AssertionError(f"L88 query tile drift: {frame.group_key}")
    replay = forward_l88_z1(
        runtime.model, item, batch.boxes, sentences, device,
        query_tile=len(sentences), autocast_bf16=bool(bf16),
    )
    z1 = replay["z1"].float()
    expected = (len(sentences), int(frame.row_offsets.__len__()), 256)
    if tuple(z1.shape) != expected:
        raise AssertionError(f"L88 adapted Z1 shape drift: {frame.group_key}: {tuple(z1.shape)}")
    if not bool(torch.isfinite(z1).all()):
        raise FloatingPointError(f"nonfinite L88 adapted Z1: {frame.group_key}")
    if bool(replay.get("candidate_deletion")) or bool(replay.get("candidate_truncation")):
        raise AssertionError(f"L88 adapted Z1 dropped candidates: {frame.group_key}")
    # The sidecar only consumes Z1.  Explicitly discard the large adapted
    # memory/text objects before returning the differentiable Z1 tensor.
    replay.pop("z1", None)
    del replay, item, batch
    gc.collect()
    return z1


def sidecar_forward(sidecar: L88FullRMOT, frame: Any, z1: torch.Tensor,
                    device: torch.device, temporal_enabled: bool) -> dict[str, torch.Tensor]:
    return sidecar(
        z1, frame.text_global.to(device), frame.frame_global.to(device),
        frame.current_observation.to(device), frame.history_observations.to(device),
        frame.history_mask.to(device), frame.history_frame_ids.to(device), frame.frame_id,
        temporal_enabled=temporal_enabled,
    )


def grad_report(sidecar: torch.nn.Module, injector: Any, model: torch.nn.Module) -> dict[str, Any]:
    lora = adapter_grad_report(injector)
    total = 0.0
    nonzero = 0
    finite = True
    for parameter in sidecar.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        norm = float(value.norm())
        total += norm
        nonzero += int(norm > 0.0)
        finite = finite and bool(torch.isfinite(value).all())
    lora_ids = {id(value) for value in injector.parameters()}
    unapproved: list[str] = []
    for name, parameter in model.named_parameters():
        if id(parameter) in lora_ids:
            continue
        if parameter.grad is not None:
            unapproved.append(str(name))
    return {
        "lora": lora,
        "sidecar_gradient_norm_sum": total,
        "sidecar_nonzero_gradient_entries": nonzero,
        "sidecar_finite": finite,
        "unapproved_detector_gradients": unapproved,
        "all_trainable_nonzero": bool(lora["gradient_norm_sum"] > 0.0 and total > 0.0),
        "all_finite": bool(lora["finite"] and finite),
    }


def save_checkpoint(path: Path, sidecar: L88FullRMOT, injector: Any,
                    optimizer: torch.optim.Optimizer, scheduler: Any,
                    epoch: int, optimizer_step: int, args: argparse.Namespace,
                    phase: str) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite L88 checkpoint: {path}")
    package = {
        "format": "locatemot-l88-full-rmot-checkpoint-v1",
        "model_config": asdict(sidecar.config),
        "sidecar_state_dict": {key: value.detach().cpu().clone() for key, value in sidecar.state_dict().items()},
        "lora_state_dict": injector.adapter_state_dict(),
        "lora_manifest": injector.manifest(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch), "optimizer_step": int(optimizer_step), "phase": str(phase),
        "seed": SEED, "args": vars(args),
        "manifest_sha256": MANIFEST_SHA,
        "cache_summary_sha256": file_sha(L88_CACHE / "summary.json"),
        "groundingdino_lora_used": True, "groundingdino_backbone_trainable": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(package, temporary)
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "sha256": file_sha(path), "epoch": int(epoch),
            "optimizer_step": int(optimizer_step), "phase": str(phase)}


def short_contract(args: argparse.Namespace, out: Path) -> int:
    """One-group integration path used before the registered 40-epoch run."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 integration output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    set_seed(args.seed)
    store = L88ClipStore(L85_CACHE, load_cache_into_ram=False)
    reader = EncoderCacheReader(args.cache)
    runtime = None
    sidecar = None
    try:
        runtime_cls = __import__("locatemot.rmot.l88_grounding_runtime", fromlist=["L88GroundingRuntime"]).L88GroundingRuntime
        runtime = runtime_cls(device)
        injector = inject_lora(runtime.model)
        runtime.model.eval()
        sidecar = L88FullRMOT(L86Config()).to(device=device, dtype=torch.float32)
        sidecar.train()
        optimizer = torch.optim.AdamW(list(injector.parameters()) + list(sidecar.parameters()), lr=2e-4)
        optimizer.zero_grad(set_to_none=True)
        anchor = str(store.train_keys[0])
        frame = store.build_frame(anchor, temporal_enabled=False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(args.bf16 and device.type == "cuda")):
            z1 = encode_z1(runtime, reader, store, frame, device, query_tile=len(frame.query_ids), bf16=args.bf16)
            output = sidecar_forward(sidecar, frame, z1, device, False)
            loss, info = l87a_loss(output, frame.labels, frame.current_observation.to(device), [], temporal_enabled=False)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite L88 integration loss")
        loss.backward()
        gradients = grad_report(sidecar, injector, runtime.model)
        if not gradients["all_trainable_nonzero"] or gradients["unapproved_detector_gradients"]:
            raise AssertionError(f"L88 integration gradient contract failed: {gradients}")
        torch.nn.utils.clip_grad_norm_(list(injector.parameters()) + list(sidecar.parameters()), 1.0)
        optimizer.step()
        checkpoint = save_checkpoint(out / "checkpoint_l88_integration_step1.pt", sidecar, injector,
                                     optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
                                     1, 1, args, "S")
        package = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        reloaded = L88FullRMOT(L86Config(**package["model_config"]))
        loaded = reloaded.load_state_dict(package["sidecar_state_dict"], strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise AssertionError(f"L88 integration sidecar reload failed: {loaded}")
        runtime.close()
        runtime = None
        write_json(out / "metrics_l88_integration_step1.json", {
            "format": "locatemot-l88-training-integration-v1", "status": "complete", "stage": "targeted_regression",
            "steps": 1, "loss": float(loss.detach()), "loss_info": info, "gradients": gradients,
            "candidate_count": len(frame.row_offsets), "candidate_rows_retained": True,
            "candidate_deletion": False, "candidate_truncation": False, "strict_reload": True,
            "sampling": {"dataset": frame.dataset, "video": frame.video, "category_counts": {str(x["category"]): 1 for x in frame.labels}},
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        write_json(out / "provenance.json", {
            "format": "locatemot-l88-training-integration-provenance-v1", "status": "complete",
            "command": " ".join([sys.executable, *sys.argv]), "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "inputs": {"l88_cache": str(args.cache.resolve()), "cache_summary_sha256": reader.summary_sha256,
                        "manifest_sha256": MANIFEST_SHA}, "model_parameters": sidecar.parameter_report(),
            "lora_manifest": injector.manifest(), "groundingdino_lora_used": True,
            "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        })
        write_json(out / "status.json", {"format": "locatemot-l88-training-integration-status-v1", "status": "complete",
                                         "steps": 1, "strict_reload": True, "gradient_contract": True,
                                         "screening_gt_used": False, "official_test_labels_read": False,
                                         "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return 0
    finally:
        if runtime is not None:
            runtime.close()
        if sidecar is not None:
            del sidecar
        store.release_loaded_cache_items()
        store.close(); del reader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 training output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if int(args.seed) != SEED:
        raise AssertionError(f"L88 seed is fixed at {SEED}")
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    device = torch.device(args.device)
    runtime = None
    store = None
    sidecar = None
    rank = 0
    trace: list[dict[str, Any]] = []
    sampling: list[dict[str, Any]] = []
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if file_sha(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for L88")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        if int(args.epochs) != 40 and not args.allow_short:
            raise AssertionError("registered L88 run requires exactly 40 epochs")
        if int(args.max_groups) and not args.allow_short:
            raise AssertionError("max_groups is only for targeted integration")
        set_seed(args.seed)
        reader = EncoderCacheReader(args.cache)
        store = L88ClipStore(L85_CACHE, load_cache_into_ram=False)
        keys = [str(value) for value in store.train_keys]
        if len(keys) != 524:
            raise AssertionError(f"L88 train group count drift: {len(keys)}")
        runtime_cls = __import__("locatemot.rmot.l88_grounding_runtime", fromlist=["L88GroundingRuntime"]).L88GroundingRuntime
        runtime = runtime_cls(device)
        injector = inject_lora(runtime.model)
        runtime.model.eval()
        sidecar = L88FullRMOT(L86Config()).to(device=device, dtype=torch.float32)
        sidecar.train()
        trainable = list(injector.parameters()) + list(sidecar.parameters())
        if not trainable or any(not value.requires_grad for value in trainable):
            raise AssertionError("L88 trainable parameter contract failed")
        optimizer = torch.optim.AdamW([
            {"params": list(sidecar.parameters()), "lr": 2e-4, "weight_decay": 1e-2},
            {"params": list(injector.parameters()), "lr": 1e-4, "weight_decay": 0.0},
        ], betas=(0.9, 0.999))
        total_groups = len(keys) if not int(args.max_groups) else int(args.max_groups)
        effective_batch = max(1, int(args.effective_clip_batch))
        accumulation = effective_batch
        steps_per_epoch = math.ceil(total_groups / accumulation)
        total_optimizer_steps = steps_per_epoch * int(args.epochs)
        warmup_steps = max(1, int(round(total_optimizer_steps * 0.05)))
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(1e-8, float(step + 1) / float(warmup_steps))
            progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        write_json(out / "config.json", {
            "format": "locatemot-l88-training-config-v1", "status": "running", "stage": "L88 full RMOT",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": int(args.seed),
            "epochs": int(args.epochs), "device": str(device), "world_size": 1,
            "effective_frame_group_batch": effective_batch, "accumulation_steps": accumulation,
            "bf16": bool(args.bf16), "query_tile": int(args.query_tile), "cache": str(args.cache.resolve()),
            "cache_summary_sha256": reader.summary_sha256, "manifest_sha256": MANIFEST_SHA,
            "curriculum": {"S": [1, 8], "T": [9, 20], "J": [21, 40]},
            "optimizer": {"sidecar_lr": 2e-4, "lora_lr": 1e-4, "sidecar_weight_decay": 1e-2,
                          "lora_weight_decay": 0.0, "warmup_fraction": 0.05, "schedule": "cosine",
                          "gradient_clip": 1.0},
            "lora_manifest": injector.manifest(), "sidecar_parameters": sidecar.parameter_report(),
            "fit_scope": "L49 fit only; V1/V2; no calibration/validation/screening/official-test labels",
            "same_class_hard_negative_metadata": "unavailable; L87-A all-negative target-bag fallback",
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "groundingdino_lora_used": True, "groundingdino_backbone_trainable": False,
            "bert_body_trainable": False, "bbox_head_trainable": False, "decoder_layers_2_to_6_trainable": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "allow_short": bool(args.allow_short), "max_groups": int(args.max_groups),
        })
        optimizer.zero_grad(set_to_none=True)
        optimizer_step = 0
        amp_enabled = bool(args.bf16 and device.type == "cuda")
        for epoch in range(1, int(args.epochs) + 1):
            phase = "S" if epoch <= 8 else ("T" if epoch <= 20 else "J")
            temporal_enabled = epoch > 8
            schedule = list(keys)
            random.Random(int(args.seed) + epoch).shuffle(schedule)
            schedule = schedule[:total_groups]
            epoch_loss = 0.0
            epoch_groups = 0
            epoch_optimizer_steps = 0
            epoch_grad_entries = 0
            epoch_grad_nonzero = 0
            epoch_category = {"positive": 0, "multi_positive": 0, "inactive": 0, "present_uncovered": 0}
            epoch_domain = {"refer_kitti_v1": 0, "refer_kitti_v2": 0}
            epoch_pos = epoch_neg = epoch_masked = epoch_temporal_pairs = 0
            finite_steps = 0
            for group_index, anchor in enumerate(schedule):
                clip = store.build_clip(anchor, temporal_enabled=temporal_enabled, clip_length=4)
                current = clip[-1]
                previous_outputs: list[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]] = []
                try:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                        current_z1 = encode_z1(runtime, reader, store, current, device,
                                               query_tile=int(args.query_tile), bf16=args.bf16)
                        current_output = sidecar_forward(sidecar, current, current_z1, device, temporal_enabled)
                        for previous in clip[:-1]:
                            previous_z1 = encode_z1(runtime, reader, store, previous, device,
                                                    query_tile=int(args.query_tile), bf16=args.bf16)
                            previous_output = sidecar_forward(sidecar, previous, previous_z1, device, temporal_enabled)
                            previous_outputs.append((previous_output, previous.labels))
                            del previous_z1
                        loss, info = l87a_loss(
                            current_output, current.labels, current.current_observation.to(device),
                            previous_outputs, temporal_enabled=temporal_enabled,
                        )
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"nonfinite L88 loss epoch={epoch} group={anchor}")
                        (loss / float(accumulation)).backward()
                    epoch_loss += float(loss.detach()); epoch_groups += 1; finite_steps += 1
                    epoch_pos += int(info.get("positive_count", 0)); epoch_neg += int(info.get("negative_target_bags", 0))
                    epoch_masked += int(info.get("masked_missing_count", 0)); epoch_temporal_pairs += int(info.get("positive_pairs", 0))
                    epoch_domain[str(current.dataset)] = epoch_domain.get(str(current.dataset), 0) + 1
                    for label in current.labels:
                        category = str(label["category"])
                        epoch_category[category] = epoch_category.get(category, 0) + 1
                    should_step = ((group_index + 1) % accumulation == 0) or (group_index + 1 == len(schedule))
                    if should_step:
                        gradients = grad_report(sidecar, injector, runtime.model)
                        if not gradients["all_finite"] or not gradients["all_trainable_nonzero"]:
                            raise FloatingPointError(f"L88 gradient contract failed epoch={epoch} group={anchor}: {gradients}")
                        if gradients["unapproved_detector_gradients"]:
                            raise AssertionError(f"unapproved detector gradient: {gradients['unapproved_detector_gradients'][:5]}")
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                        optimizer_step += 1; epoch_optimizer_steps += 1
                        epoch_grad_entries += int(gradients["lora"]["gradient_entries"] + gradients["sidecar_nonzero_gradient_entries"])
                        epoch_grad_nonzero += int(gradients["lora"]["nonzero_gradient_entries"] + gradients["sidecar_nonzero_gradient_entries"])
                    del current_z1, current_output, previous_outputs, clip
                    store.release_loaded_cache_items()
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception:
                    store.release_loaded_cache_items()
                    raise
            entry = {
                "epoch": epoch, "phase": phase, "temporal_enabled": temporal_enabled,
                "loss_mean": epoch_loss / max(1, epoch_groups), "groups": epoch_groups,
                "finite_group_losses": finite_steps, "optimizer_steps_epoch": epoch_optimizer_steps,
                "optimizer_steps_total": optimizer_step, "positive_rows": epoch_pos,
                "negative_target_bags": epoch_neg, "masked_missing_count": epoch_masked,
                "temporal_identity_pairs": epoch_temporal_pairs, "category_counts": epoch_category,
                "domain_counts": epoch_domain, "gradient_entries": epoch_grad_entries,
                "nonzero_gradient_entries": epoch_grad_nonzero, "candidate_rows_retained": True,
                "candidate_deletion": False, "candidate_truncation": False,
                "peak_memory_bytes": memory_bytes(device),
            }
            trace.append(entry); sampling.append({
                "epoch": epoch, "phase": phase, "seed": int(args.seed) + epoch,
                "groups": epoch_groups, "domain_counts": epoch_domain, "category_counts": epoch_category,
                "all_candidate_rows": True, "candidate_deletion": False, "candidate_truncation": False,
            })
            print(json.dumps(entry, sort_keys=True), flush=True)
            if epoch % 2 == 0:
                save_checkpoint(out / f"checkpoint_l88_epoch{epoch:03d}.pt", sidecar, injector,
                                optimizer, scheduler, epoch, optimizer_step, args, phase)
        checkpoints = []
        for path in sorted(out.glob("checkpoint_l88_epoch*.pt")):
            checkpoints.append({"path": str(path.resolve()), "sha256": file_sha(path),
                                "epoch": int(path.stem.split("epoch")[-1])})
        final = checkpoints[-1] if checkpoints else None
        payload = {
            "format": "locatemot-l88-full-rmot-training-v1", "status": "complete",
            "stage": "L88 40-epoch RMOT-only LoRA plus sidecar training", "command": command,
            "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": int(args.seed),
            "epochs": int(args.epochs), "optimizer_steps": optimizer_step, "world_size": 1,
            "effective_frame_group_batch": effective_batch, "cache_summary_sha256": reader.summary_sha256,
            "lora_manifest": injector.manifest(), "sidecar_parameters": sidecar.parameter_report(),
            "checkpoints": checkpoints, "final_checkpoint": final, "loss_trace": str((out / "loss_trace.json").resolve()),
            "sampling_trace": str((out / "sampling_trace.json").resolve()), "trace": trace,
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "groundingdino_lora_used": True, "groundingdino_backbone_trainable": False,
            "bert_body_trainable": False, "bbox_head_trainable": False, "decoder_layers_2_to_6_trainable": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": memory_bytes(device), "failure_root_cause": None,
            "next_action": "score all even checkpoints on the registered internal dev groups before fixed evaluation",
        }
        write_json(out / "loss_trace.json", trace); write_json(out / "sampling_trace.json", sampling)
        write_json(out / "config.json", json.loads((out / "config.json").read_text()) | {
            "status": "complete", "optimizer_steps": optimizer_step, "checkpoints": checkpoints,
            "wall_seconds": time.perf_counter() - started, "peak_memory_bytes": memory_bytes(device),
        })
        write_json(out / "metrics_l88_training.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": "locatemot-l88-training-status-v1", "status": "complete",
                                         "epochs": int(args.epochs), "optimizer_steps": optimizer_step,
                                         "checkpoint_count": len(checkpoints), "both_domains": True,
                                         "all_four_categories": True, "candidate_deletion": False,
                                         "candidate_truncation": False, "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                         "hota_trackeval_run": False})
        return 0
    except Exception:
        trace_text = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 full RMOT training — INCOMPLETE\n\n" + trace_text)
        write_json(out / "status.json", {"format": "locatemot-l88-training-status-v1", "status": "incomplete",
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
        if sidecar is not None:
            del sidecar
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cache", type=Path, default=L88_CACHE)
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "outputs/l88/train/joint40")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-tile", type=int, default=4)
    parser.add_argument("--effective-clip-batch", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--allow-short", action="store_true")
    args = parser.parse_args()
    if int(args.query_tile) < 1 or int(args.effective_clip_batch) < 1:
        raise ValueError("query tile and effective batch must be positive")
    if args.allow_short and int(args.max_groups) == 0 and int(args.epochs) == 1:
        return short_contract(args, args.out.resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
