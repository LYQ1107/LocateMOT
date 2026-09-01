#!/usr/bin/env python3
"""Bounded L80-R1 fit probe: only the frozen region sampling interface changes."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from locatemot.models.l80_r1_region import L80R1Config, L80R1RegionCorrespondence  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    EXPECTED_MANIFEST_SHA, FORBIDDEN_LABEL_FIELDS, L80BankStore, MANIFEST,
    key_only, load_fit_units, load_full_unit_for_labels, sha256_file,
)
from locatemot.rmot.l80_losses import l80_loss  # noqa: E402
from locatemot.rmot.l80_r1_runtime import (  # noqa: E402
    CLIP_SHA256, CLIP_WEIGHT, FrameFeatureCache, load_clip, raw_inputs_for_unit_r1,
)
from tools.train_l80_v12_joint import row_digest, stratified_schedule  # noqa: E402

SEED = 20260829


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def strict_reload(path: Path, device: torch.device) -> dict[str, Any]:
    package = torch.load(path, map_location=device, weights_only=False)
    config = L80R1Config(**package["model_config"])
    model = L80R1RegionCorrespondence(config).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"R1 strict reload mismatch: {result}")
    model.eval()
    return {"strict": True, "missing_keys": [], "unexpected_keys": [],
            "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "checkpoint_step": int(package.get("step", 0))}


def save_checkpoint(out: Path, model: L80R1RegionCorrespondence,
                    optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace) -> Path:
    path = out / f"checkpoint_l80_r1_step{int(step)}.pt"
    torch.save({
        "format": "locatemot-l80-r1-region-checkpoint-v1", "stage": str(args.stage),
        "step": int(step), "seed": int(args.seed), "model_config": model.config.__dict__,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "region_interface": "three frozen CLIP taps, ROI 8x8 + context 4x4 + scene",
        "clip_weights_copied": False,
    }, path)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 training output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA changed")
    if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
        raise AssertionError("CLIP SHA changed")
    if args.steps <= 0:
        raise ValueError("R1 requires a positive --steps")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    rows = load_fit_units()
    config = L80R1Config()
    model = L80R1RegionCorrespondence(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    clip_model = None
    cache = FrameFeatureCache(max_items=int(args.cache_items))
    store = L80BankStore(max_history=config.history_length)
    label_cache: dict[str, dict[str, Any]] = {}
    key_digests: dict[str, str] = {}
    checkpoints: dict[str, str] = {}
    loss_trace: list[dict[str, Any]] = []
    sampling_trace: list[dict[str, Any]] = []
    grad_norms: list[float] = []
    domain_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    finite_steps = nonzero_steps = 0
    step = 0
    save_steps = {int(x) for x in str(args.save_steps).split(",") if x.strip()}
    try:
        clip_model = load_clip(device)
        model.train()
        schedule = stratified_schedule(rows, args.steps, args.seed)
        for item in schedule:
            metadata = key_only(item)
            if FORBIDDEN_LABEL_FIELDS.intersection(metadata):
                raise AssertionError(f"fit label leaked before R1 feature construction: {item['unit_key']}")
            batch = store.build_unit(metadata)
            digest = row_digest(batch.row_keys)
            if key_digests.setdefault(batch.unit_key, digest) != digest:
                raise AssertionError(f"R1 candidate key drift: {batch.unit_key}")
            raw = raw_inputs_for_unit_r1(clip_model, batch, device, cache)
            if tuple(raw["visual_tokens"].shape) != (batch.candidate_count, 243, 768):
                raise AssertionError(f"R1 visual shape drift: {batch.unit_key}")
            if batch.candidate_count != len(batch.row_keys) or batch.candidate_count != int(raw["visual_tokens"].shape[0]):
                raise AssertionError(f"R1 candidate count drift: {batch.unit_key}")
            # Explicit feature boundary: only now attach fit expression labels.
            full = label_cache.get(batch.unit_key)
            if full is None:
                full = load_full_unit_for_labels(batch.unit_key)
                label_cache[batch.unit_key] = full
            labels = store.attach_labels(batch, full)
            history = batch.history_observations.to(device=device).clone()
            history_mask = batch.history_mask.to(device=device).clone()
            history_frames = batch.history_frame_ids.to(device=device).clone()
            observations = batch.observations.to(device=device).clone()
            output = model(raw["visual_tokens"], raw["text_tokens"], raw["text_mask"], history,
                           history_mask, history_frames, batch.frame_id)
            loss, parts = l80_loss(output, labels["labels"], labels["coverage_mask"], observations,
                                   history_mask, labels["category"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"R1 nonfinite loss at {batch.unit_key}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite_grad = True
            any_nonzero = False
            for parameter in model.parameters():
                if parameter.grad is not None:
                    finite_grad = finite_grad and bool(torch.isfinite(parameter.grad).all())
                    any_nonzero = any_nonzero or bool((parameter.grad.abs() > 0).any())
            if not finite_grad or not any_nonzero:
                raise FloatingPointError(f"R1 invalid trainable gradient at {batch.unit_key}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)))
            optimizer.step()
            step += 1; finite_steps += 1; nonzero_steps += int(any_nonzero)
            domain_counts[batch.dataset] += 1; category_counts[labels["category"]] += 1
            loss_trace.append({"step": step, "loss_finite": True, "gradient_finite": True,
                               "gradient_nonzero": True, "gradient_norm": grad_norm,
                               "candidate_count": batch.candidate_count,
                               "candidate_key_digest": digest, "visual_forward_count": cache.visual_forward_count,
                               **parts})
            sampling_trace.append({"step": step, "unit_key": batch.unit_key, "dataset": batch.dataset,
                                   "video": batch.video, "frame_id": int(batch.frame_id),
                                   "category": labels["category"], "declared_category": labels["declared_category"],
                                   "candidate_count": int(batch.candidate_count),
                                   "positive_count": int(labels["positive_count"]),
                                   "schedule_position": int(item["schedule_position"]),
                                   "schedule_kind": item["schedule_kind"], "candidate_key_digest": digest,
                                   "candidate_deletion": False, "candidate_truncation": False,
                                   "labels_after_raw_features": True})
            grad_norms.append(grad_norm)
            if step in save_steps:
                checkpoints[str(step)] = str(save_checkpoint(out, model, optimizer, step, args))
            del output, raw, labels, batch, history, history_mask, history_frames, observations
            model.zero_grad(set_to_none=True)
            if step % 64 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if step != args.steps:
            raise AssertionError(f"R1 step count mismatch {step} != {args.steps}")
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# L80-R1 training — INCOMPLETE\n\n" + traceback.format_exc() +
            "\nNo screening/official-test labels, TrackEval/HOTA, ordinary MOT or OVMOT action was run.\n")
        raise
    finally:
        cache.clear(); store._store._bank = None; store._store._text_cache = None
        if clip_model is not None:
            del clip_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if str(args.steps) not in checkpoints:
        checkpoints[str(args.steps)] = str(save_checkpoint(out, model, optimizer, args.steps, args))
    reload_audit = {name: strict_reload(Path(path), device) for name, path in checkpoints.items()}
    metrics = {
        "format": "locatemot-l80-r1-fit-metrics-v1", "status": "complete",
        "stage": str(args.stage), "command": command, "seed": int(args.seed), "steps": int(args.steps),
        "finite_steps": int(finite_steps), "nonzero_gradient_steps": int(nonzero_steps),
        "fit_units_available": len(rows), "domains_seen": dict(domain_counts),
        "categories_seen": dict(category_counts), "model": model.parameter_report(),
        "region_interface": {"roi_grid": 8, "context_grid": 4, "tokens_per_scale": 81,
                              "region_tokens": 243, "changed_from_r0": "4x4/2x2 to 8x8/4x4"},
        "learning_rate": float(args.lr), "weight_decay": float(args.weight_decay),
        "candidate_rows_complete": True, "candidate_key_drift": 0,
        "candidate_deletion": False, "candidate_truncation": False,
        "raw_dense_cache_written": False, "persistent_frame_feature_cache": False,
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "labels_source": "fit-only expression-level L49/L69 membership, attached after raw feature construction",
        "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT),
        "visual_forward_count": int(cache.visual_forward_count), "checkpoint_paths": checkpoints,
        "checkpoint_reload": reload_audit,
        "gradient_norm": {"mean": float(np.mean(grad_norms)), "max": float(np.max(grad_norms)), "finite": True},
        "elapsed_sec": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
    }
    write_json(out / "metrics_l80_r1_fit.json", metrics)
    write_json(out / "config.json", {"format": "locatemot-l80-r1-fit-config-v1", "stage": args.stage,
        "seed": args.seed, "steps": args.steps, "fit_only": True, "device": str(device),
        "model_config": config.__dict__, "cache_items": args.cache_items,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip},
        "checkpoint_schedule": args.save_steps, "candidate_set": "complete L69 rows; no top-k/NMS/deletion",
        "region_interface": "8x8 ROI + 4x4 context + scene at three frozen CLIP taps",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
    write_json(out / "reload_audit.json", {"format": "locatemot-l80-r1-reload-audit-v1", "status": "complete",
        "checkpoints": reload_audit, "strict": True, "missing_keys": [], "unexpected_keys": []})
    (out / "loss_trace.json").write_text(json.dumps(loss_trace, indent=2) + "\n")
    (out / "sampling_trace.json").write_text(json.dumps(sampling_trace, indent=2) + "\n")
    write_json(out / "provenance.json", {"format": "locatemot-l80-r1-fit-provenance-v1", "status": "complete",
        "command": command, "cwd": str(Path.cwd().resolve()),
        "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "inputs": {"manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
                   "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
                   "l69_features": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                   "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT)},
        "outputs": [str(out / "metrics_l80_r1_fit.json"), str(out / "loss_trace.json"), str(out / "sampling_trace.json")],
        "fit_only": True, "region_only_change": True,
        "labels_attached_after_raw_feature_construction": True, "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED", "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False, "old_assets_modified": False})
    write_json(out / "status.json", {"format": "locatemot-l80-r1-status-v1", "status": "complete",
        "stage": args.stage, "command": command, "outputs": list(checkpoints.values()),
        "failure_root_cause": None, "next_action": "run fixed R1 semantic evaluation",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-steps", default="100,500")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", default="r1-bounded-fit-probe")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--cache-items", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
