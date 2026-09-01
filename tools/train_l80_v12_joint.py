#!/usr/bin/env python3
"""L80 fit-only training for the raw regional correspondence model.

The script never loads calibration/validation/screening/test labels. It
constructs a label-free L69/raw-image unit first, then attaches the fit
expression membership labels for that unit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))
from locatemot.models.l80_raw_region_correspondence import L80Config, L80RawRegionCorrespondence  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    CATEGORIES, DATASETS, EXPECTED_MANIFEST_SHA, FIT_VIDEOS, FORBIDDEN_LABEL_FIELDS,
    L80BankStore, MANIFEST, key_only, load_fit_units, load_full_unit_for_labels,
    sha256_file,
)
from locatemot.rmot.l80_losses import l80_loss  # noqa: E402
from locatemot.rmot.l80_runtime import (  # noqa: E402
    CLIP_SHA256, CLIP_WEIGHT, FrameFeatureCache, load_clip, raw_inputs_for_unit,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def row_digest(row_keys: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(json.dumps([list(x) for x in row_keys], sort_keys=False).encode()).hexdigest()


def stratified_schedule(rows: list[dict[str, Any]], steps: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["dataset"]), str(row["category"]))].append(row)
    required = {(dataset, category) for dataset in DATASETS for category in CATEGORIES}
    if set(buckets) != required:
        raise AssertionError(f"fit strata changed: {sorted(set(buckets) ^ required)}")
    rng = np.random.default_rng(int(seed))
    ordered: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key in sorted(required):
        values = sorted(buckets[key], key=lambda x: (str(x["video"]), int(x["frame_id"]), int(x["query_id"]), str(x["unit_key"])))
        order = rng.permutation(len(values)).tolist()
        ordered[key] = [values[int(i)] for i in order]
    keys = sorted(required)
    result = []
    for position in range(int(steps)):
        key = keys[position % len(keys)]
        item = dict(ordered[key][(position // len(keys)) % len(ordered[key])])
        item["schedule_position"] = position
        item["schedule_kind"] = "stratified_cyclic_fit"
        result.append(item)
    return result


def full_epoch_schedule(rows: list[dict[str, Any]], epoch: int, seed: int) -> list[dict[str, Any]]:
    """Use every fit row once while grouping same frames for raw-forward reuse."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["video"]), int(row["frame_id"]))].append(row)
    rng = np.random.default_rng(int(seed) + int(epoch) * 104729)
    by_video: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in groups:
        by_video[key[0]].append(key)
    output = []
    position = 0
    for video in sorted(by_video):
        frame_keys = sorted(by_video[video], key=lambda x: x[1])
        permutation = rng.permutation(len(frame_keys)).tolist()
        for index in permutation:
            frame_key = frame_keys[int(index)]
            values = sorted(groups[frame_key], key=lambda x: (str(x["dataset"]), int(x["query_id"]), str(x["unit_key"])))
            for row in values:
                item = dict(row)
                item["schedule_position"] = position
                item["epoch"] = int(epoch)
                item["schedule_kind"] = "full_fit_grouped_by_frame"
                output.append(item)
                position += 1
    if len(output) != len(rows) or {str(x["unit_key"]) for x in output} != {str(x["unit_key"]) for x in rows}:
        raise AssertionError("full epoch schedule is not one-to-one")
    return output


def strict_reload(checkpoint: Path, device: torch.device) -> dict[str, Any]:
    package = torch.load(checkpoint, map_location=device, weights_only=False)
    config = L80Config(**package["model_config"])
    model = L80RawRegionCorrespondence(config).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict reload mismatch: {result}")
    model.eval()
    return {"strict": True, "missing_keys": [], "unexpected_keys": [],
            "model_parameter_count": sum(p.numel() for p in model.parameters())}


def save_checkpoint(out: Path, model: L80RawRegionCorrespondence, optimizer: torch.optim.Optimizer,
                    step: int, epoch: int, args: argparse.Namespace) -> Path:
    path = out / f"checkpoint_l80_step{int(step)}.pt"
    payload = {
        "format": "locatemot-l80-raw-region-correspondence-checkpoint-v1",
        "stage": str(args.stage), "step": int(step), "epoch": int(epoch), "seed": int(args.seed),
        "model_config": model.config.__dict__, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "raw_visual_encoder": "frozen local OpenAI CLIP ViT-B/16; no CLIP weights copied into checkpoint",
    }
    torch.save(payload, path)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty training output {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    start = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA changed")
    if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
        raise AssertionError("CLIP SHA changed")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    rows = load_fit_units()
    if args.epochs > 0 and args.steps > 0:
        raise ValueError("choose --steps or --epochs, not both")
    if args.epochs <= 0 and args.steps <= 0:
        raise ValueError("positive --steps or --epochs required")
    total_steps = int(args.epochs * len(rows)) if args.epochs > 0 else int(args.steps)
    config = L80Config()
    model = L80RawRegionCorrespondence(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    clip_model = load_clip(device)
    cache = FrameFeatureCache(max_items=int(args.cache_items))
    store = L80BankStore(max_history=config.history_length)
    fit_label_cache: dict[str, dict[str, Any]] = {}
    known_key_digests: dict[str, str] = {}
    loss_trace, sampling_trace, grad_norms = [], [], []
    checkpoints: dict[str, str] = {}
    finite_steps = nonzero_steps = 0
    domain_counts = Counter()
    category_counts = Counter()
    step = 0
    try:
        model.train()
        for epoch in range(args.epochs if args.epochs > 0 else 1):
            schedule = full_epoch_schedule(rows, epoch, args.seed) if args.epochs > 0 else stratified_schedule(rows, total_steps, args.seed)
            for item in schedule:
                if step >= total_steps:
                    break
                metadata = key_only(item)
                if FORBIDDEN_LABEL_FIELDS.intersection(metadata):
                    raise AssertionError(f"fit label leaked before feature construction: {item['unit_key']}")
                batch = store.build_unit(metadata)
                digest = row_digest(batch.row_keys)
                old_digest = known_key_digests.setdefault(batch.unit_key, digest)
                if old_digest != digest:
                    raise AssertionError(f"candidate key drift {batch.unit_key}")
                raw = raw_inputs_for_unit(clip_model, batch, device, cache)
                if batch.candidate_count != int(raw["visual_tokens"].shape[0]):
                    raise AssertionError(f"candidate count drift {batch.unit_key}")
                # Explicit boundary: raw feature construction completed.
                full = fit_label_cache.get(batch.unit_key)
                if full is None:
                    full = load_full_unit_for_labels(batch.unit_key)
                    fit_label_cache[batch.unit_key] = full
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
                    raise FloatingPointError(f"nonfinite loss at {batch.unit_key}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                finite_grad = True
                any_nonzero = False
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        finite_grad = finite_grad and bool(torch.isfinite(parameter.grad).all())
                        any_nonzero = any_nonzero or bool((parameter.grad.abs() > 0).any())
                if not finite_grad or not any_nonzero:
                    raise FloatingPointError(f"invalid trainable gradient at {batch.unit_key}")
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)))
                optimizer.step()
                step += 1
                finite_steps += 1
                nonzero_steps += int(any_nonzero)
                domain_counts[str(batch.dataset)] += 1
                category_counts[str(labels["category"])] += 1
                loss_trace.append({"step": step, "epoch": epoch + 1, **parts,
                                   "loss_finite": True, "gradient_finite": True,
                                   "gradient_nonzero": True, "gradient_norm": norm,
                                   "candidate_count": batch.candidate_count,
                                   "candidate_key_digest": digest, "cache_items": len(cache),
                                   "visual_forward_count": cache.visual_forward_count})
                sampling_trace.append({"step": step, "unit_key": batch.unit_key, "dataset": batch.dataset,
                                       "video": batch.video, "frame_id": batch.frame_id,
                                       "category": labels["category"], "declared_category": labels["declared_category"],
                                       "candidate_count": batch.candidate_count, "positive_count": labels["positive_count"],
                                       "schedule_position": int(item.get("schedule_position", step - 1)),
                                       "schedule_kind": item.get("schedule_kind", "unknown"),
                                       "candidate_key_digest": digest, "candidate_deletion": False,
                                       "candidate_truncation": False, "labels_after_raw_features": True})
                grad_norms.append(norm)
                save_steps = {int(x) for x in args.save_steps.split(",") if x}
                if step in save_steps:
                    checkpoint = save_checkpoint(out, model, optimizer, step, epoch + 1, args)
                    checkpoints[str(step)] = str(checkpoint)
                del output, raw, labels, batch, history, history_mask, history_frames, observations
                model.zero_grad(set_to_none=True)
                if step % 64 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if args.epochs > 0 and (epoch + 1) in {1, 5, 10, 20, 30, 40, 50, 60, 100}:
                checkpoint = save_checkpoint(out, model, optimizer, step, epoch + 1, args)
                checkpoints[f"epoch{epoch + 1:02d}"] = str(checkpoint)
        if step != total_steps:
            raise AssertionError(f"training step count mismatch {step} != {total_steps}")
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# L80 training — INCOMPLETE\n\n" + traceback.format_exc() +
            "\nNo screening/official-test labels, TrackEval/HOTA, ordinary MOT or OVMOT action was run.\n")
        raise
    finally:
        cache.clear()
        store._store._bank = None
        store._store._text_cache = None
        del clip_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if str(total_steps) not in checkpoints:
        checkpoint = save_checkpoint(out, model, optimizer, total_steps, args.epochs or 1, args)
        checkpoints[str(total_steps)] = str(checkpoint)
    reload_audit = {name: strict_reload(Path(path), device) for name, path in checkpoints.items()}
    metrics = {
        "format": "locatemot-l80-fit-metrics-v1", "status": "complete", "stage": str(args.stage),
        "command": command, "seed": int(args.seed), "steps": int(total_steps), "epochs": int(args.epochs),
        "fit_units_available": len(rows), "finite_steps": int(finite_steps), "nonzero_gradient_steps": int(nonzero_steps),
        "domains_seen": dict(domain_counts), "categories_seen": dict(category_counts),
        "model": model.parameter_report(), "learning_rate": float(args.lr), "weight_decay": float(args.weight_decay),
        "history_length": config.history_length, "candidate_rows_complete": True,
        "candidate_key_drift": 0, "candidate_deletion": False, "candidate_truncation": False,
        "raw_dense_cache_written": False, "persistent_frame_feature_cache": False,
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "labels_source": "fit-only expression-level L49/L69 membership, attached after raw feature construction",
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT),
        "visual_forward_count": int(cache.visual_forward_count),
        "checkpoint_paths": checkpoints, "checkpoint_reload": reload_audit,
        "gradient_norm": {"mean": float(np.mean(grad_norms)), "max": float(np.max(grad_norms)), "finite": True},
        "elapsed_sec": time.perf_counter() - start,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
    }
    write_json(out / "metrics_l80_fit.json", metrics)
    write_json(out / "config.json", {"format": "locatemot-l80-fit-config-v1", "stage": args.stage,
        "seed": args.seed, "steps": total_steps, "epochs": args.epochs, "fit_only": True,
        "device": str(device), "model_config": config.__dict__, "cache_items": args.cache_items,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip},
        "checkpoint_schedule": args.save_steps, "candidate_set": "complete L69 rows; no top-k/NMS/deletion",
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False})
    write_json(out / "reload_audit.json", {"format": "locatemot-l80-reload-audit-v1", "status": "complete",
        "checkpoints": reload_audit, "strict": True, "missing_keys": [], "unexpected_keys": []})
    (out / "loss_trace.json").write_text(json.dumps(loss_trace, indent=2) + "\n")
    (out / "sampling_trace.json").write_text(json.dumps(sampling_trace, indent=2) + "\n")
    write_json(out / "provenance.json", {"format": "locatemot-l80-fit-provenance-v1", "status": "complete", "command": command,
        "cwd": str(Path.cwd().resolve()), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "inputs": {"manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
                   "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
                   "l69_features": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                   "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT)},
        "outputs": [str(out / "metrics_l80_fit.json"), str(out / "loss_trace.json"), str(out / "sampling_trace.json")],
        "fit_only": True, "labels_attached_after_raw_feature_construction": True,
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False, "old_assets_modified": False})
    write_json(out / "status.json", {"format": "locatemot-l80-status-v1", "status": "complete", "stage": args.stage,
        "command": command, "inputs": [str(MANIFEST), str(CLIP_WEIGHT)],
        "outputs": list(metrics["checkpoint_paths"].values()) + [str(out / "metrics_l80_fit.json")],
        "failure_root_cause": None, "next_action": "run fixed L80 calibration/validation evaluation",
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False})
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--save-steps", default="100,250,500")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", default="bounded-fit-probe")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--cache-items", type=int, default=16)
    args = parser.parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
