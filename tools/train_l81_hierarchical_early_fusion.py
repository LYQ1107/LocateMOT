#!/usr/bin/env python3
"""L81 fit-only overfit32 and bounded 500-update probe.

The trainer intentionally imports the canonical L80 loss rather than copying
or modifying it.  It constructs a key-only L69 unit and raw feature first,
then attaches fit labels.  No calibration, validation, screening or official
test labels are loaded here.
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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l81_hierarchical_early_fusion import L81Config, L81HierarchicalEarlyFusion  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    CATEGORIES,
    DATASETS,
    EXPECTED_MANIFEST_SHA,
    FORBIDDEN_LABEL_FIELDS,
    L80BankStore,
    MANIFEST,
    key_only,
    load_fit_units,
    load_full_unit_for_labels,
    sha256_file,
)
from locatemot.rmot.l80_losses import l80_loss  # noqa: E402
from locatemot.rmot.l81_runtime import (  # noqa: E402
    CLIP_SHA256,
    CLIP_WEIGHT,
    FrameFeatureCache,
    load_clip,
    raw_inputs_for_l81,
)
from tools.train_l80_v12_joint import stratified_schedule  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
EXPECTED_CONFIG = L81Config().__dict__.copy()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def row_digest(row_keys: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(json.dumps([list(x) for x in row_keys], sort_keys=False).encode()).hexdigest()


def overfit_schedule(rows: list[dict[str, Any]], steps: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["unit_key"]))[:32]
    if len(ordered) != 32 or len({str(row["unit_key"]) for row in ordered}) != 32:
        raise AssertionError("overfit32 key selection drift")
    result = []
    for step in range(int(steps)):
        item = dict(ordered[step % len(ordered)])
        item["schedule_position"] = step
        item["schedule_kind"] = "stable_lexicographic_first32_cyclic"
        result.append(item)
    return result


def save_checkpoint(out: Path, model: L81HierarchicalEarlyFusion,
                    optimizer: torch.optim.Optimizer, step: int, stage: str,
                    args: argparse.Namespace) -> Path:
    path = out / f"checkpoint_l81_step{int(step)}.pt"
    payload = {
        "format": "locatemot-l81-hierarchical-early-fusion-checkpoint-v1",
        "stage": str(stage), "step": int(step), "seed": int(args.seed),
        "model_config": model.config.__dict__, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "raw_visual_encoder": "frozen local OpenAI CLIP ViT-B/16; no CLIP weights copied",
        "canonical_loss": "locatemot.rmot.l80_losses.l80_loss",
        "candidate_set": "complete L69 rows; no top-k/NMS/deletion",
    }
    torch.save(payload, path)
    return path


def strict_reload(path: Path, device: torch.device) -> dict[str, Any]:
    package = torch.load(path, map_location=device, weights_only=False)
    config = L81Config(**package["model_config"])
    model = L81HierarchicalEarlyFusion(config).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"L81 strict reload mismatch: {result}")
    model.eval()
    return {
        "strict": True, "missing_keys": [], "unexpected_keys": [],
        "step": int(package.get("step", 0)),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_sha256": sha256_file(path),
    }


def gradient_check(output: dict[str, torch.Tensor], labels: dict[str, Any]) -> dict[str, Any]:
    score = output["candidate_logits"]
    if not score.requires_grad:
        raise AssertionError("candidate logits are detached during fit")
    score.retain_grad()
    positive = labels["labels"].to(device=score.device).bool()
    covered = bool(labels["membership_mask"].all())
    grad = score.grad
    if grad is None:
        return {"positive_nonzero": False, "negative_nonzero": False, "minimum_positive_nonzero": False,
                "membership_masked": not covered, "membership_gradient_expected": covered,
                "valid": not covered}
    finite = bool(torch.isfinite(grad).all())
    pos_nonzero = bool((grad[positive].abs() > 0).any()) if bool(positive.any()) else True
    neg_nonzero = bool((grad[~positive].abs() > 0).any()) if bool((~positive).any()) else True
    min_nonzero = bool((grad[positive].abs() > 0).all()) if bool(positive.any()) and covered else True
    return {
        "positive_nonzero": pos_nonzero, "negative_nonzero": neg_nonzero,
        "minimum_positive_nonzero": min_nonzero, "membership_masked": not covered,
        "membership_gradient_expected": covered,
        "finite": finite, "valid": finite and (pos_nonzero or not covered) and (neg_nonzero or not covered),
        "positive_count": int(positive.sum()), "negative_count": int((~positive).sum()),
        "min_positive_grad_abs": float(grad[positive].abs().min().detach().cpu()) if bool(positive.any()) else None,
    }


def model_forward(model: L81HierarchicalEarlyFusion, raw: dict[str, Any], batch: Any,
                  device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    history = batch.history_observations.to(device=device).clone()
    history_mask = batch.history_mask.to(device=device).clone()
    history_frames = batch.history_frame_ids.to(device=device).clone()
    output = model(
        raw["visual_pyramid"], raw["local_tokens"], raw["text_tokens"], raw["text_mask"],
        history, history_mask, history_frames, int(batch.frame_id), raw["boxes_norm"],
    )
    del history, history_mask, history_frames
    return output, batch.observations.to(device=device).clone()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L81 training output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA changed")
    if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
        raise AssertionError("CLIP SHA changed")
    if EXPECTED_CONFIG != L81Config().__dict__:
        raise AssertionError("L81 pre-registered configuration changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("L81 fit requires registered GPU0 CUDA runtime")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = load_fit_units()
    if args.stage == "overfit32":
        if args.steps != 200:
            raise AssertionError("overfit32 is fixed to exactly 200 updates")
        schedule = overfit_schedule(rows, args.steps)
        selected_keys = sorted({str(x["unit_key"]) for x in schedule})
        write_json(out / "unit_keys.json", {
            "format": "locatemot-l81-overfit32-unit-keys-v1", "status": "complete",
            "selection": "stable lexicographic first 32 L80 fit unit keys",
            "unit_count": len(selected_keys), "unit_keys": selected_keys,
            "seed": int(args.seed), "labels_used_for_selection": False,
        })
    elif args.stage == "probe500":
        if args.steps != 500:
            raise AssertionError("probe500 is fixed to exactly 500 updates")
        schedule = stratified_schedule(rows, args.steps, args.seed)
    else:
        raise ValueError(f"unknown L81 stage {args.stage}")
    model = L81HierarchicalEarlyFusion(L81Config()).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    clip_model = load_clip(device)
    cache = FrameFeatureCache(max_items=int(args.cache_items))
    store = L80BankStore(max_history=model.config.history_length)
    known_key_digests: dict[str, str] = {}
    loss_trace: list[dict[str, Any]] = []
    sampling_trace: list[dict[str, Any]] = []
    checkpoints: dict[str, str] = {}
    domain_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    gradient_checks: list[dict[str, Any]] = []
    finite_steps = 0
    nonzero_steps = 0
    step = 0
    try:
        model.train()
        for item in schedule:
            if step >= int(args.steps):
                break
            metadata = key_only(item)
            if FORBIDDEN_LABEL_FIELDS.intersection(metadata):
                raise AssertionError(f"fit label leaked before raw feature: {item['unit_key']}")
            batch = store.build_unit(metadata)
            digest = row_digest(batch.row_keys)
            if known_key_digests.setdefault(batch.unit_key, digest) != digest:
                raise AssertionError(f"candidate key drift: {batch.unit_key}")
            raw = raw_inputs_for_l81(clip_model, batch, device, cache)
            if int(raw["local_tokens"].shape[0]) != batch.candidate_count:
                raise AssertionError(f"candidate count drift after raw feature: {batch.unit_key}")
            # This is the first label access for the unit, after complete raw
            # representation construction.
            full = load_full_unit_for_labels(batch.unit_key)
            labels = store.attach_labels(batch, full)
            output, current_observations = model_forward(model, raw, batch, device)
            output["candidate_logits"].retain_grad()
            loss, parts = l80_loss(
                output, labels["labels"], labels["membership_mask"], current_observations,
                batch.history_mask.to(device=device).clone(), labels["category"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite L81 loss at {batch.unit_key}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite_grad = True
            any_nonzero = False
            for parameter in model.parameters():
                if parameter.grad is not None:
                    finite_grad = finite_grad and bool(torch.isfinite(parameter.grad).all())
                    any_nonzero = any_nonzero or bool((parameter.grad.abs() > 0).any())
            check = gradient_check(output, labels)
            check["unit_key"] = batch.unit_key
            gradient_checks.append(check)
            if not finite_grad or not any_nonzero or not check["valid"]:
                raise FloatingPointError(f"invalid L81 gradient at {batch.unit_key}: {check}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)))
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite clipped L81 gradient at {batch.unit_key}")
            optimizer.step()
            step += 1; finite_steps += 1; nonzero_steps += int(any_nonzero)
            domain_counts[str(batch.dataset)] += 1
            category_counts[str(labels["category"])] += 1
            loss_trace.append({
                "step": step, **parts, "loss_finite": True, "gradient_finite": finite_grad,
                "gradient_nonzero": any_nonzero, "gradient_norm": grad_norm,
                "candidate_count": batch.candidate_count, "candidate_key_digest": digest,
                "visual_forward_count": cache.visual_forward_count, "cache_items": len(cache),
            })
            sampling_trace.append({
                "step": step, "unit_key": batch.unit_key, "dataset": batch.dataset,
                "video": batch.video, "frame_id": batch.frame_id,
                "category": labels["category"], "declared_category": labels["declared_category"],
                "candidate_count": batch.candidate_count, "positive_count": labels["positive_count"],
                "schedule_position": int(item.get("schedule_position", step - 1)),
                "schedule_kind": item.get("schedule_kind", "unknown"),
                "candidate_key_digest": digest, "candidate_deletion": False,
                "candidate_truncation": False, "labels_attached_after_raw_features": True,
                "present_uncovered_membership_masked": labels["category"] == "present_uncovered",
            })
            save_steps = {int(x) for x in args.save_steps.split(",") if x}
            if step in save_steps:
                checkpoint = save_checkpoint(out, model, optimizer, step, args.stage, args)
                checkpoints[str(step)] = str(checkpoint)
            del output, current_observations, raw, labels, full, batch, metadata
            model.zero_grad(set_to_none=True)
            if step % 32 == 0:
                gc.collect(); torch.cuda.empty_cache()
        if step != int(args.steps):
            raise AssertionError(f"L81 step count mismatch: {step} != {args.steps}")
        if args.stage == "overfit32":
            first = float(np.median([x["total"] for x in loss_trace[:20]]))
            last = float(np.median([x["total"] for x in loss_trace[-20:]]))
            if not last < first:
                raise AssertionError(f"overfit32 loss did not descend: first20={first}, last20={last}")
        if str(args.steps) not in checkpoints:
            checkpoint = save_checkpoint(out, model, optimizer, args.steps, args.stage, args)
            checkpoints[str(args.steps)] = str(checkpoint)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            f"# L81 {args.stage} — INCOMPLETE\n\n" + traceback.format_exc() +
            "\nNo calibration/validation/screening/test labels, TrackEval/HOTA, ordinary MOT or OVMOT action was run.\n")
        write_json(out / "status.json", {
            "format": "locatemot-l81-fit-status-v1", "status": "fit_learnability_fail",
            "stage": args.stage, "steps_completed": step, "command": command,
            "failure_root_cause": "see INCOMPLETE.md first traceback",
            "next_action": "stop before semantic evaluation and perform one targeted regression",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        raise
    finally:
        cache.clear(); store._store._bank = None; store._store._text_cache = None
        del clip_model
        gc.collect(); torch.cuda.empty_cache()
    reload_audit = {name: strict_reload(Path(path), device) for name, path in checkpoints.items()}
    all_required_gradient_steps = all(int(item["gradient_nonzero"]) if isinstance(item["gradient_nonzero"], bool) else bool(item["gradient_nonzero"]) for item in loss_trace)
    category_complete = set(category_counts) == set(CATEGORIES) if args.stage == "probe500" else True
    metrics = {
        "format": "locatemot-l81-fit-metrics-v1", "status": "complete", "stage": args.stage,
        "command": command, "seed": int(args.seed), "steps": int(args.steps),
        "fit_units_available": len(rows), "schedule_units": len(schedule),
        "finite_steps": int(finite_steps), "nonzero_gradient_steps": int(nonzero_steps),
        "domains_seen": dict(domain_counts), "categories_seen": dict(category_counts),
        "all_registered_categories_seen": category_complete,
        "model": model.parameter_report(), "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay), "gradient_clip": float(args.grad_clip),
        "history_length": model.config.history_length, "candidate_rows_complete": True,
        "candidate_key_drift": 0, "candidate_deletion": False, "candidate_truncation": False,
        "present_uncovered_membership_masked": True,
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "labels_source": "fit-only expression-level L49/L69 membership, attached after raw feature construction",
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT),
        "visual_forward_count": int(cache.visual_forward_count),
        "checkpoint_paths": checkpoints, "checkpoint_reload": reload_audit,
        "loss_first20_median": float(np.median([x["total"] for x in loss_trace[:20]])),
        "loss_last20_median": float(np.median([x["total"] for x in loss_trace[-20:]])),
        "all_steps_finite": finite_steps == int(args.steps),
        "all_steps_nonzero_gradient": nonzero_steps == int(args.steps),
        "per_unit_candidate_gradient_checks": gradient_checks,
        "elapsed_sec": time.perf_counter() - started,
        "peak_memory_bytes": peak_memory,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
    }
    write_json(out / "metrics_l81_fit.json", metrics)
    write_json(out / "config.json", {
        "format": "locatemot-l81-fit-config-v1", "stage": args.stage,
        "seed": args.seed, "steps": args.steps, "fit_only": True, "device": str(device),
        "model_config": model.config.__dict__, "cache_items": args.cache_items,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
                      "grad_clip": args.grad_clip, "precision": "FP32 trainable head + CUDA BF16 frozen CLIP"},
        "checkpoint_schedule": args.save_steps,
        "candidate_set": "complete L69 rows; no top-k/NMS/deletion",
        "sampler": "stable first32 cyclic" if args.stage == "overfit32" else "L80 R0 stratified_schedule",
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
    })
    write_json(out / "reload_audit.json", {"format": "locatemot-l81-reload-audit-v1", "status": "complete",
        "checkpoints": reload_audit, "strict": True, "missing_keys": [], "unexpected_keys": []})
    write_json(out / "loss_trace.json", loss_trace)
    write_json(out / "sampling_trace.json", sampling_trace)
    write_json(out / "provenance.json", {
        "format": "locatemot-l81-fit-provenance-v1", "status": "complete", "command": command,
        "cwd": str(Path.cwd().resolve()), "project_root": str(ROOT), "luna_thread": THREAD,
        "inputs": {"manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
                   "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
                   "l69_features": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                   "l48_text": str(ROOT / "outputs/l48/data/text_cache.pt"),
                   "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT),
                   "canonical_loss_module": str(ROOT / "locatemot/rmot/l80_losses.py"),
                   "canonical_loss_import": "from locatemot.rmot.l80_losses import l80_loss"},
        "outputs": [str(out / name) for name in ("metrics_l81_fit.json", "loss_trace.json", "sampling_trace.json")],
        "fit_only": True, "labels_attached_after_raw_feature_construction": True,
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        "raw_dense_cache_written": False, "persistent_frame_feature_cache_serialized": False,
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "old_assets_modified": False,
    })
    write_json(out / "status.json", {
        "format": "locatemot-l81-fit-status-v1", "status": "complete", "stage": args.stage,
        "command": command, "inputs": [str(MANIFEST), str(CLIP_WEIGHT)],
        "outputs": list(checkpoints.values()) + [str(out / "metrics_l81_fit.json")],
        "failure_root_cause": None,
        "next_action": "run exactly one 500-update probe" if args.stage == "overfit32" else "run fixed calibration/validation semantic evaluation",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
    })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("overfit32", "probe500"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-steps", default="200")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--cache-items", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
