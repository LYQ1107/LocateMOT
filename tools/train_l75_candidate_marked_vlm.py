#!/usr/bin/env python3
"""Fit-only L75 candidate-marked VLM sidecar training.

The detector is frozen and is run once per active image.  Only the final four
language-layer q/k/v/o LoRA matrices, region marker, and compact matcher are
updated.  Candidate rows are never deleted; at fit time only a deterministic
query-independent subset of negatives is used to bound language-model cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l75_candidate_marked_vlm import CandidateMarkedVLMMatcher  # noqa: E402
from locatemot.rmot.l75_data import (  # noqa: E402
    IMAGE_ROOT, L75Bank, MANIFEST_PATH, MANIFEST_SHA256, load_splits,
    make_record, sha256_file, unit_key,
)
from locatemot.rmot.l75_runtime import (  # noqa: E402
    attach_language_lora, frozen_target_digest, language_forward,
    load_locateanything, lora_state_dict, load_lora_state_dict,
    marked_visual_batch, prepare_visual, region_value_batch,
)
from locatemot.rmot.l75_train_utils import (  # noqa: E402
    gradient_row_summary, l75_loss, sample_candidate_indices,
)

SEED = 20260829
STRATA = ("positive", "multi_positive", "inactive", "present_uncovered")
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def deterministic_schedule(splits: dict[str, list[dict[str, Any]]], steps: int) -> list[dict[str, Any]]:
    """Select a fixed stratified fit schedule, then group by video for I/O."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        for category in STRATA:
            values = sorted(
                [row for row in splits["fit"] if str(row["dataset"]) == dataset
                 and str(row.get("category")) == category], key=unit_key
            )
            if not values:
                raise AssertionError(f"missing fit stratum {dataset}/{category}")
            groups[(dataset, category)] = values
    selected: list[dict[str, Any]] = []
    cursors = {key: 0 for key in groups}
    group_order = [(dataset, category) for dataset in DATASETS for category in STRATA]
    while len(selected) < int(steps):
        progressed = False
        for key in group_order:
            if len(selected) >= int(steps):
                break
            cursor = cursors[key]
            if cursor < len(groups[key]):
                selected.append(groups[key][cursor])
                cursors[key] += 1
                progressed = True
        if not progressed:
            raise AssertionError("fit schedule exhausted before requested steps")
    # Keep all category/domain counts fixed, but avoid reloading the large bank
    # on every alternating stratum.
    return sorted(selected, key=lambda row: (str(row["video"]), str(row["dataset"]),
                                             str(row.get("category")), unit_key))


def checkpoint_payload(model: Any, matcher: Any, lora_contract: dict[str, Any],
                       optimizer: Any, step: int, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "locatemot-l75-adapter-only-v1",
        "step": int(step),
        "matcher": {key: value.detach().cpu().clone() for key, value in matcher.state_dict().items()},
        "lora": lora_state_dict(model),
        "lora_contract": lora_contract,
        "optimizer": optimizer.state_dict(),
        "config": config,
    }


def unit_step(model: Any, matcher: Any, processor: Any, tokenizer: Any,
              unit: dict[str, Any], device: str, candidate_chunk: int,
              max_negatives: int, optimizer: Any,
              capture_reload: bool = False) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from PIL import Image

    bank = L75Bank(str(unit["video"]))
    reload_payload = None
    try:
        record = make_record(unit, bank, include_labels=True)
        image_path = IMAGE_ROOT / str(unit["video"]) / f"{int(unit['frame_id']):06d}.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        if record.get("image_size_declared") and record["image_size_declared"] != [image.width, image.height]:
            raise AssertionError(f"image size mismatch for {record['unit_key']}")
        rows = record["row_offsets"]
        boxes = bank.tensors["box"].index_select(
            0, torch.as_tensor(rows, dtype=torch.long)
        ).float().tolist()
        prepared = prepare_visual(model, processor, tokenizer, image,
                                  record["sentence"], boxes)
        if len(prepared["candidate_cells"]) != record["candidate_count"]:
            raise AssertionError("candidate mapping count drift")
        selected = sample_candidate_indices(record, bank, max_negatives=max_negatives)
        if record["category"] in ("positive", "multi_positive"):
            if not all(index in selected for index in record["positive_indices"]):
                raise AssertionError("a positive row was omitted from fit subset")
        selected_labels = [bool(record["labels"][index]) for index in selected]
        optimizer.zero_grad(set_to_none=True)
        matcher.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = None
        base_visual = prepared["base_visual"].to(device=device)
        total_selected = len(selected)
        if not total_selected:
            raise AssertionError("empty selected candidate set")
        # Keep every autograd graph bounded by the registered candidate chunk.
        # Positives are ordered first by sample_candidate_indices; their
        # detached logits are accumulated below for later negative chunks.
        # Never expand a chunk to the number of positives: that was the first
        # actionable cause of the formal-run OOM.
        chunk_ranges = list(range(0, total_selected, int(candidate_chunk)))
        loss_parts: list[dict[str, float]] = []
        row_gradient_parts: list[dict[str, Any]] = []
        reload_input = None
        positive_reference_values: list[torch.Tensor] = []
        score_by_index: dict[int, float] = {}
        for chunk_no, start in enumerate(chunk_ranges):
            local_indices = selected[start:start + int(candidate_chunk)]
            cells = [prepared["candidate_cells"][index] for index in local_indices]
            marked, _ = marked_visual_batch(base_visual, cells, matcher.region_marker)
            regions, region_mask = region_value_batch(marked, cells)
            hidden = language_forward(model, prepared, marked, inference=False)
            output = matcher(hidden, prepared["expression_positions"], regions, region_mask)
            output["match_logit"].retain_grad()
            labels = torch.as_tensor(
                [record["labels"][index] for index in local_indices],
                dtype=torch.float32, device=device,
            )
            loss, parts = l75_loss(
                output["match_logit"], output["absent_logit"], labels,
                record["category"], bool(record["coverage_mask"]), matcher.region_marker,
            )
            # The first streamed chunk contains every positive.  Reuse its
            # detached values for pairwise pressure on each later negative
            # chunk, without retaining the positive language graph across the
            # whole candidate set.
            if (not bool(labels.any()) and positive_reference_values
                    and bool(record["coverage_mask"])):
                positive_reference = torch.cat(positive_reference_values, dim=0).to(device=device)
                pairwise = F.softplus(
                    0.20 - positive_reference.float().unsqueeze(1) +
                    output["match_logit"].float().reshape(1, -1)
                ).mean()
                loss = loss + pairwise
                parts["pairwise"] = float(pairwise.detach().cpu())
            (loss / max(1, len(chunk_ranges))).backward()
            loss_parts.append(parts)
            row_gradient_parts.append(gradient_row_summary(
                output["match_logit"], labels, bool(record["coverage_mask"])
            ))
            for local_index, value in zip(local_indices, output["match_logit"].detach().float().cpu().tolist()):
                score_by_index[int(local_index)] = float(value)
            if bool(labels.any()):
                # Detach only the small positive score vector.  Keeping the
                # hidden graph would recreate the OOM when a unit has many
                # positive rows; every positive still received BCE/min loss
                # in its own bounded chunk.
                positive_reference_values.append(output["match_logit"].detach().float())
            if capture_reload and chunk_no == len(chunk_ranges) - 1:
                reload_input = {
                    "hidden": hidden.detach().float().cpu(),
                    "regions": regions.detach().float().cpu(),
                    "region_mask": region_mask.detach().cpu(),
                    "expected_logits": output["match_logit"].detach().float().cpu(),
                    "expression_positions": list(prepared["expression_positions"]),
                }
            del hidden, marked, regions, output, loss
        named_trainable = list(model.named_parameters()) + list(matcher.named_parameters())
        trainable = [parameter for _, parameter in named_trainable if parameter.requires_grad]
        masked_uncovered = record["category"] == "present_uncovered" and not bool(record["coverage_mask"])
        observed_gradients = [parameter for parameter in trainable if parameter.grad is not None]
        if masked_uncovered:
            # Membership is intentionally masked for a present-but-uncovered
            # unit.  It is therefore correct for the match fusion/head
            # parameters to have no gradient on this unit; do not turn the
            # missing target into a negative just to satisfy a blanket check.
            finite_grad = bool(observed_gradients) and all(
                bool(torch.isfinite(parameter.grad).all()) for parameter in observed_gradients
            )
        else:
            finite_grad = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                              for parameter in trainable)
        nonzero_grad = sum(int(parameter.grad is not None and
                                float(parameter.grad.detach().abs().sum()) > 0.0)
                           for parameter in trainable)
        if not finite_grad:
            missing = [name for name, parameter in named_trainable
                       if parameter.requires_grad and parameter.grad is None]
            nonfinite = [name for name, parameter in named_trainable
                         if parameter.requires_grad and parameter.grad is not None
                         and not bool(torch.isfinite(parameter.grad).all())]
            raise AssertionError(
                "nonfinite or missing trainable gradient; "
                f"missing={missing}; nonfinite={nonfinite}"
            )
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
            raise AssertionError("nonfinite trainable parameter after optimizer step")
        positive_scores = [score_by_index[index] for index in selected if record["labels"][index]]
        negative_scores = [score_by_index[index] for index in selected if not record["labels"][index]]
        sampled_margin = (min(positive_scores) - max(negative_scores)
                          if positive_scores and negative_scores and record["coverage_mask"] else None)
        # A compact row-level score summary is enough for the smoke trace; no
        # visual/hidden tensor is persisted.
        part_names = loss_parts[0].keys() if loss_parts else []
        mean_parts = {name: float(sum(item[name] for item in loss_parts) / max(1, len(loss_parts)))
                      for name in part_names}
        result = {
            "unit_key": record["unit_key"],
            "dataset": record["dataset"], "video": record["video"],
            "category": record["category"], "declared_category": record["declared_category"],
            "frame_id": record["frame_id"], "query_id": record["query_id"],
            "candidate_count": record["candidate_count"],
            "sampled_candidate_count": len(selected),
            "positive_count": record["positive_count"],
            "negative_sample_count": sum(not value for value in selected_labels),
            "selected_indices": selected,
            "candidate_keys_complete": len(record["row_keys"]) == record["candidate_count"],
            "candidate_rows_ordered": record["row_keys"] == sorted(record["row_keys"], key=lambda key: key[-1]),
            "candidate_truncation": False,
            "candidate_deletion": False,
            "candidate_mapping_rows": len(prepared["candidate_cells"]),
            "candidate_mapping_nonempty": sum(bool(cells) for cells in prepared["candidate_cells"]),
            "expression_token_count": len(prepared["expression_positions"]),
            "expression_span_method": prepared["expression_span_method"],
            "loss": float(sum(mean_parts.values())),
            "loss_parts": mean_parts,
            "sampled_score_mean": float(sum(score_by_index.values()) / max(1, len(score_by_index))),
            "sampled_score_std": float(torch.tensor(list(score_by_index.values()), dtype=torch.float32).std(unbiased=False)) if score_by_index else 0.0,
            "sampled_hard_margin": float(sampled_margin) if sampled_margin is not None else None,
            "sampled_hard_violation": bool(sampled_margin < 0.0) if sampled_margin is not None else None,
            "minimum_positive_score": float(min(positive_scores)) if positive_scores else None,
            "finite_loss": all(torch.isfinite(torch.tensor(list(mean_parts.values())))),
            "finite_trainable_gradients": bool(finite_grad),
            "nonzero_trainable_gradient_tensors": int(nonzero_grad),
            "gradient_missing_allowed_for_present_uncovered": bool(masked_uncovered),
            "missing_gradient_parameter_names": [name for name, parameter in named_trainable
                                                  if parameter.requires_grad and parameter.grad is None],
            "positive_rows_all_present_in_selected": all(index in selected for index in record["positive_indices"]),
            "row_gradient_checks": row_gradient_parts,
            "coverage_mask": bool(record["coverage_mask"]),
            "present_uncovered_not_negative": record["category"] != "present_uncovered" or not record["coverage_mask"],
        }
        if capture_reload:
            reload_payload = reload_input
        del prepared, image
        return result, reload_payload
    finally:
        bank.close()


def strict_reload_audit(model: Any, matcher: Any, checkpoint_path: Path,
                        reload_payload: dict[str, Any] | None) -> dict[str, Any]:
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fresh = CandidateMarkedVLMMatcher(hidden=256)
    matcher_result = fresh.load_state_dict(package["matcher"], strict=True)
    lora_result = load_lora_state_dict(model, package["lora"], strict=True)
    output_diff = None
    output_shape = None
    if reload_payload is not None:
        fresh.eval()
        current_device = next(matcher.parameters()).device
        # Compare the live and reloaded heads on the same device.  The prior
        # audit compared a CUDA result with a CPU result; after a long fit the
        # larger logits exposed normal CPU/CUDA matmul rounding (~6e-5), which
        # is not a state/reload mismatch.  Inputs remain the same canonical
        # serialized float tensors for both calls.
        fresh = fresh.to(current_device)
        with torch.no_grad():
            current = matcher(
                reload_payload["hidden"].to(current_device), reload_payload["expression_positions"],
                reload_payload["regions"].to(current_device), reload_payload["region_mask"].to(current_device),
            )["match_logit"].detach().cpu()
            before = fresh(
                reload_payload["hidden"].to(current_device), reload_payload["expression_positions"],
                reload_payload["regions"].to(current_device), reload_payload["region_mask"].to(current_device),
            )["match_logit"].detach().cpu()
        output_shape = list(before.shape)
        output_diff = float((before.float() - current.float()).abs().max())
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": package.get("format"),
        "matcher_missing_keys": list(matcher_result.missing_keys),
        "matcher_unexpected_keys": list(matcher_result.unexpected_keys),
        "lora": lora_result,
        "strict": not matcher_result.missing_keys and not matcher_result.unexpected_keys
                  and not lora_result["missing"] and not lora_result["unexpected"],
        "output_shape": output_shape,
        "max_output_diff": output_diff,
        "output_reload_tolerance": 1e-5,
        "output_reload_pass": output_diff is None or output_diff <= 1e-5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--candidate-chunk", type=int, default=2)
    parser.add_argument("--max-negatives", type=int, default=16)
    parser.add_argument("--checkpoint-steps", type=str, default="")
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {
        "format": "locatemot-l75-training-v1", "status": "running",
        "command": " ".join(sys.argv), "cwd": str(Path.cwd()), "seed": SEED,
        "inputs": {"fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
                    "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                    "image_root": str(IMAGE_ROOT), "model_dir": str(ROOT / "models/LocateAnything-3B"),
                    "manifest": str(MANIFEST_PATH), "manifest_sha256_expected": MANIFEST_SHA256},
        "outputs": {"directory": str(out)},
        "steps_requested": int(args.steps), "candidate_chunk": int(args.candidate_chunk),
        "max_negatives": int(args.max_negatives), "candidate_set_policy": "all rows retained; fit negatives only sampled",
        "calibration_labels_read": False, "validation_labels_read": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "raw_dense_cache_written": False, "candidate_deletion": False,
        "candidate_truncation": False, "token_span_alignment": "UNALIGNED",
    }
    write_json(out / "status.json", base)
    model = processor = tokenizer = None
    try:
        if Path.cwd().resolve() != ROOT.resolve():
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
            raise AssertionError("fixed manifest SHA mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU0 CUDA is required for L75 smoke")
        random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        splits = load_splits()
        schedule = deterministic_schedule(splits, args.steps)
        model, processor, tokenizer, runtime = load_locateanything("cuda:0")
        lora_contract = attach_language_lora(model, rank=8, alpha=16.0, target_layers=4)
        # The local Qwen2 training-mask helper currently returns a [1,1,S,S]
        # block mask even when the candidate-marked language batch has B>1.
        # Keep the decoder in eval mode so its batch-aware inference mask is
        # used, while language_forward still runs under enable_grad() and
        # propagates gradients through marker/LoRA.  Eval mode here does not
        # mean inference_mode and does not freeze the registered adapters.
        checkpointing = False
        model.language_model.model.eval()
        model.vision_model.eval()
        matcher = CandidateMarkedVLMMatcher(hidden=256).to("cuda:0")
        matcher.eval()
        lora_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        head_params = list(matcher.parameters())
        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": 2e-4},
            {"params": lora_params, "lr": 5e-5},
        ], weight_decay=0.0)
        base_digest_before = frozen_target_digest(model)
        config = {
            **base, "status": "configured", "runtime": runtime,
            "lora_contract": lora_contract,
            "matcher_contract": matcher.parameter_contract(),
            "optimizer": {"head_lr": 2e-4, "lora_lr": 5e-5, "weight_decay": 0.0},
            "gradient_checkpointing": checkpointing,
            "language_model_training_mode": False,
            "language_model_eval_with_grad": True,
            "visual_model_training_mode": False,
            "same_class_hard_negative_metadata": "unavailable; all-negative objectness fallback",
            "fit_only": True, "fixed_seed": SEED,
            "loss_contract": "balanced BCE + within-unit pairwise + all-positive/min-positive + inactive/absent + marker regularization",
        }
        write_json(out / "config.json", config)
        sampling_header = {
            "format": "locatemot-l75-sampling-v1", "seed": SEED,
            "steps": len(schedule),
            "scheduled_unit_keys": [unit_key(row) for row in schedule],
            "dataset_counts": {dataset: sum(str(row["dataset"]) == dataset for row in schedule) for dataset in DATASETS},
            "declared_category_counts": {category: sum(str(row.get("category")) == category for row in schedule) for category in STRATA},
            "negative_policy": "descending query-independent frozen objectness, stable row tie; all positives retained",
            "validation_used": False,
        }
        write_json(out / "sampling_trace.json", sampling_header)
        losses = []
        sampling_records = []
        active_video = None
        active_bank_path = None
        reload_payload = None
        max_peak = 0
        checkpoint_steps = {int(value) for value in args.checkpoint_steps.split(",") if value.strip()}
        checkpoint_steps.add(int(args.steps))
        checkpoint_paths = []
        checkpoint_summaries = []
        stop_reason = None
        for step, unit in enumerate(schedule, start=1):
            if active_video != str(unit["video"]):
                active_video = str(unit["video"])
                active_bank_path = str(ROOT / "outputs/l69/attempt9/budget40_features/kitti" / f"{active_video}.pt")
            result, maybe_reload = unit_step(
                model, matcher, processor, tokenizer, unit, "cuda:0",
                args.candidate_chunk, args.max_negatives, optimizer,
                capture_reload=(step in checkpoint_steps),
            )
            if maybe_reload is not None:
                reload_payload = maybe_reload
            result["step"] = step
            result["active_bank_path"] = active_bank_path
            losses.append(result)
            sampling_records.append({
                "step": step, "unit_key": result["unit_key"], "dataset": result["dataset"],
                "video": result["video"], "category": result["category"],
                "candidate_count": result["candidate_count"],
                "sampled_candidate_count": result["sampled_candidate_count"],
                "selected_indices": result["selected_indices"],
                "positive_count": result["positive_count"],
                "negative_sample_count": result["negative_sample_count"],
            })
            max_peak = max(max_peak, int(torch.cuda.max_memory_allocated()))
            if step in checkpoint_steps:
                checkpoint_path = out / f"checkpoint_l75_candidate_marked_step{step}.pt"
                torch.save(checkpoint_payload(model, matcher, lora_contract, optimizer,
                                              step, config), checkpoint_path)
                checkpoint_paths.append(str(checkpoint_path))
                checkpoint_summaries.append({
                    "step": int(step), "path": str(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                })
                if step == 500 and len(losses) >= 500:
                    first_window = losses[:100]
                    recent_window = losses[-100:]
                    first_ranking = [item for item in first_window
                                     if item["sampled_hard_margin"] is not None]
                    recent_ranking = [item for item in recent_window
                                      if item["sampled_hard_margin"] is not None]
                    first_min = [item["minimum_positive_score"] for item in first_window
                                 if item["minimum_positive_score"] is not None]
                    recent_min = [item["minimum_positive_score"] for item in recent_window
                                  if item["minimum_positive_score"] is not None]
                    if first_ranking and recent_ranking and first_min and recent_min:
                        first_violation = sum(item["sampled_hard_violation"] for item in first_ranking) / len(first_ranking)
                        recent_violation = sum(item["sampled_hard_violation"] for item in recent_ranking) / len(recent_ranking)
                        first_min_mean = sum(first_min) / len(first_min)
                        recent_min_mean = sum(recent_min) / len(recent_min)
                        monitor = {
                            "first_100_hard_violation": first_violation,
                            "steps_401_500_hard_violation": recent_violation,
                            "first_100_minimum_positive_score": first_min_mean,
                            "steps_401_500_minimum_positive_score": recent_min_mean,
                            "hard_violation_improved": recent_violation < first_violation,
                            "minimum_positive_improved": recent_min_mean > first_min_mean,
                        }
                        checkpoint_summaries[-1]["fit_100_to_500_monitor"] = monitor
                        if not monitor["hard_violation_improved"] and not monitor["minimum_positive_improved"]:
                            stop_reason = "fit_hard_negative_and_min_positive_both_failed_to_improve_100_to_500"
                            break
            if step % 5 == 0:
                torch.cuda.empty_cache()
        base_digest_after = frozen_target_digest(model)
        if base_digest_before != base_digest_after:
            raise AssertionError("frozen target base digest changed")
        if not checkpoint_paths:
            raise AssertionError("no checkpoint was written")
        reload_audit = strict_reload_audit(model, matcher, Path(checkpoint_paths[-1]), reload_payload)
        if not reload_audit["strict"] or not reload_audit["output_reload_pass"]:
            raise AssertionError(f"strict reload failed: {reload_audit}")
        nonzero_steps = sum(int(item["nonzero_trainable_gradient_tensors"] > 0) for item in losses)
        finite_steps = sum(int(item["finite_loss"] and item["finite_trainable_gradients"]) for item in losses)
        base_grad_count = sum(int(parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0)
                              for name, parameter in model.named_parameters() if "lora_" not in name)
        metrics = {
            **base, "status": "complete", "steps": len(losses),
            "runtime": runtime, "checkpoint_paths": checkpoint_paths,
            "checkpoint_summaries": checkpoint_summaries,
            "stop_reason": stop_reason,
            "finite_steps": finite_steps, "nonzero_gradient_steps": nonzero_steps,
            "finite_all_steps": finite_steps == len(losses),
            "nonzero_gradients_all_steps": nonzero_steps == len(losses),
            "both_domains": sorted({item["dataset"] for item in losses}) == list(DATASETS),
            "all_four_categories": sorted({item["category"] for item in losses}) == sorted(STRATA),
            "candidate_key_drift_count": sum(int(not item["candidate_keys_complete"] or not item["candidate_rows_ordered"]) for item in losses),
            "candidate_truncation": False, "candidate_deletion": False,
            "base_target_digest_before": base_digest_before,
            "base_target_digest_after": base_digest_after,
            "base_target_digest_unchanged": base_digest_before == base_digest_after,
            "base_nonzero_gradient_parameter_count": base_grad_count,
            "lora_parameter_count": sum(parameter.numel() for parameter in lora_params),
            "matcher_parameter_count": sum(parameter.numel() for parameter in matcher.parameters()),
            "peak_cuda_bytes": max_peak,
            "wall_time_seconds": time.perf_counter() - started,
            "throughput_steps_per_second": len(losses) / max(1e-9, time.perf_counter() - started),
            "strict_reload": reload_audit,
            "loss_trace_reference": "loss_trace.json",
            "sampling_trace_reference": "sampling_trace.json",
            "next_action": "if this is the B0 smoke, inspect the smoke gate before any registered longer fit",
        }
        write_json(out / f"metrics_l75_step{int(args.steps)}.json", metrics)
        write_json(out / "loss_trace.json", {"format": "locatemot-l75-loss-trace-v1", "steps": losses})
        write_json(out / "sampling_trace.json", {**sampling_header, "records": sampling_records})
        write_json(out / "reload_audit.json", reload_audit)
        provenance = {
            **base, "status": "complete", "runtime": runtime,
            "lora_contract": lora_contract, "matcher_contract": matcher.parameter_contract(),
            "model_file_manifest_sha256": runtime["model_manifest"]["manifest_sha256"],
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "fit_unit_count": len(schedule), "fit_only": True,
            "checkpoint_sha256": {path: sha256_file(Path(path)) for path in checkpoint_paths},
            "base_target_digest_before": base_digest_before,
            "base_target_digest_after": base_digest_after,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "raw_dense_cache_written": False, "token_span_alignment": "UNALIGNED",
        }
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", metrics)
        return 0
    except Exception as exc:
        failure = {
            **base, "status": "incomplete",
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
            "next_action": "preserve this attempt; fix only the first actionable error and retry in a new directory",
        }
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text("# INCOMPLETE\n\n" +
            f"First actionable root cause: `{failure['failure_root_cause']}`\n\n"
            "No full-model checkpoint or persistent raw feature cache was written.\n")
        return 1
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
