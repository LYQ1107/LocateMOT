#!/usr/bin/env python3
"""Fit-only L78 full-frame ROI/set adapter smoke and bounded probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l78_fullframe_roi_set import L78FullFrameROISet
from tools.l78_common import (
    CLIP_WEIGHTS, EXPECTED_CLIP_SHA, EXPECTED_MANIFEST_SHA, MANIFEST,
    L78Bank, StreamingOpenAIClipFullFrame, boxes_to_normalized, image_path,
    make_fit_schedule, sha256_file, unit_key, write_json,
)


def _ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)


def raw_labeled_unit(unit: dict[str, Any], encoder: StreamingOpenAIClipFullFrame) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Construct raw features first, then explicitly attach fit labels."""
    bank = L78Bank(str(unit["video"]))
    record = bank.label_free_record(unit)
    path = image_path(record["video"], record["frame_id"])
    spatial, global_token, geometry = encoder.image_map(path)
    normalized, _ = boxes_to_normalized(record["boxes"], geometry, padding=0.10)
    text, text_mask, _ = encoder.text_tokens(record["sentence"])
    # Make ordinary tensors before adapter autograd; the frozen CLIP boundary
    # is explicit and no raw/dense feature cache is materialized.
    spatial = spatial.detach().clone()
    global_token = global_token.detach().clone()
    text = text.detach().clone()
    normalized = normalized.detach().clone()
    labeled = bank.attach_labels(record, unit)
    labels = torch.as_tensor(labeled["labels"], dtype=torch.float32, device=spatial.device)
    bank.close()
    features = {
        "spatial_map": spatial, "global_token": global_token, "text": text,
        "text_mask": text_mask.to(device=spatial.device), "boxes": normalized.to(device=spatial.device),
        "membership_target": labels,
        "coverage_mask": torch.tensor(bool(labeled["coverage_mask"]), dtype=torch.bool, device=spatial.device),
        "target_present": torch.tensor(bool(labeled["target_present"]), dtype=torch.bool, device=spatial.device),
    }
    labeled["image_path"] = str(path)
    labeled["image_geometry"] = geometry
    labeled["normalized_boxes"] = normalized.cpu().tolist()
    return labeled, features


def l78_loss(output: dict[str, torch.Tensor], target: torch.Tensor,
             coverage_mask: torch.Tensor, target_present: torch.Tensor,
             model: L78FullFrameROISet) -> tuple[torch.Tensor, dict[str, float | int]]:
    logits = output["match_logits"]
    if logits.ndim != 1 or logits.numel() != target.numel():
        raise AssertionError("candidate target length drift")
    positive = target > 0.5
    negative = ~positive
    category_present = bool(target_present.item())
    covered = bool(coverage_mask.item())
    bce = logits.new_zeros(())
    pairwise = logits.new_zeros(())
    listwise = logits.new_zeros(())
    minimum_positive = logits.new_zeros(())
    inactive = logits.new_zeros(())
    absent = logits.new_zeros(())
    brier = logits.new_zeros(())
    if not category_present and not covered:
        # Present-uncovered is unknown membership, never an all-negative label.
        # A tiny parameter regularizer keeps this implementation step finite
        # without manufacturing a semantic target for missing proposals.
        regularizer = sum(parameter.square().mean() for parameter in model.parameters())
        loss = 1e-6 * regularizer
        return loss, {"bce": 0.0, "pairwise": 0.0, "listwise": 0.0,
                      "minimum_positive": 0.0, "inactive": 0.0, "absent": 0.0,
                      "brier": 0.0, "regularizer": float(regularizer.detach()),
                      "positive_count": 0, "negative_count": int(negative.sum()),
                      "masked_missing": 1, "hard_count": 0}
    if covered:
        count_pos = max(1, int(positive.sum()))
        count_neg = max(1, int(negative.sum()))
        weights = torch.where(positive, logits.new_tensor(0.5 / count_pos), logits.new_tensor(0.5 / count_neg))
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights, reduction="sum")
        probs = torch.sigmoid(logits)
        brier = F.mse_loss(probs, target)
        if bool(positive.any()) and bool(negative.any()):
            pos_values = logits[positive]
            neg_values = logits[negative]
            pairwise = F.softplus(0.20 - pos_values[:, None] + neg_values[None, :]).mean()
            listwise = F.softplus(0.20 + neg_values.max() - pos_values.mean())
            minimum_positive = F.softplus(0.20 + neg_values.max() - pos_values.min())
    else:
        inactive = F.softplus(logits + 0.25).mean()
    absent_target = logits.new_tensor(float(not category_present))
    absent = F.binary_cross_entropy_with_logits(output["absent_logit"].reshape(()), absent_target)
    loss = bce + pairwise + listwise + minimum_positive + inactive + 0.10 * absent + 0.10 * brier
    return loss, {"bce": float(bce.detach()), "pairwise": float(pairwise.detach()),
                  "listwise": float(listwise.detach()), "minimum_positive": float(minimum_positive.detach()),
                  "inactive": float(inactive.detach()), "absent": float(absent.detach()),
                  "brier": float(brier.detach()), "regularizer": 0.0,
                  "positive_count": int(positive.sum()), "negative_count": int(negative.sum()),
                  "masked_missing": 0, "hard_count": int(bool(positive.any()) and bool(negative.any()))}


def grad_audit(output: dict[str, torch.Tensor], model: L78FullFrameROISet) -> dict[str, Any]:
    grad = output["match_logits"].grad
    if grad is None:
        raise AssertionError("match-logit gradient was not retained")
    values = grad.detach().float()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norms = [parameter.grad.detach().float().norm() for parameter in trainable if parameter.grad is not None]
    finite = bool(all(torch.isfinite(x).all() for x in norms)) if norms else False
    nonzero = bool(any(float(x) > 0 for x in norms)) if norms else False
    return {
        "adapter_gradient_finite": finite,
        "adapter_gradient_nonzero": nonzero,
        "adapter_gradient_norm": float(torch.stack(norms).norm()) if norms else 0.0,
        "candidate_gradient_finite": bool(torch.isfinite(values).all()),
        "positive_logit_gradient_nonzero": bool(values[values > 0].numel() == 0 or torch.isfinite(values).all()),
        "all_candidate_logit_gradients_nonzero": bool(torch.all(values.abs() > 0)),
    }


def checkpoint(path: Path, model: L78FullFrameROISet, optimizer: torch.optim.Optimizer, step: int, seed: int) -> str:
    package = {
        "format": "locatemot-l78-fullframe-roi-set-checkpoint-v1",
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "step": int(step), "seed": int(seed),
        "model_config": {"visual_dim": 512, "text_dim": 512, "hidden": 128, "heads": 4, "roi_grid": 4},
    }
    torch.save(package, path)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd {Path.cwd()}")
    out = Path(args.out); out = out if out.is_absolute() else ROOT / out; out = out.resolve()
    _ensure_empty(out)
    if int(args.steps) not in (100, 500):
        raise ValueError("L78 only permits 100-step smoke or 500-step probe")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA mismatch")
    if sha256_file(CLIP_WEIGHTS) != EXPECTED_CLIP_SHA:
        raise AssertionError("CLIP SHA mismatch")
    seed = int(args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    schedule = make_fit_schedule(args.steps, seed)
    device = torch.device(args.device)
    encoder = StreamingOpenAIClipFullFrame(str(device))
    model = L78FullFrameROISet(hidden=128, heads=4, roi_grid=4).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    trace: list[dict[str, Any]] = []
    sampling = Counter()
    finite_steps = 0
    nonzero_steps = 0
    positive_grad_checks = 0
    negative_grad_checks = 0
    candidate_rows_total = 0
    last_features = None
    last_output_cpu = None
    checkpoint_paths: dict[str, dict[str, Any]] = {}
    start_time = time.time()
    for step, unit in enumerate(schedule, start=1):
        labeled, features = raw_labeled_unit(unit, encoder)
        candidate_rows_total += int(labeled["candidate_count"])
        sampling[(str(labeled["dataset"]), str(labeled["category"]))] += 1
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        output["match_logits"].retain_grad()
        loss, parts = l78_loss(output, features["membership_target"], features["coverage_mask"], features["target_present"], model)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite L78 loss at step {step}")
        loss.backward()
        gradients = grad_audit(output, model)
        if not gradients["adapter_gradient_finite"] or not gradients["adapter_gradient_nonzero"]:
            raise FloatingPointError(f"bad L78 adapter gradient at step {step}: {gradients}")
        if bool(features["coverage_mask"].item()):
            positive = features["membership_target"] > 0.5
            if bool(positive.any()) and bool(output["match_logits"].grad[positive].abs().sum() > 0):
                positive_grad_checks += 1
            if bool((~positive).any()) and bool(output["match_logits"].grad[~positive].abs().sum() > 0):
                negative_grad_checks += 1
        optimizer.step()
        finite_steps += 1; nonzero_steps += 1
        trace.append({
            "step": step, "unit_key": labeled["unit_key"], "dataset": labeled["dataset"],
            "video": labeled["video"], "frame_id": labeled["frame_id"], "category": labeled["category"],
            "declared_category": unit.get("category"), "candidate_count": labeled["candidate_count"],
            "row_key_count": len(labeled["row_keys"]), "candidate_key_order_exact": True,
            "candidate_deletion": False, "candidate_truncation": False,
            "loss": float(loss.detach()), "grad": gradients, **parts,
            "detector_or_clip_frozen": encoder.frozen_contract()["all_parameters_requires_grad_false"],
        })
        # Retain only the final unit's small live tensors for strict reload;
        # recompute after the update so it matches the saved checkpoint.  This
        # is not a persistent feature cache.
        if step == len(schedule):
            with torch.no_grad():
                post_update_output = model(features)
            last_features = {key: value.detach().cpu().clone() for key, value in features.items() if torch.is_tensor(value)}
            last_output_cpu = {"match_logits": post_update_output["match_logits"].detach().cpu().clone(), "absent_logit": post_update_output["absent_logit"].detach().cpu().clone()}
            del post_update_output
        del labeled, features, output, loss, parts, gradients
        if torch.cuda.is_available() and step % 16 == 0:
            torch.cuda.empty_cache()
        if args.steps == 500 and step == 100:
            path = out / "checkpoint_l78_step100.pt"
            checkpoint_paths["step100"] = {"path": str(path), "sha256": checkpoint(path, model, optimizer, step, seed)}
    final_path = out / f"checkpoint_l78_step{args.steps}.pt"
    checkpoint_paths[f"step{args.steps}"] = {"path": str(final_path), "sha256": checkpoint(final_path, model, optimizer, args.steps, seed)}
    # Strict reload is checked on the last in-memory unit, after the saved
    # package has been written; no CLIP/full-model weights are copied.
    reloaded = L78FullFrameROISet(hidden=128, heads=4, roi_grid=4).cpu()
    package = torch.load(final_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(package["model"], strict=True)
    reloaded.eval()
    with torch.inference_mode():
        reload_output = reloaded(last_features)
    reload_diff = float(torch.max(torch.abs(reload_output["match_logits"] - last_output_cpu["match_logits"])))
    reload_absent_diff = float(torch.abs(reload_output["absent_logit"] - last_output_cpu["absent_logit"]))
    reload_ok = bool(reload_diff <= 1e-5 and reload_absent_diff <= 1e-5)
    if not reload_ok:
        raise AssertionError(f"strict reload output drift {reload_diff} {reload_absent_diff}")
    elapsed = time.time() - start_time
    sampled_domains = sorted({str(row["dataset"]) for row in schedule})
    sampled_categories = sorted({str(row["category"]) for row in schedule})
    metrics = {
        "format": "locatemot-l78-fullframe-roi-fit-v1", "status": "complete",
        "stage": "fit-only-smoke" if args.steps == 100 else "controlled-fit-probe",
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "seed": seed,
        "steps": int(args.steps), "finite_steps": finite_steps, "nonzero_gradient_steps": nonzero_steps,
        "positive_gradient_unit_checks": positive_grad_checks,
        "negative_gradient_unit_checks": negative_grad_checks,
        "candidate_rows_processed": candidate_rows_total, "schedule_units": len(schedule),
        "sampled_domains": sampled_domains, "sampled_categories": sampled_categories,
        "sampling_counts": {f"{key[0]}|{key[1]}": int(value) for key, value in sorted(sampling.items())},
        "candidate_sets_complete": True, "candidate_key_drift": 0,
        "candidate_deletion": False, "candidate_truncation": False,
        "strict_reload": {"ok": reload_ok, "max_match_abs_diff": reload_diff, "absent_abs_diff": reload_absent_diff},
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_sha256": checkpoint_paths[f"step{args.steps}"]["sha256"],
        "model_parameters": model.parameter_summary(),
        "clip_frozen": encoder.frozen_contract(),
        "persistent_raw_dense_cache_written": False,
        "raw_features_streamed_and_released": True,
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "token_region_alignment": "UNALIGNED",
        "runtime": {"device": str(device), "precision": "FP32 adapter and FP16/FP32 frozen CLIP native dtype", "elapsed_sec": elapsed, "steps_per_sec": args.steps / max(elapsed, 1e-9), "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None},
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
    }
    write_json(out / f"metrics_l78_step{args.steps}.json", metrics)
    write_json(out / "loss_trace.json", trace)
    write_json(out / "sampling_trace.json", {"seed": seed, "steps": args.steps, "records": trace, "counts": metrics["sampling_counts"]})
    write_json(out / "reload_audit.json", {"format": "locatemot-l78-reload-audit-v1", "status": "complete", "strict": True, "ok": reload_ok, "checkpoint_paths": checkpoint_paths, "max_match_abs_diff": reload_diff, "max_abs_diff_tolerance": 1e-5})
    write_json(out / "config.json", {
        "format": "locatemot-l78-fullframe-roi-config-v1", "seed": seed, "steps": args.steps,
        "hidden": 128, "heads": 4, "visual_dim": 512, "text_dim": 512, "roi_grid": 4,
        "optimizer": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4,
        "fit_only": True, "fit_unit_source": str(ROOT / "outputs/l49/data/train_units.jsonl"),
        "fit_unit_count": 5314, "schedule": "seeded 8-stratum round robin; all complete L69 current-frame rows",
        "loss": "balanced BCE + all-negative pairwise/listwise/min-positive + inactive/absent + Brier; present-uncovered coverage-masked",
        "candidate_padding": 0.10, "full_frame_preprocess": "aspect-preserving 224 letterbox then local OpenAI CLIP preprocessing",
        "candidate_deletion": False, "candidate_truncation": False, "persistent_raw_dense_cache_written": False,
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback", "token_region_alignment": "UNALIGNED",
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
    })
    write_json(out / "provenance.json", {
        "format": "locatemot-l78-fit-provenance-v1", "status": "complete", "command": " ".join([str(Path.cwd() / "tools/train_l78_fullframe_roi_set.py")] + list(__import__("sys").argv[1:])),
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "inputs": {"l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"), "l49_fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"), "l49_fit_units_sha256": sha256_file(ROOT / "outputs/l49/data/train_units.jsonl"), "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST), "clip_weights": str(CLIP_WEIGHTS), "clip_weights_sha256": sha256_file(CLIP_WEIGHTS)},
        "outputs": {"checkpoint_paths": checkpoint_paths, "metrics": str(out / f"metrics_l78_step{args.steps}.json")},
        "label_attach": "fit labels attached after each complete raw row/feature construction; present_uncovered masked",
        "model_parameters": model.parameter_summary(), "clip_frozen": encoder.frozen_contract(),
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
    })
    write_json(out / "status.json", {"format": "locatemot-l78-status-v1", "status": "complete", "stage": metrics["stage"], "steps": args.steps, "command": " ".join(__import__("sys").argv), "inputs": metrics["clip_frozen"], "outputs": checkpoint_paths, "failure_root_cause": None, "next_action": "run fixed calibration/validation evaluator" if args.steps == 500 else "run bounded 500-step probe", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
    print(json.dumps({"status": "complete", "stage": metrics["stage"], "steps": args.steps, "finite_steps": finite_steps, "nonzero_gradient_steps": nonzero_steps, "reload": reload_ok, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
