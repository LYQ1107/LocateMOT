#!/usr/bin/env python3
"""Fit-only L77 region/word correspondence probe."""
from __future__ import annotations

import argparse
import json
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

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l77_common import (  # noqa: E402
    MANIFEST, MANIFEST_SHA256, load_fit_samples, load_splits, make_schedule,
    load_text_cache, safe_torch_load, sha256_file, write_json,
)
from tools.l77_loss import compute_loss  # noqa: E402
from locatemot.models.l77_region_cross_attention import L77RegionCrossAttention  # noqa: E402

SEED = 20260829


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return value


def model_config() -> dict[str, Any]:
    return {
        "region_dim": 512, "text_dim": 768, "hidden": 192, "heads": 4,
        "dropout": 0.0, "candidate_set": "complete L69 current-frame rows",
        "region_source": "frozen L69 clip[512] only",
        "text_source": "frozen L48 masked token_hidden[64,768]",
        "cross_attention": "candidate region query attends to all valid word tokens",
        "set_competition": "one shared TransformerEncoder layer over all candidates",
        "bounded_score": "2*tanh(raw_score/2)",
        "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "token_region_alignment": "UNALIGNED",
        "null": "absent diagnostic logit only; no filtering",
    }


def model_from_config(config: dict[str, Any], device: torch.device) -> L77RegionCrossAttention:
    model = L77RegionCrossAttention(
        region_dim=int(config["region_dim"]), text_dim=int(config["text_dim"]),
        hidden=int(config["hidden"]), heads=int(config["heads"]),
        dropout=float(config["dropout"]),
    ).to(device)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=(8, 100, 500), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    stage = "targeted_regression" if args.steps == 8 else ("fit_only_smoke" if args.steps == 100 else "fit_only_controlled_probe")
    running: dict[str, Any] = {
        "format": "locatemot-l77-region-cross-attention-train-v1",
        "status": "running", "stage": stage, "project_root": str(ROOT),
        "cwd": os.getcwd(), "command": " ".join(sys.argv), "seed": SEED,
        "steps_requested": int(args.steps), "resume": None if args.resume is None else str(args.resume),
        "training_run": True, "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "no_hota_or_trackeval": True, "raw_dense_feature_cache_written": False,
        "validation_labels_used_for_optimization": False,
        "screening_labels_used_for_optimization": False,
        "candidate_deletion": False, "candidate_truncation": False,
    }
    write_json(out / "status.json", running)
    started = time.perf_counter()
    bank_samples: list[tuple[dict[str, Any], dict[str, torch.Tensor]]] = []
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA256:
            raise AssertionError("fixed manifest SHA mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("L77 training requires GPU0")
        device = torch.device(args.device)
        if device.type != "cuda" or device.index not in (None, 0):
            raise RuntimeError(f"L77 training requires GPU0, got {device}")
        seed_everything(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        splits = load_splits()
        schedule = make_schedule(splits["fit"], args.steps, SEED)
        text_cache = load_text_cache()
        bank_samples = load_fit_samples(schedule, text_cache)
        if len(bank_samples) != int(args.steps):
            raise AssertionError("fit sample count drift")
        config = model_config()
        model = model_from_config(config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        start_step = 0
        loss_trace: list[dict[str, Any]] = []
        gradient_trace: list[dict[str, Any]] = []
        sampling_trace: list[dict[str, Any]] = []
        if args.resume is not None:
            if args.steps != 500:
                raise ValueError("resume is only allowed for the 500-step continuation")
            package = safe_torch_load(args.resume)
            if int(package.get("step", -1)) != 100 or int(package.get("seed", -1)) != SEED:
                raise AssertionError("resume checkpoint is not the registered step-100 checkpoint")
            if package.get("config") != config:
                raise AssertionError("resume configuration drift")
            model.load_state_dict(package["model"], strict=True)
            optimizer.load_state_dict(package["optimizer"])
            start_step = 100
            for name, target in (("loss_trace.json", loss_trace), ("gradient_trace.json", gradient_trace), ("sampling_trace.json", sampling_trace)):
                path = args.resume.parent / name
                if not path.exists():
                    raise FileNotFoundError(path)
                target.extend(json.loads(path.read_text()))
            if len(loss_trace) != 100 or len(gradient_trace) != 100 or len(sampling_trace) != 100:
                raise AssertionError("resume traces do not contain exactly 100 steps")
            torch.save(package, out / "checkpoint_l77_step100.pt")
        write_json(out / "config.json", {
            **running, **config, "device": str(device),
            "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-4},
            "sampler": "seeded round-robin dataset/category strata",
            "schedule_horizon": int(args.steps), "fit_unit_count_source": "L49 train_units split=fit only",
            "model_parameters": model.parameter_summary(),
        })
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_candidates = 0
        last_batch: dict[str, torch.Tensor] | None = None
        for absolute_step in range(start_step + 1, int(args.steps) + 1):
            record, data_cpu = bank_samples[absolute_step - 1]
            batch = {key: value.to(device, non_blocking=True) for key, value in data_cpu.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            n = int(record["candidate_count"])
            if output["match_logits"].shape != (n,) or len(record["row_keys"]) != n:
                raise AssertionError(f"candidate row drift at {record['unit_key']}")
            logits = output["match_logits"]
            logits.retain_grad()
            loss, parts = compute_loss(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {absolute_step}")
            loss.backward()
            named_parameters = [(name, parameter) for name, parameter in model.named_parameters()
                                if parameter.requires_grad]
            parameters = [parameter for _, parameter in named_parameters]
            non_none_grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
            finite_all_grads = all(bool(torch.isfinite(gradient).all()) for gradient in non_none_grads)
            # A present-uncovered unit deliberately masks every membership loss;
            # its score head is therefore disconnected by contract.  Require
            # finite/nonzero gradients on the active loss graph and record the
            # masked parameters instead of fabricating a gradient.
            coverage = batch["coverage_mask"].bool()
            required_parameters = (named_parameters if bool(coverage.any()) else
                                   [(name, parameter) for name, parameter in named_parameters
                                    if not name.startswith("score_head.")])
            finite_gradients = (finite_all_grads and all(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for _, parameter in required_parameters))
            nonzero_gradients = all(
                parameter.grad is not None and bool(parameter.grad.detach().abs().sum() > 0)
                for _, parameter in required_parameters)
            if not finite_gradients or not nonzero_gradients:
                raise FloatingPointError(f"invalid trainable gradients at step {absolute_step}")
            targets = batch["membership_target"] > 0.5
            logit_grad = logits.grad.detach().abs()
            pos_grad = logit_grad[targets & coverage]
            neg_grad = logit_grad[(~targets) & coverage]
            if pos_grad.numel() and not bool((pos_grad > 0).all()):
                raise FloatingPointError(f"positive gradient missing at step {absolute_step}")
            if neg_grad.numel() and not bool((neg_grad > 0).all()):
                raise FloatingPointError(f"negative gradient missing at step {absolute_step}")
            grad_row = {
                "step": absolute_step, "finite_loss": True,
                "finite_trainable_gradients": bool(finite_gradients),
                "nonzero_trainable_gradients": bool(nonzero_gradients),
                "positive_count": int((targets & coverage).sum()),
                "negative_count": int(((~targets) & coverage).sum()),
                "masked_missing_count": int((~coverage).sum()),
                "positive_logit_grad_nonzero": int((pos_grad > 0).sum()),
                "negative_logit_grad_nonzero": int((neg_grad > 0).sum()),
                "minimum_positive_gradient_nonzero": bool(pos_grad.numel() == 0 or bool((pos_grad > 0).all())),
                "gradient_contract_scope": "all_trainable_parameters" if bool(coverage.any()) else "active_loss_graph_excluding_masked_score_head",
                "zero_gradient_parameters": [name for name, parameter in named_parameters
                                               if parameter.grad is None or not bool(parameter.grad.detach().abs().sum() > 0)],
                "max_parameter_grad": float(max((parameter.grad.detach().abs().max() for parameter in parameters
                                                  if parameter.grad is not None), default=torch.tensor(0.0, device=logits.device))),
            }
            gradient_trace.append(grad_row)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            loss_trace.append({"step": absolute_step, **parts})
            sampling_trace.append({
                "step": absolute_step, "schedule_position": int(record["schedule_position"]),
                "unit_key": record["unit_key"], "dataset": record["dataset"],
                "video": record["video"], "query_id": int(record["query_id"]),
                "frame_id": int(record["frame_id"]), "declared_category": record["declared_category"],
                "derived_category": record["category"], "candidate_count": n,
                "positive_count": int(record["positive_count"]), "coverage_mask": bool(record["coverage_mask"]),
                "candidate_truncation": False, "candidate_deletion": False,
                "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
                "positive_indices": [int(value) for value in record["positive_indices"]],
            })
            total_candidates += n
            last_batch = batch
            del batch, output, loss

        final_checkpoint = out / f"checkpoint_l77_step{args.steps}.pt"
        model.eval()
        package = {
            "format": "locatemot-l77-region-cross-attention-checkpoint-v1",
            "step": int(args.steps), "seed": SEED, "model": cpu_copy(model.state_dict()),
            "optimizer": cpu_copy(optimizer.state_dict()), "config": config,
        }
        torch.save(package, final_checkpoint)
        if last_batch is None:
            raise AssertionError("missing final batch for reload check")
        with torch.inference_mode():
            before = model(last_batch)
            reloaded = model_from_config(config, device)
            loaded = safe_torch_load(final_checkpoint)
            reloaded.load_state_dict(loaded["model"], strict=True)
            reloaded.eval()
            after = reloaded(last_batch)
            output_keys = sorted(set(before) & set(after))
            shape_equal = all(tuple(before[key].shape) == tuple(after[key].shape) for key in output_keys)
            max_diff = max(float((before[key] - after[key]).abs().max()) for key in output_keys)
            reload_finite = all(bool(torch.isfinite(after[key]).all()) for key in output_keys)
        if not shape_equal or max_diff >= 1e-5 or not reload_finite:
            raise AssertionError(f"strict reload failed shape={shape_equal} diff={max_diff} finite={reload_finite}")
        write_json(out / "reload_audit.json", {
            "format": "locatemot-l77-reload-audit-v1", "status": "complete",
            "checkpoint": str(final_checkpoint), "strict": True, "shape_equal": bool(shape_equal),
            "max_abs_output_diff": max_diff, "finite": bool(reload_finite),
            "missing_keys": [], "unexpected_keys": [],
        })
        write_json(out / "loss_trace.json", loss_trace)
        write_json(out / "gradient_trace.json", gradient_trace)
        write_json(out / "sampling_trace.json", sampling_trace)
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        categories = Counter(str(row["derived_category"]) for row in sampling_trace)
        datasets = Counter(str(row["dataset"]) for row in sampling_trace)
        metrics = {
            "format": "locatemot-l77-step-metrics-v1", "status": "complete", "stage": stage,
            "steps": int(args.steps), "start_step": int(start_step),
            "finite_steps": int(sum(bool(row["finite_loss"]) for row in gradient_trace)),
            "nonzero_gradient_steps": int(sum(bool(row["nonzero_trainable_gradients"]) for row in gradient_trace)),
            "positive_gradient_rows": int(sum(row["positive_logit_grad_nonzero"] for row in gradient_trace)),
            "negative_gradient_rows": int(sum(row["negative_logit_grad_nonzero"] for row in gradient_trace)),
            "minimum_positive_gradient_checks": int(sum(bool(row["minimum_positive_gradient_nonzero"]) for row in gradient_trace)),
            "candidate_rows_processed": int(total_candidates),
            "datasets": dict(sorted(datasets.items())), "categories": dict(sorted(categories.items())),
            "candidate_key_drift": 0, "candidate_deletion": False, "candidate_truncation": False,
            "detector_or_backbone": "not loaded; frozen L69 features only",
            "trainable_parameters": model.parameter_summary()["trainable_parameters"],
            "finite_loss_all_steps": True, "nonzero_gradients_all_steps": True,
            "strict_reload": True, "no_persistent_raw_dense_cache": True,
            "wall_time_seconds": time.perf_counter() - started,
            "samples_per_second": float(args.steps / max(1e-9, time.perf_counter() - started)),
            "peak_cuda_memory_bytes": peak,
            "checkpoint": str(final_checkpoint), "checkpoint_sha256": sha256_file(final_checkpoint),
        }
        metrics_name = out / f"metrics_l77_step{args.steps}.json"
        write_json(metrics_name, metrics)
        provenance = {
            **running, "status": "complete", "manifest": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
            "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
            "l48_text_cache": str(ROOT / "outputs/l48/data/text_cache.pt"),
            "l48_text_cache_sha256": sha256_file(ROOT / "outputs/l48/data/text_cache.pt"),
            "fit_units_source": str(ROOT / "outputs/l49/data/train_units.jsonl"),
            "fit_units_total": 5314, "fit_only": True, "selected_unit_count": int(args.steps),
            "labels_used": "L69 candidate_gt intersected with L49 target_ids after native row construction; fit only",
            "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
            "token_region_alignment": "UNALIGNED", "forbidden_semantic_inputs": [
                "source_id", "pool_id", "track_id", "group_id", "query_id", "old_scores",
            ],
            "outputs": {"checkpoint": str(final_checkpoint), "metrics": str(metrics_name)},
            "wall_time_seconds": metrics["wall_time_seconds"], "peak_cuda_memory_bytes": peak,
        }
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {**metrics, "status": "complete", "failure_root_cause": None,
                                          "next_action": "run the fixed calibration/validation evaluator after the 500-step probe" if args.steps == 500 else "review smoke, then run the authorized 500-step probe"})
        print(json.dumps({"status": "complete", "stage": stage, "steps": args.steps,
                          "out": str(out), "metrics": str(metrics_name)}), flush=True)
        return 0
    except Exception as exc:
        failure = {**running, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "elapsed_seconds": time.perf_counter() - started,
                   "next_action": "fix only the first actionable L77 training-contract root cause and rerun in a new directory"}
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text("# L77 training INCOMPLETE\n\n" +
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n```text\n" + traceback.format_exc() + "```\n")
        raise
    finally:
        bank_samples.clear()


if __name__ == "__main__":
    raise SystemExit(main())
