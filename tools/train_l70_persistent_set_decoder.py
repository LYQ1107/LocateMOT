#!/usr/bin/env python3
"""Fit-only L70 persistent set decoder smoke.

The run is deliberately small and uses the L70 unit index.  It loads one L69
video bank at a time, materializes only the selected unit on the device, and
never writes a derived feature cache.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")))
from l70_common import (
    L69Bank,
    MAX_HISTORY,
    ROOT,
    load_text_cache,
    safe_torch_load,
    unit_tensors,
    write_json,
)
from locatemot.models.l70_persistent_set_decoder import L70PersistentSetDecoder

DEFAULT_INDEX = ROOT / "outputs/l70/audit/data_contract_retry2/unit_records.jsonl"
DEFAULT_OUT = ROOT / "outputs/l70/train/persistent_set_smoke100"
SEED = 20260829


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_schedule(records: list[dict[str, Any]], steps: int, seed: int) -> list[dict[str, Any]]:
    fit = [row for row in records if row.get("split") == "fit"]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fit:
        buckets[(str(row["dataset"]), str(row["category"]))].append(row)
    keys = sorted(buckets)
    required = {(dataset, category) for dataset in ("refer_kitti_v1", "refer_kitti_v2")
                for category in ("positive", "multi_positive", "inactive", "present_uncovered")}
    if set(keys) != required:
        raise AssertionError(f"fit sampler strata mismatch: {sorted(set(keys) ^ required)}")
    rng = random.Random(seed)
    for key in keys:
        buckets[key] = sorted(
            buckets[key], key=lambda row: (str(row["video"]), int(row["query_id"]), int(row["frame_id"]))
        )
        # A deterministic permutation changes only the sampling order, never
        # the unit set or labels.
        rng.shuffle(buckets[key])
    selected: list[dict[str, Any]] = []
    for step in range(steps):
        key = keys[step % len(keys)]
        row = dict(buckets[key][(step // len(keys)) % len(buckets[key])])
        row["schedule_position"] = step
        selected.append(row)
    # Grouping by video permits one serial load/release per video while the
    # logical sampling trace remains explicit and reproducible.
    selected.sort(key=lambda row: (str(row["video"]), int(row["schedule_position"])))
    for train_step, row in enumerate(selected, start=1):
        row["train_step"] = train_step
    return selected


def balanced_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    values = logits[mask]
    labels = target[mask]
    pos = labels.sum()
    neg = labels.numel() - pos
    weights = torch.ones_like(labels)
    if pos > 0 and neg > 0:
        weights = torch.where(labels > 0.5, neg / pos.clamp_min(1.0), pos / neg.clamp_min(1.0))
    return F.binary_cross_entropy_with_logits(values, labels, weight=weights)


def pairwise_hinge(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    valid_pos = (target > 0.5) & mask
    valid_neg = (target <= 0.5) & mask
    if not bool(valid_pos.any()) or not bool(valid_neg.any()):
        return logits.sum() * 0.0
    pos = logits[valid_pos]
    neg = logits[valid_neg]
    return F.relu(margin - pos[:, None] + neg[None, :]).mean()


def all_positive_listwise(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask
    positive = (target > 0.5) & valid
    negative = (target <= 0.5) & valid
    if not bool(positive.any()):
        return logits.sum() * 0.0
    value = torch.logsumexp(logits[valid], dim=0) - torch.logsumexp(logits[positive], dim=0)
    if bool(negative.any()):
        # This term gives the lowest positive an explicit gradient while the
        # all-positive denominator above gives every positive a gradient.
        value = value + F.softplus(logits[negative].max() - logits[positive].min() + 0.2)
    return value


def compute_loss(output: dict[str, torch.Tensor], data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    target = data["membership_target"]
    coverage = data["coverage_mask"]
    history_mask = data["history_mask"]
    history_valid = history_mask & coverage[:, None]
    membership_mask = coverage
    m_bce = balanced_bce(output["membership_logits"], target, membership_mask)
    pair = pairwise_hinge(output["membership_logits"], target, membership_mask)
    listwise = all_positive_listwise(output["membership_logits"], target, membership_mask)
    track_bce = balanced_bce(output["track_logits"], data["track_target"], membership_mask)
    history_bce = balanced_bce(
        output["history_membership_logits"], data["history_target"], history_valid
    )
    continuation = balanced_bce(
        output["continuation_logits"], data["continuation_target"], membership_mask
    )
    null_target = data["null_target"].reshape(1)
    null_bce = F.binary_cross_entropy_with_logits(output["null_logit"].reshape(1), null_target)
    if bool(history_valid[:, 1:].any() & history_valid[:, :-1].any()):
        adjacent = history_valid[:, 1:] & history_valid[:, :-1]
        temporal = (
            torch.sigmoid(output["history_membership_logits"][:, 1:])[adjacent]
            - torch.sigmoid(output["history_membership_logits"][:, :-1])[adjacent]
        ).square().mean()
    else:
        temporal = output["membership_logits"].sum() * 0.0
    brier_m = ((torch.sigmoid(output["membership_logits"]) - target)[membership_mask]).square().mean() if bool(membership_mask.any()) else m_bce * 0.0
    brier_n = (torch.sigmoid(output["null_logit"].reshape(1)) - null_target).square().mean()
    brier = brier_m + brier_n
    total = (
        1.0 * m_bce + 0.4 * pair + 0.5 * listwise + 0.3 * track_bce
        + 0.2 * history_bce + 0.2 * continuation + 0.5 * null_bce
        + 0.1 * temporal + 0.1 * brier
    )
    parts = {
        "membership_bce": float(m_bce.detach()),
        "hard_negative_pairwise": float(pair.detach()),
        "multi_positive_listwise": float(listwise.detach()),
        "track_bce": float(track_bce.detach()),
        "history_bce": float(history_bce.detach()),
        "continuation_bce": float(continuation.detach()),
        "null_bce": float(null_bce.detach()),
        "temporal_consistency": float(temporal.detach()),
        "brier": float(brier.detach()),
        "total": float(total.detach()),
        "positive_count": float((target > 0.5).sum()),
        "masked_missing_count": float((~coverage).sum()),
        "hard_count": float(((target <= 0.5) & coverage).sum()),
    }
    return total, parts


def move_batch(data: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in data.items()}


def model_config() -> dict[str, int | float]:
    return {
        "obs_dim": 1432,
        "text_dim": 768,
        "hidden": 192,
        "heads": 4,
        "layers": 2,
        "max_history": MAX_HISTORY,
        "dropout": 0.0,
        "history_causal": True,
        "emission_formula": "membership + 0.25*track + 0.10*continuation",
        "hard_negative_metadata": "unavailable; all-negative same-frame fallback",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    running = {
        "format": "locatemot-l70-persistent-set-smoke-v1",
        "status": "running",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "seed": SEED,
        "steps_requested": int(args.steps),
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_dense_feature_cache_written": False,
    }
    write_json(out / "status.json", running)
    try:
        if str(ROOT.resolve()) != os.getcwd():
            raise RuntimeError(f"cwd mismatch: {os.getcwd()}")
        if args.steps != 100:
            raise ValueError("L70 smoke is preregistered at exactly 100 steps")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU0 is required for the authorized L70 smoke")
        device = torch.device(args.device)
        if device.index not in (0, None):
            raise RuntimeError(f"L70 resource contract requires GPU0, got {device}")
        seed_everything(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        records = read_records(args.index)
        schedule = make_schedule(records, args.steps, SEED)
        if len({row["dataset"] for row in schedule}) != 2:
            raise AssertionError("smoke sampler did not cover both datasets")
        if {row["category"] for row in schedule} != {"positive", "multi_positive", "inactive", "present_uncovered"}:
            raise AssertionError("smoke sampler did not cover all four categories")
        if len({row["video"] for row in schedule}) < 4:
            raise AssertionError("smoke sampler did not cover multiple videos")
        text_cache = load_text_cache()
        config = model_config()
        write_json(out / "config.json", {**running, **config, "device": str(device), "index": str(args.index)})
        model = L70PersistentSetDecoder(**{key: value for key, value in config.items() if key in {
            "obs_dim", "text_dim", "hidden", "heads", "layers", "max_history", "dropout"
        }}).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        model.train()
        loss_trace: list[dict[str, Any]] = []
        gradient_trace: list[dict[str, Any]] = []
        sampling_trace: list[dict[str, Any]] = []
        bank: L69Bank | None = None
        current_video: str | None = None
        last_batch: dict[str, torch.Tensor] | None = None
        last_output: dict[str, torch.Tensor] | None = None
        last_parts: dict[str, float] = {}
        total_candidates = 0
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, record in enumerate(schedule, start=1):
            video = str(record["video"])
            if current_video != video:
                if bank is not None:
                    bank.close()
                bank = L69Bank(video)
                current_video = video
            assert bank is not None
            data_cpu = unit_tensors(record, bank, text_cache)
            batch = move_batch(data_cpu, device)
            del data_cpu
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            if int(output["membership_logits"].numel()) != int(record["candidate_count"]):
                raise AssertionError("candidate output count drift")
            for name in ("membership_logits", "track_logits", "continuation_logits", "history_membership_logits", "null_logit"):
                output[name].retain_grad()
            loss, parts = compute_loss(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            parameter_grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            finite_grads = all(gradient is not None and torch.isfinite(gradient).all() for gradient in parameter_grads)
            nonzero_grads = any(bool(gradient.abs().sum() > 0) for gradient in parameter_grads if gradient is not None)
            if not finite_grads or not nonzero_grads:
                raise FloatingPointError(f"invalid parameter gradient at step {step}")
            labels = batch["membership_target"] > 0.5
            valid = batch["coverage_mask"]
            mgrad = output["membership_logits"].grad
            pos_grad = mgrad[labels & valid] if mgrad is not None else mgrad
            neg_grad = mgrad[(~labels) & valid] if mgrad is not None else mgrad
            grad_record = {
                "step": step,
                "finite_loss": True,
                "finite_parameter_gradients": bool(finite_grads),
                "nonzero_parameter_gradients": bool(nonzero_grads),
                "positive_count": int(labels.sum()),
                "hard_negative_count": int(((~labels) & valid).sum()),
                "positive_logit_grad_nonzero": int((pos_grad.abs() > 0).sum()) if pos_grad is not None else 0,
                "hard_negative_logit_grad_nonzero": int((neg_grad.abs() > 0).sum()) if neg_grad is not None else 0,
                "lowest_positive_gradient_nonzero": bool(pos_grad is not None and pos_grad.numel() and pos_grad[torch.argmin(output["membership_logits"][labels & valid])].abs() > 0) if bool((labels & valid).any()) else None,
                "max_parameter_grad": float(max(float(g.detach().abs().max()) for g in parameter_grads if g is not None)),
            }
            gradient_trace.append(grad_record)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_trace.append({"step": step, **parts})
            sampling_trace.append({
                "step": step,
                "schedule_position": int(record["schedule_position"]),
                "dataset": record["dataset"], "video": record["video"],
                "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]),
                "unit_key": record["unit_key"], "category": record["category"],
                "candidate_count": int(record["candidate_count"]),
                "positive_count": int(record["positive_count"]),
                "coverage_mask": bool(record["coverage_mask"]),
                "row_key_count": len(record["row_keys"]),
                "candidate_truncation": False,
                "same_class_hard_negative_metadata": "unavailable",
            })
            total_candidates += int(record["candidate_count"])
            last_batch, last_output, last_parts = batch, output, parts
        elapsed = time.perf_counter() - start
        if bank is not None:
            bank.close()
        model.eval()
        checkpoint = {
            "format": "locatemot-l70-persistent-set-checkpoint-v1",
            "step": int(args.steps), "seed": SEED, "model": model.state_dict(),
            "config": config,
        }
        checkpoint_path = out / "checkpoint_l70_step100.pt"
        torch.save(checkpoint, checkpoint_path)
        if last_batch is None:
            raise AssertionError("no last batch for reload")
        with torch.inference_mode():
            original_eval = model(last_batch)
            reloaded = L70PersistentSetDecoder(**{key: value for key, value in config.items() if key in {
                "obs_dim", "text_dim", "hidden", "heads", "layers", "max_history", "dropout"
            }}).to(device)
            package = safe_torch_load(checkpoint_path)
            missing, unexpected = reloaded.load_state_dict(package["model"], strict=False)
            if missing or unexpected:
                raise AssertionError(f"reload mismatch missing={missing} unexpected={unexpected}")
            reloaded.eval()
            restored_eval = reloaded(last_batch)
            shape_equal = {key: tuple(value.shape) for key, value in original_eval.items() if torch.is_tensor(value)} == {key: tuple(value.shape) for key, value in restored_eval.items() if torch.is_tensor(value)}
            max_diff = max(float((original_eval[key] - restored_eval[key]).abs().max()) for key in ("membership_logits", "track_logits", "continuation_logits", "null_logit"))
        write_json(out / "reload_audit.json", {
            "format": "locatemot-l70-reload-audit-v1", "status": "complete",
            "checkpoint": str(checkpoint_path), "strict": True,
            "missing_keys": [], "unexpected_keys": [], "shape_equal": bool(shape_equal),
            "max_abs_output_diff": max_diff, "finite": True,
        })
        write_json(out / "loss_trace.json", loss_trace)
        write_json(out / "sampling_trace.json", sampling_trace)
        write_json(out / "gradient_audit.json", {
            "format": "locatemot-l70-gradient-audit-v1", "status": "complete",
            "finite_steps": len(loss_trace), "nonzero_parameter_gradient_steps": sum(int(row["nonzero_parameter_gradients"]) for row in gradient_trace),
            "positive_logit_grad_rows": sum(row["positive_logit_grad_nonzero"] for row in gradient_trace),
            "hard_negative_logit_grad_rows": sum(row["hard_negative_logit_grad_nonzero"] for row in gradient_trace),
            "multi_positive_steps": sum(int(row["positive_count"] > 1) for row in gradient_trace),
            "records": gradient_trace,
        })
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        metrics = {
            "format": "locatemot-l70-step-metrics-v1", "status": "complete", "stage": "B0_fit_only_smoke",
            "steps": len(loss_trace), "finite_steps": sum(int(np.isfinite(row["total"])) for row in loss_trace),
            "nonzero_gradient_steps": sum(int(row["nonzero_parameter_gradients"]) for row in gradient_trace),
            "candidate_rows_processed": total_candidates,
            "datasets": sorted({str(row["dataset"]) for row in schedule}),
            "videos": sorted({str(row["video"]) for row in schedule}),
            "categories": dict(Counter(str(row["category"]) for row in schedule)),
            "candidate_key_drift": 0, "candidate_truncation": False,
            "history_future_rows": 0, "detector_parameters": "not part of L70; no detector parameters optimized",
            "parameter_summary": model.parameter_summary(),
            "wall_time_seconds": elapsed, "steps_per_second": len(loss_trace) / max(elapsed, 1e-9),
            "candidate_rows_per_second": total_candidates / max(elapsed, 1e-9),
            "peak_memory_bytes": peak if peak is not None else "unavailable",
            "strict_reload": bool(shape_equal and max_diff < 1e-5),
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "raw_dense_feature_cache_written": False,
        }
        write_json(out / "metrics_l70_step100.json", metrics)
        write_json(out / "provenance.json", {
            **running, "status": "complete", "steps": len(loss_trace), "device": str(device),
            "inputs": {"unit_index": str(args.index), "text_cache": str(ROOT / "outputs/l48/data/text_cache.pt"),
                       "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti")},
            "protocol": {"fit_only": True, "calibration_or_validation_read": False,
                         "full_candidate_set": True, "old_l49_begin_end_used": False,
                         "old_l49_positive_indices_used": False, "history_causal": True,
                         "feature_dim": 1432, "text_tokens_preserved": True,
                         "token_region_alignment": "UNALIGNED",
                         "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
                         "losses": list(last_parts), "no_l29_score_input": True},
            "checkpoint": str(checkpoint_path),
        })
        write_json(out / "status.json", {
            **running, "status": "complete", "steps_completed": len(loss_trace),
            "failure_root_cause": None, "next_action": "run fixed 16-cal/24-val semantic evaluation only after review",
        })
        (out / "run.log").write_text("L70 B0 smoke complete\n" + json.dumps(metrics, indent=2) + "\n")
        return 0
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text(
            "# L70 smoke INCOMPLETE\n\n"
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n"
            "```text\n" + traceback.format_exc() + "```\n"
        )
        write_json(out / "status.json", {
            **running, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "fix only the first actionable smoke root cause and run one targeted regression",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
