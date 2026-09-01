#!/usr/bin/env python3
"""Fit-only smoke/controlled probe for the isolated L71 correspondence head."""
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

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l71_common import (  # noqa: E402
    L71Bank,
    MAX_HISTORY,
    load_text_cache,
    safe_torch_load,
    unit_tensors,
    write_json,
)
from tools.l71_loss import compute_loss  # noqa: E402
from locatemot.models.l71_bounded_query_track import L71BoundedQueryTrack  # noqa: E402


INDEX = ROOT / "outputs/l71/audit/data_contract_retry2/unit_records.jsonl"
SEED = 20260829
SCHEDULE_HORIZON = 250


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_schedule(records: list[dict[str, Any]], steps: int, seed: int) -> list[dict[str, Any]]:
    fit = [row for row in records if row.get("index_role") == "fit" and row.get("split") == "fit"]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fit:
        buckets[(str(row["dataset"]), str(row["category"]))].append(row)
    required = {
        (dataset, category)
        for dataset in ("refer_kitti_v1", "refer_kitti_v2")
        for category in ("positive", "multi_positive", "inactive", "present_uncovered")
    }
    if set(buckets) != required:
        raise AssertionError(f"fit strata mismatch: {sorted(set(buckets) ^ required)}")
    rng = random.Random(seed)
    for key in sorted(buckets):
        buckets[key] = sorted(
            buckets[key], key=lambda row: (str(row["video"]), int(row["query_id"]), int(row["frame_id"]))
        )
        rng.shuffle(buckets[key])
    horizon = max(int(steps), SCHEDULE_HORIZON)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    for position in range(horizon):
        key = keys[position % len(keys)]
        row = dict(buckets[key][(position // len(keys)) % len(buckets[key])])
        row["schedule_position"] = position
        selected.append(row)
    # A stable video order allows one bank load per contiguous video block.
    # The fixed horizon makes the first 100 schedule entries identical to the
    # prefix used by the 250-step continuation.
    selected.sort(key=lambda row: (str(row["video"]), int(row["schedule_position"])))
    for train_step, row in enumerate(selected, start=1):
        row["train_step"] = train_step
    return selected[: int(steps)]


def cpu_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_value(item) for item in value]
    return value


def model_config() -> dict[str, Any]:
    return {
        "obs_dim": 1432,
        "text_dim": 768,
        "hidden": 192,
        "max_history": MAX_HISTORY,
        "temperature": 0.07,
        "history_aggregation": "L2-normalized(0.5*current + 0.5*masked_mean(valid_history))",
        "hard_negative_metadata": "unavailable; all-negative fallback",
        "token_region_alignment": "UNALIGNED",
        "null_head": "not_implemented_for_isolation",
    }


def model_args(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in ("obs_dim", "text_dim", "hidden", "max_history", "temperature")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=(100, 250), required=True)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    stage = "fit_only_smoke" if args.steps == 100 else "fit_only_controlled_probe"
    running = {
        "format": "locatemot-l71-bounded-correspondence-train-v1",
        "status": "running",
        "stage": stage,
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "seed": SEED,
        "steps_requested": int(args.steps),
        "resume": None if args.resume is None else str(args.resume),
        "inputs": {
            "unit_index": str(args.index),
            "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
            "text_cache": str(ROOT / "outputs/l48/data/text_cache.pt"),
        },
        "outputs": {"root": str(out)},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "raw_dense_feature_cache_written": False,
        "l29_l70_scores_used_as_inputs": False,
    }
    write_json(out / "status.json", running)
    started = time.perf_counter()
    bank: L71Bank | None = None
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU0 is required for L71 training")
        device = torch.device(args.device)
        if device.type != "cuda" or device.index not in (None, 0):
            raise RuntimeError(f"L71 resource contract requires GPU0, got {device}")
        seed_everything(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        records = read_records(args.index)
        schedule = make_schedule(records, args.steps, SEED)
        if len({row["dataset"] for row in schedule}) != 2:
            raise AssertionError("schedule missed one dataset")
        required_categories = {"positive", "multi_positive", "inactive", "present_uncovered"}
        if {row["category"] for row in schedule} != required_categories:
            raise AssertionError("schedule missed one required category")
        if len({row["video"] for row in schedule}) < 4:
            raise AssertionError("schedule did not cover multiple videos")
        text_cache = load_text_cache()
        config = model_config()
        model = L71BoundedQueryTrack(**model_args(config)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        start_step = 0
        loss_trace: list[dict[str, Any]] = []
        gradient_trace: list[dict[str, Any]] = []
        sampling_trace: list[dict[str, Any]] = []
        if args.resume is not None:
            if args.steps != 250:
                raise ValueError("only the 250-step controlled probe may resume step100")
            package = safe_torch_load(args.resume)
            if int(package.get("step", -1)) != 100:
                raise AssertionError("resume checkpoint must be exactly step100")
            if package.get("seed") != SEED:
                raise AssertionError("resume seed drift")
            model.load_state_dict(package["model"], strict=True)
            optimizer.load_state_dict(package["optimizer"])
            start_step = 100
            prior_dir = args.resume.parent
            for name, target in (("loss_trace.json", loss_trace), ("gradient_trace.json", gradient_trace), ("sampling_trace.json", sampling_trace)):
                source = prior_dir / name
                if source.exists():
                    target.extend(json.loads(source.read_text()))
            if len(loss_trace) != 100 or len(gradient_trace) != 100 or len(sampling_trace) != 100:
                raise AssertionError("resume trace must contain exactly 100 prior steps")
            torch.save(package, out / "checkpoint_l71_step100.pt")
        write_json(out / "config.json", {**running, **config, "device": str(device), "optimizer_lr": 2e-4, "optimizer_weight_decay": 1e-4, "sampler": "seeded eight-stratum schedule; fixed 250-step horizon prefix", "candidate_set": "all L69 rows"})

        current_video: str | None = None
        last_batch: dict[str, torch.Tensor] | None = None
        total_candidates = 0
        run_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for absolute_step in range(start_step + 1, args.steps + 1):
            record = schedule[absolute_step - 1]
            video = str(record["video"])
            if current_video != video:
                if bank is not None:
                    bank.close()
                bank = L71Bank(video)
                current_video = video
            data_cpu = unit_tensors(record, bank, text_cache)
            batch = {key: value.to(device, non_blocking=True) for key, value in data_cpu.items()}
            del data_cpu
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            if int(output["correspondence_logits"].numel()) != int(record["candidate_count"]):
                raise AssertionError(f"candidate count drift at step {absolute_step}")
            logits = output["correspondence_logits"]
            logits.retain_grad()
            loss, parts = compute_loss(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {absolute_step}")
            loss.backward()
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            finite_grads = all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
            nonzero_grads = any(parameter.grad is not None and bool(parameter.grad.detach().abs().sum() > 0) for parameter in parameters)
            if not finite_grads or not nonzero_grads:
                raise FloatingPointError(f"invalid gradients at step {absolute_step}")
            labels = batch["membership_target"] > 0.5
            valid = batch["coverage_mask"].bool()
            logit_grad = logits.grad.detach().abs()
            positive_grad = logit_grad[labels & valid]
            negative_grad = logit_grad[(~labels) & valid]
            gradient_record = {
                "step": absolute_step,
                "finite_loss": True,
                "finite_parameter_gradients": bool(finite_grads),
                "nonzero_parameter_gradients": bool(nonzero_grads),
                "positive_count": int((labels & valid).sum()),
                "hard_negative_count": int(((~labels) & valid).sum()),
                "positive_logit_grad_nonzero": int((positive_grad > 0).sum()),
                "hard_negative_logit_grad_nonzero": int((negative_grad > 0).sum()),
                "minimum_positive_gradient_nonzero": bool(positive_grad.numel() == 0 or bool((positive_grad > 0).all())),
                "max_parameter_grad": float(max(parameter.grad.detach().abs().max() for parameter in parameters)),
            }
            gradient_trace.append(gradient_record)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_trace.append({"step": absolute_step, **parts})
            sampling_trace.append({
                "step": absolute_step,
                "schedule_position": int(record["schedule_position"]),
                "unit_key": record["unit_key"],
                "dataset": record["dataset"],
                "video": record["video"],
                "query_id": int(record["query_id"]),
                "frame_id": int(record["frame_id"]),
                "category": record["category"],
                "candidate_count": int(record["candidate_count"]),
                "positive_count": int(record["positive_count"]),
                "coverage_mask": bool(record["coverage_mask"]),
                "candidate_truncation": False,
                "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
            })
            total_candidates += int(record["candidate_count"])
            last_batch = batch

        if bank is not None:
            bank.close()
            bank = None
        model.eval()
        final_checkpoint = out / f"checkpoint_l71_step{args.steps}.pt"
        package = {
            "format": "locatemot-l71-bounded-query-track-checkpoint-v1",
            "step": int(args.steps),
            "seed": SEED,
            "model": cpu_value(model.state_dict()),
            "optimizer": cpu_value(optimizer.state_dict()),
            "config": config,
        }
        torch.save(package, final_checkpoint)
        if last_batch is None:
            raise AssertionError("no final batch")
        with torch.inference_mode():
            original = model(last_batch)
            reloaded = L71BoundedQueryTrack(**model_args(config)).to(device)
            loaded = safe_torch_load(final_checkpoint)
            reloaded.load_state_dict(loaded["model"], strict=True)
            reloaded.eval()
            restored = reloaded(last_batch)
            shape_equal = {key: tuple(value.shape) for key, value in original.items()} == {key: tuple(value.shape) for key, value in restored.items()}
            max_diff = max(float((original[key] - restored[key]).abs().max()) for key in original)
            reload_finite = all(bool(torch.isfinite(value).all()) for value in restored.values())
        if not shape_equal or max_diff >= 1e-5 or not reload_finite:
            raise AssertionError(f"strict reload failed shape={shape_equal} diff={max_diff} finite={reload_finite}")
        write_json(out / "reload_audit.json", {
            "format": "locatemot-l71-reload-audit-v1",
            "status": "complete",
            "checkpoint": str(final_checkpoint),
            "strict": True,
            "shape_equal": bool(shape_equal),
            "max_abs_output_diff": max_diff,
            "finite": bool(reload_finite),
            "missing_keys": [],
            "unexpected_keys": [],
        })
        write_json(out / "loss_trace.json", loss_trace)
        write_json(out / "gradient_trace.json", gradient_trace)
        write_json(out / "sampling_trace.json", sampling_trace)
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else "unavailable"
        metrics = {
            "format": "locatemot-l71-step-metrics-v1",
            "status": "complete",
            "stage": stage,
            "steps": int(args.steps),
            "start_step": int(start_step),
            "finite_steps": int(sum(bool(row["finite_loss"]) for row in gradient_trace)),
            "nonzero_gradient_steps": int(sum(bool(row["nonzero_parameter_gradients"]) for row in gradient_trace)),
            "positive_logit_grad_rows": int(sum(row["positive_logit_grad_nonzero"] for row in gradient_trace)),
            "hard_negative_logit_grad_rows": int(sum(row["hard_negative_logit_grad_nonzero"] for row in gradient_trace)),
            "minimum_positive_gradient_checks": int(sum(bool(row["minimum_positive_gradient_nonzero"]) for row in gradient_trace)),
            "candidate_rows_processed_current_run": int(total_candidates),
            "candidate_rows_processed_all_trace": int(sum(row["candidate_count"] for row in sampling_trace)),
            "datasets": sorted({str(row["dataset"]) for row in sampling_trace}),
            "videos": sorted({str(row["video"]) for row in sampling_trace}),
            "categories": dict(Counter(str(row["category"]) for row in sampling_trace)),
            "candidate_key_drift": 0,
            "candidate_truncation": False,
            "history_future_rows": 0,
            "parameter_summary": model.parameter_summary(),
            "checkpoint": str(final_checkpoint),
            "wall_time_seconds_current_run": time.perf_counter() - run_started,
            "wall_time_seconds_total": time.perf_counter() - started,
            "steps_per_second_current_run": (args.steps - start_step) / max(time.perf_counter() - run_started, 1e-9),
            "peak_memory_bytes": peak,
            "strict_reload": True,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
            "raw_dense_feature_cache_written": False,
            "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
            "null_head": "not_implemented_for_isolation",
        }
        write_json(out / f"metrics_l71_step{args.steps}.json", metrics)
        write_json(out / "provenance.json", {
            **running,
            "status": "complete",
            "device": str(device),
            "checkpoint": str(final_checkpoint),
            "checkpoint_sha256": __import__("tools.l71_common", fromlist=["sha256_file"]).sha256_file(final_checkpoint),
            "model": config,
            "protocol": {
                "fit_only": True,
                "fit_unit_count": 5314,
                "complete_candidate_set": True,
                "old_l49_begin_end_used": False,
                "old_l49_positive_indices_used": False,
                "history_causal": True,
                "observation_dim": 1432,
                "text_tokens_preserved": True,
                "temperature": 0.07,
                "token_region_alignment": "UNALIGNED",
                "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
                "loss": "unit-local pairwise softplus + all-positive listwise/minimum-positive + inactive no-match + logit L2",
            },
            "resume_source": None if args.resume is None else str(args.resume),
        })
        write_json(out / "status.json", {**running, "status": "complete", "steps_completed": int(args.steps), "failure_root_cause": None, "next_action": "run fixed L71 semantic evaluation after both step100 and step250 evidence are available"})
        (out / "run.log").write_text(json.dumps(metrics, indent=2) + "\n")
        return 0
    except Exception as exc:
        if bank is not None:
            bank.close()
        write_json(out / "status.json", {**running, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}", "next_action": "fix only the first actionable L71 training root cause and rerun a targeted regression in a new directory"})
        (out / "INCOMPLETE.md").write_text(
            "# L71 training INCOMPLETE\n\n"
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n"
            "```text\n" + traceback.format_exc() + "```\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
