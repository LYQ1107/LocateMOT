#!/usr/bin/env python3
"""L51-B1 full-fit semantic training with one explicit wider bound.

Only L49 ``split=fit`` units are loaded for optimization.  The frozen L29
teacher remains the base emission and the L51 branch is a bounded residual.
Raw CLIP crops are created and released per step; no crop/embedding cache is
written.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l51_streaming_crop_adapter import L51StreamingCropAdapter  # noqa: E402
from locatemot.rmot.l49_data import (  # noqa: E402
    L29_CHECKPOINT,
    TEXT_CACHE,
    sha256_file,
)
from tools.train_l49_kitti_rmot import L29Teacher  # noqa: E402
from tools.train_l51_streaming_crop_adapter import (  # noqa: E402
    BANK_ROOT,
    CLIP_WEIGHTS,
    DATA,
    FAST_MANIFEST,
    L28_ROOT,
    StreamingClipPatches,
    forward_item,
    losses,
    materialize_units,
)


RESIDUAL_BOUND = 0.5
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def load_fit_units() -> list[dict]:
    path = DATA / "train_units.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(str(row.get("split")) != "fit" for row in rows):
        raise AssertionError("B1 requires only split=fit train_units")
    return rows


def balanced_order(units: list[dict], seed: int) -> list[dict]:
    """Deterministic round-robin over domain/category groups, then all units."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for unit in units:
        groups[(str(unit["dataset"]), str(unit["category"]))].append(unit)
    for values in groups.values():
        values.sort(key=lambda row: (str(row["video"]), int(row["frame_id"]), int(row["query_id"])))
        rng.shuffle(values)
    keys = sorted(groups)
    order = []
    active = {key: list(values) for key, values in groups.items()}
    while any(active.values()):
        for key in keys:
            if active[key]:
                order.append(active[key].pop())
    if len(order) != len(units):
        raise AssertionError("B1 sampler dropped or duplicated fit units")
    return order


def counts(units: list[dict], field: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(str(row[field]) for row in units).items())}


def make_provenance(all_units: list[dict], sampled: list[dict], seed: int) -> dict:
    return {
        "format": "locatemot-l51-b1-full-fit-provenance-v1",
        "stage": "B1",
        "project_root": str(ROOT),
        "seed": int(seed),
        "train_manifest": str((DATA / "train_units.jsonl").resolve()),
        "train_manifest_sha256": sha256_file(DATA / "train_units.jsonl"),
        "fit_only": True,
        "fit_unit_count": len(all_units),
        "sampled_unit_count": len(sampled),
        "fit_domains": counts(all_units, "dataset"),
        "fit_categories": counts(all_units, "category"),
        "fit_videos": counts(all_units, "video"),
        "sampled_domains": counts(sampled, "dataset"),
        "sampled_categories": counts(sampled, "category"),
        "sampled_videos": counts(sampled, "video"),
        "text_cache": str(TEXT_CACHE.resolve()),
        "text_cache_sha256": sha256_file(TEXT_CACHE),
        "fixed_manifest_sha256": sha256_file(FAST_MANIFEST),
        "l29_checkpoint": str(L29_CHECKPOINT.resolve()),
        "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
        "l19_bank_root": str(BANK_ROOT.resolve()),
        "l28_cache_root": str(L28_ROOT.resolve()),
        "residual_bound": RESIDUAL_BOUND,
        "residual_initialization": "final residual projection zero; initial final_logit exactly L29 teacher",
        "image_encoder": "frozen CLIP ViT-B/16",
        "image_weights": str(CLIP_WEIGHTS),
        "image_weights_sha256": sha256_file(CLIP_WEIGHTS),
        "crop_contract": {"box_source": "L19 candidate observation box", "padding": 0.10, "boundary": "clip",
                          "persistent_raw_cache": False},
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id"],
        "official_test_labels_read": False,
        "calibration_labels_read": False,
        "validation_labels_read": False,
        "screening_gt_used": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_cache_written": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_language_mask": "UNALIGNED/not claimed",
    }


def finite_state(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(value).all().item()
               for value in model.state_dict().values() if torch.is_floating_point(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    if args.steps < 500:
        raise ValueError("B1 requires at least 500 steps")
    if sha256_file(FAST_MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("fixed manifest SHA mismatch")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.time()
    try:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        all_units = load_fit_units()
        ordered = balanced_order(all_units, args.seed)
        sampled = ordered[:args.steps]
        text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        teacher = L29Teacher(text, torch.device("cpu"))
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("B1 requires the authorized single GPU")
        encoder = StreamingClipPatches(device)
        model = L51StreamingCropAdapter(hidden=128, heads=4, layers=2,
                                        residual_bound=RESIDUAL_BOUND).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        provenance = make_provenance(all_units, sampled, args.seed)
        config = {
            "seed": args.seed, "steps": args.steps, "device": str(device), "precision": "FP32",
            "residual_bound": RESIDUAL_BOUND, "model": model.config,
            "fit_units": {"total": len(all_units), "sampled": len(sampled), "sampler": "seeded domain/category round-robin"},
            "loss": {"frame_balanced_bce": 1.0, "pairwise_hard": 1.0, "multi_positive_listwise": 0.5,
                     "min_positive": 0.5, "teacher_distillation": 0.25, "residual_l2": 0.05,
                     "inactive_bce": 0.25, "objectness_prefilter": 48, "current_score_hard_topk": 12},
            "official_test_labels_read": False, "calibration_labels_read": False,
            "validation_labels_read": False, "screening_gt_used": False,
            "ordinary_mot_ovmot_touched": False, "raw_cache_written": False,
        }
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        sampling = Counter(); trace=[]; crop_count=0; peak=0
        initial_diff = None; initial_residual = None
        model.train()
        for step, unit in enumerate(sampled, start=1):
            item = materialize_units([unit], text, teacher)[0]
            sampling[(item["dataset"], item["video"], item["category"])] += 1
            if step == 1:
                with torch.inference_mode():
                    initial_out, initial_patch = forward_item(model.eval(), encoder, item, text, device)
                    initial_diff = float((initial_out["final_logit"] - item["teacher"].to(device)).abs().max())
                    initial_residual = float(initial_out["residual"].abs().max())
                del initial_out, initial_patch
                if initial_diff != 0.0 or initial_residual != 0.0:
                    raise AssertionError(f"initial teacher contract failed: {initial_diff}, {initial_residual}")
                model.train()
            opt.zero_grad(set_to_none=True)
            output, patch = forward_item(model, encoder, item, text, device)
            output["final_logit"].retain_grad()
            total, parts, pos, hard = losses(output, item, device)
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            total.backward()
            score_grad = output["final_logit"].grad.detach().abs()
            pos_frac = float((score_grad[pos] > 1e-10).float().mean()) if len(pos) else 0.0
            hard_frac = float((score_grad[hard] > 1e-10).float().mean()) if len(hard) else 0.0
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            finite_grads = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
            if not finite_grads or not math.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
            group_grad = {}
            for name, parameter in model.named_parameters():
                group = "image_adapter" if name.startswith("image_proj") else "text_adapter" if name.startswith("text_proj") else "residual" if name.startswith("residual_head") else "other"
                current = float(parameter.grad.detach().abs().max()) if parameter.grad is not None else 0.0
                group_grad[group] = max(group_grad.get(group, 0.0), current)
            opt.step()
            crop_count += int(patch.shape[0])
            peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
            row = {"step": step, **parts, "gradient_norm": grad_norm,
                   "positive_grad_fraction": pos_frac, "hard_grad_fraction": hard_frac,
                   "residual_mean": float(output["residual"].detach().mean()),
                   "residual_max_abs": float(output["residual"].detach().abs().max()),
                   "group_grad_max": group_grad, "finite": True,
                   "candidate_count": int(len(item["y"])), "candidate_truncation": False}
            trace.append(row)
            del output, patch, total, item
            gc.collect()
            if step in (100, 250, 500):
                step_dir = out / f"step{step}"; step_dir.mkdir()
                checkpoint = step_dir / f"checkpoint_l51_b1_step{step}.pt"
                torch.save({"format": "locatemot-l51-b1-checkpoint-v1", "stage": "B1", "step": step,
                            "model": model.state_dict(), "optimizer": opt.state_dict(),
                            "config": config, "provenance": provenance}, checkpoint)
                reload_model = L51StreamingCropAdapter(hidden=128, heads=4, layers=2,
                                                       residual_bound=RESIDUAL_BOUND).cpu()
                loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
                reload_model.load_state_dict(loaded["model"], strict=True)
                reload_ok = finite_state(reload_model)
                if not reload_ok:
                    raise FloatingPointError(f"nonfinite reload at step {step}")
                step_metrics = {"stage": "B1", "step": step, "checkpoint": str(checkpoint.resolve()),
                                "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_reload": True,
                                "state_finite": reload_ok, "finite_steps": step,
                                "residual_bound": RESIDUAL_BOUND,
                                "residual_max_abs_so_far": max(x["residual_max_abs"] for x in trace),
                                "sampled_steps": step, "crop_count": crop_count}
                (step_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
                (step_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
                (step_dir / "reload_audit.json").write_text(json.dumps(step_metrics, indent=2) + "\n")
                (step_dir / f"metrics_step{step}.json").write_text(json.dumps(step_metrics, indent=2) + "\n")
        elapsed = time.time() - started
        metrics = {
            "format": "locatemot-l51-b1-metrics-v1", "stage": "B1", "status": "pass",
            "seed": args.seed, "requested_steps": args.steps, "completed_steps": len(trace),
            "finite_steps": sum(int(row["finite"]) for row in trace),
            "nonzero_gradient_steps": sum(int(row["gradient_norm"] > 0) for row in trace),
            "fit_unit_count": len(all_units), "sampled_unit_count": len(sampled),
            "sampled_domain_count": len(set(row["dataset"] for row in sampled)),
            "sampled_video_count": len(set(row["video"] for row in sampled)),
            "sampled_category_counts": counts(sampled, "category"),
            "candidate_frame_key_drift": 0, "candidate_truncation": False,
            "base_vs_initial_residual_diff": initial_diff, "initial_residual_max_abs": initial_residual,
            "residual_bound": RESIDUAL_BOUND,
            "residual_max_abs_over_run": max(row["residual_max_abs"] for row in trace),
            "positive_gradient_fraction_min": min(row["positive_grad_fraction"] for row in trace if row["positive_count"]),
            "hard_gradient_fraction_min": min(row["hard_grad_fraction"] for row in trace if row["hard_negative_count"]),
            "image_adapter_nonzero_gradient_steps": sum(row["group_grad_max"].get("image_adapter", 0) > 0 for row in trace),
            "residual_nonzero_gradient_steps": sum(row["group_grad_max"].get("residual", 0) > 0 for row in trace),
            "loss_mean": {key: float(np.mean([row[key] for row in trace])) for key in ("total", "frame_balanced_bce", "pairwise_hard", "multi_positive_listwise", "min_positive", "teacher_distillation", "residual_l2", "inactive_bce")},
            "gradient_norm_mean": float(np.mean([row["gradient_norm"] for row in trace])),
            "crop_count": crop_count, "peak_memory_bytes": peak, "elapsed_sec": elapsed,
            "steps_per_sec": len(trace) / max(elapsed, 1e-9),
            "checkpoints": {str(step): str((out / f"step{step}" / f"checkpoint_l51_b1_step{step}.pt").resolve()) for step in (100, 250, 500)},
            "official_test_labels_read": False, "calibration_labels_read": False,
            "validation_labels_read": False, "screening_gt_used": False,
            "ordinary_mot_ovmot_touched": False, "raw_cache_written": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_language_mask": "UNALIGNED/not claimed",
        }
        (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        (out / "gradient_audit.json").write_text(json.dumps({
            "finite_steps": metrics["finite_steps"], "nonzero_gradient_steps": metrics["nonzero_gradient_steps"],
            "image_adapter_nonzero_gradient_steps": metrics["image_adapter_nonzero_gradient_steps"],
            "residual_nonzero_gradient_steps": metrics["residual_nonzero_gradient_steps"],
            "positive_gradient_fraction_min": metrics["positive_gradient_fraction_min"],
            "hard_gradient_fraction_min": metrics["hard_gradient_fraction_min"],
        }, indent=2) + "\n")
        (out / "sampling_trace.json").write_text(json.dumps({
            "fit_unit_count": len(all_units), "sampled_unit_count": len(sampled),
            "fit_domain_counts": counts(all_units, "dataset"), "fit_category_counts": counts(all_units, "category"),
            "sampled_domain_counts": counts(sampled, "dataset"), "sampled_category_counts": counts(sampled, "category"),
            "sampled_video_counts": counts(sampled, "video"), "unit_keys": [str(x["unit_key"]) for x in sampled],
            "candidate_set_policy": "complete set per unit; no truncation",
        }, indent=2) + "\n")
        (out / "metrics_l51_b1.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out / "reload_audit.json").write_text(json.dumps({"checkpoints": metrics["checkpoints"], "all_saved_steps_reload": True}, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text(f"# L51 B1 incomplete\n\nFirst actionable root cause: `{type(exc).__name__}: {exc}`\n")
        raise


if __name__ == "__main__":
    main()
