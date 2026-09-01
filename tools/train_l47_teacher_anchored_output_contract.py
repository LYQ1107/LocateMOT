#!/usr/bin/env python3
"""Train the bounded L47 teacher-anchored output-contract probe.

This is intentionally a small train-only experiment.  Each optimization
sample is one complete current-frame candidate set; the frozen L29 logits are
computed once and remain an explicit input/output anchor.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
import sys
sys.path.insert(0, str(ROOT))

from locatemot.models.l47_teacher_anchored_output_contract import (  # noqa: E402
    L47TeacherAnchoredOutputContract,
    teacher_anchored_loss,
)
from tools.l47_data import (  # noqa: E402
    FAST,
    FIT_VIDEOS,
    L28,
    L29,
    SPLIT,
    build_unit,
    category,
    load_bank,
    load_l29,
    load_queries,
    load_text,
    sha256,
    smoke_refs,
)


def manual_pairwise_gradient_audit():
    """Check the sign and all-positive gradient of the hinge surrogate."""
    positive = torch.tensor([0.8, 0.5], dtype=torch.float32, requires_grad=True)
    negative = torch.tensor([0.2, -0.1], dtype=torch.float32, requires_grad=True)
    good = torch.nn.functional.softplus(
        0.1 + negative[None, :] - positive[:, None]
    ).mean()
    good.backward()
    # A positive score should be pushed up (negative derivative); a negative
    # score should be pushed down (positive derivative).
    if not bool((positive.grad < 0).all()) or not bool((negative.grad > 0).all()):
        raise AssertionError("pairwise gradient direction is incorrect")
    positive_bad = torch.tensor([0.1, 0.0])
    negative_bad = torch.tensor([0.6, 0.5])
    bad = torch.nn.functional.softplus(
        0.1 + negative_bad[None, :] - positive_bad[:, None]
    ).mean()
    if not float(good.detach()) < float(bad.detach()):
        raise AssertionError("pairwise loss does not decrease for positive>negative")
    return {
        "positive_scores": [0.8, 0.5], "negative_scores": [0.2, -0.1],
        "good_loss": float(good.detach()), "bad_loss": float(bad.detach()),
        "positive_gradients": [float(x) for x in positive.grad],
        "negative_gradients": [float(x) for x in negative.grad],
        "all_positive_nonzero": bool((positive.grad.abs() > 1e-10).all()),
        "all_negative_nonzero": bool((negative.grad.abs() > 1e-10).all()),
        "passed": True,
    }


def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def rank_step_stats(output, unit):
    teacher = output["teacher_score"].detach().float()
    score = output["final_score"].detach().float()
    labels = unit["labels"].bool()
    pos = torch.nonzero(labels, as_tuple=False).flatten()
    hard = unit["hard_indices"]
    if len(pos) and len(hard):
        td = teacher[pos, None] - teacher[hard][None, :]
        sd = score[pos, None] - score[hard][None, :]
        correct = td > 0
        return {
            "pairs": int(td.numel()),
            "teacher_correct_pairs": int(correct.sum()),
            "teacher_error_pairs": int((~correct).sum()),
            "teacher_correct_flips": int((correct & (sd < 0)).sum()),
            "teacher_error_corrections": int((~correct & (sd > 0)).sum()),
        }
    return {"pairs": 0, "teacher_correct_pairs": 0, "teacher_error_pairs": 0,
            "teacher_correct_flips": 0, "teacher_error_corrections": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unit-count", type=int, default=32)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    if args.steps < 1 or args.steps > 250:
        raise ValueError("L47 B0 is limited to 250 steps")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.set_num_threads(1)
    manual = manual_pairwise_gradient_audit()

    queries = load_queries()
    banks = {video: load_bank(video) for video in FIT_VIDEOS}
    caches = {
        video: torch.load(L28 / f"{video}.pt", map_location="cpu", weights_only=False)
        for video in FIT_VIDEOS
    }
    text_hidden, text_mask = load_text()
    teacher = load_l29(device)
    refs = smoke_refs(queries, banks, FIT_VIDEOS, limit=args.unit_count)
    if len(refs) < min(args.unit_count, 8):
        raise RuntimeError(f"only {len(refs)} train smoke refs available")
    units = []
    for query, frame_index, _labels in refs:
        units.append(build_unit(
            query, frame_index, banks[query["video"]],
            caches[query["video"]],
            teacher, text_hidden, text_mask,
        ))
    distinct_videos = sorted({unit["video"] for unit in units})
    if len(distinct_videos) < 8:
        raise RuntimeError(f"smoke sampled only {len(distinct_videos)} videos")
    category_counts = {}
    for unit in units:
        name = category(unit["labels"].numpy(), set(unit["target_ids"]))
        category_counts[name] = category_counts.get(name, 0) + 1
    del teacher, banks, caches

    model = L47TeacherAnchoredOutputContract(hidden=128, heads=4, layers=2,
                                             residual_bound=0.05, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    trace = []
    gradient_norms = []
    positive_gradient_fractions = []
    hard_gradient_fractions = []
    rank_totals = {key: 0 for key in (
        "pairs", "teacher_correct_pairs", "teacher_error_pairs",
        "teacher_correct_flips", "teacher_error_corrections")}
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        unit = units[int(rng.integers(len(units)))]
        region = unit["clip"].to(device)
        history = unit["history_clip"].to(device)
        numeric = unit["numeric"].to(device)
        teacher_score = unit["teacher"].to(device)
        labels = unit["labels"].to(device)
        hard = unit["hard_indices"].to(device)
        correct = unit["teacher_correct_pairs"].to(device)
        error = unit["teacher_error_pairs"].to(device)
        with autocast_context(device):
            output = model(
                region, history, numeric,
                text_hidden[unit["query_index"]].to(device),
                text_mask[unit["query_index"]].to(device),
                teacher_score,
            )
            output["final_score"].retain_grad()
            loss, parts = teacher_anchored_loss(
                output, labels, hard, correct, error,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        gradient_norms.append(grad_norm)
        grad = output["final_score"].grad.detach().float()
        pos = torch.nonzero(labels, as_tuple=False).flatten()
        pos_fraction = float((grad[pos].abs() > 1e-10).float().mean()) if len(pos) else 0.0
        hard_fraction = float((grad[hard].abs() > 1e-10).float().mean()) if len(hard) else 0.0
        positive_gradient_fractions.append(pos_fraction)
        hard_gradient_fractions.append(hard_fraction)
        ranks = rank_step_stats(output, unit)
        for key in rank_totals:
            rank_totals[key] += ranks[key]
        row = dict(parts)
        row.update({
            "step": step,
            "scale": float(output["scale"].detach().float()),
            "frame_offset": float(output["frame_offset"].detach().float()),
            "residual_mean": float(output["residual"].detach().float().mean()),
            "residual_max_abs": float(output["residual"].detach().float().abs().max()),
            "rank_stats": ranks,
            "gradient_norm": grad_norm,
            "positive_gradient_fraction": pos_fraction,
            "hard_gradient_fraction": hard_fraction,
            "top1_proxy": float(labels[torch.argmax(output["final_score"])]) if len(pos) else None,
            "positive_count": int(labels.sum()),
            "hard_count": int(hard.numel()),
            "video": unit["video"],
        })
        trace.append(row)

    model.eval()
    checkpoint = out / f"checkpoint_l47_teacher_anchored_step{args.steps}.pt"
    payload = {
        "format": "locatemot-l47-teacher-anchored-output-contract-v1",
        "stage": "L47-B0",
        "project_root": str(ROOT),
        "seed": args.seed,
        "steps": args.steps,
        "device": str(device),
        "fit_videos": list(FIT_VIDEOS),
        "sampled_unit_count": len(units),
        "sampled_distinct_videos": distinct_videos,
        "sample_category_counts": category_counts,
        "screening_gt_used_for_fit": False,
        "fixed_fast_manifest_used_for_training": False,
        "fast_manifest": {"path": str(FAST.resolve()), "sha256": sha256(FAST)},
        "split_manifest": {"path": str(SPLIT.resolve()), "sha256": sha256(SPLIT)},
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_index_as_feature"],
        "token_span_region_verified": False,
        "motion_language_decomposition": "not claimed; no verified motion-language mask",
        "teacher": {"checkpoint": str(L29.resolve()), "sha256": sha256(L29), "role": "frozen primary emission anchor"},
        "model_config": model.config,
        "score_contract": {
            "map": "m_prime = a*m + b_frame",
            "scale_bounds": [0.9, 1.1],
            "residual_bounds": [-0.05, 0.05],
            "residual_is_independent_scorer": False,
            "frame_offset_is_global_auditable_scalar": True,
        },
        "loss_contract": {
            "full_candidate_set_per_unit": True,
            "teacher_order_weight": 1.0,
            "teacher_error_weight": 0.5,
            "multi_positive_all_positive_gradient": True,
            "inactive_null_auxiliary_only": True,
            "distillation": True,
            "residual_l2": True,
            "frame_zero_drift": True,
            "rank_flip_penalty": True,
        },
        "manual_pairwise_gradient_audit": manual,
        "loss_mean": {key: float(np.mean([row[key] for row in trace]))
                      for key in trace[0] if key in {
                          "total", "membership_bce", "teacher_order",
                          "teacher_error_correction", "pairwise", "listwise",
                          "min_positive", "distillation", "inactive_aux",
                          "rank_flip_penalty", "scale_regularizer", "offset_regularizer",
                          "residual_l2", "frame_zero_drift"}},
        "gradient": {
            "norm_mean": float(np.mean(gradient_norms)),
            "norm_max": float(np.max(gradient_norms)),
            "nonzero_steps": int(np.count_nonzero(np.asarray(gradient_norms) > 0)),
            "positive_nonzero_fraction_mean": float(np.mean(positive_gradient_fractions)),
            "hard_nonzero_fraction_mean": float(np.mean(hard_gradient_fractions)),
        },
        "rank_totals": rank_totals,
        "residual_observed": {
            "max_abs": float(max(row["residual_max_abs"] for row in trace)),
            "mean_abs": float(np.mean([abs(row["residual_mean"]) for row in trace])),
            "scale_min": float(min(row["scale"] for row in trace)),
            "scale_max": float(max(row["scale"] for row in trace)),
            "frame_offset_min": float(min(row["frame_offset"] for row in trace)),
            "frame_offset_max": float(max(row["frame_offset"] for row in trace)),
        },
        "elapsed_sec": time.time() - start,
    }
    torch.save({"model": model.state_dict(), "config": payload}, checkpoint)
    reloaded = L47TeacherAnchoredOutputContract(**model.config).cpu()
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    payload["checkpoint"] = str(checkpoint.resolve())
    payload["checkpoint_reload"] = True
    metrics_name = "metrics_l47_smoke100.json" if args.steps == 100 else f"metrics_l47_step{args.steps}.json"
    (out / metrics_name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    (out / "config.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (out / "README.md").write_text(
        "# L47 teacher-anchored output-contract training\n\n"
        "Train-only B0 probe. Every unit is a complete current-frame candidate set; "
        "L29 remains the primary frozen score anchor.\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
