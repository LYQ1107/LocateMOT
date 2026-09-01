#!/usr/bin/env python3
"""L64 fit-only smoke using real PNG crop patch tokens and CLIP text tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
UNITS = ROOT / "outputs/l49/data/train_units.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"

import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l64_raw_patch_set import L64RawPatchSet
from tools.l64_raw_patch_common import StreamingOpenAIClip, sha256


def load_fit():
    out = []
    for line in UNITS.read_text().splitlines():
        if not line.strip():
            continue
        u = json.loads(line)
        if u.get("split") == "fit" and u.get("dataset") in ("refer_kitti_v1", "refer_kitti_v2"):
            out.append(u)
    return out


def stratified(units, seed):
    rng = random.Random(seed)
    cats = ("positive", "multi_positive", "inactive", "present_uncovered")
    buckets = {(d, c): [] for d in ("refer_kitti_v1", "refer_kitti_v2") for c in cats}
    for u in units:
        buckets.setdefault((u["dataset"], u.get("category", "unknown")), []).append(u)
    for values in buckets.values():
        rng.shuffle(values)
    order = []
    while any(buckets.get(k) for k in buckets):
        for key in sorted(buckets):
            if buckets[key]:
                order.append(buckets[key].pop())
    return order


def numeric(t, begin, end):
    return torch.cat((t["geometry"][begin:end].float(), t["motion"][begin:end].float(),
                      t["lifecycle"][begin:end].float(), t["context"][begin:end].float(),
                      t["objectness"][begin:end].float().reshape(-1, 1)), dim=1)


def load_unit(unit, encoder):
    bank = torch.load(Path(unit["bank_path"]), map_location="cpu", weights_only=False)
    t = bank["tensors"]
    begin, end = int(unit["begin"]), int(unit["end"])
    n = end - begin
    if n != int(unit["candidate_count"]):
        raise AssertionError(f"candidate count mismatch {unit['unit_key']}")
    boxes = t["box"][begin:end].float()
    patches, image = encoder.encode_unit(unit["video"], int(unit["frame_id"]), boxes.tolist())
    words, mask = encoder.text_tokens(unit["sentence"])
    if tuple(patches.shape) != (n, 16, 768):
        raise AssertionError(f"unexpected image patch shape {patches.shape}")
    numeric_values = numeric(t, begin, end)
    # Labels are touched only after PNG feature construction is complete.
    y = torch.zeros(n, dtype=torch.bool)
    positive = [int(x) for x in unit.get("positive_indices", [])]
    if any(x < 0 or x >= n for x in positive):
        raise AssertionError(f"positive index out of range {unit['unit_key']}")
    if positive:
        y[torch.as_tensor(positive, dtype=torch.long)] = True
    # CLIP runs under inference_mode; clone at the frozen-to-trainable boundary
    # so adapter autograd never receives inference tensors.
    return {"unit": unit, "patches": patches.clone(), "words": words.clone(), "mask": mask.clone(),
            "numeric": numeric_values, "target": y, "image": str(image),
            "row_offsets": list(range(begin, end))}


def balanced_bce(score, target):
    parts = []
    if target.any():
        parts.append(F.binary_cross_entropy_with_logits(score[target], torch.ones_like(score[target])))
    if (~target).any():
        parts.append(F.binary_cross_entropy_with_logits(score[~target], torch.zeros_like(score[~target])))
    return torch.stack(parts).mean() if parts else score.new_zeros(())


def loss_fn(out, target):
    score = out["relevance_logit"]
    pos = torch.nonzero(target, as_tuple=False).flatten()
    neg = torch.nonzero(~target, as_tuple=False).flatten()
    zero = score.new_zeros(())
    bce = balanced_bce(score, target)
    if len(pos) and len(neg):
        hard = neg[torch.argsort(score.detach()[neg], descending=True)[:min(24, len(neg))]]
        pair = F.softplus(0.2 + score[hard][None, :] - score[pos][:, None]).mean()
        listwise = torch.logsumexp(score, 0) - torch.logsumexp(score[pos], 0)
    else:
        hard, pair, listwise = neg, zero, zero
    min_positive = F.binary_cross_entropy_with_logits(score[pos], torch.ones_like(score[pos])) if len(pos) else zero
    inactive = balanced_bce(score, torch.zeros_like(target)) if not len(pos) else zero
    null_target = score.new_tensor(float(not target.any()))
    null_loss = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
    brier = (torch.sigmoid(score) - target.float()).square().mean()
    total = bce + 0.5 * pair + 0.5 * listwise + 0.5 * min_positive + inactive + null_loss + 0.05 * brier
    return total, {"total": float(total.detach()), "bce": float(bce.detach()),
                   "pairwise": float(pair.detach()), "listwise": float(listwise.detach()),
                   "minimum_positive": float(min_positive.detach()), "inactive": float(inactive.detach()),
                   "null": float(null_loss.detach()), "brier": float(brier.detach()),
                   "positive_count": int(pos.numel()), "hard_count": int(hard.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd {Path.cwd()}")
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    if sha256(MANIFEST) != EXPECTED_MANIFEST:
        raise AssertionError("manifest SHA mismatch")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    units = stratified(load_fit(), args.seed)
    if len(units) < 8:
        raise RuntimeError("insufficient fit units")
    out.mkdir(parents=True)
    device = torch.device("cuda:0")
    encoder = StreamingOpenAIClip(device, batch_size=32)
    detector_frozen = all(not p.requires_grad for p in encoder.model.parameters())
    model = L64RawPatchSet(hidden=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    trace, sampling = [], Counter(); finite = nonzero = 0
    start = time.time(); torch.cuda.reset_peak_memory_stats(device); model.train()
    for step in range(1, args.steps + 1):
        item = load_unit(units[(step - 1) % len(units)], encoder)
        patches = item["patches"].to(device); words = item["words"].to(device)
        mask = item["mask"].to(device); numeric_values = item["numeric"].to(device)
        target = item["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(patches, words, mask, numeric_values)
        loss, parts = loss_fn(output, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss step {step}")
        loss.backward()
        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        if not norms or not all(torch.isfinite(x) for x in norms) or not any(float(x) > 0 for x in norms):
            raise FloatingPointError(f"bad adapter gradient step {step}")
        optimizer.step(); finite += 1; nonzero += 1
        sampling[(item["unit"]["dataset"], item["unit"].get("category", "unknown"))] += 1
        trace.append({"step": step, "unit_key": item["unit"]["unit_key"], "dataset": item["unit"]["dataset"],
                      "category": item["unit"].get("category"), "candidate_count": int(target.numel()),
                      "loss": float(loss.detach()), "grad_norm": float(torch.stack(norms).norm()), **parts})
        del item, patches, words, mask, numeric_values, target, output, loss
    elapsed = time.time() - start
    ck = out / f"checkpoint_l64_raw_patch_step{args.steps}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": args.steps,
                "seed": args.seed, "format": "locatemot-l64-raw-patch-set-v1"}, ck)
    reload_model = L64RawPatchSet(hidden=128).cpu()
    reload_model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"], strict=True)
    reload_ok = True
    sampled = units[:args.steps]
    payload = {"format": "locatemot-l64-raw-patch-set-smoke-v1", "status": "complete",
               "stage": "fit-only-smoke", "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
               "seed": args.seed, "steps": args.steps, "finite_steps": finite,
               "nonzero_gradient_steps": nonzero, "checkpoint": str(ck), "checkpoint_sha256": sha256(ck),
               "checkpoint_reload": reload_ok, "fit_units_total": len(units),
               "sampled_units": int(min(args.steps, len(units))),
               "sampled_domains": sorted({u["dataset"] for u in sampled}),
               "sampled_categories": sorted({u.get("category") for u in sampled}),
               "sampling_counts": {f"{d}|{c}": int(n) for (d, c), n in sampling.items()},
               "candidate_sets_complete": True, "candidate_key_drift": 0, "candidate_truncation": False,
               "persistent_raw_dense_cache_written": False, "screening_gt_used": False,
               "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
               "detector_frozen": detector_frozen, "adapter_parameter_count": sum(p.numel() for p in model.parameters()),
               "image_patch_shape": ["N", 16, 768], "text_token_shape": [77, 512],
               "image_weight": "/home/lwr/.cache/clip/ViT-B-16.pt", "image_weight_sha256": sha256(Path("/home/lwr/.cache/clip/ViT-B-16.pt")),
               "crop_rule": "L19 box + 10% padding + clip-to-image; transient PNG only",
               "token_span_alignment": "UNALIGNED", "runtime": {"device": str(device), "precision": "FP32 adapter; frozen CLIP output", "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "elapsed_sec": elapsed, "steps_per_sec": args.steps / max(elapsed, 1e-9)},
               "loss_trace": trace}
    (out / f"metrics_l64_step{args.steps}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (out / "sampling_trace.json").write_text(json.dumps(payload["sampling_counts"], indent=2) + "\n")
    (out / "reload_audit.json").write_text(json.dumps({"checkpoint": str(ck), "reload_ok": reload_ok, "strict": True, "checkpoint_sha256": sha256(ck)}, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps({"seed": args.seed, "steps": args.steps, "hidden": 128, "heads": 4, "image_dim": 768, "text_dim": 512, "numeric_dim": 32, "fit_only": True, "candidate_set_complete": True, "same_class_hard_negative_metadata": "unavailable", "screening_gt_used": False}, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps({"project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST), "train_units": str(UNITS), "train_units_sha256": sha256(UNITS), "fit_only": True, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "persistent_raw_dense_cache_written": False}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "metrics": str(out / f"metrics_l64_step{args.steps}.json"), "checkpoint": str(ck), "finite_steps": finite, "nonzero_gradient_steps": nonzero}, indent=2), flush=True)


if __name__ == "__main__":
    main()
