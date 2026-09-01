#!/usr/bin/env python3
"""Train the train-only Stage L33 query-conditioned hard-negative probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l33_query_hard_negative_probe import L33QueryHardNegativeProbe
from tools.eval_l31_bounded_identity_fusion import feature_np
from tools.train_l28_track_set_decoder import build_queries, state_at, targets_for_state

CACHE_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
TEXT_ROOT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
MEMBERSHIP_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_rows(cache, cutoff):
    ptr, frames = cache["track_ptr"].numpy(), cache["obs_frame"].numpy()
    result = []
    for track in range(len(ptr) - 1):
        begin, end = int(ptr[track]), int(ptr[track + 1])
        end_pos = int(np.searchsorted(frames[begin:end], int(cutoff), side="right"))
        if end_pos:
            start = max(0, end_pos - 1)
            result.append((begin + end_pos - 1, np.arange(begin + start, begin + end_pos - 1, dtype=np.int64)))
    return result


def candidate_visual_features(cache, cutoff):
    """Build query-independent [frozen pair, geometry/motion/lifecycle/objectness]."""
    x = cache["obs_features"].float().numpy()
    rows = latest_rows(cache, cutoff)
    values = []
    for row, earlier in rows:
        previous = int(earlier[-1]) if len(earlier) else -1
        pair = feature_np(x[previous:previous + 1], x[row:row + 1])[0] if previous >= 0 else np.zeros(26, np.float32)
        numeric = x[row, 1408:1432].astype(np.float32)
        values.append(np.concatenate((pair, numeric)))
    result = np.asarray(values, np.float32)
    if result.ndim != 2 or result.shape[1] != 50:
        raise AssertionError(f"unexpected L33 visual feature shape {result.shape}")
    return result


def candidate_features(cache, cutoff, membership_logits):
    """Build [current membership, frozen pair, geometry/motion/lifecycle/objectness]."""
    current = np.asarray(membership_logits, np.float32).reshape(-1, 1)
    visual = candidate_visual_features(cache, cutoff)
    if len(current) != len(visual):
        raise AssertionError("membership/visual candidate alignment mismatch")
    return np.concatenate((current, visual), axis=1)


def balanced_bce(logits, target):
    target = target.bool(); parts = []
    if target.any():
        parts.append(F.binary_cross_entropy_with_logits(logits[target], target[target].float()))
    if (~target).any():
        parts.append(F.binary_cross_entropy_with_logits(logits[~target], target[~target].float()))
    return torch.stack(parts).mean() if parts else logits.new_zeros(())


def loss_for_group(model, features, qh, qm, target, stale, counters):
    with torch.no_grad():
        prelim = model(features, qh, qm)["relevance_logits"]
    pos = torch.nonzero(target, as_tuple=False).flatten()
    neg = torch.nonzero(~target, as_tuple=False).flatten()
    hard = neg[torch.argsort(prelim[neg], descending=True)[:min(12, len(neg))]] if len(neg) else neg
    out = model(features, qh, qm); logits = out["relevance_logits"]; zero = logits.new_zeros(())
    membership = balanced_bce(logits, target)
    pair = F.softplus(0.5 + logits[hard][None, :] - logits[pos][:, None]).mean() if len(pos) and len(hard) else zero
    set_loss = torch.logsumexp(logits, 0) - torch.logsumexp(logits[pos], 0) if len(pos) else zero
    null_target = logits.new_tensor(float(not target.any()))
    null_loss = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
    inactive_loss = (F.binary_cross_entropy_with_logits(logits[stale], torch.zeros_like(logits[stale]))
                     if stale.any() else zero)
    total = membership + pair + 0.5 * set_loss + 0.3 * null_loss + 0.2 * inactive_loss
    counters.update(positive=int(target.sum()), negative=int((~target).sum()),
                    hard_negative=int(len(hard)), multi_positive=int(len(pos) > 1),
                    null=int(not target.any()), inactive=int(stale.sum()))
    return total, {"total": float(total.detach()), "current_membership_bce": float(membership.detach()),
                   "pairwise_hard": float(pair.detach()), "multi_positive_set": float(set_loss.detach()),
                   "null": float(null_loss.detach()), "inactive": float(inactive_loss.detach())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-root", required=True)
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    queries = build_queries()
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden, text_mask = text["token_hidden"], text["attention_mask"].bool(); del text
    cache_by_video = {video: torch.load(CACHE_ROOT / f"{video}.pt", map_location="cpu", weights_only=False)
                      for video in sorted({q["video"] for q in queries})}
    device = torch.device(args.device)
    membership = L29FrameMembershipSetDecoder().to(device)
    membership.load_state_dict(torch.load(MEMBERSHIP_CHECKPOINT, map_location=device, weights_only=False)["model"])
    membership.eval()
    for parameter in membership.parameters(): parameter.requires_grad_(False)
    model = L33QueryHardNegativeProbe(hidden=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed); trace = []; gradient = []; counters = Counter(); started = time.time(); model.train()
    for step in range(args.steps):
        losses = []; parts = []
        for _ in range(2):
            query = queries[int(rng.integers(len(queries)))]; cache = cache_by_video[query["video"]]
            cutoff = int(rng.choice(np.unique(cache["obs_frame"].numpy())))
            obs, obs_mask, obs_time, selected_gt, selected_frames = state_at(cache, cutoff)
            track_y, member_y = targets_for_state(selected_gt, selected_frames, query)
            latest = obs_mask.long().sum(1).clamp_min(1) - 1
            current_y = member_y[torch.arange(len(member_y)), latest]
            with torch.inference_mode():
                base = membership(obs.to(device), obs_mask.to(device), obs_time.to(device),
                                  text_hidden[int(query["text_index"])].to(device),
                                  text_mask[int(query["text_index"])].to(device))
            features = candidate_features(cache, cutoff, base["current_membership_logits"].float().cpu().numpy())
            value, part = loss_for_group(model, torch.as_tensor(features, device=device),
                                         text_hidden[int(query["text_index"])].to(device),
                                         text_mask[int(query["text_index"])].to(device),
                                         current_y.to(device),
                                         (track_y & ~current_y).to(device), counters)
            losses.append(value); parts.append(part)
        loss = torch.stack(losses).mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step(); gradient.append(norm)
        trace.append({"step": step + 1, **{k: float(np.mean([p[k] for p in parts])) for k in parts[0]}})
    checkpoint = out / f"checkpoint_query_hard_negative_probe_step{args.steps}.pt"
    config = {"format": "locatemot-l33-query-hard-negative-probe-v1", "steps": args.steps, "seed": args.seed,
              "train_video_count": 15, "train_query_count": len(queries), "cache_root": str(CACHE_ROOT.resolve()),
              "cache_manifest_sha256": sha(CACHE_ROOT / "manifest.json"), "manifest": str(MANIFEST.resolve()),
              "manifest_sha256": sha(MANIFEST), "membership_checkpoint": str(MEMBERSHIP_CHECKPOINT.resolve()),
              "screening_gt_used_for_fit": False, "input_dim": 51, "hidden": 128,
              "input_schema": ["current_membership", "frozen_fragment_pair_26", "geometry_7", "motion_8", "lifecycle_8", "objectness"],
              "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key", "l30_association_score"],
              "query_schema": "word-level text_tokens.pt sequence; no verified static/motion alignment mask",
              "loss_group": "video/query/frame", "counters": dict(counters),
              "loss_mean": {k: float(np.mean([r[k] for r in trace])) for k in trace[0] if k != "step"},
              "gradient_norm": {"mean": float(np.mean(gradient)), "max": float(np.max(gradient)),
                                "nonzero_steps": int(np.count_nonzero(np.asarray(gradient) > 0))},
              "elapsed_sec": time.time() - started}
    torch.save({"model": model.state_dict(), "config": config}, checkpoint)
    reload_model = L33QueryHardNegativeProbe(hidden=128).to(device)
    reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    config.update({"checkpoint": str(checkpoint.resolve()), "checkpoint_reload": True})
    (out / f"metrics_query_hard_negative_probe_step{args.steps}.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
