#!/usr/bin/env python3
"""Train the bounded Stage L32 agreement gate on train-only L28 cache."""
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
from locatemot.models.l32_agreement_gate import L32AgreementGate
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.eval_l31_bounded_identity_fusion import feature_np
from tools.train_l28_track_set_decoder import build_queries, state_at, targets_for_state

CACHE_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
TEXT_ROOT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
MEMBERSHIP_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
ASSOC_CHECKPOINT = ROOT / "outputs/l30/train/fragment_probe_step500/checkpoint_fragment_probe_step500.pt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_association(cache, weight, bias):
    x = cache["obs_features"].float().numpy()
    frame = cache["obs_frame"].numpy()
    ptr = cache["track_ptr"].numpy()
    assoc = np.zeros(len(frame), np.float32)
    for track in range(len(ptr) - 1):
        begin, end = int(ptr[track]), int(ptr[track + 1])
        rows = sorted(range(begin, end), key=lambda r: (int(frame[r]), r))
        previous, last_frame = -1, -1
        for row in rows:
            if previous >= 0 and int(frame[row]) > last_frame:
                pair = feature_np(x[previous:previous + 1], x[row:row + 1])[0]
                assoc[row] = float(pair @ weight + bias)
            if int(frame[row]) > last_frame:
                previous, last_frame = row, int(frame[row])
    cache["assoc_by_row"] = torch.as_tensor(assoc)
    return cache


def latest_meta(cache, cutoff):
    ptr, frames = cache["track_ptr"].numpy(), cache["obs_frame"].numpy()
    assoc = cache["assoc_by_row"].numpy()
    result = []
    for track in range(len(ptr) - 1):
        begin, end = int(ptr[track]), int(ptr[track + 1])
        eligible = np.flatnonzero(frames[begin:end] <= cutoff) + begin
        if not len(eligible): continue
        row = int(eligible[-1]); feat = cache["obs_features"][row].float().numpy()
        result.append((row, float(assoc[row]), float(cutoff - int(frames[row])),
                       float(np.linalg.norm(feat[1415:1423])),
                       float(np.linalg.norm(feat[1423:1431]))))
    return result


def gate_features(cache, cutoff, raw):
    meta = latest_meta(cache, cutoff)
    if len(meta) != len(raw): raise AssertionError("state/cache track alignment mismatch")
    assoc = np.asarray([x[1] for x in meta], np.float32)
    recency = np.asarray([x[2] for x in meta], np.float32)
    motion = np.asarray([x[3] for x in meta], np.float32)
    lifecycle = np.asarray([x[4] for x in meta], np.float32)
    visual = np.tanh(assoc / 3.0).astype(np.float32)
    return torch.as_tensor(np.stack((raw, assoc, recency / 8.0, motion,
                                     lifecycle, visual), axis=1), dtype=torch.float32)


def balanced_bce(logits, target):
    target = target.bool(); pieces = []
    if target.any(): pieces.append(F.binary_cross_entropy_with_logits(logits[target], target[target].float()))
    if (~target).any(): pieces.append(F.binary_cross_entropy_with_logits(logits[~target], target[~target].float()))
    return torch.stack(pieces).mean() if pieces else logits.new_zeros(())


def loss_for_group(gate, features, membership, target, stale):
    out = gate(features, membership); final, reject = out["final"], out["reject_logit"]
    zero = final.new_zeros(()); pos = torch.nonzero(target, as_tuple=False).flatten()
    neg = torch.nonzero(~target, as_tuple=False).flatten()
    hard = neg[torch.argsort(final.detach()[neg], descending=True)[:min(12, len(neg))]] if len(neg) else neg
    pair = F.softplus(0.5 + final[hard][None, :] - final[pos][:, None]).mean() if len(pos) and len(hard) else zero
    listwise = torch.logsumexp(final, 0) - torch.logsumexp(final[pos], 0) if len(pos) else zero
    bce = balanced_bce(final, target)
    reject_loss = F.binary_cross_entropy_with_logits(reject, target.float())
    stale_loss = F.binary_cross_entropy_with_logits(reject, (~stale).float()) if stale.any() else zero
    total = bce + pair + 0.5 * listwise + 0.5 * reject_loss + 0.2 * stale_loss
    return total, {"total": float(total.detach()), "membership_bce": float(bce.detach()),
                   "pairwise": float(pair.detach()), "multi_positive_set": float(listwise.detach()),
                   "reject": float(reject_loss.detach()), "stale_suppression": float(stale_loss.detach()),
                   "positive": int(target.sum()), "negative": int((~target).sum()),
                   "hard_negative": int(len(hard)), "multi_positive": int(target.sum() > 1),
                   "null": int(not target.any())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=100); parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out; out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    queries = build_queries()
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden, text_mask = text["token_hidden"], text["attention_mask"].bool(); del text
    assoc_state = torch.load(ASSOC_CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    weight, bias = assoc_state["linear.weight"].numpy().reshape(-1), float(assoc_state["linear.bias"].item())
    cache_by_video = {}
    for video in sorted({q["video"] for q in queries}):
        cache = torch.load(CACHE_ROOT / f"{video}.pt", map_location="cpu", weights_only=False)
        cache_by_video[video] = add_association(cache, weight, bias)
    device = torch.device(args.device)
    membership = L29FrameMembershipSetDecoder().to(device)
    membership.load_state_dict(torch.load(MEMBERSHIP_CHECKPOINT, map_location=device, weights_only=False)["model"]); membership.eval()
    for parameter in membership.parameters(): parameter.requires_grad_(False)
    gate = L32AgreementGate().to(device); optimizer = torch.optim.AdamW(gate.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed); trace, gradients, counters = [], [], Counter(); started = time.time(); gate.train()
    for step in range(args.steps):
        batch = [queries[int(rng.integers(len(queries)))] for _ in range(2)]; losses, parts = [], []
        for query in batch:
            cache = cache_by_video[query["video"]]; cutoff = int(rng.choice(np.unique(cache["obs_frame"].numpy())))
            obs, obs_mask, obs_time, selected_gt, selected_frames = state_at(cache, cutoff)
            track_y, member_y = targets_for_state(selected_gt, selected_frames, query)
            latest = obs_mask.long().sum(1).clamp_min(1) - 1
            current_y = member_y[torch.arange(len(member_y)), latest]
            with torch.inference_mode():
                base = membership(obs.to(device), obs_mask.to(device), obs_time.to(device),
                                  text_hidden[int(query["text_index"])].to(device), text_mask[int(query["text_index"])].to(device))
                raw = base["current_membership_logits"].float().cpu()
            features = gate_features(cache, cutoff, raw.numpy())
            value, part = loss_for_group(gate, features.to(device), raw.to(device), current_y.to(device), (track_y & ~current_y).to(device))
            losses.append(value); parts.append(part)
        loss = torch.stack(losses).mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
        gradients.append(float(torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0))); optimizer.step()
        keys = ("total", "membership_bce", "pairwise", "multi_positive_set", "reject", "stale_suppression")
        trace.append({"step": step + 1, **{key: float(np.mean([part[key] for part in parts])) for key in keys}})
        for part in parts:
            for key in ("positive", "negative", "hard_negative", "multi_positive", "null"): counters[key] += part[key]
    checkpoint = out / f"checkpoint_agreement_gate_step{args.steps}.pt"
    config = {"format": "locatemot-l32-agreement-gate-v1", "steps": args.steps, "seed": args.seed, "device": str(device),
              "train_video_count": 15, "train_query_count": len(queries), "cache_root": str(CACHE_ROOT.resolve()),
              "cache_manifest_sha256": sha(CACHE_ROOT / "manifest.json"), "membership_checkpoint": str(MEMBERSHIP_CHECKPOINT.resolve()),
              "association_checkpoint": str(ASSOC_CHECKPOINT.resolve()), "screening_gt_used_for_fit": False,
              "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"],
              "input_schema": ["membership", "association", "recency", "motion_consistency", "lifecycle", "visual_agreement"],
              "residual_bound": [-0.25, 0.25], "loss_group": "video/query/frame", "counters": dict(counters),
              "loss_mean": {key: float(np.mean([row[key] for row in trace])) for key in trace[0] if key != "step"},
              "gradient_norm": {"mean": float(np.mean(gradients)), "max": float(np.max(gradients)), "nonzero_steps": int(np.count_nonzero(np.asarray(gradients) > 0))},
              "elapsed_sec": time.time() - started}
    torch.save({"model": gate.state_dict(), "config": config}, checkpoint)
    reload_gate = L32AgreementGate().to(device); reload_gate.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    config.update({"checkpoint": str(checkpoint.resolve()), "checkpoint_reload": True})
    (out / f"metrics_agreement_gate_step{args.steps}.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n"); print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__": main()
