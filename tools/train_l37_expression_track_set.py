#!/usr/bin/env python3
"""Train the small expression-level persistent-track set decoder for L37."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l37_expression_track_set import L37ExpressionTrackSet
from tools.train_l28_track_set_decoder import build_queries, state_at

SPLIT = ROOT / "outputs/l16/data/protocol/split_manifest.json"
TEXT_ROOT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
CACHE_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
AUDIT = ROOT / "outputs/l37/audit/expression_supervision_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not logits.numel():
        return logits.new_zeros(())
    target = target.float()
    pos = target.bool()
    pieces = []
    if pos.any():
        pieces.append(F.binary_cross_entropy_with_logits(logits[pos], target[pos]))
    if (~pos).any():
        pieces.append(F.binary_cross_entropy_with_logits(logits[~pos], target[~pos]))
    return torch.stack(pieces).mean() if pieces else logits.new_zeros(())


def targets_for_state(selected_gt, selected_frames, query, cutoff):
    n = len(selected_gt)
    length = len(selected_gt[0]) if n else 0
    target = query["target"]
    sequence_y = torch.zeros(n, dtype=torch.bool)
    current_mask = torch.zeros(n, dtype=torch.bool)
    current_y = torch.zeros(n, dtype=torch.bool)
    member_y = torch.zeros((n, length), dtype=torch.bool)
    for i in range(n):
        for j in range(length):
            frame, gt = selected_frames[i][j], selected_gt[i][j]
            if frame < 0 or gt is None:
                continue
            positive = gt in target.get(frame, set())
            member_y[i, j] = positive
            sequence_y[i] |= positive
            if frame == cutoff:
                current_mask[i] = True
                current_y[i] = positive
    return sequence_y, current_mask, current_y, member_y


def sample_loss(model, query, cache, text_hidden, text_mask, device, rng, use_amp):
    cutoff = int(rng.choice(cache["obs_frame"].numpy()))
    obs, obs_mask, obs_time, selected_gt, selected_frames = state_at(cache, cutoff, history=8)
    sequence_y, current_mask, current_y, member_y = targets_for_state(
        selected_gt, selected_frames, query, cutoff)
    obs = obs.to(device); obs_mask = obs_mask.to(device); obs_time = obs_time.to(device)
    sequence_y = sequence_y.to(device); current_mask = current_mask.to(device)
    current_y = current_y.to(device); member_y = member_y.to(device)
    qh = text_hidden[query["text_index"]].to(device)
    qm = text_mask[query["text_index"]].to(device)
    amp_context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if use_amp else contextlib.nullcontext())
    with torch.no_grad(), amp_context:
        prelim = model(obs, obs_mask, obs_time, qh, qm)["current_membership_logits"]
    active = torch.nonzero(current_mask, as_tuple=False).flatten()
    current_scores = prelim[active]
    pos = active[current_y[active]] if len(active) else active
    neg = active[~current_y[active]] if len(active) else active
    hard = neg[torch.argsort(current_scores[~current_y[active]], descending=True)[:min(24, len(neg))]] if len(neg) else neg

    amp_context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if use_amp else contextlib.nullcontext())
    with amp_context:
        out = model(obs, obs_mask, obs_time, qh, qm)
    z = out["current_membership_logits"].new_zeros(())
    current_logits = out["current_membership_logits"][active]
    current_target = current_y[active]
    membership = balanced_bce(current_logits, current_target)
    sequence = balanced_bce(out["sequence_logits"], sequence_y)
    stale_target = ~current_mask
    stale = balanced_bce(out["stale_logits"], stale_target)
    if len(pos) and len(hard):
        pair = F.softplus(0.5 + out["current_membership_logits"][hard][None, :] -
                          out["current_membership_logits"][pos][:, None]).mean()
    else:
        pair = z
    if len(pos):
        # This reduction keeps every positive in the objective while allowing
        # a multi-positive frame to win as a set.
        positive_log = out["current_membership_logits"][pos]
        all_log = out["current_membership_logits"][active]
        set_loss = (torch.logsumexp(all_log, 0) - torch.logsumexp(positive_log, 0)
                    - F.logsigmoid(positive_log).mean())
    else:
        set_loss = z
    history_mask = obs_mask
    history = balanced_bce(out["membership_logits"][history_mask], member_y[history_mask])
    null_target = torch.tensor(float(not current_y[active].any()), device=device)
    null_loss = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
    cont_target = (sequence_y & ~current_y).float()
    continuation = F.binary_cross_entropy_with_logits(out["continuation_logits"], cont_target)
    total = membership + sequence + pair + 0.5 * set_loss + 0.5 * history + \
        0.35 * stale + 0.3 * null_loss + 0.15 * continuation
    parts = {
        "total": float(total.detach()), "current_membership_bce": float(membership.detach()),
        "sequence_bce": float(sequence.detach()), "pairwise": float(pair.detach()),
        "multi_positive_set": float(set_loss.detach()), "history_membership_bce": float(history.detach()),
        "stale_suppression": float(stale.detach()), "null": float(null_loss.detach()),
        "continuation": float(continuation.detach()), "current_candidates": int(len(active)),
        "current_positives": int(current_y[active].sum()) if len(active) else 0,
        "sequence_positives": int(sequence_y.sum()), "stale_tracks": int(stale_target.sum()),
        "hard_negatives": int(len(hard)), "multi_positive_frame": int(current_y[active].sum() > 1) if len(active) else 0,
        "null_frame": int(not current_y[active].any()) if len(active) else 1, "cutoff": cutoff,
    }
    return total, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    if not AUDIT.exists():
        raise FileNotFoundError(AUDIT)
    audit = json.loads(AUDIT.read_text())
    if not audit["decision"]["expression_level_supervision_available"]:
        raise RuntimeError("expression-level supervision audit did not pass")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    queries = build_queries()
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden = text["token_hidden"].float()
    text_mask = text["attention_mask"].bool()
    del text
    cache_by_video = {}
    for video in sorted({q["video"] for q in queries}):
        cache_by_video[video] = torch.load(CACHE_ROOT / f"{video}.pt", map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = L37ExpressionTrackSet(hidden=128, history=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    trace, grad_norms = [], []
    start = time.time()
    model.train()
    autocast_enabled = device.type == "cuda"
    for step in range(args.steps):
        query = queries[int(rng.integers(len(queries)))]
        loss, parts = sample_loss(model, query, cache_by_video[query["video"]],
                                  text_hidden, text_mask, device, rng,
                                  use_amp=autocast_enabled)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not np.isfinite(grad):
            raise FloatingPointError(f"nonfinite gradient at step {step + 1}")
        optimizer.step()
        grad_norms.append(grad)
        parts["step"] = step + 1
        trace.append(parts)
    checkpoint = out / f"checkpoint_l37_expression_track_set_step{args.steps}.pt"
    payload = {
        "format": "locatemot-l37-expression-track-set-v1",
        "stage": "expression-level-rmot-train-only-smoke",
        "seed": args.seed, "steps": args.steps, "device": str(device),
        "hidden": 128, "history_length": 8, "train_query_count": len(queries),
        "train_video_count": len(cache_by_video), "cache_root": str(CACHE_ROOT.resolve()),
        "cache_manifest_sha256": sha(CACHE_ROOT / "manifest.json"),
        "text_tokens": str((TEXT_ROOT / "text_tokens.pt").resolve()),
        "split_manifest_sha256": sha(SPLIT),
        "expression_supervision_audit": str(AUDIT.resolve()),
        "expression_supervision_audit_sha256": sha(AUDIT),
        "screening_gt_used_for_fit": False,
        "fast_manifest_used_for_training": False,
        "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"],
        "token_level_alignment_verified": False,
        "motion_language_decomposition": "not claimed; no verified motion-language mask",
        "loss_mean": {k: float(np.mean([x[k] for x in trace])) for k in trace[0] if k in ("total", "current_membership_bce", "sequence_bce", "pairwise", "multi_positive_set", "history_membership_bce", "stale_suppression", "null", "continuation")},
        "sampling_totals": {k: int(sum(x[k] for x in trace)) for k in ("current_candidates", "current_positives", "sequence_positives", "stale_tracks", "hard_negatives", "multi_positive_frame", "null_frame")},
        "gradient_norm": {"mean": float(np.mean(grad_norms)), "max": float(np.max(grad_norms)), "nonzero_steps": int(np.count_nonzero(np.asarray(grad_norms) > 0)), "steps": len(grad_norms)},
        "elapsed_sec": time.time() - start,
    }
    torch.save({"model": model.state_dict(), "config": payload}, checkpoint)
    reloaded = L37ExpressionTrackSet(hidden=128, history=8).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    reloaded.eval()
    payload["checkpoint"] = str(checkpoint.resolve())
    payload["checkpoint_reload"] = True
    (out / f"metrics_l37_smoke{args.steps}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
