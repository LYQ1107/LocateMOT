#!/usr/bin/env python3
"""Train the L29 current-frame membership/set decoder on train-only cache."""
from __future__ import annotations

import argparse
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
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.train_l26_crossmodal_adapter import load_expressions
from tools.train_l28_track_set_decoder import CACHE_ROOT, SPLIT, TEXT_ROOT, state_at


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_queries():
    split = json.loads(SPLIT.read_text())["kitti_v2"]
    train_videos = {str(x) for x in split["train"]}
    text_manifest = json.loads((TEXT_ROOT / "text_manifest.json").read_text())["expressions"]
    text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"])
                  for x in text_manifest}
    result = []
    for row in load_expressions():
        video = str(row["video"]); key = (video, str(row["expression"]))
        if video in train_videos and key in text_index:
            result.append({"video": video, "expression": str(row["expression"]),
                           "target": {int(k): {str(x) for x in v}
                                      for k, v in row.get("label", {}).items()},
                           "text_index": text_index[key]})
    if len(result) != 7757:
        raise AssertionError(f"expected 7757 train queries, found {len(result)}")
    return result


def balanced_bce(logits, target):
    target = target.float(); pos = target.bool(); neg = ~pos
    terms = []
    if pos.any():
        terms.append(F.binary_cross_entropy_with_logits(logits[pos], target[pos]))
    if neg.any():
        terms.append(F.binary_cross_entropy_with_logits(logits[neg], target[neg]))
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def targets_for_state(selected_gt, selected_frames, query, cutoff):
    n, length = len(selected_gt), len(selected_gt[0])
    current = torch.zeros(n, dtype=torch.bool)
    stale = torch.zeros(n, dtype=torch.bool)
    member = torch.zeros((n, length), dtype=torch.bool)
    for i in range(n):
        latest = -1; latest_gt = None
        for j in range(length):
            frame, gt = selected_frames[i][j], selected_gt[i][j]
            if frame < 0 or gt is None:
                continue
            member[i, j] = gt in query["target"].get(frame, set())
            if frame >= latest:
                latest, latest_gt = frame, gt
        if latest >= 0 and latest < cutoff:
            stale[i] = True
        current[i] = latest == cutoff and latest_gt in query["target"].get(cutoff, set())
    return current, stale, member


def loss_for_query(model, query, cache, hidden, text_mask, device, rng):
    cutoff = int(rng.choice(np.unique(cache["obs_frame"].numpy())))
    obs, obs_mask, obs_time, selected_gt, selected_frames = state_at(cache, cutoff)
    current_y, stale_y, member_y = targets_for_state(selected_gt, selected_frames,
                                                      query, cutoff)
    obs, obs_mask, obs_time = obs.to(device), obs_mask.to(device), obs_time.to(device)
    current_y, stale_y, member_y = current_y.to(device), stale_y.to(device), member_y.to(device)
    qh = hidden[query["text_index"]].to(device)
    qm = text_mask[query["text_index"]].to(device)
    out = model(obs, obs_mask, obs_time, qh, qm)
    current = out["current_membership_logits"]
    valid = ~stale_y
    hard_pool = torch.nonzero((~current_y) & valid, as_tuple=False).flatten()
    with torch.no_grad():
        prelim = current.detach()
        hard = hard_pool[torch.argsort(prelim[hard_pool], descending=True)[:min(24, len(hard_pool))]]
    zero = current.new_zeros(())
    current_bce = balanced_bce(current[valid], current_y[valid]) if valid.any() else zero
    pos = torch.nonzero(current_y, as_tuple=False).flatten()
    pairwise = (F.softplus(.5 + current[hard][None, :] - current[pos][:, None]).mean()
                if len(pos) and len(hard) else zero)
    listwise = (torch.logsumexp(current[valid], 0) - torch.logsumexp(current[pos], 0)
                if len(pos) and valid.any() else zero)
    stale_loss = balanced_bce(out["stale_logits"], stale_y)
    member_loss = balanced_bce(out["membership_logits"][obs_mask], member_y[obs_mask])
    null_target = torch.tensor(float(not current_y.any()), device=device)
    null_loss = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
    cont_target = (obs_mask.sum(1) > 1).float()
    cont_loss = F.binary_cross_entropy_with_logits(out["continuation_logits"], cont_target)
    temporal = zero
    if obs_mask.shape[1] > 1:
        overlap = obs_mask[:, 1:] & obs_mask[:, :-1]
        if overlap.any():
            temporal = (out["membership_logits"][:, 1:] -
                        out["membership_logits"][:, :-1])[overlap].abs().mean()
    total = (current_bce + pairwise + .5 * listwise + .7 * stale_loss +
             .5 * member_loss + .3 * null_loss + .2 * cont_loss + .05 * temporal)
    parts = {"total": float(total.detach()), "current_bce": float(current_bce.detach()),
             "pairwise": float(pairwise.detach()), "listwise": float(listwise.detach()),
             "stale": float(stale_loss.detach()), "membership_bce": float(member_loss.detach()),
             "null": float(null_loss.detach()), "continuation": float(cont_loss.detach()),
             "temporal": float(temporal.detach()), "positive_current": int(current_y.sum()),
             "negative_current": int((~current_y).sum()), "stale_tracks": int(stale_y.sum()),
             "hard_negatives": int(len(hard))}
    return total, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True); ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    queries = build_queries()
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden, text_mask = text["token_hidden"], text["attention_mask"].bool()
    cache_by_video = {v: torch.load(CACHE_ROOT / f"{v}.pt", map_location="cpu", weights_only=False)
                      for v in sorted({q["video"] for q in queries})}
    device = torch.device(args.device); model = L29FrameMembershipSetDecoder().to(device)
    rng = np.random.default_rng(args.seed); trainable = queries
    model.train(); rows = []; grads = []; start = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    for _ in range(args.steps):
        batch = [trainable[int(rng.integers(len(trainable)))] for _ in range(2)]
        values = []; parts = []
        for query in batch:
            value, part = loss_for_query(model, query, cache_by_video[query["video"]],
                                         hidden, text_mask, device, rng)
            values.append(value); parts.append(part)
        loss = torch.stack(values).mean(); model.zero_grad(set_to_none=True)
        loss.backward(); grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
        optimizer.step()
        rows.append({k: float(np.mean([p[k] for p in parts]))
                     for k in ("total", "current_bce", "pairwise", "listwise", "stale",
                               "membership_bce", "null", "continuation", "temporal")})
    checkpoint = out / f"checkpoint_frame_membership_step{args.steps}.pt"
    payload = {"format": "locatemot-l29-frame-membership-decoder-v1", "stage": "train-only",
               "cache_root": str(CACHE_ROOT.resolve()), "cache_manifest_sha256": sha(CACHE_ROOT / "manifest.json"),
               "split_manifest_sha256": sha(SPLIT), "text_tokens": str((TEXT_ROOT / "text_tokens.pt").resolve()),
               "seed": args.seed, "steps": args.steps, "device": str(device), "train_query_count": len(trainable),
               "screening_gt_used_for_fit": False, "emission_contract": "current_observation_membership",
               "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"],
               "motion_language_decomposition": "not claimed; no verified motion token mask",
               "loss": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "total"},
               "gradient_norm": {"mean": float(np.mean(grads)), "max": float(np.max(grads)),
                                 "nonzero_steps": int(np.count_nonzero(np.asarray(grads) > 0))},
               "elapsed_sec": time.time() - start}
    torch.save({"model": model.state_dict(), "config": payload}, checkpoint)
    reload_model = L29FrameMembershipSetDecoder().to(device)
    reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    payload["checkpoint"] = str(checkpoint.resolve()); payload["checkpoint_reload"] = True
    (out / f"metrics_frame_membership_step{args.steps}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
