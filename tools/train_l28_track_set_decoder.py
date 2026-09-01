#!/usr/bin/env python3
"""Train a small L28 track-set decoder on the train-only sequence cache."""
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
from locatemot.models.l28_track_set_decoder import L28TrackSetDecoder
from tools.train_l26_crossmodal_adapter import load_expressions

SPLIT = ROOT / "outputs/l16/data/protocol/split_manifest.json"
TEXT_ROOT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
CACHE_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_queries():
    split = json.loads(SPLIT.read_text())["kitti_v2"]
    train_videos = {str(x) for x in split["train"]}
    text_manifest = json.loads((TEXT_ROOT / "text_manifest.json").read_text())["expressions"]
    text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"])
                  for x in text_manifest}
    queries = []
    for row in load_expressions():
        video = str(row["video"])
        key = (video, str(row["expression"]))
        if video in train_videos and key in text_index:
            queries.append({"video": video, "expression": str(row["expression"]),
                            "target": {int(k): {str(x) for x in v}
                                       for k, v in row.get("label", {}).items()},
                            "text_index": text_index[key]})
    if len(queries) != 7757:
        raise AssertionError(f"expected 7757 train queries, found {len(queries)}")
    return queries


def state_at(cache, cutoff, history=8):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    feature = cache["obs_features"]
    n, dim = len(ptr) - 1, int(feature.shape[1])
    values = torch.zeros((n, history, dim), dtype=torch.float32)
    mask = torch.zeros((n, history), dtype=torch.bool)
    times = torch.zeros((n, history), dtype=torch.float32)
    gt_ids = cache["obs_gt_ids"]
    selected_gt = [[None] * history for _ in range(n)]
    selected_frames = [[-1] * history for _ in range(n)]
    for track_index in range(n):
        begin, end = int(ptr[track_index]), int(ptr[track_index + 1])
        eligible = np.flatnonzero(frames[begin:end] <= int(cutoff)) + begin
        if not len(eligible):
            continue
        chosen = eligible[-history:]
        offset = history - len(chosen)
        values[track_index, offset:] = feature[torch.as_tensor(chosen)].float()
        mask[track_index, offset:] = True
        denom = max(1.0, float(cutoff) + 1.0)
        times[track_index, offset:] = torch.as_tensor(
            np.asarray(frames[chosen], np.float32) / denom)
        for j, row in enumerate(chosen.tolist(), offset):
            selected_gt[track_index][j] = gt_ids[row]
            selected_frames[track_index][j] = int(frames[row])
    valid = mask.any(1)
    return values[valid], mask[valid], times[valid], [selected_gt[i] for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()], [selected_frames[i] for i in torch.nonzero(valid, as_tuple=False).flatten().tolist()]


def targets_for_state(selected_gt, selected_frames, queries):
    n, length = len(selected_gt), len(selected_gt[0])
    track_y = torch.zeros(n, dtype=torch.bool)
    member_y = torch.zeros((n, length), dtype=torch.bool)
    for i in range(n):
        for j in range(length):
            frame, gt = selected_frames[i][j], selected_gt[i][j]
            if frame < 0 or gt is None:
                continue
            member_y[i, j] = gt in queries["target"].get(frame, set())
        track_y[i] = bool(member_y[i].any())
    return track_y, member_y


def balanced_bce(logits, target):
    if not len(logits):
        return logits.new_zeros(())
    target = target.float()
    pos, neg = target.bool(), ~target.bool()
    parts = []
    if pos.any():
        parts.append(F.binary_cross_entropy_with_logits(logits[pos], target[pos]))
    if neg.any():
        parts.append(F.binary_cross_entropy_with_logits(logits[neg], target[neg]))
    return torch.stack(parts).mean() if parts else logits.new_zeros(())


def loss_for_batch(model, query, cache, text_hidden, text_mask, device, rng):
    frame_values = cache["obs_frame"].numpy()
    cutoff = int(rng.choice(np.unique(frame_values)))
    obs, obs_mask, obs_time, selected_gt, selected_frames = state_at(cache, cutoff)
    track_y, member_y = targets_for_state(selected_gt, selected_frames, query)
    obs, obs_mask, obs_time = obs.to(device), obs_mask.to(device), obs_time.to(device)
    track_y, member_y = track_y.to(device), member_y.to(device)
    qh = text_hidden[query["text_index"]].to(device)
    qm = text_mask[query["text_index"]].to(device)
    with torch.no_grad():
        prelim = model(obs, obs_mask, obs_time, qh, qm)["track_logits"]
    pos = torch.nonzero(track_y, as_tuple=False).flatten(); neg = torch.nonzero(~track_y, as_tuple=False).flatten()
    hard = neg[torch.argsort(prelim[neg], descending=True)[:min(24, len(neg))]] if len(neg) else neg
    out = model(obs, obs_mask, obs_time, qh, qm)
    logits = out["track_logits"]; z = logits.new_zeros(())
    track_bce = balanced_bce(logits, track_y)
    pair = F.softplus(.5 + logits[hard][None, :] - logits[pos][:, None]).mean() if len(pos) and len(hard) else z
    listwise = torch.logsumexp(logits, 0) - torch.logsumexp(logits[pos], 0) if len(pos) else z
    mlogits = out["membership_logits"][obs_mask]
    my = member_y[obs_mask]
    membership = balanced_bce(mlogits, my)
    null_target = torch.tensor(float(not track_y.any()), device=device)
    null_loss = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
    cont_target = (obs_mask.sum(1) > 1).float()
    cont_loss = F.binary_cross_entropy_with_logits(out["continuation_logits"], cont_target)
    temporal = z
    if obs_mask.shape[1] > 1:
        valid = obs_mask[:, 1:] & obs_mask[:, :-1]
        if valid.any():
            temporal = (out["membership_logits"][:, 1:] -
                        out["membership_logits"][:, :-1])[valid].abs().mean()
    total = track_bce + pair + .5 * listwise + .5 * membership + .3 * null_loss + .2 * cont_loss + .05 * temporal
    parts = {
        "total": float(total.detach()), "track_bce": float(track_bce.detach()),
        "pairwise": float(pair.detach()), "listwise": float(listwise.detach()),
        "membership_bce": float(membership.detach()), "null": float(null_loss.detach()),
        "continuation": float(cont_loss.detach()), "temporal": float(temporal.detach()),
        "positive_tracks": int(track_y.sum()), "negative_tracks": int((~track_y).sum()),
        "hard_negatives": int(len(hard)), "cutoff": cutoff,
    }
    return total, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    queries = build_queries()
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden = text["token_hidden"]  # keep frozen text storage in fp16
    text_mask = text["attention_mask"].bool()
    del text
    cache_by_video = {}
    for video in sorted({q["video"] for q in queries}):
        cache_by_video[video] = torch.load(CACHE_ROOT / f"{video}.pt",
                                           map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = L28TrackSetDecoder().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device,
                                          weights_only=False)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    rows, gradients = [], []
    start = time.time()
    trainable = [q for q in queries if q["video"] in cache_by_video]
    model.train()
    for step in range(args.steps):
        batch = [trainable[int(rng.integers(len(trainable)))] for _ in range(2)]
        losses, parts = [], []
        for query in batch:
            value, part = loss_for_batch(model, query, cache_by_video[query["video"]],
                                         text_hidden, text_mask, device, rng)
            losses.append(value); parts.append(part)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
        optimizer.step()
        rows.append({key: float(np.mean([part[key] for part in parts]))
                     for key in ("total", "track_bce", "pairwise", "listwise",
                                 "membership_bce", "null", "continuation", "temporal")})
    checkpoint = out / f"checkpoint_track_set_step{args.steps}.pt"
    payload = {
        "format": "locatemot-l28-track-set-decoder-v1",
        "stage": "train-only-track-set-smoke",
        "cache_root": str(CACHE_ROOT.resolve()),
        "cache_manifest_sha256": sha(CACHE_ROOT / "manifest.json"),
        "text_tokens": str((TEXT_ROOT / "text_tokens.pt").resolve()),
        "split_manifest_sha256": sha(SPLIT), "seed": args.seed, "steps": args.steps,
        "device": str(device), "train_query_count": len(trainable),
        "screening_gt_used_for_fit": False,
        "excluded_semantic_shortcuts": ["pool_id", "source_id", "group_id", "state_key"],
        "motion_language_decomposition": "not claimed; no verified motion token mask",
        "loss": {key: float(np.mean([row[key] for row in rows]))
                 for key in rows[0] if key not in ("total",)},
        "gradient_norm": {"mean": float(np.mean(gradients)),
                          "max": float(np.max(gradients)),
                          "nonzero_steps": int(np.count_nonzero(np.asarray(gradients) > 0))},
        "elapsed_sec": time.time() - start,
    }
    torch.save({"model": model.state_dict(), "config": payload}, checkpoint)
    reload_model = L28TrackSetDecoder().to(device)
    reload_model.load_state_dict(torch.load(checkpoint, map_location=device,
                                             weights_only=False)["model"])
    payload["checkpoint"] = str(checkpoint.resolve())
    payload["checkpoint_reload"] = True
    (out / f"metrics_track_set_step{args.steps}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
