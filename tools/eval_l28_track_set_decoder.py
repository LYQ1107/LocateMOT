#!/usr/bin/env python3
"""Candidate-level evaluation of a trained L28 track-set decoder."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l28_track_set_decoder import L28TrackSetDecoder
from tools.audit_l28_identity_bank import BANK_ROOT, load_labels
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5 as TEXT_ROOT
from tools.train_l28_track_set_decoder import state_at

SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"


def auc(scores, labels):
    scores = np.asarray(scores, np.float64); labels = np.asarray(labels, bool)
    pos = scores[labels]; neg = scores[~labels]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(order), np.float64); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def build_cache(video):
    path = BANK_ROOT / f"{video}.pt"
    bank = torch.load(path, map_location="cpu", weights_only=False)
    tensors = bank["tensors"]; count = int(tensors["track_id"].numel())
    labels, _ = load_labels(path, count, tensors=tensors)
    feature_names = ("clip", "history_clip", "uidm_h", "geometry",
                     "motion", "lifecycle", "objectness")
    feature = torch.cat([tensors[name].float().reshape(count, -1)
                         for name in feature_names], dim=1).half()
    track_ids = tensors["track_id"].long().numpy()
    frames = tensors["frame"].long().numpy()
    by_track = defaultdict(list)
    for row, track in enumerate(track_ids.tolist()):
        by_track[int(track)].append(row)
    ids = sorted(by_track); ptr = [0]; ordered = []; gt = []
    for track in ids:
        ordered.extend(by_track[track]); ptr.append(ptr[-1] + len(by_track[track]))
        gt.extend(labels[row] for row in by_track[track])
    order = torch.as_tensor(np.asarray(ordered, np.int64))
    frame_to_rows = defaultdict(list)
    for row in range(count):
        frame_to_rows[int(frames[row])].append(row)
    return {
        "track_ids": torch.as_tensor(np.asarray(ids, np.int64)),
        "track_ptr": torch.as_tensor(np.asarray(ptr, np.int64)),
        "obs_features": feature[order].contiguous(),
        "obs_frame": torch.as_tensor(frames[order.numpy()], dtype=torch.int32),
        "obs_gt_ids": gt, "frame_to_rows": frame_to_rows,
        "source_bank": str(path.resolve()),
    }


def valid_track_indices(cache, cutoff):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def evaluate_model(checkpoint, cap, device, seed, emission="track"):
    entries = make_entries()
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden = text["token_hidden"]; text_mask = text["attention_mask"].bool(); del text
    caches = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    model = L28TrackSetDecoder().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=False)["model"])
    model.eval()
    videos = {}
    records = []
    entries_by_video = defaultdict(list)
    for entry in entries:
        if entry["split"] == "screening":
            entries_by_video[str(entry["video"])].append(entry)
    used = 0
    for video, video_entries in entries_by_video.items():
        if used >= cap:
            break
        if video not in videos:
            videos[video] = build_cache(video)
        seq = videos[video]
        by_entry = []
        frame_union = set()
        for entry in video_entries:
            data = caches[(entry["video"], entry["expression"])]
            grouped = {int(frame): idx for frame, idx in frame_groups(data)}
            by_entry.append((entry, data, grouped))
            frame_union.update(grouped)
        for frame in sorted(frame_union):
            if used >= cap:
                break
            obs, obs_mask, obs_time, _, _ = state_at(seq, frame)
            with torch.inference_mode():
                encoded = model.encode_observations(
                    obs.to(device), obs_mask.to(device), obs_time.to(device))
            valid_tracks = valid_track_indices(seq, frame)
            for entry, data, grouped in by_entry:
                if frame not in grouped or used >= cap:
                    continue
                idx = grouped[frame]
                qh = text_hidden[int(entry["query_index"])].to(device)
                qm = text_mask[int(entry["query_index"])].to(device)
                with torch.inference_mode():
                    output = model.forward_encoded(encoded, encoded[1], qh, qm)
                    track_scores = output["track_logits"].float().cpu().numpy()
                    membership_scores = output["membership_logits"].float().cpu().numpy()
                score_by_track = {
                    int(seq["track_ids"][track_index]): float(track_scores[i])
                    for i, track_index in enumerate(valid_tracks)}
                if emission == "membership_latest":
                    # The membership target is aligned to an observation at a
                    # concrete frame.  Emit the latest observation available
                    # at this cutoff instead of the stale "ever relevant in
                    # history" track target used by track_logits.
                    score_by_track = {}
                    for i, track_index in enumerate(valid_tracks):
                        latest = torch.nonzero(obs_mask[i], as_tuple=False).flatten()
                        if len(latest):
                            score_by_track[int(seq["track_ids"][track_index])] = float(
                                membership_scores[i, int(latest[-1])])
                row_scores = np.asarray(
                    [score_by_track.get(int(track), -20.0)
                     for track in data["track_id"][idx]], np.float32)
                records.append({"frame": int(frame),
                                "query_index": int(entry["query_index"]),
                                "score": row_scores,
                                "label": data["label"][idx].astype(bool),
                                "track_id": data["track_id"][idx].astype(np.int64)})
                used += 1
    flat_s = np.concatenate([x["score"] for x in records]) if records else np.zeros(0)
    flat_y = np.concatenate([x["label"] for x in records]) if records else np.zeros(0, bool)
    top1 = []; top5 = []; strict = []; best = []; aps = []; selected = flat_s >= 0
    for record in records:
        order = np.argsort(-record["score"], kind="stable")
        y = record["label"]; p = record["score"][y]; n = record["score"][~y]
        if y.any():
            top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any()))
            pos = np.flatnonzero(y); ordered = y[order]; positions = np.flatnonzero(ordered)
            aps.append(float(np.mean([(ordered[:x + 1]).mean() for x in positions])))
            if len(n):
                strict.append(float(p.min() - n.max())); best.append(float(p.max() - n.max()))
    tp = int((selected & flat_y).sum()); fp = int((selected & ~flat_y).sum())
    return {
        "frame_units": len(records), "candidate_rows": int(len(flat_y)),
        "positive_rows": int(flat_y.sum()), "roc_auc": auc(flat_s, flat_y),
        "top1_frame_recall": float(np.mean(top1)) if top1 else None,
        "top5_frame_recall": float(np.mean(top5)) if top5 else None,
        "frame_average_precision": float(np.mean(aps)) if aps else None,
        "strict_min_positive_margin": float(np.mean(strict)) if strict else None,
        "best_positive_margin": float(np.mean(best)) if best else None,
        "hard_violation_rate": float(np.mean(np.asarray(strict) < 0)) if strict else None,
        "zero_threshold": {"predictions": int(selected.sum()),
                           "positive_predictions": int((selected & flat_y).sum()),
                           "predictions_per_positive": float(selected.sum() / max(1, flat_y.sum()))},
        "zero_threshold_precision": float(tp / max(1, tp + fp)),
        "zero_threshold_recall": float(tp / max(1, flat_y.sum())),
        "screening_gt_used_for_strategy_selection": False,
        "emission": emission,
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--emission", choices=("track", "membership_latest"),
                    default="track")
    args = ap.parse_args()
    result = evaluate_model(args.checkpoint, args.cap, torch.device(args.device),
                            args.seed, args.emission)
    payload = {"format": "locatemot-l28-track-set-candidate-eval-v1",
               "checkpoint": str(Path(args.checkpoint).resolve()),
               "score_root": str(SCORE_ROOT.resolve()), "cap": args.cap,
               "device": args.device, "metrics": result}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
