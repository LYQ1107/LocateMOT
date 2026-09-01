#!/usr/bin/env python3
"""No-training L29 frame-membership/set replay on the fixed fast manifest."""
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
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.audit_l28_identity_bank import BANK_ROOT, load_labels
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5 as TEXT_ROOT
from tools.train_l28_track_set_decoder import state_at

SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
CHECKPOINT = ROOT / "outputs/l28/train/track_set_step1000/checkpoint_track_set_step1000.pt"
OUT = ROOT / "outputs/l29/replay/frame_membership_set_replay.json"


def build_cache(video):
    path = BANK_ROOT / f"{video}.pt"
    bank = torch.load(path, map_location="cpu", weights_only=False)
    tensors = bank["tensors"]
    count = int(tensors["track_id"].numel())
    track_ids = tensors["track_id"].long().numpy()
    frames = tensors["frame"].long().numpy()
    by_track = defaultdict(list)
    for row, track in enumerate(track_ids.tolist()):
        by_track[int(track)].append(row)
    ids = sorted(by_track); ptr = [0]; ordered = []
    for track in ids:
        ordered.extend(by_track[track]); ptr.append(ptr[-1] + len(by_track[track]))
    order = torch.as_tensor(np.asarray(ordered, np.int64))
    feature = torch.cat([
        tensors[name].float().reshape(count, -1)
        for name in ("clip", "history_clip", "uidm_h", "geometry", "motion",
                     "lifecycle", "objectness")], dim=1).half()
    return {"track_ids": torch.as_tensor(np.asarray(ids, np.int64)),
            "track_ptr": torch.as_tensor(np.asarray(ptr, np.int64)),
            "obs_features": feature[order].contiguous(),
            "obs_frame": torch.as_tensor(frames[order.numpy()], dtype=torch.int32),
            "obs_gt_ids": [None] * len(ordered)}


def valid_tracks(cache, cutoff):
    ptr = cache["track_ptr"].numpy(); frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def select_threshold(records, key):
    values = np.concatenate([r[key] for r in records if len(r[key])])
    labels = np.concatenate([r["label"] for r in records if len(r["label"])])
    candidates = np.unique(values)
    if len(candidates) > 256:
        candidates = np.quantile(values, np.linspace(0, 1, 256))
    best = None
    for threshold in candidates:
        selected = values >= threshold
        tp = int((selected & labels).sum()); fp = int((selected & ~labels).sum())
        fn = int((~selected & labels).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        item = (f1, -float(threshold), float(threshold), tp, fp, fn)
        if best is None or item > best:
            best = item
    return {"threshold": best[2], "f1": best[0], "tp": best[3],
            "fp": best[4], "fn": best[5], "source": "calibration_only"}


def frame_metrics(records, key, threshold, null_threshold=None):
    tp = fp = fn = 0; selected_count = 0; empty = 0; null_accept = 0
    frame_recalls = []; transitions = defaultdict(list)
    for r in records:
        score = r[key]; y = r["label"]
        chosen = score >= threshold
        if null_threshold is not None and len(score) and float(score.max()) < null_threshold:
            chosen = np.zeros_like(chosen, dtype=bool)
        selected_count += int(chosen.sum())
        tp += int((chosen & y).sum()); fp += int((chosen & ~y).sum())
        fn += int((~chosen & y).sum())
        empty += int(not chosen.any())
        null_accept += int(not y.any() and chosen.any())
        frame_recalls.append(float((chosen & y).any()) if y.any() else float(not chosen.any()))
        transitions[int(r["query_index"])].append((int(r["frame"]),
                                                      set(r["track_id"][chosen].tolist())))
    switches = 0
    query_recall = {}
    for q, seq in transitions.items():
        seq.sort(); previous = None; hits = []
        for _, current in seq:
            if previous and current and current != previous:
                switches += 1
            previous = current
            hits.append(float(bool(current)))
        query_recall[str(q)] = float(np.mean(hits)) if hits else 0.0
    return {"selected": selected_count, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "false_positive_candidates_per_frame": fp / max(1, len(records)),
            "empty_output_rate": empty / max(1, len(records)),
            "null_frame_false_acceptance": null_accept / max(1, len(records)),
            "predictions_per_positive": selected_count / max(1, tp + fn),
            "identity_switches_transition_diagnostic": switches,
            "query_frame_recall": query_recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-cap", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    entries = make_entries()
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden, mask = text["token_hidden"], text["attention_mask"].bool()
    caches = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    model = L29FrameMembershipSetDecoder().to(args.device)
    state = torch.load(CHECKPOINT, map_location=args.device, weights_only=False)["model"]
    model.load_l28_checkpoint(state)
    model.eval()

    by_video = defaultdict(list)
    for e in entries:
        by_video[str(e["video"])].append(e)
    all_screen_units = []
    grouped_by_entry = {}
    for e in entries:
        data = caches[(e["video"], e["expression"])]
        grouped = {int(f): idx for f, idx in frame_groups(data)}
        grouped_by_entry[(str(e["video"]), str(e["expression"]))] = grouped
        if e["split"] == "screening":
            all_screen_units.extend((str(e["video"]), str(e["expression"]), int(f))
                                  for f in grouped)
    all_screen_units.sort()
    chosen_indices = np.linspace(0, len(all_screen_units) - 1,
                                 min(args.screen_cap, len(all_screen_units)),
                                 dtype=int)
    chosen = {all_screen_units[int(i)] for i in chosen_indices}
    calibration = []; screening = []; seq_cache = {}
    for video, video_entries in by_video.items():
        seq_cache[video] = build_cache(video)
        seq = seq_cache[video]
        frame_union = set()
        for e in video_entries:
            grouped = grouped_by_entry[(video, str(e["expression"]))]
            if e["split"] == "calibration" or any((video, str(e["expression"]), f) in chosen
                                                    for f in grouped):
                frame_union.update(grouped)
        for frame in sorted(frame_union):
            obs, obs_mask, obs_time, _, _ = state_at(seq, frame)
            with torch.inference_mode():
                encoded = model.encode_observations(obs.to(args.device), obs_mask.to(args.device),
                                                    obs_time.to(args.device))
            valid = valid_tracks(seq, frame)
            for e in video_entries:
                key = (video, str(e["expression"]))
                grouped = grouped_by_entry[key]
                if frame not in grouped:
                    continue
                if e["split"] == "screening" and (video, str(e["expression"]), frame) not in chosen:
                    continue
                qh = hidden[int(e["query_index"])].to(args.device)
                qm = mask[int(e["query_index"])].to(args.device)
                with torch.inference_mode():
                    out = model.forward_encoded(encoded, encoded[1], qh, qm)
                raw_by_track = {int(seq["track_ids"][ti]): float(out["current_membership_logits"][i].cpu())
                                for i, ti in enumerate(valid)}
                set_by_track = {int(seq["track_ids"][ti]): float(out["set_membership_logits"][i].cpu())
                                for i, ti in enumerate(valid)}
                data = caches[(e["video"], e["expression"])]
                idx = grouped[frame]
                track = data["track_id"][idx].astype(np.int64)
                r = {"video": video, "expression": str(e["expression"]),
                     "query_index": int(e["query_index"]), "frame": int(frame),
                     "raw": np.asarray([raw_by_track.get(int(t), -20.0) for t in track], np.float32),
                     "set": np.asarray([set_by_track.get(int(t), -20.0) for t in track], np.float32),
                     "label": data["label"][idx].astype(bool), "track_id": track}
                (calibration if e["split"] == "calibration" else screening).append(r)
    raw_threshold = select_threshold(calibration, "raw")
    set_threshold = select_threshold(calibration, "set")
    null_values = [float(r["set"].max()) for r in calibration if not r["label"].any() and len(r["set"])]
    null_threshold = float(np.quantile(null_values, .95)) if null_values else None
    strategies = {
        "l28_latest_membership": frame_metrics(screening, "raw", raw_threshold["threshold"]),
        "l29_frame_aligned_set": frame_metrics(screening, "set", set_threshold["threshold"], null_threshold),
        "l29_frame_aligned_set_no_null": frame_metrics(screening, "set", set_threshold["threshold"]),
    }
    control = json.loads((ROOT / "outputs/l28/eval/sequence_decoder_control_final/control.json").read_text())
    control_b = control["models"]["B_F4_bounded_residual"]["strategies"]
    strategies["l27_raw_threshold_control_B"] = control_b["l27_threshold"]
    strategies["l27_top2_control_B"] = control_b["l27_top2"]
    payload = {"format": "locatemot-l29-frame-membership-replay-v1",
               "checkpoint": str(CHECKPOINT.resolve()),
               "manifest": str((ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json").resolve()),
               "calibration_queries": 64, "screening_queries": 96,
               "screening_frame_units": len(screening), "screening_gt_used_for_selection": False,
               "calibration": {"raw": raw_threshold, "set": set_threshold,
                               "set_null_q95": null_threshold,
                               "calibration_frame_units": len(calibration)},
               "strategies": strategies,
               "provenance": {"l27_control": "immutable sequence_decoder_control_final/control.json",
                              "l28_checkpoint_reused_without_training": True,
                              "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"]}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
