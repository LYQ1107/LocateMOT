#!/usr/bin/env python3
"""Stage L29 read-only audit of current-frame emission alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l28_track_set_decoder import L28TrackSetDecoder
from tools.audit_l28_identity_bank import BANK_ROOT
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5 as TEXT_ROOT
from tools.train_l28_track_set_decoder import state_at

CHECKPOINT = ROOT / "outputs/l28/train/track_set_step1000/checkpoint_track_set_step1000.pt"
SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
OUT = ROOT / "outputs/l29/audit/emission_contract.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    ids = sorted(by_track)
    ptr = [0]
    ordered = []
    for track in ids:
        ordered.extend(by_track[track])
        ptr.append(ptr[-1] + len(by_track[track]))
    order = torch.as_tensor(np.asarray(ordered, np.int64))
    features = torch.cat([
        tensors[name].float().reshape(count, -1)
        for name in ("clip", "history_clip", "uidm_h", "geometry", "motion",
                     "lifecycle", "objectness")], dim=1).half()
    return {
        "track_ids": torch.as_tensor(np.asarray(ids, np.int64)),
        "track_ptr": torch.as_tensor(np.asarray(ptr, np.int64)),
        "obs_features": features[order].contiguous(),
        "obs_frame": torch.as_tensor(frames[order.numpy()], dtype=torch.int32),
        # state_at is shared with the frozen L28 replay helper.  Contract
        # auditing does not read labels, but the helper requires this key.
        "obs_gt_ids": [None] * len(ordered),
        "source_bank": str(path.resolve()),
    }


def valid_track_indices(cache, cutoff):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT, f"wrong project root: {Path.cwd()}"
    entries = make_entries()
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu",
                      weights_only=False)
    text_hidden = text["token_hidden"]
    text_mask = text["attention_mask"].bool()
    caches = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    model = L28TrackSetDecoder().to(args.device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=args.device,
                                     weights_only=False)["model"])
    model.eval()

    by_video = defaultdict(list)
    for entry in entries:
        if entry["split"] == "screening":
            by_video[str(entry["video"])].append(entry)
    videos = {}
    records = []
    sample_rows = []
    duplicate_keys = Counter()
    stale_rows = 0
    total_rows = 0
    nonfinite = 0
    used = 0
    for video, video_entries in by_video.items():
        if used >= args.cap:
            break
        videos[video] = build_cache(video)
        seq = videos[video]
        entry_groups = []
        frame_union = set()
        for entry in video_entries:
            data = caches[(entry["video"], entry["expression"])]
            grouped = {int(frame): idx for frame, idx in frame_groups(data)}
            entry_groups.append((entry, data, grouped))
            frame_union.update(grouped)
        for frame in sorted(frame_union):
            if used >= args.cap:
                break
            obs, obs_mask, obs_time, _, _ = state_at(seq, frame)
            with torch.inference_mode():
                encoded = model.encode_observations(
                    obs.to(args.device), obs_mask.to(args.device), obs_time.to(args.device))
            valid = valid_track_indices(seq, frame)
            latest_frame = {}
            track_emission = {}
            member_emission = {}
            with torch.inference_mode():
                for entry, data, grouped in entry_groups:
                    if frame not in grouped or used >= args.cap:
                        continue
                    idx = grouped[frame]
                    qh = text_hidden[int(entry["query_index"])].to(args.device)
                    qm = text_mask[int(entry["query_index"])].to(args.device)
                    output = model.forward_encoded(encoded, encoded[1], qh, qm)
                    track_scores = output["track_logits"].float().cpu().numpy()
                    member_scores = output["membership_logits"].float().cpu().numpy()
                    for i, track_index in enumerate(valid):
                        begin = int(seq["track_ptr"][track_index])
                        end = int(seq["track_ptr"][track_index + 1])
                        observed = np.flatnonzero(
                            seq["obs_frame"][begin:end].numpy() <= int(frame)) + begin
                        if not len(observed):
                            continue
                        latest_obs = int(observed[-1])
                        latest_pos = int(torch.nonzero(obs_mask[i], as_tuple=False)
                                         .flatten()[-1])
                        track = int(seq["track_ids"][track_index])
                        latest_frame[track] = int(seq["obs_frame"][latest_obs])
                        track_emission[track] = float(track_scores[i])
                        member_emission[track] = float(member_scores[i, latest_pos])
                    for row in range(len(idx)):
                        row_track = int(data["track_id"][idx][row])
                        key = (video, int(entry["query_index"]), int(frame), row_track,
                               latest_frame.get(row_track, -1))
                        duplicate_keys[key] += 1
                        latest = latest_frame.get(row_track, -1)
                        stale = latest >= 0 and latest < int(frame)
                        stale_rows += int(stale)
                        total_rows += 1
                        ts = track_emission.get(row_track, -20.0)
                        ms = member_emission.get(row_track, -20.0)
                        nonfinite += int(not np.isfinite(ts) or not np.isfinite(ms))
                        if len(sample_rows) < 30:
                            sample_rows.append({
                                "video": video, "query_index": int(entry["query_index"]),
                                "frame": int(frame), "track_id": row_track,
                                "latest_observation_frame": int(latest),
                                "stale": bool(stale), "track_logit": ts,
                                "latest_membership_logit": ms,
                                "label": bool(data["label"][idx][row]),
                            })
                    used += 1
    duplicate_total = sum(max(0, count - 1) for count in duplicate_keys.values())
    duplicate_key_count = sum(count > 1 for count in duplicate_keys.values())
    payload = {
        "format": "locatemot-l29-emission-contract-audit-v1",
        "project_root": str(ROOT),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha(CHECKPOINT),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST),
        "score_root": str(SCORE_ROOT.resolve()),
        "replay": {
            "split": "screening",
            "cap_frame_units": args.cap,
            "frame_units": used,
            "candidate_rows": total_rows,
            "nonfinite_emissions": nonfinite,
            "track_vs_latest_membership": True,
            "threshold_selection": "none in contract audit; calibration-only required later",
        },
        "alignment": {
            "row_key_fields": ["video", "query_index", "frame", "track_id",
                               "latest_observation_frame"],
            "duplicate_emission_keys": duplicate_key_count,
            "duplicate_rows_over_key": duplicate_total,
            "stale_rows": stale_rows,
            "stale_row_rate": stale_rows / max(1, total_rows),
            "observation_unique_for_each_track_frame": duplicate_total == 0,
        },
        "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"],
        "screening_gt_used_for_selection": False,
        "sample_rows": sample_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    readme = out.parent / "README.md"
    readme.write_text(
        "# L29 emission contract audit\n\n"
        "Read-only replay of the frozen L28 step-1000 checkpoint. It compares the "
        "historical track logit with the latest observation-aligned membership logit. "
        "No training, bank write, threshold fitting, or screening-GT model selection "
        "was performed. See `emission_contract.json` for provenance and row samples.\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
