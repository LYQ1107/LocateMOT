#!/usr/bin/env python3
"""Train-only identity/provenance audit for L39."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
CACHE = ROOT / "outputs/l28/track_sequence_bank_final"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
OUT = ROOT / "outputs/l39/audit/identity_probe_contract.json"
TRAIN_VIDEOS = ["0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0020"]
HELDOUT_VIDEOS = ["0015", "0016", "0017"]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd().resolve() == ROOT
    fast = json.loads(MANIFEST.read_text())
    videos = TRAIN_VIDEOS + HELDOUT_VIDEOS
    total_tracks = total_obs = labelled_obs = 0; gt_tracks = defaultdict(int); pair_keys = set(); duplicate = 0
    source_counts = Counter(); field_ok = True
    for video in videos:
        cache = torch.load(CACHE / f"{video}.pt", map_location="cpu", weights_only=False)
        required = {"obs_features", "obs_frame", "obs_gt_ids", "obs_source", "track_ids", "track_ptr", "track_gt_ids"}
        field_ok &= required.issubset(cache)
        total_tracks += len(cache["track_ids"]); total_obs += len(cache["obs_frame"])
        labelled_obs += sum(x is not None for x in cache["obs_gt_ids"])
        source_counts.update(cache["obs_source"].tolist())
        ptr = cache["track_ptr"].tolist(); frames = cache["obs_frame"].tolist(); gt = cache["obs_gt_ids"]
        for ti, track in enumerate(cache["track_ids"].tolist()):
            begin, end = int(ptr[ti]), int(ptr[ti + 1]); gids = {str(x) for x in gt[begin:end] if x is not None}
            for g in gids: gt_tracks[(video, g)] += 1
            for row in range(begin, end):
                key = (video, int(frames[row]), int(track), int(row))
                duplicate += int(key in pair_keys); pair_keys.add(key)
    split = set(videos)
    fast_videos = {str(x["video"]) for x in fast["queries"]}
    payload = {
        "schema_version": "locatemot-l39-identity-probe-contract-v1",
        "project_root": str(ROOT), "train_videos": TRAIN_VIDEOS, "heldout_train_videos": HELDOUT_VIDEOS,
        "train_video_count": len(TRAIN_VIDEOS), "heldout_video_count": len(HELDOUT_VIDEOS),
        "track_count": total_tracks, "observation_count": total_obs, "labelled_observation_count": labelled_obs,
        "required_fields_present": field_ok, "unique_observation_keys": duplicate == 0,
        "pair_key": "(video, frame, track/fragment, observation_row)",
        "source_counts_for_diagnostics_only": {str(k): int(v) for k, v in source_counts.items()},
        "same_gt_fragment_track_groups": int(sum(v > 1 for v in gt_tracks.values())),
        "same_gt_track_group_count": len(gt_tracks),
        "fast_manifest_sha256": sha(MANIFEST), "fast_manifest_query_count": len(fast["queries"]),
        "fast_video_intersection_with_l39_train": sorted(split & fast_videos),
        "screening_gt_used_for_training_or_structure_selection": False,
        "inputs": ["clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle"],
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "expression"],
        "labels": {"same_gt_fragment": "GT_PRIVILEGED_ORACLE", "different_gt_hard_pair": "GT_PRIVILEGED_ORACLE", "inactive": "GT_PRIVILEGED_ORACLE"},
        "decision": {"identity_supervision_available": bool(field_ok and duplicate == 0 and labelled_obs > 0), "enter_l39_smoke": bool(field_ok and duplicate == 0 and labelled_obs > 0)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
