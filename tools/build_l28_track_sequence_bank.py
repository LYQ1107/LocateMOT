#!/usr/bin/env python3
"""Build a compact, train-only persistent track sequence cache for L28."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.audit_l28_identity_bank import (BANK_ROOT, RECORD_ROOTS, load_labels,
                                            record_path)

SPLIT = ROOT / "outputs/l16/data/protocol/split_manifest.json"
OUT = ROOT / "outputs/l28/track_sequence_bank_final"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.time()
    split = json.loads(SPLIT.read_text())["kitti_v2"]
    videos = [str(x) for x in split["train"]]
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for vi, video in enumerate(sorted(videos), 1):
        bank_path = BANK_ROOT / f"{video}.pt"
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        tensors = bank["tensors"]; count = int(tensors["track_id"].numel())
        labels, label_source = load_labels(bank_path, count, tensors=tensors)
        feature_names = ("clip", "history_clip", "uidm_h", "geometry",
                         "motion", "lifecycle", "objectness")
        pieces = [tensors[name].float().reshape(count, -1)
                  for name in feature_names]
        flat_features = torch.cat(pieces, dim=1).half()
        track_ids = tensors["track_id"].long().numpy()
        frame = tensors["frame"].long().numpy()
        source = tensors["pool_id"].long().numpy()
        by_track = {}
        for row, track in enumerate(track_ids.tolist()):
            by_track.setdefault(int(track), []).append(row)
        ids = sorted(by_track)
        track_ptr = [0]; kept_rows = []; track_gt_ids = []
        for track in ids:
            indices = by_track[track]
            kept_rows.extend(indices)
            track_ptr.append(track_ptr[-1] + len(indices))
            track_gt_ids.append(sorted({labels[r] for r in indices if labels[r] is not None}))
        order = np.asarray(kept_rows, np.int64)
        payload = {
            "format": "locatemot-l28-train-track-sequence-v1",
            "video": video, "source_bank": str(bank_path.resolve()),
            "source_bank_sha256": sha(bank_path), "labels_source": str(label_source),
            "labels_are_train_supervision_only": True,
            "track_ids": torch.as_tensor(np.asarray(ids, np.int64)),
            "track_ptr": torch.as_tensor(np.asarray(track_ptr, np.int64)),
            "obs_features": flat_features[torch.as_tensor(order)].contiguous(),
            "obs_frame": torch.as_tensor(frame[order], dtype=torch.int32),
            "obs_source": torch.as_tensor(source[order], dtype=torch.int8),
            "obs_gt_ids": [labels[int(r)] for r in order.tolist()],
            "track_gt_ids": track_gt_ids,
            "feature_schema": list(feature_names),
            "feature_dim": int(flat_features.shape[1]),
            "history_length": 8,
        }
        destination = OUT / f"{video}.pt"
        torch.save(payload, destination)
        rows.append({
            "video": video, "tracks": len(ids), "observations": len(order),
            "feature_dim": int(flat_features.shape[1]),
            "source_bank": str(bank_path.resolve()), "source_bank_sha256": sha(bank_path),
            "labels_source": str(label_source),
        })
        del bank, tensors, flat_features, payload
        print(f"[l28-track-cache] {video} {vi}/{len(videos)} tracks={len(ids)} obs={len(order)}",
              flush=True)
    manifest = {
        "format": "locatemot-l28-train-track-sequence-manifest-v1",
        "split_manifest": str(SPLIT.resolve()), "split_manifest_sha256": sha(SPLIT),
        "videos": rows, "video_count": len(rows), "screening_videos_written": [],
        "gt_used_for": "train supervision cache only",
        "screening_gt_used_for_training": False, "elapsed_sec": time.time() - start,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "COMPLETE").write_text("train-only\n")
    print(json.dumps({"out": str(OUT), "videos": len(rows),
                      "elapsed_sec": manifest["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
