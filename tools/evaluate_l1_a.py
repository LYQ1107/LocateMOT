#!/usr/bin/env python
"""Stage L1-A: subset/reactivation analysis + detection manifest + reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.evaluation.full_video_subsets import analyze_splits, aggregate  # noqa: E402

DANCETRACK = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"
DLA_CACHE = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla"
VARIANTS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]


def detection_manifest(protocol, split):
    split_cfg = json.load(open(os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{split}.json")))
    vids = [v["video_id"] for v in split_cfg["videos"]]
    h = hashlib.sha256()
    n_frames = 0
    n_cands = 0
    if protocol == "dla":
        from locatemot.data.token_cache import cache_key, read_frame_cache
        for vid in vids:
            for t in range(1, 5000):
                fr = read_frame_cache(DLA_CACHE, cache_key("dancetrack", vid, t, "person"))
                if fr is None:
                    continue
                meta = fr["meta"]
                h.update(f"{vid}/{t}/{meta['candidate_count']}/{meta['query']}".encode())
                n_frames += 1
                n_cands += meta["candidate_count"]
    else:
        base = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/detections_ctrl"
        for vid in vids:
            d = os.path.join(base, vid)
            if not os.path.isdir(d):
                continue
            for n in sorted(os.listdir(d)):
                data = open(os.path.join(d, n), "rb").read()
                h.update(f"{vid}/{n}".encode() + data)
                n_frames += 1
                n_cands += len(data.strip().splitlines())
    return {
        "protocol": protocol,
        "split": split,
        "frame_count": n_frames,
        "candidate_count": n_cands,
        "sha256": h.hexdigest(),
        "generator": "LocateAnything-3B 783f656d" if protocol == "dla" else "YOLOX-X bytetrack_dance_model.pth.tar",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--protocols", default="dla,ctrl")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    variants = args.variants.split(",")
    protocols = [p for p in args.protocols.split(",") if p]
    out_dir = os.path.join(ROOT, "outputs", "l1_a")
    os.makedirs(out_dir, exist_ok=True)

    manifests = []
    for protocol in protocols:
        manifests.append(detection_manifest(protocol, args.split))
    with open(os.path.join(out_dir, "detection_manifest.json"), "w") as f:
        json.dump(manifests, f, indent=2)

    split_cfg = json.load(open(os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{args.split}.json")))
    vids = [v["video_id"] for v in split_cfg["videos"]]
    gt_root = os.path.join(DANCETRACK, "train" if args.split == "calibration" else args.split)
    tracker_root = os.path.join(out_dir, "trackeval")
    analyzed = analyze_splits(gt_root, tracker_root, variants, vids, protocols=protocols)
    rows = aggregate(analyzed)
    with open(os.path.join(out_dir, f"subset_results_{args.split}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, f"reactivation_results_{args.split}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "video", "frame", "oid", "gap", "prev_tid", "new_tid", "id_kept"])
        for key, data in analyzed.items():
            for r in data["reactivations"]:
                w.writerow([key, r["video"], r["frame"], r["oid"], r["gap"],
                            r["prev_tid"], r["new_tid"], r["id_kept"]])
    print(json.dumps(rows, indent=2))
    print("[eval] done")


if __name__ == "__main__":
    main()
