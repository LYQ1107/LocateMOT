"""Stage L6: build compact per-video frame-sequence datasets for UIDM.

Each video is written as an independent pickle containing only what the
model-in-the-loop identity-dynamics trainer needs:

  - per-frame candidate boxes / PBD / gen score
  - per-frame GT mapping (candidate index -> GT id, GT boxes)

No U0/GT paired views are precomputed; the identity-dynamics model builds
its own persistent states during training and inference.

Usage:
  python tools/build_l6_sequences.py \
      --manifest outputs/l1_c/fixed_candidate_manifest/dancetrack_train.jsonl \
      --domain dancetrack_train --out outputs/l6/data
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import read_frame_cache  # noqa: E402


def load_manifest(path):
    by_video = defaultdict(list)
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: int(e["frame"]))
    return by_video


def build_video(entries):
    frames = []
    for e in entries:
        n = int(e["candidate_count"]) if "candidate_count" in e else len(e["boxes"])
        key = e.get("cache_key") or (
            f"{e['dataset']}/{e['video_id']}/"
            f"{int(e['frame']):05d}/{e['protocol']}")
        fr = read_frame_cache(e["cache_root"], key)
        pbd = np.zeros((n, 2048), np.float16)
        gen = np.zeros(n, np.float32)
        if fr is not None:
            feats = fr["features"]
            if "pbd_box_end_last" in feats:
                p = np.asarray(feats["pbd_box_end_last"], np.float32).reshape(-1, 2048)
                pbd[:min(n, len(p))] = p[:min(n, len(p))].astype(np.float16)
            if "gen_score" in feats:
                g = np.asarray(feats["gen_score"], np.float32).reshape(-1)
                gen[:min(n, len(g))] = g[:min(n, len(g))]
        cand_gt = [None] * n
        for gid, ci in e.get("matched", {}).items():
            if isinstance(ci, dict):
                ci = ci.get("candidate", -1)
            ci = int(ci)
            if 0 <= ci < n:
                cand_gt[ci] = gid
        frames.append({
            "frame": int(e["frame"]),
            "boxes": np.asarray(e["boxes"], np.float32).reshape(n, 4),
            "pbd": pbd,
            "gen": gen,
            "cand_gt": cand_gt,
            "gt_boxes": {str(k): list(v) for k, v in e.get("gt_boxes", {}).items()},
        })
    return {
        "video_id": entries[0]["video_id"],
        "image_size": list(entries[0].get("image_size", [1280, 720])),
        "frames": frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", default="outputs/l6/data")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    by_video = load_manifest(args.manifest)
    if args.max_videos:
        by_video = dict(list(by_video.items())[:args.max_videos])
    out_dir = os.path.join(args.out, args.domain)
    os.makedirs(out_dir, exist_ok=True)
    index = {"domain": args.domain, "manifest": args.manifest, "videos": {}}
    vids = sorted(by_video)
    for vi, vid in enumerate(vids):
        rec = build_video(by_video[vid])
        path = os.path.join(out_dir, f"{vid}.pkl")
        with open(path, "wb") as f:
            pickle.dump(rec, f, protocol=4)
        index["videos"][vid] = {
            "path": path, "frames": len(rec["frames"]),
            "image_size": rec["image_size"],
        }
        if (vi + 1) % 20 == 0 or vi + 1 == len(vids):
            print(f"[l6data] {args.domain} {vi+1}/{len(vids)}", flush=True)
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[l6data] done domain={args.domain} videos={len(vids)}", flush=True)


if __name__ == "__main__":
    main()
