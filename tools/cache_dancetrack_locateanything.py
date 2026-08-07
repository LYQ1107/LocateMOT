#!/usr/bin/env python
"""Stage L1-A: LocateAnything-3B full-video candidate cache for DanceTrack.

Usage:
  python tools/cache_dancetrack_locateanything.py --gpu 3 --split train --shard 0 --num-shards 4
  python tools/cache_dancetrack_locateanything.py --gpu 3 --split pilot --videos dancetrack0086,dancetrack0002 --max-frames 60

Pilot writes per-query protocols (person_d1/d2/d3); main cache writes protocol
`person` with the query chosen in calibration. Resume is per-frame via .complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import cache_key, exists, write_frame_cache  # noqa: E402
from locatemot.models.object_tokens.extractor import ObjectTokenExtractor  # noqa: E402

MODEL_COMMIT = "783f656d127ee498137b5ff52603ce36c292d317"
DANCETRACK = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"
QUERIES = {
    "d1": "Locate all the instances that matches the following description: person.",
    "d2": "Locate all the instances that matches the following description: a person.",
    "d3": "Locate all the instances that matches the following description: people.",
}


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_gt(video_id: str, split: str):
    """Returns {frame: [(x1,y1,x2,y2,obj_id)]} for class=1."""
    data_dir = "train" if split == "calibration" else split
    gt_path = os.path.join(DANCETRACK, data_dir, video_id, "gt", "gt.txt")
    out = defaultdict(list)
    for line in open(gt_path):
        p = line.strip().split(",")
        if len(p) < 9:
            continue
        fid, oid = int(p[0]), int(p[1])
        cls = int(p[7])
        if cls != 1:
            continue
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        if w <= 0 or h <= 0:
            continue
        out[fid].append((x, y, x + w, y + h, oid))
    return out


def frames_of(video_id: str, split: str):
    data_dir = "train" if split == "calibration" else split
    img_dir = os.path.join(DANCETRACK, data_dir, video_id, "img1")
    names = sorted(os.listdir(img_dir))
    return [os.path.join(img_dir, n) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/models/LocateAnything-3B")
    ap.add_argument("--cache-root", default="/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla")
    ap.add_argument("--split", choices=["train", "calibration", "val", "pilot"], required=True)
    ap.add_argument("--videos", default="")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--query-id", default="d1")
    ap.add_argument("--protocol", default="person")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="outputs/l1_a")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.cache_root, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    if args.split == "pilot":
        vids = [v.strip() for v in args.videos.split(",") if v.strip()]
        split_of = {}
        for s in ("train", "calibration"):
            for v in vids:
                if os.path.isdir(os.path.join(DANCETRACK, s, v)):
                    split_of[v] = s
    else:
        cfg_path = os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{args.split}.json")
        cfg = json.load(open(cfg_path))
        vids = [v["video_id"] for v in cfg["videos"]]
        split_of = {v: args.split for v in vids}

    all_jobs = []
    for vid in vids:
        sp = split_of[vid]
        for i, path in enumerate(frames_of(vid, sp)):
            if args.max_frames and i >= args.max_frames:
                break
            fid = int(os.path.splitext(os.path.basename(path))[0])
            all_jobs.append((sp, vid, fid, path))
    all_jobs = [j for i, j in enumerate(all_jobs) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] jobs: {len(all_jobs)}", flush=True)

    if not all_jobs:
        print("[shard] no jobs")
        return

    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    ckpt_hash = json.dumps({
        f: hashlib.sha256(open(os.path.join(args.model, f), "rb").read()).hexdigest()[:12]
        for f in sorted(os.listdir(args.model))
        if f.startswith("model-") and f.endswith(".safetensors")
    }, sort_keys=True)
    extractor = ObjectTokenExtractor(
        model, tok, proc, model_dir=args.model,
        model_commit=MODEL_COMMIT, checkpoint_hash=ckpt_hash, seed=20260806,
    )
    query = QUERIES[args.query_id]
    proto = args.protocol

    rows = []
    done = 0
    for sp, vid, fid, jpg in all_jobs:
        key = cache_key("dancetrack", vid, fid, proto)
        if exists(args.cache_root, key):
            done += 1
            continue
        gt = read_gt(vid, sp).get(fid, [])
        image = Image.open(jpg).convert("RGB")
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        result = extractor.extract(
            image, question=query, semantic_label=proto, source_frame=f"{vid}/{fid:05d}",
            generation_mode="hybrid", max_new_tokens=args.max_new_tokens,
            temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.1,
            in_token_limit=4096,
        )
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        tokens = result["object_tokens"]
        features = {
            "pbd_coord_mean_last": _stack(tokens, "pbd_coordinate_mean_feature"),
            "pbd_coord_mean_penultimate": _stack(tokens, "pbd_coordinate_mean_penultimate_feature"),
            "pbd_box_end_last": _stack(tokens, "pbd_box_end_feature"),
            "region": _stack(tokens, "region_feature"),
            "geometry": _stack(tokens, "geometry_feature"),
            "gen_score": np.asarray([t.generation_score or 0.0 for t in tokens], dtype=np.float32),
            "boxes": np.asarray([t.box_xyxy for t in tokens], dtype=np.float32),
            "normalized_boxes": np.asarray([t.normalized_box for t in tokens], dtype=np.float32),
        }
        features = {k: v for k, v in features.items() if v is not None and len(v) > 0}
        matched = {}
        for oid, gtb in [(g[4], g[:4]) for g in gt]:
            best_idx, best_iou = None, 0.0
            for i, tb in enumerate(tokens):
                iou = _iou(tb.box_xyxy, gtb)
                if iou > best_iou:
                    best_idx, best_iou = i, iou
            if best_idx is not None:
                matched[str(oid)] = {"candidate": best_idx, "iou": round(best_iou, 4)}
        meta = {
            "dataset": "dancetrack", "video_id": vid, "frame": int(fid), "protocol": proto,
            "query": query, "image_size": list(image.size), "candidate_count": len(tokens),
            "gt_object_ids": [g[4] for g in gt],
            "gt_boxes": {str(g[4]): list(g[:4]) for g in gt},
            "matched_candidates": matched,
            "model_commit": MODEL_COMMIT, "checkpoint_hash": ckpt_hash,
            "seconds": round(elapsed, 3), "peak_gpu_gb": round(peak, 3),
            "split": sp,
        }
        write_frame_cache(args.cache_root, key, features, meta)
        rows.append([key, round(elapsed, 3), round(peak, 3), len(tokens)])
        done += 1
        if done % 20 == 0:
            print(f"[shard {args.shard}] done={done}/{len(all_jobs)} last={key} {elapsed:.2f}s", flush=True)

    with open(os.path.join(args.out, f"dla_cache_runtime_{args.split}_shard{args.shard}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "seconds", "peak_gpu_gb", "tokens"])
        w.writerows(rows)
    print(f"[shard {args.shard}] finished, done={done}", flush=True)


def _stack(tokens, attr):
    vals = [getattr(t, attr) for t in tokens]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return np.asarray(vals, dtype=np.float32)


if __name__ == "__main__":
    main()
