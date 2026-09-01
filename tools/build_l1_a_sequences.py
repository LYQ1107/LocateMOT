#!/usr/bin/env python
"""Stage L1-A: freeze DanceTrack video-level disjoint splits (32 train / 8
calibration from official train, 25 official val fully held-out)."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANCETRACK = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"
SEED = 20260806


def read_gt(seq_dir: str):
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    per_frame = defaultdict(list)
    n_boxes = 0
    for line in open(gt_path):
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue
        fid = int(parts[0])
        cls = int(parts[7]) if len(parts) > 7 else 1
        if cls != 1:
            continue
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        if w <= 0 or h <= 0:
            continue
        per_frame[fid].append([x, y, w, h])
        n_boxes += 1
    return per_frame, n_boxes


def video_stats(split: str):
    rows = []
    base = os.path.join(DANCETRACK, split)
    for vid in sorted(os.listdir(base)):
        if not os.path.isdir(os.path.join(base, vid)):
            continue
        per_frame, n_boxes = read_gt(os.path.join(base, vid))
        frames = sorted(per_frame)
        densities = [len(per_frame[f]) for f in frames]
        n_imgs = len(os.listdir(os.path.join(base, vid, "img1")))
        rows.append({
            "video_id": vid,
            "frames": n_imgs,
            "gt_frames": len(frames),
            "gt_boxes": n_boxes,
            "mean_density": float(np.mean(densities)) if densities else 0.0,
            "max_density": int(max(densities)) if densities else 0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "configs", "data"))
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "l1_a_split_report.md"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    train_stats = video_stats("train")
    val_stats = video_stats("val")
    # stratify by density tercile for the 8 calibration videos
    densities = np.array([r["mean_density"] for r in train_stats])
    q1 = float(np.quantile(densities, 1 / 3))
    q2 = float(np.quantile(densities, 2 / 3))
    buckets = defaultdict(list)
    for r in train_stats:
        if r["mean_density"] < q1:
            buckets["low"].append(r)
        elif r["mean_density"] < q2:
            buckets["med"].append(r)
        else:
            buckets["high"].append(r)
    calib = []
    for b, n in (("low", 2), ("med", 3), ("high", 3)):
        rng.shuffle(buckets[b])
        calib.extend(buckets[b][:n])
    rng.shuffle(calib)
    calib_ids = {r["video_id"] for r in calib}
    train_ids = [r["video_id"] for r in train_stats if r["video_id"] not in calib_ids]
    rng.shuffle(train_ids)

    def dump(name, ids, stats):
        by_id = {r["video_id"]: r for r in stats}
        entries = []
        for vid in ids:
            s = by_id[vid]
            entries.append({
                "video_id": vid,
                "dataset": "dancetrack",
                "frames": s["frames"],
                "gt_frames": s["gt_frames"],
                "gt_boxes": s["gt_boxes"],
                "mean_density": round(s["mean_density"], 4),
                "max_density": s["max_density"],
            })
        path = os.path.join(args.out_dir, f"l1_a_dancetrack_{name}.json")
        json.dump({"split": name, "seed": args.seed, "videos": entries}, open(path, "w"), indent=2)
        return path

    p_train = dump("train", train_ids, train_stats)
    p_calib = dump("calibration", [r["video_id"] for r in calib], train_stats)
    p_val = dump("val", [r["video_id"] for r in val_stats], val_stats)

    with open(args.report, "w") as f:
        f.write(f"# Stage L1-A DanceTrack Split Report\n\n")
        f.write(f"生成时间：2026-08-07，seed={args.seed}\n\n")
        f.write("划分规则：video-level disjoint；train/calibration 来自官方 train（40 视频），"
                "official val（25 视频）全程 held-out，不用于训练/阈值/checkpoint 选择。\n\n")
        for name, p in [("train", p_train), ("calibration", p_calib), ("val", p_val)]:
            js = json.load(open(p))
            n_v = len(js["videos"])
            n_f = sum(v["frames"] for v in js["videos"])
            n_b = sum(v["gt_boxes"] for v in js["videos"])
            f.write(f"## {name}: {n_v} videos, {n_f} frames, {n_b} GT boxes\n")
            f.write("| video_id | frames | gt_frames | gt_boxes | mean_density | max_density |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for v in js["videos"]:
                f.write(f"| {v['video_id']} | {v['frames']} | {v['gt_frames']} | {v['gt_boxes']} "
                        f"| {v['mean_density']:.2f} | {v['max_density']} |\n")
            f.write("\n")
        train_stats_all = {r["video_id"]: r for r in train_stats}
        calib_ids2 = [r["video_id"] for r in calib]
        f.write("## Calibration selection rationale\n\n")
        f.write("按 GT mean density 分桶（tercile），从低/中/高各选 2/3/3 个视频，"
                "保证 calibration 覆盖低密度、中密度、高密度场景。\n")
        f.write("train/calibration 各自 video-level disjoint；同一视频不会同时出现在两个 split。\n")
    print(f"train={len(train_ids)} calibration={len(calib)} val={len(val_stats)}")
    print("train:", ",".join(train_ids))
    print("calibration:", ",".join([r["video_id"] for r in calib]))


if __name__ == "__main__":
    main()
