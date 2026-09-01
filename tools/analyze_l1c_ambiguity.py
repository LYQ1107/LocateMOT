"""Stage L1-C: IoU / PBD ambiguity bucket analysis on association outputs.

For every GT detection in DanceTrack val, compute:
  margin_iou = top1 IoU(candidate, GT) - top2 IoU(candidate, GT) among
               candidates in the same frame;
  margin_pbd = top1 PBD cosine - top2 PBD cosine (same candidate set).
Then per method (from track eval txt), check whether the candidate matched to
this GT carries the correct track ID.

Usage:
  python tools/analyze_l1c_ambiguity.py --methods C0,C1,C2,C3,C4,UA
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
MANIFEST = os.path.join(ROOT, "outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl")
TRACKEVAL = os.path.join(ROOT, "outputs/l1_c/trackeval")


def load_manifest():
    by_video = defaultdict(list)
    with open(MANIFEST) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])
    return by_video


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_tracker(path):
    rows = defaultdict(list)
    if not os.path.exists(path):
        return rows
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        rows[int(float(p[0]))].append((int(float(p[1])), list(map(float, p[2:6]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="C0,C1,C2,C3,C4,UA")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/l1_c"))
    args = ap.parse_args()
    methods = args.methods.split(",")
    by_video = load_manifest()

    # precompute per-frame GT margins
    margins = {}
    for vid, entries in by_video.items():
        for e in entries:
            fr = e["frame"]
            boxes = np.asarray(e["boxes"], dtype=np.float64)
            for gid, m in e.get("matched", {}).items():
                ci = int(m["candidate"])
                gt = e["gt_boxes"][gid]
                ious = [iou(gt, boxes[j]) for j in range(len(boxes))]
                ious_sorted = sorted(ious, reverse=True)
                top1 = ious_sorted[0] if ious_sorted else 0.0
                top2 = ious_sorted[1] if len(ious_sorted) > 1 else 0.0
                margins[(vid, fr, gid)] = {
                    "iou_margin": top1 - top2,
                    "iou_top1": top1,
                    "candidate_idx": ci,
                }

    # compute PBD margins from cache features (approx via manifest? needs cache read)
    # For now store only IoU margins; PBD margin filled by eval script with cache.
    buckets = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.01)]
    out_rows = []
    for method in methods:
        per_bucket = defaultdict(lambda: {"total": 0, "correct": 0})
        for vid, entries in by_video.items():
            path = os.path.join(TRACKEVAL, method, f"{vid}.txt")
            tracker = load_tracker(path)
            gid_to_tid = {}
            for e in entries:
                fr = e["frame"]
                preds = sorted(tracker.get(fr, []))
                pred_by_idx = {i: tid for i, (tid, _b) in enumerate(preds)}
                for gid, m in e.get("matched", {}).items():
                    key = (vid, fr, gid)
                    if key not in margins:
                        continue
                    mm = margins[key]
                    ci = mm["candidate_idx"]
                    tid = pred_by_idx.get(ci)
                    if tid is None:
                        continue
                    correct = (gid not in gid_to_tid) or (gid_to_tid[gid] == tid)
                    b = next((b for b in buckets
                              if mm["iou_margin"] >= b[0] and mm["iou_margin"] < b[1]),
                             buckets[-1])
                    per_bucket[b]["total"] += 1
                    per_bucket[b]["correct"] += int(correct)
                    gid_to_tid[gid] = tid
        row = {"method": method}
        for b in buckets:
            d = per_bucket[b]
            acc = d["correct"] / max(1, d["total"])
            row[f"iou_{b[0]:.2f}-{b[1]:.2f}_acc"] = round(acc, 4)
            row[f"iou_{b[0]:.2f}-{b[1]:.2f}_n"] = d["total"]
        out_rows.append(row)
        print(row, flush=True)
    with open(os.path.join(args.out, "per_iou_ambiguity.csv"), "w") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
if __name__ == "__main__":
    main()
