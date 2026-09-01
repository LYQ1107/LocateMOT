#!/usr/bin/env python
"""Stage L1-A: evaluate LocateAnything DanceTrack candidate pilot."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402

DLA_CACHE = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla"
PILOT_VIDEOS = ["dancetrack0052", "dancetrack0082", "dancetrack0096"]
QUERY_IDS = ["d1", "d2", "d3"]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    out = os.path.join(ROOT, "outputs", "l1_a")
    os.makedirs(out, exist_ok=True)
    rows = []
    for qid in QUERY_IDS:
        proto = f"person_{qid}"
        total_gt = 0
        recall = {0.3: 0, 0.5: 0, 0.7: 0}
        total_cand = 0
        total_frames = 0
        total_seconds = 0.0
        peak_gpu = 0.0
        per_frame_gt = []
        small_hits = small_gt = 0
        dense_frames = 0
        dense_recall = {0.5: 0, 0.7: 0}
        dense_gt = 0
        dup_pairs = 0
        dup_pairs_total = 0
        # iterate actual cache frames
        for vid in PILOT_VIDEOS:
            for fid in range(1, 1000):
                fr = read_frame_cache(DLA_CACHE, cache_key("dancetrack", vid, fid, proto))
                if fr is None:
                    # stop when 10 consecutive missing (pilot only ran 30 frames)
                    if fid > 35:
                        break
                    continue
                meta = fr["meta"]
                boxes = fr["features"]["boxes"]
                gt_boxes = meta["gt_boxes"]
                total_frames += 1
                total_cand += meta["candidate_count"]
                total_seconds += meta.get("seconds", 0.0)
                peak_gpu = max(peak_gpu, meta.get("peak_gpu_gb", 0.0))
                per_frame_gt.append(len(gt_boxes))
                # recall
                for oid, gtb in gt_boxes.items():
                    g = gtb
                    best = 0.0
                    for b in boxes:
                        best = max(best, _iou(b, g))
                    total_gt += 1
                    for th in recall:
                        if best >= th:
                            recall[th] += 1
                    area = (g[2]-g[0])*(g[3]-g[1])
                    if area < 32 * 32:
                        small_gt += 1
                        if best >= 0.5:
                            small_hits += 1
                # duplicate rate: pairs of candidates with IoU > 0.5
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        dup_pairs_total += 1
                        if _iou(boxes[i], boxes[j]) > 0.5:
                            dup_pairs += 1
                # high-density frames
                if len(gt_boxes) >= 10:
                    dense_frames += 1
                    dense_gt += len(gt_boxes)
                    for oid, gtb in gt_boxes.items():
                        best = max([_iou(b, gtb) for b in boxes] or [0.0])
                        for th in (0.5, 0.7):
                            if best >= th:
                                dense_recall[th] += 1
        rows.append({
            "query_id": qid,
            "query": meta.get("query", ""),
            "frames": total_frames,
            "gt_objects": total_gt,
            "candidates": total_cand,
            "candidates_per_frame": round(total_cand / max(1, total_frames), 3),
            "recall_0.3": round(recall[0.3] / max(1, total_gt), 4),
            "recall_0.5": round(recall[0.5] / max(1, total_gt), 4),
            "recall_0.7": round(recall[0.7] / max(1, total_gt), 4),
            "precision": round(total_gt / max(1, total_cand), 4),
            "small_recall_0.5": round(small_hits / max(1, small_gt), 4) if small_gt else 0.0,
            "small_gt": small_gt,
            "dense_frames": dense_frames,
            "dense_recall_0.5": round(dense_recall[0.5] / max(1, dense_gt), 4) if dense_gt else 0.0,
            "dense_recall_0.7": round(dense_recall[0.7] / max(1, dense_gt), 4) if dense_gt else 0.0,
            "duplicate_rate": round(dup_pairs / max(1, dup_pairs_total), 4) if dup_pairs_total else 0.0,
            "avg_seconds": round(total_seconds / max(1, total_frames), 3),
            "peak_gpu_gb": round(peak_gpu, 2),
            "fps": round(total_frames / max(1e-6, total_seconds), 2),
        })
    with open(os.path.join(out, "candidate_recall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: r["recall_0.5"])
    print(f"BEST_QUERY={best['query_id']} recall05={best['recall_0.5']}")


if __name__ == "__main__":
    main()
