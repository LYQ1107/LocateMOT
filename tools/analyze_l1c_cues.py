"""Stage L1-C: per-event cue disagreement and ambiguity analysis.

For every GT-valid association event (GT object matched to a candidate in a
frame), compute prediction-side features:
  iou_top1 / iou_top2 / iou_margin (over candidates vs GT box)
  pbd_top1 / pbd_top2 / pbd_margin (track-reference PBD cosine over candidates)
  candidate_count, same-category count, object size, temporal gap
and per-method correctness (track-id continuity). Outputs CSVs for
IoU/PBD ambiguity, cue taxonomy and root-cause feature table.

Usage:
  python tools/analyze_l1c_cues.py --methods IoU,Motion,RawPBD,IoUPBD,B6,UAF \
      --split val --feature-cache /data3/.../LocateMOT_L1A/cache_dla --protocol person
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x)
    return x / n if n > 1e-6 else x


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


def load_manifest(path):
    by_video = defaultdict(list)
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])
    return by_video


def pbd_feats(entry, cache_root, protocol):
    key = cache_key(entry["dataset"], entry["video_id"], int(entry["frame"]),
                    protocol)
    fr = read_frame_cache(cache_root, key)
    if fr is None:
        return None, None
    f = fr["features"]
    be = f.get("pbd_box_end_last")
    co = f.get("pbd_coord_mean_last")
    return (np.asarray(be, dtype=np.float32) if be is not None else None,
            np.asarray(co, dtype=np.float32) if co is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--feature-cache",
                    default="/data3/testdata/vranlee/.MOTSynth.partial/"
                            "LocateMOT_L1A/cache_dla")
    ap.add_argument("--protocol", default="person")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/l1_c"))
    args = ap.parse_args()
    methods = args.methods.split(",")
    by_video = load_manifest(os.path.join(
        ROOT, "outputs/l1_c/fixed_candidate_manifest",
        f"dancetrack_{args.split}.jsonl"))

    events = []  # one row per (video, frame, gt, method)
    for method in methods:
        for vid, entries in by_video.items():
            tracker = load_tracker(os.path.join(
                ROOT, "outputs/l1_c/trackeval", method, f"{vid}.txt"))
            gid_to_tid = {}
            last_ci = {}  # gid -> (entry, candidate_idx) of most recent past frame
            for e in entries:
                fr = int(e["frame"])
                preds = tracker.get(fr, [])
                pred_by_idx = {i: tid for i, (tid, _b) in enumerate(preds)}
                boxes = np.asarray(e["boxes"], dtype=np.float64)
                be, co = pbd_feats(e, args.feature_cache, args.protocol)
                for gid, m in e.get("matched", {}).items():
                    ci = int(m["candidate"])
                    gt = e["gt_boxes"][gid]
                    prev_entry, prev_ci = last_ci.get(gid, (None, None))
                    ious = [iou(gt, boxes[j]) for j in range(len(boxes))]
                    ious_sorted = sorted(ious, reverse=True)
                    iou_top1 = ious_sorted[0] if ious_sorted else 0.0
                    iou_top2 = ious_sorted[1] if len(ious_sorted) > 1 else 0.0
                    iou_margin = iou_top1 - iou_top2
                    # PBD margin: track reference = GT object's previous
                    # matched candidate feature in this video.
                    pbd_top1 = pbd_top2 = pbd_margin = 0.0
                    if prev_entry is not None and be is not None:
                        prev_be, _ = pbd_feats(prev_entry, args.feature_cache,
                                               args.protocol)
                        if prev_be is not None and prev_ci < len(prev_be):
                            ref = norm(prev_be[prev_ci])
                            sims = [float(np.dot(ref, norm(be[j])))
                                    for j in range(len(be))]
                            sims_sorted = sorted(sims, reverse=True)
                            pbd_top1 = sims_sorted[0] if sims_sorted else 0.0
                            pbd_top2 = sims_sorted[1] if len(sims_sorted) > 1 else 0.0
                            pbd_margin = pbd_top1 - pbd_top2
                    # Cue correctness (candidate selection, not ID continuity)
                    iou_correct = (int(np.argmax(ious)) == ci)
                    pbd_correct = False
                    if prev_entry is not None and be is not None:
                        prev_be, _ = pbd_feats(prev_entry, args.feature_cache,
                                               args.protocol)
                        if prev_be is not None and prev_ci < len(prev_be):
                            ref = norm(prev_be[prev_ci])
                            sims = [float(np.dot(ref, norm(be[j])))
                                    for j in range(len(be))]
                            pbd_correct = (int(np.argmax(sims)) == ci)
                    tid = pred_by_idx.get(ci)
                    method_correct = (gid not in gid_to_tid) or (gid_to_tid[gid] == tid)
                    if tid is not None:
                        gid_to_tid[gid] = tid
                    if tid is None:
                        continue
                    last_ci[gid] = (e, ci)
                    w, h = e.get("image_size", [1280, 720])
                    obj_size = math.sqrt(
                        max(0, gt[2]-gt[0]) * max(0, gt[3]-gt[1]) / (w * h))
                    gap = fr - (prev_entry["frame"] if prev_entry is not None else fr)
                    events.append({
                        "method": method, "video": vid, "frame": fr, "gt": gid,
                        "correct": int(method_correct),
                        "iou_top1": round(iou_top1, 4),
                        "iou_margin": round(iou_margin, 4),
                        "pbd_top1": round(pbd_top1, 4),
                        "pbd_margin": round(pbd_margin, 4),
                        "iou_correct": int(iou_correct),
                        "pbd_correct": int(pbd_correct),
                        "num_candidates": len(boxes),
                        "obj_size": round(float(obj_size), 5),
                        "gap": int(gap),
                    })

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cue_events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)

    # ambiguity buckets (frozen from calibration quantiles not applied here;
    # report raw margins and both fixed buckets)
    iou_buckets = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.01)]
    pbd_buckets = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.08), (0.08, 1.01)]

    def bucket(v, bs):
        return next((b for b in bs if v >= b[0] and v < b[1]), bs[-1])

    rows = []
    for method in methods:
        evs = [e for e in events if e["method"] == method]
        row = {"method": method, "n": len(evs)}
        for b in iou_buckets:
            sub = [e for e in evs if bucket(e["iou_margin"], iou_buckets) == b]
            row[f"iou_{b[0]:.2f}-{b[1]:.2f}_acc"] = (
                round(sum(e["correct"] for e in sub) / max(1, len(sub)), 4))
            row[f"iou_{b[0]:.2f}-{b[1]:.2f}_n"] = len(sub)
        for b in pbd_buckets:
            sub = [e for e in evs if bucket(e["pbd_margin"], pbd_buckets) == b]
            row[f"pbd_{b[0]:.2f}-{b[1]:.2f}_acc"] = (
                round(sum(e["correct"] for e in sub) / max(1, len(sub)), 4))
            row[f"pbd_{b[0]:.2f}-{b[1]:.2f}_n"] = len(sub)
        both = [e for e in evs if e["iou_correct"] and e["pbd_correct"]]
        po = [e for e in evs if e["pbd_correct"] and not e["iou_correct"]]
        io = [e for e in evs if e["iou_correct"] and not e["pbd_correct"]]
        bothw = [e for e in evs if not e["iou_correct"] and not e["pbd_correct"]]
        row.update({
            "both_correct_n": len(both), "pbd_only_n": len(po),
            "iou_only_n": len(io), "both_wrong_n": len(bothw),
        })
        rows.append(row)
    with open(os.path.join(out_dir, "cue_ambiguity_summary.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
