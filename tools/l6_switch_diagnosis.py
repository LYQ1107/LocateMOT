"""Stage L6: characterise ID switches from tracker outputs + GT.

For every GT identity, find frames where the tracker ID changes (switch).
For each switch, measure the preceding context:
  - gap since the identity was last observed in candidates
  - maximum IoU of the GT box with other GT boxes in the previous frame
    (crowding)
  - whether the previous frame had no candidate for this identity
    (detection gap)

Usage:
  python tools/l6_switch_diagnosis.py \
      --manifest outputs/l1_c/fixed_candidate_manifest/mot17_train.jsonl \
      --tracker outputs/l6/trackeval/uidm_ep5_mot17/trackers/mot17_train \
      --domain mot17 --out outputs/l6/switch_mot17.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def iou(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ar = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    br = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ar + br - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    by_video = defaultdict(list)
    with open(args.manifest) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    stats = defaultdict(list)
    total_switches = 0
    per_video = {}
    for vid, entries in by_video.items():
        entries.sort(key=lambda e: int(e["frame"]))
        path = os.path.join(args.tracker, f"{vid}.txt")
        if not os.path.exists(path):
            continue
        trk = defaultdict(dict)  # gid -> {frame: track_id}
        cand_of = defaultdict(dict)  # gid -> {frame: cand_idx}
        for e in entries:
            fr = int(e["frame"])
            for gid in e.get("gt_boxes", {}):
                cand_of[gid][fr] = None
            for gid, m in e.get("matched", {}).items():
                ci = int(m["candidate"]) if isinstance(m, dict) \
                    else int(m)
                cand_of[gid][fr] = ci
        with open(path) as f:
            for line in f:
                p = line.strip().split(",")
                fr = int(float(p[0])); tid = int(float(p[1]))
                # candidate index is not directly in tracker output; we can
                # only assign via GT matching later; skip here.
        # map tracker rows back to candidates via box IoU with GT boxes
        cand_trk = {}  # (frame, cand_idx) -> tid
        with open(path) as f:
            for line in f:
                p = line.strip().split(",")
                fr = int(float(p[0])); tid = int(float(p[1]))
                box = [float(p[2]), float(p[3]),
                       float(p[2]) + float(p[4]),
                       float(p[3]) + float(p[5])]
                cand_trk.setdefault(fr, []).append((tid, box))
        frames = [int(e["frame"]) for e in entries]
        prev_gt = None
        for gid in cand_of:
            gid_switches = 0
            last_tid = None
            last_obs_frame = None
            for fr in sorted(cand_of[gid]):
                ci = cand_of[gid][fr]
                tid = None
                if ci is not None:
                    # match by best IoU with the GT box of this identity
                    gb = entries[[i for i, e in enumerate(entries)
                                  if int(e["frame"]) == fr][0]] \
                        .get("gt_boxes", {}).get(gid)
                    if gb is None:
                        tid = None
                    else:
                        best = None
                        best_iou = 0.3
                        for t, box in cand_trk.get(fr, []):
                            v = iou(gb, box)
                            if v > best_iou:
                                best_iou = v
                                best = t
                        tid = best
                if tid is not None:
                    if last_tid is not None and tid != last_tid:
                        gid_switches += 1
                        total_switches += 1
                        gap = 0 if last_obs_frame is None else \
                            fr - last_obs_frame
                        # crowding in previous frame
                        prev_i = [i for i, e in enumerate(entries)
                                  if int(e["frame"]) == fr - 1]
                        max_iou_other = 0.0
                        det_gap = last_obs_frame is not None and \
                            fr - last_obs_frame > 1
                        if prev_i:
                            pgb = entries[prev_i[0]].get("gt_boxes", {})
                            gb = entries[prev_i[0]].get("gt_boxes", {}).get(gid)
                            if gb is not None:
                                for og, ob in pgb.items():
                                    if og != gid:
                                        max_iou_other = max(max_iou_other,
                                                            iou(gb, ob))
                        stats["gap_at_switch"].append(gap)
                        stats["crowd_iou_prev"].append(max_iou_other)
                        stats["det_gap"].append(int(det_gap))
                    last_tid = tid
                    last_obs_frame = fr
            if gid_switches:
                per_video[vid] = per_video.get(vid, 0) + gid_switches
    out = {
        "domain": args.domain,
        "total_switches": total_switches,
        "per_video": per_video,
        "gap_at_switch_mean": float(np.mean(stats["gap_at_switch"])) if
        stats["gap_at_switch"] else None,
        "gap_at_switch_hist": np.histogram(
            stats["gap_at_switch"], bins=[0, 1, 2, 5, 10, 31, 1000])[0]
        .tolist() if stats["gap_at_switch"] else None,
        "crowd_iou_prev_mean": float(np.mean(stats["crowd_iou_prev"])) if
        stats["crowd_iou_prev"] else None,
        "det_gap_fraction": float(np.mean(stats["det_gap"])) if
        stats["det_gap"] else None,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
