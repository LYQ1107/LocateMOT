"""Stage L5: build clip-level, GT-anchored paired-view training data.

For every video of a domain manifest we run two trajectory sources:

  gt  : per-view tracks are built from the manifest's GT identity mapping
        (candidate -> GT id).  Histories contain observations of the same GT
        identity, so targets are GT anchored by construction.
  u0  : per-view tracks are the frozen L1DK (U0) base-tracker rollout, so
        histories may contain association errors (input evidence); the target
        of a track at frame t is still GT anchored through the candidate's
        manifest GT id (a track's label is the GT id of its most recent
        observation, falling back to its dominant history GT id).

Every emitted frame sample stores compact references (frame index + candidate
index) into the video candidate table; the trainer materialises observation
windows on the fly.  PBD (pbd_box_end_last) is stored once per candidate in
float16.

Usage:
  python tools/build_l5_clips.py \
      --manifest outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl \
      --out outputs/l5/clips/bdd100k_train.pkl \
      --domain bdd100k
  python tools/build_l5_clips.py \
      --manifest outputs/l1_c/fixed_candidate_manifest/dancetrack_calibration.jsonl \
      --out outputs/l5/clips/dancetrack_calibration.pkl \
      --domain dancetrack
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

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402
from locatemot.models.l1d_association import compute_affinity_features  # noqa: E402
from locatemot.tracking.association import hungarian_max  # noqa: E402
from locatemot.tracking.motion import KalmanBoxTracker7  # noqa: E402

WEIGHTS = (0.4, 0.2, 0.4)
THRESHOLD = 0.25
MAX_AGE = 30
MAX_OBS = 16


def _iou(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ar = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    br = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ar + br - inter
    return inter / union if union > 0 else 0.0


def load_manifest(path):
    by_video = defaultdict(list)
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])
    return by_video


def category_of(cats, gid):
    if gid in cats:
        return cats[gid]
    return "person"


def video_specs(entries, domain):
    """Return spec names to build for a video."""
    if domain == "bdd100k":
        cats = set()
        for e in entries:
            for gid in e.get("gt_boxes", {}):
                cats.add(category_of(e.get("gt_categories", {}), gid))
        ordered = [c for c in ("car", "truck", "bus", "pedestrian", "motor",
                               "bike", "trailer", "rider", "other vehicle",
                               "train", "other person") if c in cats]
        return ["ALL"] + [f"cat:{c}" for c in ordered]
    # DanceTrack / MOT17 / MOT20: top-2 longest GT instances
    gt_len = defaultdict(int)
    for e in entries:
        for gid in e.get("gt_boxes", {}):
            gt_len[gid] += 1
    inst = [g for g, _ in sorted(gt_len.items(), key=lambda x: -x[1])[:2]]
    return ["ALL"] + [f"inst:{gid}" for gid in inst]


def spec_keep(entry, spec):
    """Return full-candidate indices kept by a spec."""
    n = int(entry["candidate_count"])
    cand_gt = [None] * n
    for gid, m in entry.get("matched", {}).items():
        ci = int(m["candidate"])
        if 0 <= ci < n:
            cand_gt[ci] = gid
    if spec == "ALL":
        return list(range(n)), cand_gt
    if spec.startswith("cat:"):
        c = spec.split(":", 1)[1]
        keep = [i for i in range(n) if cand_gt[i] is not None
                and category_of(entry.get("gt_categories", {}), cand_gt[i]) == c]
        return keep, cand_gt
    if spec.startswith("inst:"):
        gids = set(spec.split(":", 1)[1].split(","))
        keep = [i for i in range(n) if cand_gt[i] in gids]
        return keep, cand_gt
    raise ValueError(spec)


class U0Track:
    __slots__ = ("tid", "true_gt", "last_box", "prev_box", "age", "hits",
                 "ref_pbd", "anchor_pbd", "last_frame", "kalman", "history",
                 "gt_counts")

    def __init__(self, tid, gt, box, pbd, frame, cand_idx, gen=0.0,
                 log_ncand=0.0, fp_pos=0):
        self.tid = tid
        self.true_gt = gt
        self.last_box = box
        self.prev_box = box
        self.age = 1
        self.hits = 1
        self.ref_pbd = pbd
        self.anchor_pbd = pbd
        self.last_frame = frame
        self.kalman = KalmanBoxTracker7(box) if box is not None else None
        self.history = [(frame, fp_pos, cand_idx, box, pbd, gt, gen,
                         log_ncand)]
        self.gt_counts = defaultdict(int)
        if gt is not None:
            self.gt_counts[gt] += 1

    def dominant_gt(self):
        if not self.gt_counts:
            return None
        return max(self.gt_counts.items(), key=lambda x: x[1])[0]


def compute_view_base(tracks, cands, image_size, cur_frame, use_kalman=True):
    """Compute L1DK base affinity for a frame."""
    T = len(tracks)
    N = len(cands)
    if T == 0 or N == 0:
        return None
    tb = np.stack([t.last_box for t in tracks])
    pb = np.stack([t.prev_box for t in tracks])
    rb = np.stack([t.ref_pbd for t in tracks])
    ab = np.stack([t.anchor_pbd for t in tracks])
    cb = np.stack([c["box"] for c in cands])
    cp = np.stack([c["pbd"] for c in cands])
    cg = np.asarray([c["gen"] for c in cands], np.float32)
    gaps = np.asarray([max(1, cur_frame - t.last_frame) for t in tracks], np.float32)
    ages = np.asarray([t.age for t in tracks], np.float32)
    hits = np.asarray([t.hits for t in tracks], np.float32)
    pred = None
    if use_kalman:
        pred = np.stack([t.kalman.predict() for t in tracks])
    else:
        vel = tb - pb
        pred = tb + vel * np.maximum(gaps[:, None], 1.0)
    feats = compute_affinity_features(
        tb, cb, rb, ab, cp, cg, gaps, ages, hits, pb, WEIGHTS,
        image_size, motion_pred_boxes=pred)
    return feats


def build_u0_sample(tracks, cands, keep, cand_gt, image_size, cur_frame,
                    source="u0", gt_boxes=None):
    """Build per-frame sample from current U0 tracks + candidates."""
    T = len(tracks)
    N = len(cands)
    if T == 0 or N == 0:
        return None
    feats = compute_view_base(tracks, cands, image_size, cur_frame,
                              use_kalman=(source == "u0"))
    base = feats["base"]
    row_label = np.full(T, -1, np.int64)
    col_label = np.full(N, -1, np.int64)
    # GT-anchored target: the GT identity of the GT box with highest IoU to
    # the track's current box (never the tracker's integer id, never the
    # history majority which can be corrupted by an early switch).
    cur_gt = []
    for t in tracks:
        best = None
        best_iou = 0.3
        if gt_boxes:
            for gid, gb in gt_boxes.items():
                iou = _iou(t.last_box, gb)
                if iou > best_iou:
                    best_iou = iou
                    best = gid
        cur_gt.append(best)
    for i, t in enumerate(tracks):
        target_gt = cur_gt[i]
        if source == "gt":
            target_gt = t.true_gt
        for j, c in enumerate(cands):
            if c["gt"] is not None and c["gt"] == target_gt:
                row_label[i] = j
                col_label[j] = i
                break
    base_correct = np.zeros(T, bool)
    for i in range(T):
        if row_label[i] >= 0:
            base_correct[i] = int(np.argmax(base[i])) == row_label[i]
    return {
        "keep": np.asarray(keep, np.int64),
        "pair_feats": feats["pair_feats"],
        "track_feats": feats["track_feats"],
        "cand_feats": feats["cand_feats"],
        "base": base,
        "row_label": row_label,
        "col_label": col_label,
        "base_correct": base_correct,
        "track_gt": [t.true_gt for t in tracks],
        "track_dom_gt": [t.dominant_gt() for t in tracks],
        "track_cur_gt": cur_gt,
        "track_hist": [t.history[-MAX_OBS:] for t in tracks],
        "track_tid": [t.tid for t in tracks],
    }


def build_video(entries, domain, max_obs=MAX_OBS):
    """Return video record with candidate table + per-view samples."""
    cand_table = []
    for e in entries:
        n = int(e["candidate_count"])
        boxes = np.asarray(e["boxes"], np.float32).reshape(n, 4)
        pbd = np.zeros((n, 2048), np.float16)
        gen = np.zeros(n, np.float32)
        feats = None
        fr = read_frame_cache(
            e["cache_root"],
            cache_key(e["dataset"], e["video_id"], int(e["frame"]), e["protocol"]))
        if fr is not None:
            feats = fr["features"]
        if feats is not None and "pbd_box_end_last" in feats:
            p = np.asarray(feats["pbd_box_end_last"], np.float32).reshape(n, 2048)
            pbd = p.astype(np.float16)
        if feats is not None and "gen_score" in feats:
            g = np.asarray(feats["gen_score"], np.float32).reshape(-1)
            gen[:min(n, len(g))] = g[:min(n, len(g))]
        gt = [None] * n
        for gid, m in e.get("matched", {}).items():
            ci = int(m["candidate"])
            if 0 <= ci < n:
                gt[ci] = gid
        cand_table.append({
            "frame": int(e["frame"]),
            "box": boxes,
            "pbd": pbd,
            "gen": gen,
            "gt": gt,
            "gt_box": e.get("gt_boxes", {}),
            "matched": {gid: int(m["candidate"]) for gid, m in e.get("matched", {}).items()},
        })
    specs = video_specs(entries, domain)
    views = {}
    for spec in specs:
        # GT-anchored view
        gt_samples = []
        # per-frame candidate lists and cand_gt maps
        keep_by_frame = []
        cand_gt_by_frame = []
        for e in entries:
            keep, cg = spec_keep(e, spec)
            keep_by_frame.append(keep)
            cand_gt_by_frame.append(cg)
        for fp in range(len(entries)):
            e = entries[fp]
            keep = keep_by_frame[fp]
            cand_gt = cand_gt_by_frame[fp]
            if not keep:
                continue
            # GT tracks with obs before fp
            by_gt = defaultdict(list)
            for fp0 in range(fp):
                for ci in keep_by_frame[fp0]:
                    g = cand_gt_by_frame[fp0][ci]
                    if g is not None:
                        by_gt[g].append((fp0, ci))
            tracks = []
            for gid, hist in by_gt.items():
                hist = hist[-max_obs:]
                obs_boxes = []
                obs_pbd = []
                for fp0, ci in hist:
                    obs_boxes.append(cand_table[fp0]["box"][ci])
                    obs_pbd.append(cand_table[fp0]["pbd"][ci])
                t = U0Track(0, gid, None, None, 0, -1)
                t.last_box = np.asarray(obs_boxes[-1], np.float64)
                t.prev_box = np.asarray(obs_boxes[-2] if len(obs_boxes) > 1 else obs_boxes[-1],
                                        np.float64)
                t.ref_pbd = obs_pbd[-1]
                t.anchor_pbd = obs_pbd[0]
                t.last_frame = int(entries[hist[-1][0]]["frame"])
                t.age = len(obs_boxes)
                t.hits = len(obs_boxes)
                t.kalman = None
                t.true_gt = gid
                t.history = [
                    (int(entries[fp0]["frame"]), fp0, ci,
                     entries[fp0]["boxes"][ci],
                     cand_table[fp0]["pbd"][ci], gid,
                     float(cand_table[fp0]["gen"][ci]),
                     np.log1p(len(keep_by_frame[fp0])))
                    for fp0, ci in hist]
                t.gt_counts[gid] = len(hist)
                tracks.append(t)
            cands = [{"box": np.asarray(entries[fp]["boxes"][i], np.float64),
                      "pbd": cand_table[fp]["pbd"][i],
                      "gen": float(cand_table[fp]["gen"][i]), "gt": cand_gt[i]}
                     for i in keep]
            sample = build_u0_sample(tracks, cands, keep, cand_gt_by_frame[fp],
                                     tuple(e["image_size"]), int(e["frame"]),
                                     source="gt", gt_boxes=e.get("gt_boxes", {}))
            if sample is not None:
                sample["frame"] = fp
                sample["frame_id"] = int(e["frame"])
                sample["source"] = "gt"
                gt_samples.append(sample)
        # U0 rollout view
        u0_samples = []
        u0_tracks = []
        next_tid = 1
        for fp in range(len(entries)):
            e = entries[fp]
            keep = keep_by_frame[fp]
            cand_gt = cand_gt_by_frame[fp]
            if not keep:
                continue
            image_size = tuple(e["image_size"])
            active = []
            for t in u0_tracks:
                if int(e["frame"]) - t.last_frame <= MAX_AGE:
                    active.append(t)
            u0_tracks = active
            cands = [{"box": np.asarray(entries[fp]["boxes"][i], np.float64),
                      "pbd": cand_table[fp]["pbd"][i],
                      "gen": float(cand_table[fp]["gen"][i]), "gt": cand_gt[i]}
                     for i in keep]
            sample = build_u0_sample(u0_tracks, cands, keep, cand_gt, image_size,
                                     int(e["frame"]), source="u0",
                                     gt_boxes=e.get("gt_boxes", {}))
            if sample is not None:
                sample["frame"] = fp
                sample["frame_id"] = int(e["frame"])
                sample["source"] = "u0"
                u0_samples.append(sample)
            # update / birth
            assigns = []
            if u0_tracks and cands:
                feats = compute_view_base(u0_tracks, cands, image_size,
                                          int(e["frame"]), use_kalman=True)
                assigns = hungarian_max(feats["base"], THRESHOLD)
            matched_c = set()
            for ti, ci in assigns:
                t = u0_tracks[ti]
                c = cands[ci]
                t.kalman.update(c["box"])
                t.prev_box = t.last_box
                t.last_box = c["box"]
                t.ref_pbd = c["pbd"]
                t.age += 1
                t.hits += 1
                t.last_frame = int(e["frame"])
                t.history.append((int(e["frame"]), fp, keep[ci], c["box"],
                                  c["pbd"], c["gt"], float(c["gen"]),
                                  np.log1p(len(keep))))
                if c["gt"] is not None:
                    t.gt_counts[c["gt"]] += 1
                matched_c.add(ci)
            for ci, c in enumerate(cands):
                if ci in matched_c:
                    continue
                t = U0Track(next_tid, c["gt"], c["box"], c["pbd"],
                            int(e["frame"]), keep[ci], float(c["gen"]),
                            np.log1p(len(keep)), fp_pos=fp)
                next_tid += 1
                u0_tracks.append(t)
            for i, t in enumerate(u0_tracks):
                if i not in {ti for ti, _ in assigns}:
                    t.kalman.update(None)
        views[spec] = {"gt": gt_samples, "u0": u0_samples}
    return {"image_size": list(entries[0].get("image_size", [1280, 720])),
            "cands": cand_table, "views": views}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--videos", nargs="*", default=None)
    args = ap.parse_args()
    by_video = load_manifest(args.manifest)
    if args.videos:
        by_video = {v: by_video[v] for v in args.videos if v in by_video}
    elif args.max_videos:
        by_video = dict(list(by_video.items())[:args.max_videos])
    out = {"domain": args.domain, "videos": {}}
    for vid, entries in by_video.items():
        rec = build_video(entries, args.domain)
        n_gt = sum(len(v["gt"]) for v in rec["views"].values())
        n_u0 = sum(len(v["u0"]) for v in rec["views"].values())
        print(f"[clip] {vid} views={list(rec['views'])} gt_samples={n_gt} "
              f"u0_samples={n_u0}", flush=True)
        out["videos"][vid] = rec
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"[clip] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
