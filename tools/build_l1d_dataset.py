"""Stage L1-D: build training/eval data from the fixed candidate manifest.

Phase 1 (--dump-raw): read caches once, store per-frame raw candidate
features/GT (no association decisions).
Phase 2 (--simulate): run the calibrated base tracker offline (exactly the
shared AC shell: all candidates output, unmatched -> birth, gap > max_age ->
terminated), compute EGRA features and GT-supervised labels, optionally write
MOTChallenge tracker outputs for TrackEval.

Usage:
  python tools/build_l1d_dataset.py dump-raw \
      --manifest outputs/l1_c/fixed_candidate_manifest/dancetrack_calibration.jsonl \
      --out outputs/l1_d/raw/dancetrack_calibration.pkl
  python tools/build_l1d_dataset.py simulate \
      --raw outputs/l1_d/raw/dancetrack_calibration.pkl \
      --out outputs/l1_d/data/dancetrack_calibration.pkl \
      --weights 0.5,0.2,0.3 --threshold 0.3 \
      --write-trackers outputs/l1_d/calib_base/CAL
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


def load_manifest(path):
    by_video = defaultdict(list)
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])
    return by_video


def dump_raw(manifest_path, out_path):
    by_video = load_manifest(manifest_path)
    dataset = None
    frames = []
    for vid, entries in by_video.items():
        for e in entries:
            if dataset is None:
                dataset = e["dataset"]
            fr = read_frame_cache(
                e["cache_root"],
                cache_key(e["dataset"], vid, int(e["frame"]), e["protocol"]))
            if fr is None:
                print(f"[warn] missing cache {vid} frame {e['frame']}", flush=True)
                continue
            feats = fr["features"]
            boxes = np.asarray(feats.get("boxes", np.zeros((0, 4), np.float32)),
                               dtype=np.float32).reshape(-1, 4)
            n = len(boxes)
            pbd = np.zeros((n, 2048), np.float32)
            if "pbd_box_end_last" in feats:
                pbd = np.asarray(feats["pbd_box_end_last"], dtype=np.float32).reshape(n, 2048)
            gen = np.zeros(n, np.float32)
            if "gen_score" in feats:
                g = np.asarray(feats["gen_score"], dtype=np.float32).reshape(-1)
                gen[: min(n, len(g))] = g[: min(n, len(g))]
            cand_gt = [None] * n
            for gid, m in e.get("matched", {}).items():
                ci = int(m["candidate"])
                if 0 <= ci < n:
                    cand_gt[ci] = gid
            frames.append({
                "dataset": dataset,
                "video_id": vid,
                "frame": int(e["frame"]),
                "image_size": list(e.get("image_size", [1280, 720])),
                "boxes": boxes,
                "pbd_be": pbd,
                "gen": gen,
                "cand_gt": cand_gt,
                "gt_matched": {gid: int(m["candidate"]) for gid, m in e.get("matched", {}).items()},
                "gt_boxes": e.get("gt_boxes", {}),
            })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(frames, f, protocol=4)
    print(f"[raw] {len(frames)} frames -> {out_path}", flush=True)


class Track:
    __slots__ = ("tid", "true_gt", "last_box", "prev_box", "age", "hits",
                 "ref_pbd", "anchor_pbd", "last_frame", "kalman")

    def __init__(self, tid, true_gt, box, pbd):
        self.tid = tid
        self.true_gt = true_gt
        self.last_box = box
        self.prev_box = box
        self.age = 1
        self.hits = 1
        self.ref_pbd = pbd
        self.anchor_pbd = pbd
        self.last_frame = 0
        self.kalman = None


def simulate(raw_path, out_path, weights, threshold, max_age,
             write_trackers=None, use_kalman=True):
    with open(raw_path, "rb") as f:
        frames = pickle.load(f)
    wi, wp, wm = (float(x) for x in weights.split(","))
    weights = (wi, wp, wm)
    samples = []
    next_tid = 1
    per_video_trackers = defaultdict(list)
    by_video = defaultdict(list)
    for fr in frames:
        by_video[fr["video_id"]].append(fr)

    for vid, vframes in by_video.items():
        tracks = []
        for fr in vframes:
            image_size = tuple(fr["image_size"])
            cands = [{
                "box": fr["boxes"][i], "pbd": fr["pbd_be"][i],
                "gen": float(fr["gen"][i]), "gt": fr["cand_gt"][i],
            } for i in range(len(fr["boxes"]))]
            N = len(cands)
            # active tracks: not terminated
            active = []
            for t in tracks:
                gap = fr["frame"] - t.last_frame
                if gap <= max_age:
                    active.append(t)
            tracks = active
            T = len(tracks)
            matched_t = set()
            matched_c = set()
            pred_boxes = None
            if T and use_kalman:
                pred_boxes = np.stack([t.kalman.predict() for t in tracks])
            assigns = []
            sample = None
            if T and N:
                tb = np.stack([t.last_box for t in tracks])
                pb = np.stack([t.prev_box for t in tracks])
                rb = np.stack([t.ref_pbd for t in tracks])
                ab = np.stack([t.anchor_pbd for t in tracks])
                cb = np.stack([c["box"] for c in cands])
                cp = np.stack([c["pbd"] for c in cands])
                cg = np.asarray([c["gen"] for c in cands], np.float32)
                gaps = np.asarray([fr["frame"] - t.last_frame for t in tracks], np.float32)
                ages = np.asarray([t.age for t in tracks], np.float32)
                hits = np.asarray([t.hits for t in tracks], np.float32)
                feats = compute_affinity_features(
                    tb, cb, rb, ab, cp, cg, gaps, ages, hits, pb, weights,
                    image_size, motion_pred_boxes=pred_boxes)
                base = feats["base"]
                assigns = hungarian_max(base, threshold)
                # labels
                row_label = np.full(T, -1, np.int64)
                col_label = np.full(N, -1, np.int64)
                true_gt_to_track = {}
                for i, t in enumerate(tracks):
                    if t.true_gt is not None:
                        true_gt_to_track[t.true_gt] = i
                for gid, ci in fr["gt_matched"].items():
                    if gid in true_gt_to_track:
                        row_label[true_gt_to_track[gid]] = ci
                        col_label[ci] = true_gt_to_track[gid]
                base_correct = np.zeros(T, bool)
                for i in range(T):
                    if row_label[i] >= 0:
                        base_correct[i] = int(np.argmax(base[i])) == row_label[i]
                sample = {
                    "dataset": fr["dataset"],
                    "video_id": vid,
                    "frame": fr["frame"],
                    "pair_feats": feats["pair_feats"],
                    "track_feats": feats["track_feats"],
                    "cand_feats": feats["cand_feats"],
                    "base": feats["base"],
                    "row_label": row_label,
                    "col_label": col_label,
                    "base_correct": base_correct,
                    "track_tids": np.asarray([t.tid for t in tracks], np.int64),
                    "track_last_boxes": tb.astype(np.float32),
                    "track_true_gt": [t.true_gt for t in tracks],
                    "n_cand": N,
                }
            # update matched tracks
            for ti, ci in assigns:
                t = tracks[ti]
                c = cands[ci]
                t.kalman.update(c["box"])
                t.prev_box = t.last_box
                t.last_box = c["box"]
                t.ref_pbd = c["pbd"]
                t.age += 1
                t.hits += 1
                t.last_frame = fr["frame"]
                matched_t.add(ti)
                matched_c.add(ci)
                per_video_trackers[vid].append(
                    f"{fr['frame']},{t.tid},{c['box'][0]:.2f},{c['box'][1]:.2f},"
                    f"{c['box'][2]-c['box'][0]:.2f},{c['box'][3]-c['box'][1]:.2f},"
                    f"{c['gen']:.4f},-1,-1,-1\n")
            # births
            for ci, c in enumerate(cands):
                if ci in matched_c:
                    continue
                t = Track(next_tid, c["gt"], c["box"], c["pbd"])
                t.kalman = KalmanBoxTracker7(c["box"])
                t.last_frame = fr["frame"]
                next_tid += 1
                tracks.append(t)
                per_video_trackers[vid].append(
                    f"{fr['frame']},{t.tid},{c['box'][0]:.2f},{c['box'][1]:.2f},"
                    f"{c['box'][2]-c['box'][0]:.2f},{c['box'][3]-c['box'][1]:.2f},"
                    f"{c['gen']:.4f},-1,-1,-1\n")
            # unmatched tracks: kalman observation update None (shared shell)
            for i, t in enumerate(tracks):
                if i not in matched_t:
                    t.kalman.update(None)
            # retain all tracks including newly born; active filter happens next frame
            if sample is not None:
                samples.append(sample)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(samples, f, protocol=4)
    if write_trackers:
        os.makedirs(write_trackers, exist_ok=True)
        for vid, lines in per_video_trackers.items():
            with open(os.path.join(write_trackers, f"{vid}.txt"), "w") as f:
                f.writelines(lines)
    n_sup = sum(int((s["row_label"] >= 0).sum()) for s in samples)
    n_bc = sum(int(s["base_correct"].sum()) for s in samples)
    print(f"[sim] frames={len(samples)} supervised={n_sup} "
          f"base_correct={n_bc} ({n_bc / max(1, n_sup):.4f}) -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("dump-raw")
    p1.add_argument("--manifest", required=True)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("simulate")
    p2.add_argument("--raw", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--weights", required=True)
    p2.add_argument("--threshold", type=float, default=0.3)
    p2.add_argument("--max-age", type=int, default=30)
    p2.add_argument("--write-trackers", default="")
    args = ap.parse_args()
    if args.cmd == "dump-raw":
        dump_raw(args.manifest, args.out)
    else:
        simulate(args.raw, args.out, args.weights, args.threshold,
                 args.max_age, args.write_trackers or None)


if __name__ == "__main__":
    main()
