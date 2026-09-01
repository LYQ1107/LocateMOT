"""Stage L4: build paired specification views for restriction-equivariant training.

For every frame of a domain manifest, this simulates the shared L1DK base
tracker twice:
  - full view:  all candidates (spec = ALL);
  - restricted view: candidates kept by a specification (category / instance).

Each emitted paired sample contains the EGRA feature tensors of both views,
GT-supervised row/column labels per view, and permutation-free alignment
between the views:
  - common_cand: (full_candidate_idx, restricted_candidate_idx) pairs;
  - common_track: (full_track_idx, restricted_track_idx) pairs aligned by
    the privileged GT identity the track was born for (training-only oracle).

Usage:
  python tools/build_l4_pairs.py \
      --raw outputs/l1_d/raw/bdd100k_train.pkl \
      --manifest outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl \
      --out outputs/l4/data/bdd_pairs.pkl --dataset bdd100k
  python tools/build_l4_pairs.py \
      --raw outputs/l1_d/raw/dancetrack_calibration.pkl \
      --manifest outputs/l1_c/fixed_candidate_manifest/dancetrack_calibration.jsonl \
      --out outputs/l4/data/dance_pairs.pkl --dataset dancetrack
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

from locatemot.models.l1d_association import compute_affinity_features  # noqa: E402
from locatemot.tracking.association import hungarian_max  # noqa: E402
from locatemot.tracking.motion import KalmanBoxTracker7  # noqa: E402
from tools.build_l1d_dataset import Track, load_manifest  # noqa: E402

WEIGHTS = (0.4, 0.2, 0.4)
THRESHOLD = 0.25
MAX_AGE = 30
SPEC_ALL = 0
SPEC_CAT = 1
SPEC_INST = 2


def load_categories(manifest_path):
    """(video_id, frame) -> {gid: category}."""
    out = {}
    with open(manifest_path) as f:
        for line in f:
            e = json.loads(line)
            out[(e["video_id"], int(e["frame"]))] = e.get("gt_categories", {})
    return out


def category_of(cats, gid):
    if gid in cats:
        return cats[gid]
    return "person"


def step_tracker(tracks, next_tid, fr, keep, weights=WEIGHTS,
                 threshold=THRESHOLD, max_age=MAX_AGE, use_kalman=True):
    """Advance one tracker view and return (sample or None, tracks, next_tid)."""
    image_size = tuple(fr["image_size"])
    cands = [{
        "box": fr["boxes"][i], "pbd": fr["pbd_be"][i],
        "gen": float(fr["gen"][i]), "gt": fr["cand_gt"][i],
    } for i in keep]
    full_to_compact = {full_i: j for j, full_i in enumerate(keep)}
    N = len(cands)
    active = []
    for t in tracks:
        if fr["frame"] - t.last_frame <= max_age:
            active.append(t)
    tracks = active
    T = len(tracks)
    sample = None
    matched_t = set()
    matched_c = set()
    assigns = []
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
        pred_boxes = None
        if use_kalman:
            pred_boxes = np.stack([t.kalman.predict() for t in tracks])
        feats = compute_affinity_features(
            tb, cb, rb, ab, cp, cg, gaps, ages, hits, pb, weights,
            image_size, motion_pred_boxes=pred_boxes)
        base = feats["base"]
        assigns = hungarian_max(base, threshold)
        row_label = np.full(T, -1, np.int64)
        col_label = np.full(N, -1, np.int64)
        true_gt_to_track = {}
        for i, t in enumerate(tracks):
            if t.true_gt is not None:
                true_gt_to_track[t.true_gt] = i
        gt_matched_local = {}
        for gid, ci in fr["gt_matched"].items():
            j = full_to_compact.get(int(ci))
            if j is not None:
                gt_matched_local[gid] = j
        for gid, ci in gt_matched_local.items():
            if gid in true_gt_to_track:
                row_label[true_gt_to_track[gid]] = ci
                col_label[ci] = true_gt_to_track[gid]
        base_correct = np.zeros(T, bool)
        for i in range(T):
            if row_label[i] >= 0:
                base_correct[i] = int(np.argmax(base[i])) == row_label[i]
        sample = {
            "pair_feats": feats["pair_feats"],
            "track_feats": feats["track_feats"],
            "cand_feats": feats["cand_feats"],
            "base": feats["base"],
            "row_label": row_label,
            "col_label": col_label,
            "base_correct": base_correct,
            "track_true_gt": [t.true_gt for t in tracks],
            "n_cand": N,
        }
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
    for ci, c in enumerate(cands):
        if ci in matched_c:
            continue
        t = Track(next_tid, c["gt"], c["box"], c["pbd"])
        t.kalman = KalmanBoxTracker7(c["box"])
        t.last_frame = fr["frame"]
        next_tid += 1
        tracks.append(t)
    for i, t in enumerate(tracks):
        if i not in matched_t:
            t.kalman.update(None)
    return sample, tracks, next_tid


def top_k_instance_ids(frames, k=2):
    gt_len = defaultdict(int)
    for fr in frames:
        for gid in fr["gt_boxes"]:
            gt_len[gid] += 1
    return [g for g, _ in sorted(gt_len.items(), key=lambda x: -x[1])[:k]]


def build_pairs(frames, manifest_path, dataset, instance_k=2):
    cats = load_categories(manifest_path)
    by_video = defaultdict(list)
    for fr in frames:
        by_video[fr["video_id"]].append(fr)
    pairs = []
    for vid, vframes in by_video.items():
        vframes.sort(key=lambda f: f["frame"])
        inst_ids = top_k_instance_ids(vframes, k=instance_k)
        full_tracks = []
        full_tid = 1
        rest_state = {}
        for fr in vframes:
            frame_cats = cats.get((vid, fr["frame"]), {})
            all_keep = list(range(len(fr["boxes"])))
            full_sample, full_tracks, full_tid = step_tracker(
                full_tracks, full_tid, fr, all_keep)
            specs = []
            if dataset == "bdd100k":
                seen = set()
                for i in all_keep:
                    gid = fr["cand_gt"][i]
                    if gid is None:
                        continue
                    c = category_of(frame_cats, gid)
                    if c not in seen:
                        seen.add(c)
                        specs.append((f"cat:{c}", SPEC_CAT))
            else:
                if inst_ids:
                    specs.append(("inst:auto", SPEC_INST))
            for spec_name, spec_idx in specs:
                keep = []
                if spec_name.startswith("cat:"):
                    c = spec_name.split(":", 1)[1]
                    keep = [i for i in all_keep
                            if fr["cand_gt"][i] is not None
                            and category_of(frame_cats, fr["cand_gt"][i]) == c]
                elif spec_name == "inst:auto":
                    keep = [i for i in all_keep if fr["cand_gt"][i] in inst_ids]
                else:
                    raise ValueError(spec_name)
                if not keep:
                    continue
                st = rest_state.setdefault(
                    spec_name, {"tracks": [], "tid": 1})
                rest_sample, st["tracks"], st["tid"] = step_tracker(
                    st["tracks"], st["tid"], fr, keep)
                if full_sample is None or rest_sample is None:
                    continue
                full_gt_idx = {g: i for i, g in enumerate(full_sample["track_true_gt"])
                               if g is not None}
                rest_gt_idx = {g: i for i, g in enumerate(rest_sample["track_true_gt"])
                               if g is not None}
                common_track = [(full_gt_idx[g], rest_gt_idx[g])
                                for g in sorted(set(full_gt_idx) & set(rest_gt_idx))]
                if not common_track:
                    continue
                common_cand = [(i, j) for j, i in enumerate(keep)]
                pairs.append({
                    "dataset": dataset,
                    "video_id": vid,
                    "frame": fr["frame"],
                    "spec_family": spec_name,
                    "spec_idx": spec_idx,
                    "full": full_sample,
                    "rest": rest_sample,
                    "common_cand": np.asarray(common_cand, np.int64),
                    "common_track": np.asarray(common_track, np.int64),
                })
        print(f"[l4-pairs {dataset}] {vid} frames={len(vframes)} "
              f"pairs={sum(1 for p in pairs if p['video_id']==vid)}", flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True, choices=["bdd100k", "dancetrack",
                                                         "mot17", "mot20"])
    ap.add_argument("--instance-k", type=int, default=2)
    args = ap.parse_args()
    with open(args.raw, "rb") as f:
        frames = pickle.load(f)
    pairs = build_pairs(frames, args.manifest, args.dataset,
                        instance_k=args.instance_k)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(pairs, f, protocol=4)
    spec_counts = defaultdict(int)
    for p in pairs:
        spec_counts[p["spec_family"]] += 1
    print(f"[l4-pairs] {args.dataset}: {len(pairs)} pairs", flush=True)
    for k, v in sorted(spec_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {v}", flush=True)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
