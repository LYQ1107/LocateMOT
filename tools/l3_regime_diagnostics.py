"""Stage L3: per-regime method/cue preference diagnostics.

For each domain, frame-level prediction-side regime features are computed
from the fixed candidate manifests and cached PBD features. Frames are
bucketed by regime; for each method (C0/C1/C2/C3/L1DK/EGRA) the windowed
AssA (TrackEval-consistent, H frames) is computed from the saved AC tracker
outputs. This shows whether the best method/cue changes across regimes.

Usage:
  python tools/l3_regime_diagnostics.py \
      --domains dancetrack_val,mot17_train,mot20_train,bdd100k_train \
      --out outputs/l3/regime_diagnostics.json --horizon 16
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

from tools.run_l2_oracle import windowed_metrics  # noqa: E402

DOMAINS = {
    "dancetrack_val": {
        "manifest": "outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl",
        "raw": "outputs/l1_d/raw/dancetrack_val.pkl",
        "trackers": "outputs/l1_c/trackeval",
        "variants": ["C0", "C1", "C2", "C3", "L1DK_BASE", "L1DK_d03"],
    },
    "mot17_train": {
        "manifest": "outputs/l1_c/fixed_candidate_manifest/mot17_train.jsonl",
        "raw": "outputs/l1_d/raw/mot17_train.pkl",
        "trackers": "outputs/l2/baseline_AC/mot17_train",
        "variants": ["C0", "C1", "C2", "C3", "L1DK", "L1DK_d03"],
    },
    "mot20_train": {
        "manifest": "outputs/l1_c/fixed_candidate_manifest/mot20_train.jsonl",
        "raw": "outputs/l1_d/raw/mot20_train.pkl",
        "trackers": "outputs/l2/baseline_AC/mot20_train",
        "variants": ["C0", "C1", "C2", "C3", "L1DK", "L1DK_d03"],
    },
    "bdd100k_train": {
        "manifest": "outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl",
        "raw": "outputs/l1_d/raw/bdd100k_train.pkl",
        "trackers": "outputs/l2/baseline_AC/bdd100k_train",
        "variants": ["C0", "C1", "C2", "C3", "L1DK", "L1DK_d03"],
    },
}


def load_tracker_rows(root, vid):
    p = os.path.join(root, f"{vid}.txt")
    if not os.path.exists(p):
        return None
    rows = defaultdict(list)
    for line in open(p):
        q = line.strip().split(",")
        if len(q) < 6:
            continue
        fr = int(float(q[0]))
        tid = int(float(q[1]))
        x1, y1, x2, y2 = map(float, q[2:6])
        rows[fr].append((tid, [x1, y1, x1 + x2, y1 + y2]))
    return rows


def frame_features(entry, prev_boxes, pbd_by_video_frame, image_size):
    boxes = np.asarray(entry["boxes"], np.float64).reshape(-1, 4)
    n = len(boxes)
    iw, ih = entry.get("image_size", [1280, 720])
    diag = float(np.hypot(iw, ih)) + 1e-6
    feats = {
        "n_cand": n,
        "gap": 1,
        "size": 0.0,
        "iou_amb": 0.0,
        "pbd_amb": 0.0,
        "motion": 0.0,
        "sem_div": 0,
    }
    if n == 0:
        return feats
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    feats["size"] = float(np.sqrt(np.maximum(areas, 0)).mean() / diag)
    # pairwise IoU ambiguity
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            iw_ = max(0.0, ix2 - ix1); ih_ = max(0.0, iy2 - iy1)
            inter = iw_ * ih_
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            sims.append(inter / max(1e-6, union))
    feats["iou_amb"] = float(np.mean(sims)) if sims else 0.0
    # PBD ambiguity: mean pairwise cosine
    key = (entry["video_id"], int(entry["frame"]))
    pbd = pbd_by_video_frame.get(key)
    if pbd is not None and len(pbd) > 1:
        pbd = pbd / (np.linalg.norm(pbd, axis=-1, keepdims=True) + 1e-6)
        sim = pbd @ pbd.T
        iu = np.triu_indices(len(pbd), 1)
        feats["pbd_amb"] = float(sim[iu].mean()) if len(iu[0]) else 0.0
    # motion proxy: mean center displacement vs previous frame candidates
    if prev_boxes is not None and len(prev_boxes):
        cc = (boxes[:, :2] + boxes[:, 2:]) / 2
        pc = (prev_boxes[:, :2] + prev_boxes[:, 2:]) / 2
        d = np.sqrt(((cc[:, None] - pc[None]) ** 2).sum(-1))
        feats["motion"] = float(np.min(d, axis=1).mean() / diag)
    # semantic diversity from GT categories (audit-only axis)
    gc = entry.get("gt_categories", {})
    feats["sem_div"] = len(set(gc.values()))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="dancetrack_val,mot17_train,mot20_train,bdd100k_train")
    ap.add_argument("--out", default="outputs/l3/regime_diagnostics.json")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    domains = args.domains.split(",")

    # global regime feature stats for quantile bucketing
    all_feats = []
    for dom in domains:
        cfg = DOMAINS[dom]
        with open(cfg["raw"], "rb") as f:
            raw = pickle.load(f)
        pbd_by = {}
        for fr in raw:
            pbd_by[(fr["video_id"], int(fr["frame"]))] = fr["pbd_be"]
        with open(cfg["manifest"]) as f:
            entries = [json.loads(l) for l in f]
        by_vid = defaultdict(list)
        for e in entries:
            by_vid[e["video_id"]].append(e)
        for v, evs in by_vid.items():
            evs.sort(key=lambda e: e["frame"])
            prev = None
            for idx, e in enumerate(evs):
                f = frame_features(e, prev, pbd_by, e.get("image_size"))
                if idx + 1 < len(evs):
                    f["gap"] = max(1, evs[idx + 1]["frame"] - e["frame"])
                all_feats.append(f)
                prev = np.asarray(e["boxes"], np.float64).reshape(-1, 4) \
                    if e["candidate_count"] else prev

    keys = ["n_cand", "iou_amb", "pbd_amb", "motion", "size", "gap", "sem_div"]
    q = {k: (np.percentile([f[k] for f in all_feats], 50)) for k in keys}
    print("global medians:", {k: round(v, 4) for k, v in q.items()})

    results = {}
    for dom in domains:
        cfg = DOMAINS[dom]
        with open(cfg["raw"], "rb") as f:
            raw = pickle.load(f)
        pbd_by = {}
        for fr in raw:
            pbd_by[(fr["video_id"], int(fr["frame"]))] = fr["pbd_be"]
        with open(cfg["manifest"]) as f:
            entries = [json.loads(l) for l in f]
        by_vid = defaultdict(list)
        for e in entries:
            by_vid[e["video_id"]].append(e)
        # preload tracker rows
        tr_rows = {}
        for v in by_vid:
            tr_rows[v] = {
                var: load_tracker_rows(os.path.join(cfg["trackers"], var), v)
                for var in cfg["variants"]}
        bucket_sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        n_windows = 0
        for v, evs in by_vid.items():
            evs.sort(key=lambda e: e["frame"])
            prev = None
            for i, e in enumerate(evs):
                f = frame_features(e, prev, pbd_by, e.get("image_size"))
                prev = np.asarray(e["boxes"], np.float64).reshape(-1, 4) \
                    if e["candidate_count"] else prev
                if i % args.stride != 0:
                    continue
                if e["candidate_count"] == 0:
                    continue
                # build window [i, i+H)
                gts = []
                for k in range(args.horizon):
                    if i + k >= len(evs):
                        break
                    e2 = evs[i + k]
                    gts.append([(gid, np.asarray(box, np.float64))
                                for gid, box in e2.get("gt_boxes", {}).items()])
                if not gts:
                    continue
                bucket = tuple(
                    "hi" if f[k] > q[k] else "lo" for k in keys)
                for var in cfg["variants"]:
                    rows = tr_rows[v][var]
                    if rows is None:
                        continue
                    dets = []
                    for k in range(len(gts)):
                        e2 = evs[i + k]
                        r = rows.get(int(e2["frame"]))
                        if r is None:
                            dets.append([])
                            continue
                        dets.append([(tid, box) for tid, box in r])
                    if not any(dets):
                        continue
                    w = windowed_metrics(dets, gts)
                    bucket_sums[bucket][var][0] += w["assa"]
                    bucket_sums[bucket][var][1] += 1
                n_windows += 1
                if n_windows >= args.max_windows:
                    break
            if n_windows >= args.max_windows:
                break
        out_buckets = {}
        for b, varstats in bucket_sums.items():
            out_buckets["/".join(b)] = {
                var: (s[0] / s[1]) if s[1] else None
                for var, s in varstats.items()}
        results[dom] = {
            "n_windows": n_windows,
            "buckets": out_buckets,
        }
        print(f"[{dom}] windows={n_windows} buckets={len(out_buckets)}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"medians": q, "results": results}, f, indent=2, default=str)
    print("saved", args.out)


if __name__ == "__main__":
    main()
