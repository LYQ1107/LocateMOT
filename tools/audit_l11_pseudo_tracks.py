"""Stage L11: quality audit of pseudo-tracklets (5-10 videos).

Evaluation-only use of C-TAO base_and_novel annotations (never used for
pseudo supervision): latent GT identity of an unmatched candidate is the
C-TAO track with max IoU >= LATENT_IOU at the same frame.

The precision check uses `link_id` (raw linker output before the
GT-overlap exclusion), so it measures the linker itself on the
GT-covered subset.  Training uses the more conservative `pseudo_id`.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from tools.build_l11_pseudo_tracks import (  # noqa: E402
    DEFAULT_GT_JSON, load_gt, iou)

DEFAULT_DATA = ROOT / "outputs" / "l10" / "data" / "tao_train"
DEFAULT_PSEUDO = ROOT / "outputs" / "l11" / "data" / "pseudo_tracks"
LATENT_IOU = 0.40


def audit_video(name, rec, side, latent, scene_by_key):
    out = {"name": name, "n_frames": len(rec["frames"]),
           "pseudo_cands": 0, "link_cands": 0, "new": 0, "existing": 0,
           "tracklets": 0, "lengths": [], "cycle": [], "mean_app": [],
           "dup_identity_frames": 0, "pseudo_frames": 0,
           "pairs_same": 0, "pairs_total": 0,
           "cand_majority_ok": 0, "cand_latent": 0}
    scene_dir = scene_by_key.get(rec["video_id"][len("train-"):])
    # latent GT per frame per candidate
    latent_by_cand = []
    for fr in rec["frames"]:
        fn = (f"train/{scene_dir}/frame{int(fr['frame']):04d}.jpg"
              if scene_dir else None)
        gts = latent.get(fn, []) if fn else []
        lat = []
        for b in fr["boxes"]:
            best, best_iou = None, 0.0
            for gid, gb in gts:
                v = iou(b, gb)
                if v > best_iou:
                    best_iou, best = v, gid
            lat.append(best if best_iou >= LATENT_IOU else None)
        latent_by_cand.append(lat)
    tr_by_id = {t["id"]: t for t in side.get("tracklet_stats", [])}
    seen = set()
    for fi, fr in enumerate(rec["frames"]):
        sc = side["frames"][fi]
        active = set()
        lat_counts = defaultdict(int)
        for j, pid in enumerate(sc["link_id"]):
            if pid is None:
                continue
            out["link_cands"] += 1
            active.add(pid)
            lat = latent_by_cand[fi][j]
            if lat is not None:
                out["cand_latent"] += 1
                lat_counts[lat] += 1
            if sc["pseudo_id"][j] is not None:
                out["pseudo_cands"] += 1
        if active:
            out["pseudo_frames"] += 1
            if any(c >= 2 for c in lat_counts.values()):
                out["dup_identity_frames"] += 1
        # per-tracklet stats once
        for j, pid in enumerate(sc["link_id"]):
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            tr = tr_by_id.get(pid)
            if tr is None:
                continue
            out["tracklets"] += 1
            out["new"] += 1
            out["existing"] += max(0, tr["len"] - 1)
            out["lengths"].append(tr["len"])
            out["cycle"].append(tr["cycle_rate"])
            out["mean_app"].append(tr["mean_app"])
            with_lat = []
            for fi2, fr2 in enumerate(rec["frames"]):
                sc2 = side["frames"][fi2]
                for j2, pid2 in enumerate(sc2["link_id"]):
                    if pid2 == pid and latent_by_cand[fi2][j2] is not None:
                        with_lat.append(latent_by_cand[fi2][j2])
            if with_lat:
                maj = Counter(with_lat).most_common(1)[0][0]
                out["cand_majority_ok"] += sum(1 for x in with_lat
                                               if x == maj)
            n_pair = len(with_lat) * (len(with_lat) - 1) // 2
            same = sum(1 for i in range(len(with_lat))
                       for j in range(i + 1, len(with_lat))
                       if with_lat[i] == with_lat[j])
            out["pairs_total"] += n_pair
            out["pairs_same"] += same
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--max-videos", type=int, default=8)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--pseudo-dir", default=str(DEFAULT_PSEUDO))
    ap.add_argument("--novel-json", default=DEFAULT_GT_JSON)
    ap.add_argument("--out", default=str(ROOT / "results" / "l11"
                                         / "pseudo_quality.json"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    pseudo_dir = Path(args.pseudo_dir)
    index = json.loads((data_dir / "index.json").read_text())
    names = sorted(index["videos"].keys())
    if args.videos:
        names = [n for n in names if n in set(args.videos)]
    names = names[:args.max_videos]
    print("[l11audit] loading base_and_novel GT ...", flush=True)
    latent, scenes = load_gt()
    print(f"[l11audit] latent files={len(latent)} scenes={len(scenes)}",
          flush=True)
    rows = []
    for name in names:
        rec = pickle.load(open(data_dir / f"{name}.pkl", "rb"))
        side = pickle.load(open(pseudo_dir / f"{name}.pkl", "rb"))
        rows.append(audit_video(name, rec, side, latent, scenes))
    keys = ["pseudo_cands", "link_cands", "new", "existing", "tracklets",
            "pseudo_frames", "dup_identity_frames", "pairs_same",
            "pairs_total", "cand_majority_ok", "cand_latent"]
    agg = {"videos": len(rows)}
    for k in keys:
        agg[k] = sum(r[k] for r in rows)
    lens = [x for r in rows for x in r["lengths"]]
    cyc = [x for r in rows for x in r["cycle"]]
    app = [x for r in rows for x in r["mean_app"]]
    agg["mean_len"] = float(np.mean(lens)) if lens else 0.0
    agg["median_len"] = float(np.median(lens)) if lens else 0.0
    agg["mean_cycle"] = float(np.mean(cyc)) if cyc else 0.0
    agg["mean_app"] = float(np.mean(app)) if app else 0.0
    agg["pair_same_precision"] = (agg["pairs_same"] / agg["pairs_total"]
                                  if agg["pairs_total"] else 0.0)
    agg["majority_precision"] = (agg["cand_majority_ok"] / agg["cand_latent"]
                                 if agg["cand_latent"] else 0.0)
    agg["new_rate"] = agg["new"] / max(1, agg["new"] + agg["existing"])
    agg["unique_id_ratio"] = agg["tracklets"] / max(1, agg["link_cands"])
    agg["dup_identity_rate"] = (agg["dup_identity_frames"]
                                / max(1, agg["pseudo_frames"]))
    agg["latent_coverage"] = agg["cand_latent"] / max(1, agg["link_cands"])
    print(json.dumps(agg, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"aggregate": agg, "per_video": rows}, f, indent=2)


if __name__ == "__main__":
    main()
