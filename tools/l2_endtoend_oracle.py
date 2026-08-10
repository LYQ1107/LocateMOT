"""Stage L2: end-to-end greedy oracle (receding-horizon privileged controller).

At every conflict frame the controller enumerates candidate actions, rolls
each forward H frames with the frozen base policy, and applies the action
with the highest windowed AssA.  Non-conflict frames use the base action.
This gives a practical upper bound on what a perfect causal utility student
could achieve end-to-end (modulo receding-horizon approximation).

Usage:
  python tools/l2_endtoend_oracle.py \
      --raw outputs/l1_d/raw/dancetrack_val.pkl \
      --domain dancetrack_val --out outputs/l2/oracle \
      --videos 6 --horizon 16 --tracker-dir outputs/l2/oracle/trackers/ORACLE_H16
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.run_l2_oracle import (  # noqa: E402
    L2Track,
    apply_assignment,
    complete_assignment,
    conflict_components,
    generate_actions,
    run_base_frame,
    windowed_metrics,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", default="outputs/l2/oracle")
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--max-actions", type=int, default=6)
    ap.add_argument("--max-components", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--tracker-dir", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    with open(args.raw, "rb") as f:
        frames = pickle.load(f)
    by_video = defaultdict(list)
    for fr in frames:
        by_video[fr["video_id"]].append(fr)
    vids = sorted(by_video)[: args.videos]

    results = []
    t0 = time.time()
    for vi, vid in enumerate(vids):
        vframes = by_video[vid]
        tracks = []
        next_tid = 1
        base_rows = []  # (frame, tid, box) per candidate
        oracle_rows = []
        n_conflict = 0
        for idx, fr in enumerate(vframes):
            tracks[:] = [t for t in tracks if t.status != "TERMINATED"]
            cands = [{
                "box": np.asarray(fr["boxes"][i], np.float64),
                "pbd": np.asarray(fr["pbd_be"][i], np.float32),
                "gen": float(fr["gen"][i]),
                "gt": fr["cand_gt"][i],
            } for i in range(len(fr["boxes"]))]
            assigns, base, feats, pred_boxes = run_base_frame(
                tracks, cands, fr["image_size"], cur_frame=fr["frame"])
            comps = conflict_components(base, 0.25) if base is not None else []
            chosen = dict()
            if comps:
                n_conflict += 1
                for comp in comps[: args.max_components]:
                    actions = generate_actions(
                        base, assigns, comp,
                        [c["gt"] for c in cands],
                        [t.true_gt for t in tracks],
                        rng=rng)[: args.max_actions]
                    best = None
                    best_u = -1.0
                    for act in actions:
                        tr = [L2Track.from_snapshot(t.snapshot()) for t in tracks]
                        ca = [dict(c) for c in cands]
                        full = complete_assignment(base, act, assigns)
                        nid, _ = apply_assignment(
                            tr, ca, full, fr["frame"], 10_000_000)
                        dets, gts = [], []
                        for k in range(1, args.horizon + 1):
                            if idx + k >= len(vframes):
                                break
                            f2 = vframes[idx + k]
                            c2 = [{
                                "box": np.asarray(f2["boxes"][i], np.float64),
                                "pbd": np.asarray(f2["pbd_be"][i], np.float32),
                                "gen": float(f2["gen"][i]),
                                "gt": f2["cand_gt"][i],
                            } for i in range(len(f2["boxes"]))]
                            a2, b2, _, _ = run_base_frame(
                                tr, c2, f2["image_size"], cur_frame=f2["frame"])
                            tid_map = {ci: tr[ti].tid for ti, ci in a2}
                            nid, born = apply_assignment(
                                tr, c2, a2, f2["frame"], nid)
                            tid_map.update(born)
                            dets.append(
                                [(tid_map[ci], c["box"].copy())
                                 for ci, c in enumerate(c2)])
                            gts.append(
                                [(gid, np.asarray(b, np.float64))
                                 for gid, b in f2.get("gt_boxes", {}).items()])
                        w = windowed_metrics(dets, gts)
                        if w["assa"] > best_u:
                            best_u = w["assa"]
                            best = act
                    if best is not None:
                        for ti, ci in best:
                            chosen[ti] = ci
            # build final assignment: chosen edges + base fill
            fixed = [(ti, ci) for ti, ci in chosen.items()]
            final = complete_assignment(base, fixed, assigns)
            base_tid = {}
            for ti, ci in assigns:
                base_tid[ci] = tracks[ti].tid
            next_tid, born = apply_assignment(
                tracks, cands, final, fr["frame"], next_tid)
            tid_map = {}
            for ti, ci in final:
                tid_map[ci] = tracks[ti].tid
            tid_map.update(born)
            for ci, c in enumerate(cands):
                base_rows.append((fr["frame"], base_tid.get(ci, -1),
                                  c["box"].copy()))
                oracle_rows.append((fr["frame"], tid_map[ci], c["box"].copy()))
        # assemble per-frame dets from rows
        fr_list = [f["frame"] for f in vframes]
        db, dob = [], []
        gb = []
        for fr in fr_list:
            db.append([(tid, box) for f, tid, box in base_rows if f == fr])
            dob.append([(tid, box) for f, tid, box in oracle_rows if f == fr])
            gb.append([(gid, np.asarray(b, np.float64))
                       for gid, b in next(f["gt_boxes"] for f in vframes
                                         if f["frame"] == fr).items()])
        wb = windowed_metrics(db, gb)
        wo = windowed_metrics(dob, gb)
        results.append({
            "video": vid, "n_frames": len(vframes), "n_conflict": n_conflict,
            "base": wb, "oracle": wo,
            "gain_assa": wo["assa"] - wb["assa"],
            "gain_idf1": wo["idf1"] - wb["idf1"],
            "idsw_base": wb["idsw"], "idsw_oracle": wo["idsw"],
        })
        if args.tracker_dir:
            os.makedirs(args.tracker_dir, exist_ok=True)
            with open(os.path.join(args.tracker_dir, f"{vid}.txt"), "w") as f:
                for r in oracle_rows:
                    x1, y1, x2, y2 = r[2]
                    f.write(f"{r[0]},{r[1]},{x1:.3f},{y1:.3f},"
                            f"{x2-x1:.3f},{y2-y1:.3f},1.0,-1,-1,-1\n")
        print(f"[{args.domain}] {vid} conflicts={n_conflict} "
              f"gain={wo['assa']-wb['assa']:+.4f} "
              f"idsw {wb['idsw']}->{wo['idsw']} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    agg = {
        "domain": args.domain,
        "n_videos": len(results),
        "horizon": args.horizon,
        "per_video": results,
        "mean_gain_assa": float(np.mean([r["gain_assa"] for r in results])),
        "mean_gain_idf1": float(np.mean([r["gain_idf1"] for r in results])),
        "sum_idsw_base": int(sum(r["idsw_base"] for r in results)),
        "sum_idsw_oracle": int(sum(r["idsw_oracle"] for r in results)),
    }
    with open(os.path.join(args.out, f"e2e_{args.domain}.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()
