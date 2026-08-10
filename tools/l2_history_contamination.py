"""Stage L2: history contamination audit.

Replays L1DK base and EGRA (L1DK_d03) on the same AC shell, classifies every
base->EGRA correction, and cross-tabulates local helpfulness with the
pre-correction predicted-trajectory contamination state (purity, past IDSW,
age, fragments, dominant GT).

Usage:
  python tools/l2_history_contamination.py \
      --raw outputs/l1_d/raw/dancetrack_val.pkl \
      --domain dancetrack_val --out outputs/l2/oracle
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

from tools.run_l2_oracle import (  # noqa: E402
    L2Track,
    apply_assignment,
    replay_video,
    track_contamination,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", default="outputs/l2/oracle")
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    import torch
    from locatemot.models.l1d_association import L1DAssociator
    device = torch.device("cpu")
    model = L1DAssociator()
    ck = torch.load(
        os.path.join(ROOT, "outputs/l1_d/checkpoints/l1d_k/final.pt"),
        map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    with open(args.raw, "rb") as f:
        frames = pickle.load(f)
    by_video = defaultdict(list)
    for fr in frames:
        by_video[fr["video_id"]].append(fr)
    vids = sorted(by_video)
    if args.max_videos:
        vids = vids[: args.max_videos]

    rows = []
    for vid in vids:
        records, _ = replay_video(by_video[vid], egra_model=model, device=device)
        for r in records:
            if r["base"] is None or "egra_assigns" not in r:
                continue
            bm = {ti: ci for ti, ci in r["assigns"]}
            em = {ti: ci for ti, ci in r["egra_assigns"]}
            all_t = sorted(set(bm) | set(em))
            for ti in all_t:
                if bm.get(ti) == em.get(ti):
                    continue
                cont = r["contam"][ti]
                snap = r["track_snaps"][ti]
                base_gt = None if bm.get(ti) is None else r["cand_gt"][bm[ti]]
                egra_gt = None if em.get(ti) is None else r["cand_gt"][em[ti]]
                # local classification relative to the track's birth identity
                if egra_gt == snap["true_gt"] and base_gt != snap["true_gt"]:
                    kind = "helpful"
                elif base_gt == snap["true_gt"] and egra_gt != snap["true_gt"]:
                    kind = "harmful"
                elif egra_gt == base_gt:
                    kind = "same_gt"
                elif egra_gt == snap["true_gt"] or egra_gt == cont["dominant_gt"]:
                    kind = "helpful"
                else:
                    kind = "other"
                rows.append({
                    "video": vid,
                    "frame": r["frame"],
                    "kind": kind,
                    "true_gt": snap["true_gt"],
                    "dominant_gt": cont["dominant_gt"],
                    "current_gt": cont["current_gt"],
                    "purity": cont["purity"],
                    "n_gt_hits": cont["n_gt_hits"],
                    "past_idsw": cont["past_idsw"],
                    "fragments": cont["fragments"],
                    "age": cont["age"],
                    "hits": cont["hits"],
                    "base_gt": base_gt,
                    "egra_gt": egra_gt,
                })
        print(f"[{args.domain}] {vid} corrections={sum(1 for r in rows if r['video']==vid)}",
              flush=True)

    agg = {
        "domain": args.domain,
        "n_videos": len(vids),
        "n_corrections": len(rows),
        "by_kind": dict(sorted(
            (k, sum(1 for r in rows if r["kind"] == k))
            for k in set(r["kind"] for r in rows))),
        "purity": {},
        "idsw": {},
        "age": {},
    }

    def bucket_table(key, bins, labels):
        out = {}
        for lab in labels:
            out[lab] = {"n": 0, "helpful": 0, "harmful": 0}
        for r in rows:
            v = r[key]
            lab = labels[-1]
            for b, l in zip(bins, labels):
                if v <= b:
                    lab = l
                    break
            out[lab]["n"] += 1
            if r["kind"] == "helpful":
                out[lab]["helpful"] += 1
            elif r["kind"] == "harmful":
                out[lab]["harmful"] += 1
        return out

    agg["purity"] = bucket_table("purity", [0.5, 0.8, 0.95],
                                 ["<0.5", "0.5-0.8", "0.8-0.95", ">=0.95"])
    agg["idsw"] = bucket_table("past_idsw", [0, 1, 3],
                               ["0", "1", "2-3", ">=4"])
    agg["age"] = bucket_table("age", [3, 10, 30], ["1-3", "4-10", "11-30", ">30"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"contamination_{args.domain}.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)
    with open(os.path.join(args.out, f"corrections_{args.domain}.pkl"), "wb") as f:
        pickle.dump(rows, f, protocol=4)
    print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()
