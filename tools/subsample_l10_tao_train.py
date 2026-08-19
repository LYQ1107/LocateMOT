"""Stage L10: cap hard negatives per frame in the TAO-train OVMOT stream.

Keeps ALL GT-matched candidates (positives) plus the top-K unmatched
candidates by detection score (hard negatives), preserving the original
detection order.  This keeps ~100% of the positive supervision while
making the LocateAnything crop-PBD cache tractable (the bottleneck of the
L10 pipeline).  Evaluation remains on the full TAO-val candidate protocol.

Usage:
  python tools/subsample_l10_tao_train.py --keep-unmatched 16
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
DATA = ROOT / "outputs" / "l10" / "data" / "tao_train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-unmatched", type=int, default=16)
    args = ap.parse_args()
    K = args.keep_unmatched
    index = json.loads((DATA / "index.json").read_text())
    n_videos = n_frames = n_cand = n_keep = n_pos = 0
    for vname, info in index["videos"].items():
        p = Path(info["path"])
        rec = pickle.load(open(p, "rb"))
        changed = False
        for fr in rec["frames"]:
            n = len(fr["boxes"])
            if n == 0:
                continue
            n_cand += n
            keep = []
            pos = 0
            # score-descending rank among unmatched
            order = sorted(range(n),
                           key=lambda j: float(fr["gen"][j]), reverse=True)
            unmatched_seen = 0
            for j in order:
                if fr["cand_gt"][j] is not None:
                    keep.append(j)
                    pos += 1
                elif unmatched_seen < K:
                    keep.append(j)
                    unmatched_seen += 1
            pos = sum(1 for j in keep if fr["cand_gt"][j] is not None)
            keep = sorted(keep)
            if len(keep) == n:
                n_keep += n
                continue
            for k in ("boxes", "gen", "label", "clip", "pbd"):
                fr[k] = np.asarray(fr[k])[keep]
            fr["cand_gt"] = [fr["cand_gt"][j] for j in keep]
            changed = True
            n_keep += len(keep)
            n_pos += pos
        if changed:
            tmp = p.with_suffix(".tmp2")
            with open(tmp, "wb") as f:
                pickle.dump(rec, f)
            os.replace(tmp, p)
        n_videos += 1
        n_frames += len(rec["frames"])
    print(f"[subsample] K={K} videos={n_videos} frames={n_frames} "
          f"before={n_cand} after={n_keep} pos_kept={n_pos} "
          f"cands/frame={n_keep/max(1,n_frames):.2f}")


if __name__ == "__main__":
    main()
