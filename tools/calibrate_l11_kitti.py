"""Stage L11: calibrate query-conditioned CLIP candidate filtering.

Uses only official Refer-KITTI-V2 TRAIN sequences (never the 4 eval
sequences 0005/0011/0013/0019).  For each expression-frame, candidates
are ranked by CLIP crop-sentence similarity; we sweep per-frame top-k
and minimum similarity and report query precision and target recall.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.build_l10_refer_kitti import load_labels  # noqa: E402

DATA = ROOT / "outputs" / "l11" / "data" / "rmot_kitti"
EVAL_SEQS = {"0005", "0011", "0013", "0019"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "l11"
                                         / "kitti_calibration.json"))
    ap.add_argument("--max-seqs", type=int, default=0)
    args = ap.parse_args()
    exp_meta = json.loads((DATA / "expressions.json").read_text())
    seqs = [s for s in sorted(exp_meta) if s not in EVAL_SEQS]
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    groups = []  # per (seq, expr, frame): (sims, positives, target ids)
    for seq in seqs:
        rec = pickle.load(open(DATA / f"{seq}.pkl", "rb"))
        frames = {fr["frame"]: fr for fr in rec["frames"]}
        for e in exp_meta[seq]:
            spec = np.asarray(e["spec"], np.float32)
            spec = spec / (np.linalg.norm(spec) + 1e-9)
            for fs, ids in e["label"].items():
                fr = frames.get(int(fs))
                if fr is None:
                    continue
                n = len(fr["boxes"])
                if n == 0:
                    continue
                clip = np.asarray(fr["clip"], np.float32)
                clip = clip / (np.linalg.norm(clip, axis=1, keepdims=True)
                               + 1e-9)
                sims = clip @ spec
                tgt = {str(x) for x in ids}
                pos = np.asarray(
                    [1.0 if (g is not None and g in tgt) else 0.0
                     for g in fr["cand_gt"]], np.float32)
                groups.append((sims, pos, tgt, fr["cand_gt"]))
    print(f"[l11calib] groups={len(groups)}", flush=True)
    table = []
    for topk in (3, 5, 8, 10, 12, 15, 20):
        for cmin in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            tot_c = tot_tp = 0
            tot_targets = 0
            tot_target_hit = 0
            for sims, pos, tgt, cand_gt in groups:
                keep = sims >= cmin
                if keep.sum() > topk:
                    idx = np.argsort(-sims[keep])[:topk]
                    kk = np.nonzero(keep)[0][idx]
                else:
                    kk = np.nonzero(keep)[0]
                if len(kk) == 0:
                    continue
                tot_c += len(kk)
                tot_tp += int(pos[kk].sum())
                hit_targets = {g for j in kk
                               if (g := cand_gt[j]) is not None and g in tgt}
                tot_target_hit += len(hit_targets)
                tot_targets += len(tgt)
            prec = tot_tp / max(1, tot_c)
            rec = tot_target_hit / max(1, tot_targets)
            table.append({"topk": topk, "cmin": cmin, "cands": tot_c,
                          "query_prec": round(prec, 4),
                          "target_recall": round(rec, 4)})
    best = None
    for r in table:
        if r["query_prec"] >= 0.10:
            if best is None or r["target_recall"] > best["target_recall"]:
                best = r
    print("topk cmin cands qprec trec")
    for r in table:
        print(r["topk"], r["cmin"], r["cands"], r["query_prec"],
              r["target_recall"])
    print("BEST:", best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"table": table, "best": best}, f, indent=2)


if __name__ == "__main__":
    main()
