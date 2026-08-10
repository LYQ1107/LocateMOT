"""Stage L4: official TrackEval (ALL mode) for trained spec-eq models.

For each tag (a2/a5/u0) and each AC domain, runs the OnlineTracker shell
(eval_l3) with the shared checkpoint, lays out MOTChallenge files, and runs
official TrackEval (run_l1d_trackeval).

Usage:
  python tools/eval_l4_ac.py --tag a5 \
      --ckpt outputs/l4/checkpoints/a5/final.pt \
      --out outputs/l4/trackeval/a5 --gpu 9
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"

DOMAINS = {
    "dance_l3": ("dancetrack_val",
                 "outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl",
                 30),
    "bdd_l3": ("bdd100k_train",
               "outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl",
               5),
    "mot17_l3": ("mot17_train",
                 "outputs/l1_c/fixed_candidate_manifest/mot17_train.jsonl",
                 30),
    "mot20_l3": ("mot20_train",
                 "outputs/l1_c/fixed_candidate_manifest/mot20_train.jsonl",
                 30),
}


def run(cmd):
    print("[eval_l4_ac]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=9)
    args = ap.parse_args()
    out = ROOT / args.out
    tracker_root = out / "trackers"
    eval_root = out / "trackeval"
    for label, (domain, manifest, fps) in DOMAINS.items():
        src_dir = tracker_root / label
        run([PY, ROOT / "tools/eval_l3.py", "--model", "u0",
             "--ckpt", args.ckpt, "--manifest", manifest,
             "--out", src_dir, "--gpu", args.gpu])
        variant_dir = eval_root / label / "U0"
        variant_dir.mkdir(parents=True, exist_ok=True)
        for p in src_dir.glob("*.txt"):
            dst = variant_dir / p.name
            if not dst.exists():
                shutil.copyfile(p, dst)
        run([PY, ROOT / "tools/run_l1d_trackeval.py", "--split", label,
             "--manifest", manifest, "--tracker-root", eval_root / label,
             "--variants", "U0", "--fps", str(fps)])
    print("[eval_l4_ac] done", flush=True)


if __name__ == "__main__":
    main()
