"""Stage L5 Route A: official TrackEval for the temporal identity model.

Usage:
  python tools/eval_l5_route_a.py --tag route_a_base \
      --ckpt outputs/l5/checkpoints/route_a_base/final.pt \
      --model-size base --out outputs/l5/trackeval/route_a_base --gpu 6
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
    print("[eval_l5] ", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-size", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--domains", default="dance,bdd,mot17,mot20")
    args = ap.parse_args()
    out = ROOT / args.out
    tracker_root = out / "trackers"
    eval_root = out / "trackeval"
    domain_map = {"dance": "dance_l3", "bdd": "bdd_l3",
                  "mot17": "mot17_l3", "mot20": "mot20_l3"}
    for key in args.domains.split(","):
        label = domain_map[key.strip()]
        domain, manifest, fps = DOMAINS[label]
        split = f"{args.tag}_{label}"
        src_dir = tracker_root / label
        run([PY, ROOT / "tools/eval_l3.py", "--model", "l5",
             "--model-size", args.model_size,
             "--ckpt", args.ckpt, "--manifest", manifest,
             "--out", src_dir, "--gpu", args.gpu])
        variant_dir = eval_root / label / "U0"
        variant_dir.mkdir(parents=True, exist_ok=True)
        for p in src_dir.glob("*.txt"):
            dst = variant_dir / p.name
            if not dst.exists():
                shutil.copyfile(p, dst)
        run([PY, ROOT / "tools/run_l1d_trackeval.py", "--split", split,
             "--manifest", manifest, "--tracker-root", eval_root / label,
             "--variants", "U0", "--fps", str(fps)])
    print("[eval_l5] done", flush=True)


if __name__ == "__main__":
    main()
