"""Stage L7: closed-set regression for the unified CLIP-front-end UIDM.

Same four-domain TrackEval protocol as eval_l6_uidm, but candidates and
appearance tokens come from outputs/l7/data/clip_closed (frozen CLIP),
so the shared app_dim=512 checkpoint can be evaluated on ordinary MOT.

Usage:
  python tools/eval_l7_closed_clip.py --tag unified \
      --ckpt outputs/l7/checkpoints/ovmot_joint/latest.pt \
      --out outputs/l7/trackeval/unified_regression --gpu 6
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l6_uidm import UIDM  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"

DOMAINS = {
    "dance": ("dancetrack_val",
              "outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl", 30),
    "bdd": ("bdd100k_train",
            "outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl", 5),
    "mot17": ("mot17_train",
              "outputs/l1_c/fixed_candidate_manifest/mot17_train.jsonl", 30),
    "mot20": ("mot20_train",
              "outputs/l1_c/fixed_candidate_manifest/mot20_train.jsonl", 30),
}
CLIP_DOMAINS = {
    "dance": "dancetrack_val",
    "bdd": "bdd100k_train",
    "mot17": "mot17_train",
    "mot20": "mot20_train",
}
SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def run_tracker(ckpt_path, data_dir, out_dir, gpu, new_margin=0.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    size = cfg.get("model", "large")
    model = UIDM(**SIZES[size],
                 no_interaction=cfg.get("no_interaction", False),
                 use_cue_rel=cfg.get("use_cue_rel", False),
                 app_dim=cfg.get("app_dim", 512)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    files = sorted(Path(data_dir).glob("**/*.pkl"))
    t0 = time.time()
    for vi, pkl_path in enumerate(files):
        rec = pickle.load(open(pkl_path, "rb"))
        tracker = OnlineTracker(variant="UIDM", uidm=model,
                                device=str(device),
                                output_all_candidates=True)
        tracker.uidm_new_margin = new_margin
        tracker.l1d_weights = (0.4, 0.2, 0.4)
        tracker.l1d_threshold = 0.25
        rows = []
        for fr in rec["frames"]:
            cands = []
            for j in range(len(fr["boxes"])):
                x1, y1, x2, y2 = [float(v) for v in fr["boxes"][j]]
                cands.append({
                    "box": [x1, y1, x2, y2],
                    "features": {
                        "pbd_be": np.asarray(fr["clip"][j], np.float32),
                        "gen": float(fr["gen"][j]),
                    },
                })
            tracker.image_size = rec["image_size"]
            outputs = tracker.process_frame(int(fr["frame"]), cands)
            for o in outputs:
                x1, y1, x2, y2 = o["box"]
                rows.append([fr["frame"], o["track_id"], x1, y1,
                             x2 - x1, y2 - y1,
                             float(o.get("score", 1.0)), -1, -1, -1])
        with open(out_dir / f"{rec['video_id']}.txt", "w") as f:
            for r in rows:
                f.write(",".join(
                    f"{v:.3f}" if isinstance(v, float) else str(v)
                    for v in r) + "\n")
        if (vi + 1) % 20 == 0 or vi + 1 == len(files):
            print(f"[l7closed] {vi+1}/{len(files)} elapsed={time.time()-t0:.0f}s",
                  flush=True)
    print("[l7closed] tracker done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--domains", default="dance,bdd,mot17,mot20")
    ap.add_argument("--new-margin", type=float, default=0.0)
    args = ap.parse_args()
    out = Path(args.out)
    tracker_root = out / "trackers"
    eval_root = out / "trackeval"
    os.makedirs(tracker_root, exist_ok=True)
    for key in args.domains.split(","):
        label, manifest, fps = DOMAINS[key.strip()]
        src_dir = tracker_root / label
        src_dir.mkdir(parents=True, exist_ok=True)
        dom = CLIP_DOMAINS[key.strip()]
        run_tracker(args.ckpt, f"outputs/l7/data/clip_eval/{dom}",
                    src_dir, args.gpu, args.new_margin)
        split = f"{args.tag}_{label}"
        variant_dir = eval_root / label / "U0"
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        for p in src_dir.glob("*.txt"):
            shutil.copyfile(p, variant_dir / p.name)
        subprocess.run(
            [PY, str(ROOT / "tools/run_l1d_trackeval.py"),
             "--split", split, "--manifest", manifest,
             "--tracker-root", str(eval_root / label),
             "--variants", "U0", "--fps", str(fps)], check=True)
    print("[l7closed] done", flush=True)


if __name__ == "__main__":
    main()
