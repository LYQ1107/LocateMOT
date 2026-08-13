"""Stage L6 UIDM: fresh-tracker TrackEval across heterogeneous domains.

Usage:
  python tools/eval_l6_uidm.py --tag uidm_pilot \
      --ckpt outputs/l6/checkpoints/uidm_pilot/latest.pt --model-size base \
      --out outputs/l6/trackeval/uidm_pilot --gpu 7
"""
from __future__ import annotations

import argparse
import json
import os
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
from tools.eval_l3 import build_candidates  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"

DOMAINS = {
    "dance": ("dancetrack_val",
              "outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl",
              30),
    "bdd": ("bdd100k_train",
            "outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl",
            5),
    "mot17": ("mot17_train",
              "outputs/l1_c/fixed_candidate_manifest/mot17_train.jsonl",
              30),
    "mot20": ("mot20_train",
              "outputs/l1_c/fixed_candidate_manifest/mot20_train.jsonl",
              30),
    "tao": ("tao_amodal_train",
            "outputs/l4/manifests/tao_amodal_train_l4.jsonl",
            1),
}

SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def run_tracker(ckpt, model_size, manifest, out_dir, gpu, new_margin=0.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    size = model_size or cfg.get("model", "base")
    model = UIDM(**SIZES[size],
                 no_interaction=cfg.get("no_interaction", False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    by_video = {}
    with open(manifest) as f:
        for line in f:
            e = json.loads(line)
            by_video.setdefault(e["video_id"], []).append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: int(e["frame"]))
    t0 = time.time()
    for vi, (vid, entries) in enumerate(by_video.items()):
        tracker = OnlineTracker(variant="UIDM", uidm=model,
                                device=str(device),
                                output_all_candidates=True)
        tracker.uidm_new_margin = new_margin
        tracker.l1d_weights = (0.4, 0.2, 0.4)
        tracker.l1d_threshold = 0.25
        rows = []
        for entry in entries:
            cands, image_size = build_candidates(entry)
            tracker.image_size = image_size
            outputs = tracker.process_frame(int(entry["frame"]), cands)
            for o in outputs:
                x1, y1, x2, y2 = o["box"]
                rows.append([entry["frame"], o["track_id"], x1, y1,
                             x2 - x1, y2 - y1,
                             float(o.get("score", 1.0)), -1, -1, -1])
        with open(out_dir / f"{vid}.txt", "w") as f:
            for r in rows:
                f.write(",".join(
                    f"{v:.3f}" if isinstance(v, float) else str(v)
                    for v in r) + "\n")
        if (vi + 1) % 10 == 0 or vi + 1 == len(by_video):
            print(f"[eval_l6] {vid} {vi+1}/{len(by_video)} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    print("[eval_l6] tracker done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-size", default=None,
                    choices=["small", "base", "large"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--domains", default="dance,bdd,mot17,mot20")
    ap.add_argument("--new-margin", type=float, default=0.0)
    args = ap.parse_args()
    out = ROOT / args.out
    tracker_root = out / "trackers"
    eval_root = out / "trackeval"
    os.makedirs(tracker_root, exist_ok=True)
    ck_path = Path(args.ckpt)
    if not ck_path.exists():
        raise SystemExit(f"checkpoint not found: {ck_path}")
    print(f"[eval_l6] tag={args.tag} ckpt={ck_path.resolve()}", flush=True)
    for key in args.domains.split(","):
        label, manifest, fps = DOMAINS[key.strip()]
        split = f"{args.tag}_{label}"
        src_dir = tracker_root / label
        src_dir.mkdir(parents=True, exist_ok=True)
        run_tracker(args.ckpt, args.model_size, manifest, src_dir, args.gpu,
                    args.new_margin)
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
             "--variants", "U0", "--fps", str(fps)],
            check=True)
    print("[eval_l6] done", flush=True)


if __name__ == "__main__":
    main()
