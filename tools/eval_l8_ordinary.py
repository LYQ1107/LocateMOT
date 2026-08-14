"""Stage L8: ordinary MOT regression with the shared L8 checkpoint.

Same four-domain TrackEval protocol as L6/L7, but candidates carry both PBD
and CLIP tokens and the tracker uses the Unified Observation Adapter with a
fixed closed-set category spec.  Identity core and adapter come from one
shared L8 checkpoint.

Usage:
  python tools/eval_l8_ordinary.py --ckpt outputs/l8/checkpoints/.../latest.pt \
      --out outputs/l8/trackeval/... --gpu 0
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

from locatemot.models.l8_unified import L8UnifiedUIDM  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402
from tools.eval_l3 import build_candidates  # noqa: E402
from tools.train_l8_uidm import _specs  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
MANIFEST = ROOT / "outputs" / "l1_c" / "fixed_candidate_manifest"
CLIP_EVAL = ROOT / "outputs" / "l7" / "data" / "clip_eval"
DOMAINS = {
    "dance": ("dancetrack_val", "dancetrack_val", "dancetrack_val", 30,
              "person"),
    "bdd": ("bdd100k_train", "bdd100k_train", "bdd100k_train", 5,
            "person, car, truck, bus, rider, bicycle, motorcycle, train"),
    "mot17": ("mot17_train", "mot17_train", "mot17_train", 30, "person"),
    "mot20": ("mot20_train", "mot20_train", "mot20_train", 30, "person"),
}
SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def load_manifest(domain):
    by_video = {}
    with open(MANIFEST / f"{domain}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            by_video.setdefault(e["video_id"], []).append(e)
    for v in by_video:
        by_video[v].sort(key=lambda x: int(x["frame"]))
    return by_video


def run_tracker(model, by_video, clip_dir, out_dir, gpu, spec_emb):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    spec = torch.as_tensor(spec_emb[None], device=device)
    t0 = time.time()
    for vi, (vid, entries) in enumerate(by_video.items()):
        clip_rec = pickle.load(open(clip_dir / f"{vid}.pkl", "rb"))
        cf = {fr["frame"]: fr for fr in clip_rec["frames"]}
        tracker = OnlineTracker(
            variant="UIDM", uidm=model.uidm, device=str(device),
            output_all_candidates=True,
            uidm_adapter=model.adapter, uidm_spec=spec.cpu().numpy()[0])
        tracker.uidm_new_margin = 0.0
        tracker.l1d_weights = (0.4, 0.2, 0.4)
        tracker.l1d_threshold = 0.25
        rows = []
        for entry in entries:
            frame = int(entry["frame"])
            cands, image_size = build_candidates(entry)
            cfr = cf[frame]
            assert len(cands) == len(cfr["boxes"])
            for j, c in enumerate(cands):
                c["features"]["clip"] = np.asarray(cfr["clip"][j], np.float32)
            tracker.image_size = image_size
            outputs = tracker.process_frame(frame, cands)
            for o in outputs:
                x1, y1, x2, y2 = o["box"]
                rows.append([frame, o["track_id"], x1, y1, x2 - x1,
                             y2 - y1, float(o.get("score", 1.0)),
                             -1, -1, -1])
        with open(out_dir / f"{vid}.txt", "w") as f:
            for r in rows:
                f.write(",".join(
                    f"{v:.3f}" if isinstance(v, float) else str(v)
                    for v in r) + "\n")
        if (vi + 1) % 10 == 0 or vi + 1 == len(by_video):
            print(f"[l8ord] {vid} {vi+1}/{len(by_video)} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    print("[l8ord] tracker done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--domains", default="dance,bdd,mot17,mot20")
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model = L8UnifiedUIDM(**SIZES[cfg.get("model", "base")],
                          mode=cfg.get("mode", "unified"))
    model.load_state_dict(ck["model"])
    out = Path(args.out)
    tracker_root = out / "trackers"
    eval_root = out / "trackeval"
    for key in args.domains.split(","):
        label, manifest_dom, clip_dom, fps, spec_text = DOMAINS[key.strip()]
        src_dir = tracker_root / label
        src_dir.mkdir(parents=True, exist_ok=True)
        spec_emb = _specs([spec_text], device="cpu")[0]
        by_video = load_manifest(manifest_dom)
        if args.max_videos:
            by_video = dict(list(by_video.items())[:args.max_videos])
        run_tracker(model, by_video, CLIP_EVAL / clip_dom, src_dir,
                    args.gpu, spec_emb)
        split = f"{Path(args.out).name}_{label}"
        variant_dir = eval_root / label / "U0"
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        for p in src_dir.glob("*.txt"):
            shutil.copyfile(p, variant_dir / p.name)
        subprocess.run(
            [PY, str(ROOT / "tools/run_l1d_trackeval.py"),
             "--split", split, "--manifest",
             str(MANIFEST / f"{manifest_dom}.jsonl"),
             "--tracker-root", str(eval_root / label),
             "--variants", "U0", "--fps", str(fps)], check=True)
    print("[l8ord] done", flush=True)


if __name__ == "__main__":
    main()
