"""Stage L3: run U0/U1 AC tracker on a domain manifest and write outputs.

Usage:
  python tools/eval_l3.py --model u1 --ckpt outputs/l3/checkpoints/u1/final.pt \
      --manifest outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl \
      --out outputs/l3/trackers/u1/dancetrack_val --gpu 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import read_frame_cache  # noqa: E402
from locatemot.models.l1d_association import L1DAssociator  # noqa: E402
from locatemot.models.l4_spec_eq import L4SpecEqAssociator  # noqa: E402
from locatemot.models.l3_unified import L3Associator  # noqa: E402
from locatemot.models.l5_route_a import L5TemporalAssociator  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402


def build_candidates(entry):
    root = entry["cache_root"]
    key = entry.get("cache_key") or (
        f"{entry['dataset']}/{entry['video_id']}/{int(entry['frame']):05d}/{entry['protocol']}")
    fr = read_frame_cache(root, key)
    if fr is None:
        return [], entry.get("image_size", [1280, 720])
    feats = fr["features"]
    boxes = np.asarray(feats.get("boxes", np.zeros((0, 4))), dtype=np.float64)
    n = len(boxes)
    cands = []
    for i in range(n):
        f = {
            "pbd": np.asarray(feats["pbd_coord_mean_last"][i], dtype=np.float32)
            if "pbd_coord_mean_last" in feats and len(feats["pbd_coord_mean_last"]) > i
            else np.zeros(2048, np.float32),
            "pbd_be": np.asarray(feats["pbd_box_end_last"][i], dtype=np.float32)
            if "pbd_box_end_last" in feats and len(feats["pbd_box_end_last"]) > i
            else np.zeros(2048, np.float32),
            "region": np.asarray(feats["region"][i], dtype=np.float32)
            if "region" in feats and len(feats["region"]) > i else np.zeros(4608, np.float32),
            "geom": np.asarray(feats["geometry"][i], dtype=np.float32)
            if "geometry" in feats and len(feats["geometry"]) > i
            else np.zeros(5, np.float32),
            "gen": float(feats["gen_score"][i]) if "gen_score" in feats
            and len(feats["gen_score"]) > i else 0.0,
        }
        cands.append({"box": boxes[i], "features": f, "index": i})
    return cands, entry.get("image_size", [1280, 720])


def load_model(model_type, ckpt, device):
    if model_type == "u0":
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = ck["model"] if "model" in ck else ck
        if "spec_embed.weight" in state:
            model = L4SpecEqAssociator(n_spec=3, d_spec=16)
        else:
            model = L1DAssociator()
        model.load_state_dict(state)
    else:
        model = L3Associator(use_spec=False)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = ck["model"] if "model" in ck else ck
        model.load_state_dict(state)
    return model.to(device).eval()


def load_l5(ckpt, model_size, device):
    sizes = {
        "small": dict(d_model=128, temporal_layers=2, set_layers=2,
                      n_heads=4, ffn_dim=512),
        "base": dict(d_model=256, temporal_layers=4, set_layers=4,
                     n_heads=8, ffn_dim=1024),
        "large": dict(d_model=384, temporal_layers=6, set_layers=6,
                      n_heads=8, ffn_dim=1536),
    }
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model_size = model_size or cfg.get("model", "base")
    model = L5TemporalAssociator(
        **sizes[model_size],
        delta_scale=cfg.get("delta_scale", 0.6)).to(device)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["u0", "u1", "l5"], required=True)
    ap.add_argument("--model-size", default=None,
                    choices=["small", "base", "large"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=8)
    ap.add_argument("--delta-scale", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "l5":
        model = load_l5(args.ckpt, args.model_size, device)
        variant = "L5"
    else:
        model = load_model(args.model, args.ckpt, device)
        variant = "L1D"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_video = {}
    with open(args.manifest) as f:
        for line in f:
            e = json.loads(line)
            by_video.setdefault(e["video_id"], []).append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])
    t0 = time.time()
    for vid, entries in by_video.items():
        tracker = OnlineTracker(variant=variant,
                                l1d=model if variant == "L1D" else None,
                                l5=model if variant == "L5" else None,
                                device=str(device),
                                output_all_candidates=True)
        tracker.l1d_weights = (0.4, 0.2, 0.4)
        tracker.l1d_threshold = args.threshold
        tracker.l1d_delta_scale = args.delta_scale
        tracker.l1d_rel_threshold = 0.0
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
        with open(out / f"{vid}.txt", "w") as f:
            for r in rows:
                f.write(",".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                 for v in r) + "\n")
        print(f"[eval_l3 {args.model}] {vid} frames={len(entries)} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
    print("[eval_l3] done", flush=True)


if __name__ == "__main__":
    main()
