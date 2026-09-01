"""Stage L1-C: run association methods on a fixed candidate manifest.

All methods consume the exact same candidate set (boxes/scores/features) from
the frozen cache. Only track IDs may differ (association-controlled protocol).

Usage:
  python tools/run_l1c_tracker.py --variant C2 \
      --manifest outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl \
      --out outputs/l1_c/trackeval/C2 --gpu 1
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
from locatemot.models.track_decoder.relation_track_decoder import RelationTrackDecoderModel  # noqa: E402
from locatemot.models.l1d_association import L1DAssociator  # noqa: E402
from locatemot.models.ua_decoder import UnifiedAssociationDecoder  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402


def load_manifest(path):
    by_video = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            by_video.setdefault(e["video_id"], []).append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])
    return by_video


def build_candidates(entry):
    root = entry["cache_root"]
    key = f"{entry['dataset']}/{entry['video_id']}/{int(entry['frame']):05d}/{entry['protocol']}"
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


def load_ua(ckpt, device):
    model = UnifiedAssociationDecoder()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state)
    return model.to(device).eval()


def load_b6(device):
    model = RelationTrackDecoderModel(use_pbd_base=True, use_region_geom=True, residual=True)
    ck = torch.load(ROOT / "outputs/l0_d/checkpoints/b6/best.pt",
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def load_l1d(ckpt, device):
    model = L1DAssociator()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--ua-ckpt", default="")
    ap.add_argument("--l1d-ckpt", default="")
    ap.add_argument("--l1d-weights", default="0.7,0.3,0.0")
    ap.add_argument("--l1d-delta-scale", type=float, default=0.6)
    ap.add_argument("--l1d-rel-threshold", type=float, default=0.0)
    ap.add_argument("--l1d-threshold", type=float, default=-1.0)
    ap.add_argument("--calibration", default="")
    ap.add_argument("--ac", action="store_true",
                    help="association-controlled: output all candidates")
    ap.add_argument("--new-margin", type=float, default=0.0,
                    help="subtract from NEW logits at inference (shared calibration)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variant = args.variant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ua = None
    b6 = None
    l1d = None
    if variant == "UA":
        ua = load_ua(args.ua_ckpt, device)
    elif variant in ("C4", "T2"):
        b6 = load_b6(device)
    elif variant == "L1D":
        l1d = load_l1d(args.l1d_ckpt, device)

    calib = {}
    if args.calibration and os.path.exists(args.calibration):
        calib = json.load(open(args.calibration))

    by_video = load_manifest(args.manifest)
    t0 = time.time()
    total_frames = 0
    for vid, entries in by_video.items():
        tracker = OnlineTracker(
            variant=variant, b6=b6, ua=ua, l1d=l1d, device=str(device),
            output_all_candidates=args.ac)
        tracker.new_margin = args.new_margin
        if variant == "L1D":
            wi, wp, wm = (float(x) for x in args.l1d_weights.split(","))
            tracker.l1d_weights = (wi, wp, wm)
            tracker.l1d_threshold = (
                float(args.l1d_threshold) if args.l1d_threshold >= 0
                else float(calib.get("l1d_thresh", 0.3)))
            tracker.l1d_delta_scale = args.l1d_delta_scale
            tracker.l1d_rel_threshold = args.l1d_rel_threshold
        if variant == "C2":
            tracker.pbd_thresh = float(calib.get("pbd_thresh", 0.3))
        if variant == "C3":
            tracker.iou_w = float(calib.get("iou_w", 0.5))
            tracker.pbd_w = float(calib.get("pbd_w", 0.5))
            tracker.c3_thresh = float(calib.get("c3_thresh", 0.3))
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
            total_frames += 1
        with open(out / f"{vid}.txt", "w") as f:
            for r in rows:
                f.write(",".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                 for v in r) + "\n")
        print(f"[{variant}] {vid}: frames={len(entries)} "
              f"elapsed={(time.time() - t0):.1f}s", flush=True)
    print(f"[{variant}] done videos={len(by_video)} frames={total_frames} "
          f"seconds={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
