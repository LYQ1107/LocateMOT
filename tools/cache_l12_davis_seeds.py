"""Stage L12: seed identity tokens (PBD) for DAVIS prompt types.

For every first-frame object seed and prompt type:
  mask : bbox crop with background masked out (black)
  box  : tight bbox crop
  point: square crop centered at the deterministic interior point

Writes outputs/l12/data/davis_seed_pbd.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.cache_l9_tao_pbd import _extract_crop, load_model  # noqa: E402
from tools.build_l10_refer_kitti import CLIP_MEAN, CLIP_STD, encode_clip  # noqa: E402

DATA = ROOT / "outputs" / "l12" / "data" / "davis"
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/DAVIS/DAVIS/JPEGImages/480p")
OUT = ROOT / "outputs" / "l12" / "data" / "davis_seed_pbd.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    extractor = load_model(args.gpu)
    videos = sorted(p.stem for p in DATA.glob("*.pkl"))
    if args.max_videos:
        videos = videos[:args.max_videos]
    out = {}
    t0 = time.time()
    n = 0
    import clip as clipmod
    cm, _ = clipmod.load("ViT-B/32", device="cuda")
    cm.eval()
    for vi, vid in enumerate(videos):
        rec = pickle.load(open(DATA / f"{vid}.pkl", "rb"))
        m = np.asarray(Image.open(FRAMES / vid / "00000.jpg").convert("RGB"))
        H, W = m.shape[:2]
        img = Image.fromarray(m)
        out[vid] = {}
        for oid, seed in rec["seeds"].items():
            box = seed["box"]
            x1, y1, x2, y2 = box
            mask = seed["mask"]
            out[vid][str(oid)] = {}
            crops = {}
            # box prompt
            f = _extract_crop(extractor, img, box)
            if f is not None and f["pbd_box_end_last"] is not None:
                out[vid][str(oid)]["box"] = \
                    f["pbd_box_end_last"].tolist()
                crops["box"] = _crop_arr(m, box)
                n += 1
            # mask prompt: black out background
            masked = m.copy()
            for c in range(3):
                masked[:, :, c] = np.where(mask > 0, masked[:, :, c], 0)
            f = _extract_crop(extractor, Image.fromarray(masked), box)
            if f is not None and f["pbd_box_end_last"] is not None:
                out[vid][str(oid)]["mask"] = f["pbd_box_end_last"].tolist()
                crops["mask"] = _crop_arr(masked, box)
                n += 1
            # point prompt: square crop centered at the point
            px, py = seed["point"]
            side = min(x2 - x1, y2 - y1)
            if side < 2:
                side = max(2, min(W, H) // 8)
            h = side // 2
            sq = [max(0, px - h), max(0, py - h),
                  min(W, px + h), min(H, py + h)]
            f = _extract_crop(extractor, img, sq)
            if f is not None and f["pbd_box_end_last"] is not None:
                out[vid][str(oid)]["point"] = f["pbd_box_end_last"].tolist()
                crops["point"] = _crop_arr(m, sq)
                n += 1
            if crops:
                cfeats = encode_clip(cm, list(crops.values()), "cuda")
                for k, arr in zip(crops.keys(), cfeats):
                    out[vid][str(oid)][k + "_clip"] = \
                        arr.astype(np.float32).tolist()
        print(f"[l12seeds] {vid} {vi+1}/{len(videos)} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"[l12seeds] done crops={n}", flush=True)


def _crop_arr(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        x2 = min(img.shape[1], x1 + 2)
        y2 = min(img.shape[0], y1 + 2)
    return cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    main()
