"""Stage L12: CLIP crop features for DAVIS val candidates (in-place).

Fills `fr["clip"]` in outputs/l12/data/davis/<video>.pkl for every
candidate box.  PBD features are cached separately by
tools/cache_l12_davis_pbd.py (LocateAnything-3B).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.build_l10_refer_kitti import CLIP_MEAN, CLIP_STD, encode_clip  # noqa: E402

DATA = ROOT / "outputs" / "l12" / "data" / "davis"
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/DAVIS/DAVIS/JPEGImages/480p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import clip
    device = "cuda"
    cm, _ = clip.load("ViT-B/32", device=device)
    cm.eval()
    videos = sorted(p.stem for p in DATA.glob("*.pkl"))
    if args.max_videos:
        videos = videos[:args.max_videos]
    for vi, vid in enumerate(videos):
        p = DATA / f"{vid}.pkl"
        rec = pickle.load(open(p, "rb"))
        crops = []
        spans = []
        for fr in rec["frames"]:
            arr = cv2.imread(str(FRAMES / vid / f"{fr['frame']:05d}.jpg"),
                             cv2.IMREAD_COLOR)
            if arr is None:
                arr = np.zeros((2, 2, 3), np.uint8)
            H, W = arr.shape[:2]
            start = len(crops)
            for b in fr["boxes"]:
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    x2 = min(W, max(x2, x1 + 2))
                    y2 = min(H, max(y2, y1 + 2))
                crops.append(arr[y1:y2, x1:x2] if x2 > x1 and y2 > y1
                             else np.zeros((2, 2, 3), np.uint8))
            spans.append((start, len(fr["boxes"])))
        feats = encode_clip(cm, crops, device)
        for fr, (start, n) in zip(rec["frames"], spans):
            fr["clip"] = feats[start:start + n].astype(np.float16)
        with open(p, "wb") as f:
            pickle.dump(rec, f, protocol=4)
        print(f"[l12clip] {vid} {vi+1}/{len(videos)}", flush=True)


if __name__ == "__main__":
    main()
