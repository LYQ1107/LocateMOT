"""Stage L12: LocateAnything-3B crop-PBD features for DAVIS candidates.

Same extraction as L9 TAO PBD cache; writes
outputs/l12/cache/davis_pbd/<video>/<frame:05d>/pbd_full with rows
aligned to the candidate order in outputs/l12/data/davis/<video>.pkl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import cache_key, write_frame_cache  # noqa: E402
from locatemot.models.object_tokens.extractor import ObjectTokenExtractor  # noqa: E402
from tools.cache_l9_tao_pbd import _ckpt_hash, _extract_crop, _pick_token, load_model  # noqa: E402

MODEL_COMMIT = "783f656d127ee498137b5ff52603ce36c292d317"
MODEL_DIR = ROOT / "models" / "LocateAnything-3B"
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/DAVIS/DAVIS/JPEGImages/480p")
DATA = ROOT / "outputs" / "l12" / "data" / "davis"
CACHE = ROOT / "outputs" / "l12" / "cache" / "davis_pbd"
QUERY = "Locate the main object in the image."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    extractor = load_model(args.gpu)
    videos = sorted(p.stem for p in DATA.glob("*.pkl"))
    if args.videos:
        videos = [v for v in videos if v in set(args.videos)]
    if args.max_videos:
        videos = videos[:args.max_videos]
    print(f"[l12pbd] videos={videos}", flush=True)
    t0 = time.time()
    n_crops = 0
    for vi, vid in enumerate(videos):
        rec = pickle.load(open(DATA / f"{vid}.pkl", "rb"))
        for fi, fr in enumerate(rec["frames"]):
            frame = fr["frame"]
            key = cache_key("davis", vid, frame, "pbd_full")
            n = len(fr["boxes"])
            # write_frame_cache stores the tensor as pbd_full.safetensors.
            # Also validate candidate_count: an earlier empty-DLA run can
            # leave a valid-looking zero-row cache after the pkl is rebuilt.
            tensor_path = (CACHE / vid / f"{frame:05d}" /
                           "pbd_full.safetensors")
            meta_path = (CACHE / vid / f"{frame:05d}" /
                         "pbd_full.meta.json")
            if tensor_path.exists() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if int(meta.get("candidate_count", -1)) == n:
                        continue
                except (OSError, ValueError, TypeError):
                    pass
            feats = {"pbd_box_end_last": np.zeros((n, 2048), np.float32)}
            img = Image.open(FRAMES / vid / f"{frame:05d}.jpg").convert("RGB")
            for j, b in enumerate(fr["boxes"]):
                f = _extract_crop(extractor, img, b)
                if f is not None and f["pbd_box_end_last"] is not None:
                    feats["pbd_box_end_last"][j] = f["pbd_box_end_last"]
                n_crops += 1
            write_frame_cache(
                str(CACHE), key, feats,
                {"candidate_count": n, "video": vid, "frame": frame})
        print(f"[l12pbd] {vid} {vi+1}/{len(videos)} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[l12pbd] done crops={n_crops} elapsed={time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
