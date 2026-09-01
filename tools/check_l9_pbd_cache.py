"""One-time L9 regression: verify TAO PBD cache key/dim/finite/alignment.

Usage: python tools/check_l9_pbd_cache.py [--frames 24]
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import read_frame_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root",
                    default=str(ROOT / "outputs" / "l9" / "cache"
                                / "tao_val_pbd"))
    ap.add_argument("--src-root",
                    default=str(ROOT / "outputs" / "l7" / "data"
                                / "tao_val"))
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()
    root = args.cache_root
    keys = [p[len(root) + 1:-len(".safetensors")]
            for p in glob.glob(root + "/tao/*/*/*.safetensors")]
    assert len(keys) >= args.frames, f"need {args.frames} frames, have {len(keys)}"
    pbd_key = "pbd_box_end_last"
    for k in keys[:args.frames]:
        d = read_frame_cache(root, k)
        feats, meta = d["features"], d["meta"]
        n = int(meta["candidate_count"])
        assert pbd_key in feats, f"{k}: missing {pbd_key}"
        assert tuple(feats[pbd_key].shape) == (n, 2048), f"{k}: dim"
        assert torch.isfinite(feats[pbd_key]).all().item(), f"{k}: non-finite"
        assert tuple(feats["boxes"].shape) == (n, 4), f"{k}: boxes"
        vid = k.split("/")[1]
        frame = int(k.split("/")[2])
        src = pickle.load(open(Path(args.src_root) / f"{vid}.pkl", "rb"))
        fr = next(f for f in src["frames"] if int(f["frame"]) == frame)
        assert len(fr["boxes"]) == n, f"{k}: candidate count mismatch"
        dbox = np.abs(np.asarray(feats["boxes"], np.float32) -
                      np.asarray(fr["boxes"], np.float32)).max()
        assert dbox <= 1.0, f"{k}: box mismatch {dbox}"
        dclip = np.abs(np.asarray(feats["clip"], np.float32) -
                       np.asarray(fr["clip"], np.float32)).max()
        assert dclip <= 1e-3, f"{k}: clip mismatch"
    print(f"PBD cache regression OK: {args.frames} frames, key={pbd_key}, "
          "dim=2048, finite, candidate-aligned (float16 precision)")


if __name__ == "__main__":
    main()
