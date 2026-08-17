"""Merge the crop-PBD cache into the TAO-train OVMOT pkls (pbd field).

Usage: python tools/merge_l9_train_pbd.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402

DATA = ROOT / "outputs" / "l9" / "data" / "tao_train"
CACHE = ROOT / "outputs" / "l9" / "cache" / "tao_train_pbd"


def main():
    index = json.loads((DATA / "index.json").read_text())
    n_miss = 0
    n_frames = 0
    for vname, info in index["videos"].items():
        p = Path(info["path"])
        rec = pickle.load(open(p, "rb"))
        changed = False
        for fr in rec["frames"]:
            key = cache_key("tao", vname, int(fr["frame"]), "pbd_full")
            d = read_frame_cache(str(CACHE), key)
            if d is None:
                n_miss += 1
                continue
            pbd = np.asarray(d["features"]["pbd_box_end_last"], np.float32)
            if len(pbd) != len(fr["boxes"]):
                n_miss += 1
                continue
            fr["pbd"] = pbd.astype(np.float16)
            changed = True
            n_frames += 1
        if changed:
            with open(p, "wb") as f:
                pickle.dump(rec, f)
    print(f"[merge] frames_merged={n_frames} cache_miss={n_miss} "
          f"videos={len(index['videos'])}")


if __name__ == "__main__":
    main()
