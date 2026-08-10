"""Repair matched_candidates in cached frames whose GT was missed due to
the str/int frame-id bug (dancetrack / mot17 / mot20).  No re-inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
CACHE = Path(
    "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/cache_dla")

from locatemot.data.token_cache import read_frame_cache, write_frame_cache  # noqa: E402
from tools.cache_l1b_locateanything import Loaders, _iou  # noqa: E402


def main():
    loaders = Loaders()
    fixed = 0
    for ds in ("dancetrack", "mot17", "mot20"):
        base = CACHE / ds
        if not base.exists():
            continue
        for meta_path in base.rglob("*.meta.json"):
            rel = meta_path.relative_to(CACHE)
            key = str(rel).rsplit(".meta.json", 1)[0]
            data = read_frame_cache(str(CACHE), key)
            if data is None:
                continue
            meta = data["meta"]
            vid = meta["video_id"]
            fid = meta["frame"]
            _, gt = loaders.frame_and_gt(ds, vid, fid)
            if not gt:
                continue
            feats = data["features"]
            boxes = np.asarray(feats.get("boxes", np.zeros((0, 4))))
            matched = {}
            for oid, gtb in [(g[4], g[:4]) for g in gt]:
                best_idx, best_iou = None, 0.0
                for i in range(len(boxes)):
                    iou = _iou(boxes[i].tolist(), gtb)
                    if iou > best_iou:
                        best_idx, best_iou = i, iou
                if best_idx is not None:
                    matched[str(oid)] = {"candidate": best_idx,
                                         "iou": round(best_iou, 4)}
            if matched:
                meta["gt_object_ids"] = [g[4] for g in gt]
                meta["gt_boxes"] = {str(g[4]): list(g[:4]) for g in gt}
                meta["matched_candidates"] = matched
                feats_np = {
                    k: (v.numpy() if hasattr(v, "numpy")
                        else np.asarray(v))
                    for k, v in feats.items()
                }
                write_frame_cache(str(CACHE), key, feats_np, meta)
                fixed += 1
    print("REPAIR_DONE", fixed)


if __name__ == "__main__":
    main()
