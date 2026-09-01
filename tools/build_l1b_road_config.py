"""Stage L1-B road multi-class expansion config (BDD + TAO)."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
OUT = ROOT / "configs" / "l1_b"
SEED = 20260806
BDD_LABELS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
                  "annotations/box_track_20")
BDD_IMAGES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
                  "bdd100k/images/track")
TAO = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
           "TAO-Amodal")


def bdd_all(max_frames=40):
    out = {}
    for p in sorted((BDD_LABELS / "train").glob("*.json")):
        vname = p.stem
        if not (BDD_IMAGES / "train" / vname).exists():
            continue
        v = json.loads(p.read_text())
        if len(v) < max_frames:
            continue
        idx = [round(i * (len(v) - 1) / (max_frames - 1))
               for i in range(max_frames)]
        out[vname] = sorted({int(v[i]["frameIndex"]) for i in idx})
    return out


def tao_top(rng, n_videos=100, max_frames=40):
    d = json.loads((TAO / "annotations" / "train.json").read_text())
    imgs = {}
    for im in d["images"]:
        imgs.setdefault(im["video"], []).append(im)
    cands = []
    for vname, ims in imgs.items():
        if len(ims) >= max_frames:
            cands.append(vname)
    cands.sort()
    rng.shuffle(cands)
    out = {}
    for vname in cands[:n_videos]:
        ims = sorted(imgs[vname], key=lambda x: x["frame_index"])
        idx = [round(i * (len(ims) - 1) / (max_frames - 1))
               for i in range(max_frames)]
        out[vname] = [{"frame_index": int(ims[i]["frame_index"]),
                       "file_name": ims[i]["file_name"]}
                      for i in sorted({idx[i] for i in range(max_frames)})]
    return out


def main():
    rng = random.Random(SEED)
    cfg = {
        "bdd100k_train": bdd_all(),
        "tao_amodal_train": tao_top(rng),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "road_v2_videos.json").write_text(json.dumps(cfg, indent=1))
    n = {k: sum(len(v["frames"]) if isinstance(v, dict) else len(v)
                for v in val.values())
         for k, val in cfg.items()}
    print("ROAD_CONFIG_DONE", n, "total", sum(n.values()))


if __name__ == "__main__":
    main()
