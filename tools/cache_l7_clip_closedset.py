"""Stage L7: cache frozen CLIP ViT-B/32 crop embeddings for closed-set data.

Reads the L6 per-video sequence pickles (outputs/l6/data/<domain>/*.pkl),
loads each source image, crops the candidate boxes and encodes them with
the frozen CLIP image encoder.  Writes a sibling pickle under --out with the
same structure plus a 'clip' field (float16, [N,512]) replacing the PBD
token as the open-vocabulary appearance evidence.

Usage:
  python tools/cache_l7_clip_closedset.py --gpus 4,5,6,7 \
      --domains bdd100k_train dancetrack_calibration dancetrack_train \
                mot17_train mot20_train \
      --out outputs/l7/data/clip_closed
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

DANCETRACK = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
MOT17 = Path("/data1/LWR/vranlee/DATASETS/JDE/MOT17")
MOT20 = Path("/data1/LWR/vranlee/M4FTMoveOut4Doing/ByteTrack-mbt/datasets/"
             "mix_mot20_ch/mot20_train")
BDD_IMAGES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
                  "bdd100k/images/track")
BDD_LABELS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
                  "annotations/box_track_20")


def frame_image_path(domain, video_id, frame):
    fid = int(frame)
    if domain == "dancetrack_train":
        return DANCETRACK / "train" / video_id / "img1" / f"{fid:08d}.jpg"
    if domain == "dancetrack_calibration":
        return DANCETRACK / "train" / video_id / "img1" / f"{fid:08d}.jpg"
    if domain == "mot17_train":
        return MOT17 / "train" / video_id / "img1" / f"{fid:06d}.jpg"
    if domain == "mot20_train":
        return MOT20 / video_id / "img1" / f"{fid:06d}.jpg"
    if domain == "bdd100k_train":
        return _bdd_path(video_id, fid)
    raise ValueError(domain)


_bdd_index = {}


def _bdd_path(video_id, frame_index):
    if video_id not in _bdd_index:
        lab = json.loads((BDD_LABELS / "train" / f"{video_id}.json").read_text())
        _bdd_index[video_id] = {
            int(f["frameIndex"]): BDD_IMAGES / "train" / video_id / f["name"]
            for f in lab}
    return _bdd_index[video_id].get(frame_index)


def worker(gpu, jobs, out_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import clip
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    import cv2
    mean = torch.as_tensor([0.48145466, 0.4578275, 0.40821073],
                           device=device)[None, :, None, None]
    std = torch.as_tensor([0.26862954, 0.26130258, 0.27577711],
                          device=device)[None, :, None, None]

    def encode(crops, batch=512):
        out = np.zeros((len(crops), 512), np.float16)
        for i in range(0, len(crops), batch):
            chunk = []
            for arr in crops[i:i + batch]:
                h, w = arr.shape[:2]
                scale = 224.0 / min(h, w)
                nh, nw = int(round(h * scale)), int(round(w * scale))
                im = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_CUBIC)
                y = max(0, (nh - 224) // 2)
                x = max(0, (nw - 224) // 2)
                chunk.append(im[y:y + 224, x:x + 224].astype(np.float32))
            t = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2) / 255.0
            t = (t.to(device) - mean) / std
            with torch.no_grad():
                out[i:i + batch] = model.encode_image(
                    t).float().cpu().numpy().astype(np.float16)
        return out

    for domain, src_path in jobs:
        rec = pickle.load(open(src_path, "rb"))
        out_frames = []
        crops = []
        spans = []
        for fr in rec["frames"]:
            n = len(fr["boxes"])
            spans.append((len(crops), n))
            if n:
                img_path = frame_image_path(domain, rec["video_id"],
                                            fr["frame"])
                try:
                    arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if arr is None:
                        with Image.open(img_path) as im:
                            arr = np.asarray(im.convert("RGB"))[:, :, ::-1]
                    H, W = arr.shape[:2]
                    for b in fr["boxes"]:
                        x1, y1, x2, y2 = [int(v) for v in b]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            x2 = min(W, max(x2, x1 + 2))
                            y2 = min(H, max(y2, y1 + 2))
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            crops.append(np.zeros((2, 2, 3), np.uint8))
                        else:
                            crops.append(arr[y1:y2, x1:x2])
                except Exception as e:
                    print(f"[clip] fail {img_path}: {e}", flush=True)
        feats_all = encode(crops)
        for fr, (start, count) in zip(rec["frames"], spans):
            feats = feats_all[start:start + count]
            nfr = dict(fr)
            nfr["clip"] = feats
            out_frames.append(nfr)
        out_rec = dict(rec)
        out_rec["frames"] = out_frames
        out_path = out_dir / domain / f"{rec['video_id']}.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(out_rec, f)
        os.replace(tmp, out_path)
        print(f"[clip:{gpu}] {domain}/{rec['video_id']} "
              f"frames={len(out_frames)}", flush=True)


def dry_run(jobs):
    import clip
    missing = 0
    for domain, src_path in jobs:
        rec = pickle.load(open(src_path, "rb"))
        for fr in rec["frames"][:2]:
            p = frame_image_path(domain, rec["video_id"], fr["frame"])
            if p is None or not p.exists():
                missing += 1
                print(f"[dry] missing {domain} {rec['video_id']} {fr['frame']}",
                      flush=True)
    print(f"[dry] done missing={missing}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--data-dir", default="outputs/l6/data")
    ap.add_argument("--out", default="outputs/l7/data/clip_closed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    jobs = []
    for domain in args.domains:
        src_dir = Path(args.data_dir) / domain
        for p in sorted(src_dir.glob("*.pkl")):
            jobs.append((domain, str(p)))
    if args.dry_run:
        dry_run(jobs)
        return
    gpus = [int(x) for x in args.gpus.split(",")]
    shards = [[] for _ in gpus]
    for i, job in enumerate(jobs):
        shards[i % len(gpus)].append(job)
    procs = []
    for gpu, shard in zip(gpus, shards):
        p = torch.multiprocessing.Process(
            target=worker, args=(gpu, shard, out_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    # build per-domain index.json (L6 dataset loader requirement)
    for domain in args.domains:
        dom_dir = out_dir / domain
        index = {"videos": {}}
        for p in sorted(dom_dir.glob("*.pkl")):
            rec = pickle.load(open(p, "rb"))
            index["videos"][rec["video_id"]] = {
                "path": str(p), "frames": len(rec["frames"])}
        with open(dom_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)
    print(f"[clip] done {len(jobs)} videos", flush=True)


if __name__ == "__main__":
    main()
