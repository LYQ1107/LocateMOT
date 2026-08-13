"""Stage L7: CLIP crop embeddings for the full closed-set evaluation
manifests (same candidate boxes as the L6 regression protocol).

Usage:
  python tools/cache_l7_clip_eval.py --gpus 4,5,6,7 \
      --domains dancetrack_val bdd100k_train mot17_train mot20_train \
      --out outputs/l7/data/clip_eval
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
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
MANIFEST = Path("outputs/l1_c/fixed_candidate_manifest")

_bdd = {}


def bdd_path(vid, frame_index):
    if vid not in _bdd:
        lab = json.loads((BDD_LABELS / "train" / f"{vid}.json").read_text())
        _bdd[vid] = {
            int(f["frameIndex"]): BDD_IMAGES / "train" / vid / f["name"]
            for f in lab}
    return _bdd[vid].get(int(frame_index))


def img_path(dataset, vid, frame):
    if dataset == "dancetrack":
        return DANCETRACK / "val" / vid / "img1" / f"{int(frame):08d}.jpg"
    if dataset == "mot17":
        return MOT17 / "train" / vid / "img1" / f"{int(frame):06d}.jpg"
    if dataset == "mot20":
        return MOT20 / vid / "img1" / f"{int(frame):06d}.jpg"
    if dataset == "bdd100k":
        return bdd_path(vid, frame)
    raise ValueError(dataset)


def load_videos(domain):
    by_video = defaultdict(list)
    with open(MANIFEST / f"{domain}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda x: int(x["frame"]))
    return by_video


def worker(gpu, jobs, out_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import clip
    device = "cuda"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
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
                if h < 2 or w < 2:
                    arr = np.zeros((2, 2, 3), arr.dtype)
                    h = w = 2
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

    for domain, vid, entries in jobs:
        dom_dir = out_dir / domain
        dom_dir.mkdir(parents=True, exist_ok=True)
        out_path = dom_dir / f"{vid}.pkl"
        if out_path.exists():
            continue
        frames = []
        crops = []
        spans = []
        for e in entries:
            boxes = np.asarray(e["boxes"], np.float32).reshape(-1, 4)
            scores = np.asarray(e["scores"], np.float32).reshape(-1)
            spans.append((len(crops), len(boxes)))
            path = img_path(e["dataset"], vid, int(e["frame"]))
            try:
                arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if arr is None:
                    with Image.open(path) as im:
                        arr = np.asarray(im.convert("RGB"))[:, :, ::-1]
                H, W = arr.shape[:2]
                for b in boxes:
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
            except Exception as ex:
                print(f"[clipeval] fail {path}: {ex}", flush=True)
                crops.extend([np.zeros((2, 2, 3), np.uint8)] * len(boxes))
            frames.append({"frame": int(e["frame"]), "boxes": boxes,
                           "gen": scores})
        feats = encode(crops)
        for fr, (start, count) in zip(frames, spans):
            fr["clip"] = feats[start:start + count]
        rec = {"video_id": vid,
               "image_size": [int(x) for x in entries[0]["image_size"]],
               "frames": frames}
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rec, f)
        os.replace(tmp, out_path)
        print(f"[clipeval:{gpu}] {domain}/{vid} frames={len(frames)}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/l7/data/clip_eval")
    args = ap.parse_args()
    out_dir = Path(args.out)
    gpus = [int(x) for x in args.gpus.split(",")]
    all_jobs = []
    for domain in args.domains:
        by_video = load_videos(domain)
        all_jobs.extend((domain, vid, entries)
                        for vid, entries in by_video.items())
    shards = [[] for _ in gpus]
    for i, job in enumerate(all_jobs):
        shards[i % len(gpus)].append(job)
    procs = []
    for gpu, shard in zip(gpus, shards):
        p = torch.multiprocessing.Process(
            target=worker, args=(gpu, shard, out_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    for domain in args.domains:
        dom_dir = out_dir / domain
        index = {"videos": {}}
        for p in sorted(dom_dir.glob("*.pkl")):
            rec = pickle.load(open(p, "rb"))
            index["videos"][rec["video_id"]] = {
                "path": str(p), "frames": len(rec["frames"])}
        with open(dom_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)
    print("[clipeval] done", flush=True)


if __name__ == "__main__":
    main()
