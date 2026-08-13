"""Stage L7: build TAO per-video sequences for OVMOT (official protocol).

Inputs (read-only, symlink-friendly):
  - official TAO v1-val GT: masa/data/tao/annotations/tao_val_lvis_v1_classes.json
  - official TAO v0.5 train GT: TAO-Amodal/annotations/train.json
  - TAO frames: TAO-Amodal/frames
  - public Detic dets (val): masa/results/public_dets/tao_val_dets/...
  - generated Detic dets (train): --dets-root (per-frame pickles:
    {'det_bboxes': [N,5], 'det_labels': [N]}), path layout
    <dets-root>/<split>/<dataset>/<video_stem>/frame%04d.pth

Outputs per-video pickles:
  frames: [{frame, boxes [N,4] xyxy, gen [N], clip [N,512] fp16,
            label [N] lvis-v1 id, cand_gt [N] gt-track-id-or-None,
            gt_boxes {gt_track_id: [x1,y1,x2,y2]}}]

The appearance token is frozen CLIP ViT-B/32 crop embedding (open-vocabulary
semantic interface), replacing the closed-set PBD token for OVMOT.

Usage (multi-GPU):
  python tools/build_l7_tao.py --split val --gpus 4,5,6,7
      --out outputs/l7/data/tao_val
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

TAO_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal")
MASA_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa")
VAL_GT = MASA_ROOT / "data/tao/annotations/tao_val_lvis_v1_classes.json"
TRAIN_GT = TAO_ROOT / "annotations/train.json"
FRAMES = TAO_ROOT / "frames"
VAL_DETS = (MASA_ROOT / "results/public_dets/tao_val_dets/teta_50_internms"
            / "detic_tao_val_det")


def load_gt(split):
    if split == "val":
        gt = json.load(open(VAL_GT))
        anns = defaultdict(list)
        for a in gt["annotations"]:
            anns[a["image_id"]].append(a)
        return gt, anns
    gt = json.load(open(TRAIN_GT))
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    return gt, anns


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    den = ar + br - inter
    return inter / den if den > 1e-9 else 0.0


def match_dets(dets, gt_boxes):
    """Greedy one-to-one IoU match; returns per-det gt track id or None."""
    scores = []
    for j, d in enumerate(dets):
        for gid, gb in gt_boxes.items():
            v = iou(d, gb)
            if v >= 0.5:
                scores.append((v, j, gid))
    scores.sort(reverse=True)
    used_d, used_g, out = set(), set(), [None] * len(dets)
    for v, j, gid in scores:
        if j in used_d or gid in used_g:
            continue
        used_d.add(j)
        used_g.add(gid)
        out[j] = gid
    return out


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def prepare_video(vid, images, anns, dets_root, frames_root):
    """Returns (frame_meta, crops) where frame_meta carries everything except
    the CLIP feature, and crops is a flat list of uint8 RGB numpy arrays."""
    import cv2
    frames = []
    crops = []
    crop_span = []  # per frame (start, count)
    for img in images:
        fname = img["file_name"].rsplit("/", 1)[-1]
        stem = fname.replace(".jpg", "")
        if stem.startswith("frame"):
            fidx = int(stem[5:])
        else:
            fidx = int(img["frame_index"])
        img_path = frames_root / img["file_name"]
        parts = img["file_name"].split("/")
        det_name = f"frame{fidx:04d}.pth" if stem.startswith("frame") \
            else f"{stem}.pth"
        det_pth = (Path(dets_root) / parts[0] / parts[1] / parts[2]
                   / det_name)
        try:
            det = pickle.load(open(det_pth, "rb"))
        except FileNotFoundError:
            det = {"det_bboxes": torch.zeros((0, 5)),
                   "det_labels": torch.zeros((0,), dtype=torch.long)}
        boxes = det["det_bboxes"].numpy().astype(np.float32)
        labels = det["det_labels"].numpy().astype(np.int64)
        g_anns = anns.get(img["id"], [])
        gt_boxes = {}
        for a in g_anns:
            x, y, w, h = a["bbox"]
            gt_boxes[str(a["track_id"])] = [x, y, x + w, y + h]
        if len(boxes):
            keep = boxes[:, 4] >= 0.05
            boxes = boxes[keep]
            labels = labels[keep]
        cand_gt = match_dets(boxes[:, :4], gt_boxes) if len(boxes) else []
        span = [len(crops), 0]
        try:
            img_arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_arr is None:
                img_arr = np.asarray(Image.open(img_path).convert("RGB"))[:, :, ::-1]
            H, W = img_arr.shape[:2]
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    x2 = min(W, max(x2, x1 + 2))
                    y2 = min(H, max(y2, y1 + 2))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    crops.append(np.zeros((2, 2, 3), np.uint8))
                else:
                    crops.append(img_arr[y1:y2, x1:x2])
            span[1] = len(boxes)
        except Exception as e:
            print(f"[l7tao] crop fail {img_path}: {e}", flush=True)
        crop_span.append(tuple(span))
        frames.append({
            "frame": fidx,
            "boxes": boxes[:, :4].astype(np.float32),
            "gen": boxes[:, 4].astype(np.float32),
            "label": labels.astype(np.int32),
            "cand_gt": cand_gt,
            "gt_boxes": gt_boxes,
        })
    return frames, crops, crop_span


def encode_crops(model, crops, batch=512):
    """CLIP ViT-B/32 preprocessing + batched fp16 image encoding."""
    import cv2
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
            im = im[y:y + 224, x:x + 224].astype(np.float32)
            chunk.append(im)
        t = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2) / 255.0
        t = t.to(model.logit_scale.device)
        mean = torch.as_tensor(CLIP_MEAN, device=t.device)[None, :, None, None]
        std = torch.as_tensor(CLIP_STD, device=t.device)[None, :, None, None]
        t = (t - mean) / std
        with torch.no_grad():
            out[i:i + batch] = model.encode_image(t).float().cpu().numpy() \
                .astype(np.float16)
    return out


def worker(gpu, videos, out_dir, gt, anns, dets_root):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import clip
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    id2img = defaultdict(list)
    for img in gt["images"]:
        id2img[img["video_id"]].append(img)
    for img_list in id2img.values():
        img_list.sort(key=lambda x: int(x["frame_index"]))
    for vname, vid in videos:
        out_path = out_dir / f"{vname}.pkl"
        if out_path.exists():
            continue
        frames, crops, crop_span = prepare_video(
            vid, id2img.get(vid, []), anns, dets_root, FRAMES)
        clip_feats = encode_crops(model, crops)
        for fr, (start, count) in zip(frames, crop_span):
            fr["clip"] = clip_feats[start:start + count]
        w = id2img.get(vid, [{}])[0].get("width", 1280)
        h = id2img.get(vid, [{}])[0].get("height", 720)
        rec = {"video_id": vname, "image_size": [w, h], "frames": frames}
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rec, f)
        os.replace(tmp, out_path)
        print(f"[l7tao:{gpu}] {vname} frames={len(frames)} "
              f"dets={sum(len(fr['boxes']) for fr in frames)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--out", default="outputs/l7/data/tao_val")
    ap.add_argument("--dets-root", default=str(VAL_DETS))
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt, anns = load_gt(args.split)
    vids = sorted(gt["videos"], key=lambda v: v["id"])
    if args.max_videos:
        vids = vids[:args.max_videos]
    items = [(v["name"].replace("/", "-"), v["id"]) for v in vids]
    gpus = [int(x) for x in args.gpus.split(",")]
    shards = [[] for _ in gpus]
    for i, item in enumerate(items):
        shards[i % len(gpus)].append(item)
    procs = []
    for gpu, shard in zip(gpus, shards):
        p = torch.multiprocessing.Process(
            target=worker, args=(gpu, shard, out_dir, gt, anns, args.dets_root))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    # index.json
    index = {"videos": {}}
    for vname, vid in items:
        p = out_dir / f"{vname}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                rec = pickle.load(f)
            index["videos"][vname] = {"path": str(p), "frames": len(rec["frames"])}
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"[l7tao] done {len(index['videos'])} videos", flush=True)


if __name__ == "__main__":
    main()
