"""Stage L9: OVMOT training stream from the L6 TAO-train pkls.

Detic DLA dets on TAO train are unavailable (torchvision roi_align OOM in
this environment), so we reuse the 105 L6 TAO-train videos: their
LocateAnything-generated boxes + GT, with freshly cached crop-based PBD
identity tokens and frozen CLIP crop features.  This directly adapts the
shared core to the crop-PBD observation distribution used by the TAO val
full-observation evaluation.

Output: outputs/l9/data/tao_train/*.pkl with
  frames: {frame (file-derived), boxes, gen, clip, cand_gt, gt_boxes,
           pbd (zeros placeholder; filled from the crop cache)}
plus index.json.  The crop-PBD cache is then produced by
tools/cache_l9_tao_pbd.py --data-dir outputs/l9/data/tao_train ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

L6 = ROOT / "outputs" / "l6" / "data" / "tao_amodal_train"
TRAIN_GT = ("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
            "TAO-Amodal/annotations/train.json")
OUT = ROOT / "outputs" / "l9" / "data" / "tao_train"
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
              "TAO-Amodal/frames")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    den = ar + br - inter
    return inter / den if den > 1e-9 else 0.0


def match_dets(dets, gt_boxes):
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


def encode_clip(model, crops, device, batch=512):
    import cv2
    import torch
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
        t = t.to(device)
        mean = torch.as_tensor(CLIP_MEAN, device=device)[None, :, None, None]
        std = torch.as_tensor(CLIP_STD, device=device)[None, :, None, None]
        t = (t - mean) / std
        with torch.no_grad():
            out[i:i + batch] = model.encode_image(
                t).float().cpu().numpy().astype(np.float16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import cv2
    import torch
    import clip
    device = "cuda"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    gt = json.load(open(TRAIN_GT))
    vid_name2id = {v["name"].replace("/", "-"): v["id"] for v in gt["videos"]}
    vid_id2imgs = {}
    for im in gt["images"]:
        vid_id2imgs.setdefault(im["video_id"], []).append(im)
    for v in vid_id2imgs.values():
        v.sort(key=lambda x: int(x["frame_index"]))

    pkls = sorted(L6.glob("*.pkl"))
    if args.max_videos:
        pkls = pkls[:args.max_videos]
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"videos": {}}
    name_map = {}
    for p in pkls:
        rec = pickle.load(open(p, "rb"))
        vname = rec["video_id"]
        key = f"train-AVA-{vname}" if "/" not in vname else vname
        vid = vid_name2id.get(key) or vid_name2id.get(vname)
        if vid is None:
            # try suffix match
            for n, vid0 in vid_name2id.items():
                if n.endswith("/" + vname) or n.endswith("-" + vname):
                    vid = vid0
                    break
        if vid is None:
            print(f"[l9train] skip {vname}", flush=True)
            continue
        imgs = vid_id2imgs.get(vid, [])
        file2frame = {}
        for im in imgs:
            stem = im["file_name"].rsplit("/", 1)[-1].replace(".jpg", "")
            if stem.startswith("frame"):
                fidx = int(stem[5:])
            else:
                fidx = int(im["frame_index"])
            file2frame[fidx] = im["file_name"]
        frames_out = []
        for fr in rec["frames"]:
            src_frame = int(fr["frame"])
            # l6 pkl frame is the 0-based annotation frame_index; the
            # frame file is src_frame+1 (frameXXXX)
            fname = file2frame.get(src_frame) or file2frame.get(
                src_frame + 1)
            if fname is None:
                continue
            stem = fname.rsplit("/", 1)[-1].replace(".jpg", "")
            out_frame = int(stem[5:]) if stem.startswith("frame") \
                else src_frame
            boxes = np.asarray(fr["boxes"], np.float32)
            gen = np.asarray(fr["gen"], np.float32)
            cand_gt = match_dets(boxes, fr.get("gt_boxes", {}))
            frames_out.append({
                "frame": out_frame, "boxes": boxes, "gen": gen,
                "cand_gt": cand_gt, "gt_boxes": fr.get("gt_boxes", {}),
                "pbd": np.zeros((len(boxes), 2048), np.float16),
            })
        # CLIP features
        crops = []
        crop_span = []
        for fr in frames_out:
            fname = file2frame.get(fr["frame"]) or file2frame.get(
                fr["frame"] - 1)
            img_path = FRAMES / fname
            arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if arr is None:
                arr = np.zeros((2, 2, 3), np.uint8)
            H, W = arr.shape[:2]
            span = [len(crops), len(fr["boxes"])]
            for b in fr["boxes"]:
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    crops.append(np.zeros((2, 2, 3), np.uint8))
                else:
                    crops.append(arr[y1:y2, x1:x2])
            crop_span.append(span)
        clip_feats = encode_clip(model, crops, device)
        for fr, (start, n) in zip(frames_out, crop_span):
            fr["clip"] = clip_feats[start:start + n]
        out_rec = {"video_id": vname,
                   "image_size": list(rec["image_size"]),
                   "frames": frames_out}
        out_path = OUT / f"{vname}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(out_rec, f)
        index["videos"][vname] = {"path": str(out_path),
                                  "frames": len(frames_out)}
        name_map[vname] = next(
            (n for n, vid0 in vid_name2id.items() if vid0 == vid), vname)
        print(f"[l9train] {vname} frames={len(frames_out)} "
              f"cands={sum(len(f['boxes']) for f in frames_out)}",
              flush=True)
    with open(OUT / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    with open(OUT / "video_name_map.json", "w") as f:
        json.dump(name_map, f, indent=2)
    print(f"[l9train] done videos={len(index['videos'])}", flush=True)


if __name__ == "__main__":
    main()
