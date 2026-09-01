"""Stage L10: build the full TAO-train OVMOT stream from DLA dets.

Unlike the L9 stream (105 videos, LocateAnything-generated boxes), this
builder uses the DLA (Detic-SwinB, MASA checkpoint) detections that were
generated for all 500 TAO train videos, and aligns every candidate to the
continuous C-TAO base annotations (COVTrack ICCV 2025) so that identity
supervision is dense and consistent with the TAO-val full-PBD protocol.

Output format matches the L9 OVMOT pkls:
  video_id  : GT video name with "/" replaced by "-"
  image_size: [w, h]
  frames    : [{frame, boxes [N,4] xyxy, gen [N], label [N],
                cand_gt [N] (C-TAO/TAO track id or None),
                gt_boxes {track_id: xyxy}, clip [N,512] fp16,
                pbd zeros [N,2048] fp16 placeholder}]

Usage (one process per shard; each shard uses its own GPU):
  python tools/build_l10_tao_train.py --gpus 4,6,7
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
import cv2

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

TAO_GT = ("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
          "TAO-Amodal/annotations/train.json")
CTAO_JSON = ("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/"
             "external_annotations/covtrack/ctao_base.json")
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
              "TAO-Amodal/frames")
DEFAULT_DETS = ROOT / "outputs" / "l10" / "cache" / "tao_train_candidates"

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    den = ar + br - inter
    return inter / den if den > 1e-9 else 0.0


def match_dets(dets, gt_boxes):
    """Greedy one-to-one IoU>=0.5 match; per-det gt track id or None."""
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


def encode_clip(model, crops, device, batch=512):
    """Frozen CLIP ViT-B/32 crop encoding (fp16), same as L7/L9 builders."""
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
            out[i:i + batch] = model.encode_image(t).float().cpu().numpy() \
                .astype(np.float16)
    return out


def det_path_for_image(dets_root, file_name):
    parts = file_name.split("/")
    stem = parts[-1].replace(".jpg", "")
    if stem.startswith("frame"):
        fname = f"frame{int(stem[5:]):04d}.pth"
    else:
        fname = f"{stem}.pth"
    return Path(dets_root) / parts[0] / parts[1] / parts[2] / fname


def worker(gpu, videos, gt_images, ctao_by_file, tao_anns_by_img,
           dets_root, out_dir, require_complete):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import clip
    device = "cuda"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    # image list per video
    by_vid = defaultdict(list)
    for im in gt_images:
        by_vid[im["video_id"]].append(im)
    for lst in by_vid.values():
        lst.sort(key=lambda x: int(x["frame_index"]))

    for vname, vid in videos:
        out_key = vname.replace("/", "-")
        out_path = out_dir / f"{out_key}.pkl"
        if out_path.exists():
            print(f"[l10:{gpu}] skip existing {out_key}", flush=True)
            continue
        imgs = by_vid.get(vid, [])
        missing = 0
        frames = []
        crops = []
        spans = []
        for im in imgs:
            dp = det_path_for_image(dets_root, im["file_name"])
            if not dp.exists():
                missing += 1
                det = None
            else:
                det = pickle.load(open(dp, "rb"))
            boxes = det["det_bboxes"].numpy().astype(np.float32) \
                if det is not None else np.zeros((0, 5), np.float32)
            labels = det["det_labels"].numpy().astype(np.int64) \
                if det is not None else np.zeros((0,), np.int64)
            if len(boxes):
                keep = boxes[:, 4] >= 0.05
                boxes = boxes[keep]
                labels = labels[keep]
            fname = im["file_name"]
            ctao_anns = ctao_by_file.get(fname, [])
            gt_boxes = {}
            for a in ctao_anns:
                x, y, w, h = a["bbox"]
                gt_boxes[str(a["track_id"])] = [x, y, x + w, y + h]
            if not gt_boxes:
                for a in tao_anns_by_img.get(im["id"], []):
                    x, y, w, h = a["bbox"]
                    gt_boxes[str(a["track_id"])] = [x, y, x + w, y + h]
            cand_gt = match_dets(boxes[:, :4], gt_boxes) if len(boxes) else []
            stem = fname.rsplit("/", 1)[-1].replace(".jpg", "")
            frame = int(stem[5:]) if stem.startswith("frame") \
                else int(im["frame_index"])
            start = len(crops)
            img_path = FRAMES / fname
            arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if arr is None:
                arr = np.zeros((2, 2, 3), np.uint8)
            H, W = arr.shape[:2]
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
                    crops.append(arr[y1:y2, x1:x2])
            spans.append((start, len(boxes)))
            frames.append({
                "frame": frame,
                "boxes": boxes[:, :4].astype(np.float32),
                "gen": boxes[:, 4].astype(np.float32),
                "label": labels.astype(np.int32),
                "cand_gt": cand_gt,
                "gt_boxes": gt_boxes,
                "pbd": np.zeros((len(boxes), 2048), np.float16),
            })
        if require_complete and missing:
            print(f"[l10:{gpu}] skip {out_key} missing_dets={missing} "
                  f"({len(imgs)})", flush=True)
            continue
        clip_feats = encode_clip(model, crops, device)
        for fr, (start, count) in zip(frames, spans):
            fr["clip"] = clip_feats[start:start + count]
        w = imgs[0].get("width", 1280) if imgs else 1280
        h = imgs[0].get("height", 720) if imgs else 720
        rec = {"video_id": out_key, "image_size": [w, h], "frames": frames}
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rec, f)
        os.replace(tmp, out_path)
        n_det = sum(len(fr["boxes"]) for fr in frames)
        n_match = sum(1 for fr in frames for g in fr["cand_gt"] if g is not None)
        print(f"[l10:{gpu}] {out_key} frames={len(frames)} "
              f"dets={n_det} matched={n_match} missing={missing}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,6,7")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "l10" / "data"
                                         / "tao_train"))
    ap.add_argument("--dets-root", default=str(DEFAULT_DETS))
    ap.add_argument("--gt-json", default=TAO_GT)
    ap.add_argument("--ctao-json", default=CTAO_JSON)
    ap.add_argument("--require-complete", action="store_true",
                    help="skip videos whose DLA dets are not complete")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dets_root = Path(args.dets_root)

    gt = json.load(open(args.gt_json))
    ctao = json.load(open(args.ctao_json))
    ctao_by_file = {}
    for im in ctao["images"]:
        ctao_by_file[im["file_name"]] = im["id"]
    ctao_anns = defaultdict(list)
    for a in ctao["annotations"]:
        ctao_anns[a["image_id"]].append(a)
    ctao_by_file = {f: ctao_anns[i] for f, i in ctao_by_file.items()}
    tao_anns_by_img = defaultdict(list)
    for a in gt["annotations"]:
        tao_anns_by_img[a["image_id"]].append(a)

    videos = sorted(gt["videos"], key=lambda v: v["id"])
    if args.max_videos:
        videos = videos[:args.max_videos]
    if args.worker:
        shard = [v for i, v in enumerate(videos)
                 if i % args.num_shards == args.shard]
        worker(args.gpu, [(v["name"], v["id"]) for v in shard],
               gt["images"], ctao_by_file, tao_anns_by_img,
               dets_root, out_dir, args.require_complete)
        return
    gpus = [int(x) for x in args.gpus.split(",")]
    import subprocess
    procs = []
    for si, gpu in enumerate(gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker", "--gpu", str(gpu), "--shard", str(si),
            "--num-shards", str(len(gpus)),
            "--out", str(out_dir), "--dets-root", str(dets_root),
            "--gt-json", args.gt_json, "--ctao-json", args.ctao_json,
        ]
        if args.require_complete:
            cmd.append("--require-complete")
        log = (ROOT / "outputs" / "l10" / "logs" /
               f"build_l10_worker{si}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "wb") as f:
            p = subprocess.Popen(cmd, env=env, stdout=f,
                                 stderr=subprocess.STDOUT)
        procs.append(p)
    for p in procs:
        p.wait()
    print("[l10] builder workers done", flush=True)

    # rebuild index (pick up anything produced by a previous partial run)
    index = {"videos": {}}
    for p in sorted(out_dir.glob("*.pkl")):
        rec = pickle.load(open(p, "rb"))
        index["videos"][rec["video_id"]] = {
            "path": str(p), "frames": len(rec["frames"])}
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"[l10] index videos={len(index['videos'])}", flush=True)


if __name__ == "__main__":
    main()
