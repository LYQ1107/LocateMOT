"""Stage L11: Refer-KITTI/V2 candidate-front-end repair.

Builds reduced, higher-precision RMOT candidate pkls from the L10 DLA
detections:
  1. LVIS category whitelist (data-driven from KITTI GT on train seqs,
     ids are LVIS v1 ids; DLA labels are 0-based -> +1).
  2. Cross-category NMS (greedy by det score, IoU >= NMS_IOU).
  3. Det score threshold + top-K per frame.
  4. CLIP crop features for the kept candidates (same ViT-B/32 as L10).
  5. Query-conditioned CLIP top-k / min-sim calibration on train seqs
     (official eval sequences never used for calibration).

Output: outputs/l11/data/rmot_kitti/<seq>.pkl with the same schema as
the L10 KITTI pkls plus `orig_idx` (original DLA candidate index, so the
existing PBD cache rows can be selected).
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

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.build_l10_refer_kitti import (  # noqa: E402
    CLIP_MEAN, CLIP_STD, encode_clip, iou, load_expressions, load_labels,
    match_dets)

KITTI_IMGS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/KITTI_tracking"
                  "/training/image_02")
DETS_ROOT = ROOT / "outputs" / "l10" / "cache" / "kitti_dets"
OUT = ROOT / "outputs" / "l11" / "data" / "rmot_kitti"
EVAL_SEQS = {"0005", "0011", "0013", "0019"}

# LVIS v1 category ids (label+1) relevant to KITTI tracking classes.
WHITELIST = {173, 207, 692, 800, 922, 1114, 1115, 1123,
             94, 703, 701, 1120, 1179, 793}


def cross_nms(boxes, scores, thr=0.7):
    """Greedy NMS across categories; returns keep mask (bool)."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(-scores)
    keep = np.zeros(n, bool)
    for i in order:
        if keep[i]:
            continue
        keep[i] = True
        for j in order:
            if keep[j] or j == i:
                continue
            if iou(boxes[i], boxes[j]) >= thr:
                keep[j] = False
    return keep


def build_seq(seq, frames, dets, labels, cm, device, score_thr=0.05,
              topk=30):
    out_frames = []
    crops = []
    spans = []
    for frame in frames:
        det = dets.get(frame)
        if det is None:
            boxes = np.zeros((0, 5), np.float32)
            labs = np.zeros((0,), np.int64)
        else:
            boxes = det["det_bboxes"].numpy().astype(np.float32)
            labs = det["det_labels"].numpy().astype(np.int64)
        if len(boxes):
            lvis = labs.astype(np.int64) + 1
            keep = (boxes[:, 4] >= score_thr) & np.isin(lvis, list(WHITELIST))
            boxes, labs = boxes[keep], labs[keep]
            if len(boxes):
                keep = cross_nms(boxes[:, :4], boxes[:, 4])
                boxes, labs = boxes[keep], labs[keep]
            if len(boxes) > topk:
                idx = np.argsort(-boxes[:, 4])[:topk]
                boxes, labs = boxes[idx], labs[idx]
        orig_idx = []
        det0 = dets.get(frame)
        if det0 is not None and len(det0["det_bboxes"]):
            all_boxes = det0["det_bboxes"].numpy().astype(np.float32)
            for b in boxes[:, :4]:
                best, bi = -1.0, -1
                for j, ab in enumerate(all_boxes[:, :4]):
                    v = iou(b, ab)
                    if v > best:
                        best, bi = v, j
                orig_idx.append(bi if best >= 0.5 else -1)
        gt_boxes = {}
        img_path = KITTI_IMGS / seq / f"{frame:06d}.png"
        arr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if arr is None:
            arr = np.zeros((2, 2, 3), np.uint8)
        H, W = arr.shape[:2]
        for cls, tid, x1, y1, w, h in labels.get(frame, []):
            px1, py1 = x1 * W, y1 * H
            px2, py2 = (x1 + w) * W, (y1 + h) * H
            gt_boxes[str(tid)] = [px1, py1, px2, py2]
        cand_gt = match_dets(boxes[:, :4], gt_boxes) if len(boxes) else []
        start = len(crops)
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                x2 = min(W, max(x2, x1 + 2))
                y2 = min(H, max(y2, y1 + 2))
            crops.append(arr[y1:y2, x1:x2] if x2 > x1 and y2 > y1
                         else np.zeros((2, 2, 3), np.uint8))
        spans.append((start, len(boxes)))
        out_frames.append({
            "frame": frame,
            "boxes": boxes[:, :4].astype(np.float32),
            "gen": boxes[:, 4].astype(np.float32),
            "label": labs.astype(np.int32),
            "cand_gt": cand_gt,
            "gt_boxes": gt_boxes,
            "orig_idx": np.asarray(orig_idx, np.int32),
            "pbd": np.zeros((len(boxes), 2048), np.float16),
        })
    clip = encode_clip(cm, crops, device)
    for fr, (start, count) in zip(out_frames, spans):
        fr["clip"] = clip[start:start + count]
    return {"video_id": seq, "image_size": [W, H], "frames": out_frames}


def spec_encode(cm, sentences, device):
    out = {}
    for i in range(0, len(sentences), 256):
        chunk = sentences[i:i + 256]
        import clip
        toks = clip.tokenize(chunk).to(device)
        with torch.no_grad():
            feats = cm.encode_text(toks).float().cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        for t, f in zip(chunk, feats):
            out[t] = f.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-out", default=str(ROOT / "results" / "l11"
                                               / "kitti_calibration.json"))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import clip
    device = "cuda"
    cm, _ = clip.load("ViT-B/32", device=device)
    cm.eval()
    OUT.mkdir(parents=True, exist_ok=True)

    # all sequences with dets
    seqs = sorted(d.name for d in DETS_ROOT.iterdir() if d.is_dir())
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    print(f"[l11rk] seqs={seqs}", flush=True)

    # expressions + specs (train + eval)
    all_exps = {s: load_expressions(s) for s in seqs}
    sentences = sorted({e["sentence"] for s in all_exps
                        for e in all_exps[s].values()})
    specs = spec_encode(cm, sentences, device)
    exp_meta = {}
    for s in seqs:
        meta = []
        for name, e in all_exps[s].items():
            m = dict(e)
            m["spec"] = specs[e["sentence"]].tolist()
            meta.append(m)
        exp_meta[s] = meta
    with open(OUT / "expressions.json", "w") as f:
        json.dump(exp_meta, f)

    for si, seq in enumerate(seqs):
        out_path = OUT / f"{seq}.pkl"
        if out_path.exists():
            print(f"[l11rk] skip {seq}", flush=True)
            continue
        labels = load_labels(seq)
        dets = {}
        for p in sorted((DETS_ROOT / seq).glob("*.pth")):
            dets[int(p.stem)] = pickle.load(open(p, "rb"))
        frames = sorted(dets)
        rec = build_seq(seq, frames, dets, labels, cm, device,
                        args.score_thr, args.topk)
        with open(out_path, "wb") as f:
            pickle.dump(rec, f)
        print(f"[l11rk] {seq} frames={len(frames)} "
              f"cands={sum(len(x['boxes']) for x in rec['frames'])} "
              f"{si+1}/{len(seqs)}", flush=True)

    if args.calibrate:
        print("[l11rk] calibration is a separate step: "
              "tools/calibrate_l11_kitti.py", flush=True)


if __name__ == "__main__":
    main()
