"""Stage L10: build Refer-KITTI-V2 RMOT caches for the shared UIDM.

Creates:
- outputs/l10/data/rmot_kitti/<seq>.pkl  (frames with DLA dets, CLIP,
  PBD zeros, C-TAO-style KITTI GT alignment)
- outputs/l10/data/rmot_kitti/expressions.json (861 official seqmap
  queries + all expressions, with CLIP text specs)
- outputs/l10/data/rmot_kitti/gt_template/<seq>/<expr>/gt.txt (official
  RMOT TrackEval GT)

Usage:
  python tools/build_l10_refer_kitti.py --gpus 4,6,7
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

KITTI_IMGS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/KITTI_tracking"
                  "/training/image_02")
V2_ROOT = Path("/data1/LWR/vranlee/MFT2025/REFER-MFT25/refer-kitti-v2")
DETS_ROOT = ROOT / "outputs" / "l10" / "cache" / "kitti_dets"
OUT = ROOT / "outputs" / "l10" / "data" / "rmot_kitti"
TRAIN_LIST = (Path("/data1/LWR/vranlee/SERVER_ONLY/avis/"
                   "LocateMOT_reference_repos") / "temp_rmot" /
              "datasets" / "data_path" / "refer-kitti-v2.train")
SEQMAP = (Path("/data1/LWR/vranlee/SERVER_ONLY/avis/"
               "LocateMOT_reference_repos") / "temp_rmot" /
          "datasets" / "data_path" / "seqmap.txt")
EVAL_SEQS = {"0005", "0011", "0013", "0019"}

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


def load_labels(seq):
    """labels_with_ids -> {frame: [(class, track_id, xyxy)]}."""
    d = V2_ROOT / "labels_with_ids" / "image_02" / seq
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.txt")):
        frame = int(p.stem)
        rows = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            tid = int(float(parts[1]))
            x1, y1, w, h = [float(v) for v in parts[2:6]]
            rows.append((cls, tid, x1, y1, w, h))
        out[frame] = rows
    return out


def load_expressions(seq):
    d = V2_ROOT / "expression" / seq
    if not d.is_dir():
        return {}
    out = {}
    for p in sorted(d.glob("*.json")):
        obj = json.loads(p.read_text())
        out[p.stem] = {
            "expression": p.stem,
            "sentence": obj.get("sentence", p.stem),
            "raw_sentence": obj.get("raw_sentence", ""),
            "label": {str(k): [str(x) for x in v]
                      for k, v in obj.get("label", {}).items()},
            "ignore": obj.get("ignore", []),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,6,7")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true",
                    help="only the 4 official evaluation sequences")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gt_template").mkdir(parents=True, exist_ok=True)

    # frame list per sequence: official train list for train seqs,
    # all frames for the 4 official eval sequences.
    seq_frames = {}
    for line in TRAIN_LIST.read_text().splitlines():
        if not line.strip():
            continue
        rel = line.replace("KITTI/training/image_02/", "")
        seq, fname = rel.split("/", 1)
        seq_frames.setdefault(seq, []).append(int(fname.replace(".png", "")))
    for seq in EVAL_SEQS:
        img_dir = KITTI_IMGS / seq
        if img_dir.is_dir():
            seq_frames[seq] = sorted(
                int(p.stem) for p in img_dir.glob("*.png"))
    seqs = sorted(seq_frames)
    if args.eval_only:
        seqs = [s for s in seqs if s in EVAL_SEQS]
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]

    # expressions (all) + official seqmap
    all_exps = {s: load_expressions(s) for s in seqs}
    seqmap = [l.strip() for l in SEQMAP.read_text().splitlines() if l.strip()]

    # CLIP text specs for every expression (all 9.7k) on one GPU
    gpu0 = int(args.gpus.split(",")[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu0)
    import clip
    device = "cuda"
    cm, _ = clip.load("ViT-B/32", device=device)
    cm.eval()
    spec_cache = {}
    todo = sorted({e["sentence"] for s in all_exps
                   for e in all_exps[s].values()})
    for i in range(0, len(todo), 256):
        chunk = todo[i:i + 256]
        toks = clip.tokenize(chunk).to(device)
        with torch.no_grad():
            feats = cm.encode_text(toks).float().cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        for t, f in zip(chunk, feats):
            spec_cache[t] = f.astype(np.float32)
    print(f"[l10rkit] specs={len(spec_cache)}", flush=True)

    # per-sequence video pkl
    exp_meta = {}
    for seq in seqs:
        labels = load_labels(seq)
        img_dir = KITTI_IMGS / seq
        frames = []
        crops = []
        spans = []
        missing_dets = 0
        for frame in sorted(seq_frames[seq]):
            det_p = DETS_ROOT / seq / f"{frame:06d}.pth"
            if not det_p.exists():
                missing_dets += 1
                det = None
            else:
                det = pickle.load(open(det_p, "rb"))
            boxes = det["det_bboxes"].numpy().astype(np.float32) \
                if det is not None else np.zeros((0, 5), np.float32)
            det_labels = det["det_labels"].numpy().astype(np.int64) \
                if det is not None else np.zeros((0,), np.int64)
            if len(boxes):
                keep = boxes[:, 4] >= 0.05
                boxes = boxes[keep]
                det_labels = det_labels[keep]
            gt_boxes = {}
            img_path = img_dir / f"{frame:06d}.png"
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
                if x2 - x1 < 2 or y2 - y1 < 2:
                    crops.append(np.zeros((2, 2, 3), np.uint8))
                else:
                    crops.append(arr[y1:y2, x1:x2])
            spans.append((start, len(boxes)))
            frames.append({
                "frame": frame,
                "boxes": boxes[:, :4].astype(np.float32),
                "gen": boxes[:, 4].astype(np.float32),
                "label": det_labels.astype(np.int32),
                "cand_gt": cand_gt,
                "gt_boxes": gt_boxes,
                "pbd": np.zeros((len(boxes), 2048), np.float16),
            })
        clip_feats = encode_clip(cm, crops, device)
        for fr, (start, count) in zip(frames, spans):
            fr["clip"] = clip_feats[start:start + count]
        rec = {"video_id": seq, "image_size": [W, H], "frames": frames,
               "missing_dets": missing_dets}
        with open(OUT / f"{seq}.pkl", "wb") as f:
            pickle.dump(rec, f)
        exps = list(all_exps[seq].values())
        exp_meta[seq] = [
            {**e, "spec": spec_cache[e["sentence"]].tolist()} for e in exps
        ]
        print(f"[l10rkit] {seq} frames={len(frames)} "
              f"dets={sum(len(f['boxes']) for f in frames)} "
              f"missing={missing_dets} exps={len(exps)}", flush=True)

    with open(OUT / "expressions.json", "w") as f:
        json.dump(exp_meta, f, indent=1)

    # GT templates for official seqmap entries
    n_gt = 0
    for line in seqmap:
        seq, expr = line.split("+", 1)
        if seq not in all_exps:
            continue
        e = all_exps[seq].get(expr)
        if e is None:
            continue
        labels = load_labels(seq)
        d = OUT / "gt_template" / seq / expr
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        labels = load_labels(seq)
        img_dir = KITTI_IMGS / seq
        # first image gives W,H
        first = sorted(seq_frames[seq])[0]
        _arr = cv2.imread(str(img_dir / f"{first:06d}.png"), cv2.IMREAD_COLOR)
        H0, W0 = _arr.shape[:2] if _arr is not None else (375, 1242)
        for frame_s, ids in e["label"].items():
            frame = int(frame_s)
            idset = {int(x) for x in ids}
            for cls, tid, x1, y1, w, h in labels.get(frame, []):
                if tid in idset:
                    rows.append((frame + 1, tid, x1 * W0, y1 * H0,
                                 w * W0, h * H0, 1, 1, 1))
        rows.sort()
        with open(d / "gt.txt", "w") as f:
            for r in rows:
                f.write(",".join(
                    f"{v:.3f}" if isinstance(v, float) else str(v)
                    for v in r) + "\n")
        n_gt += 1
    print(f"[l10rkit] gt_templates={n_gt} done", flush=True)


if __name__ == "__main__":
    main()
