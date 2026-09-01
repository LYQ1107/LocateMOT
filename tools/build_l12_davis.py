"""Stage L12: build DAVIS 2017 val prompt-seeded tracking pkls.

For each val video:
  - candidates: DLA (Detic-SwinB) detections per frame (same shared
    observation stream as MOT/OVMOT/RMOT);
  - GT: object masks from DAVIS Annotations/480p; per-object bbox;
  - controlled prompt types derived from the FIRST-frame GT mask:
      mask = GT mask (0/1),
      box  = tight bbox of mask,
      point= first interior pixel scanning the bbox (deterministic).
  - stores seeds and per-frame GT object boxes for identity metrics.

Output: outputs/l12/data/davis/<video>.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.build_l10_refer_kitti import iou, match_dets  # noqa: E402

DAVIS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/DAVIS/DAVIS")
VAL_LIST = DAVIS / "ImageSets" / "2017" / "val.txt"
MASKS = DAVIS / "Annotations" / "480p"
FRAMES = DAVIS / "JPEGImages" / "480p"
DETS = ROOT / "outputs" / "l12" / "cache" / "davis_dets"
OUT = ROOT / "outputs" / "l12" / "data" / "davis"


def load_mask(path):
    # DAVIS annotations are palette PNGs; PIL returns palette indices
    # (the actual object ids), while cv2 returns palette colors.
    return np.asarray(Image.open(path), np.uint8)


def mask_bbox(mask, obj_id):
    ys, xs = np.nonzero(mask == obj_id)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
            int(ys.max()) + 1]


def interior_point(mask, obj_id, box):
    x1, y1, x2, y2 = box
    for y in range(y1, y2):
        for x in range(x1, x2):
            if mask[y, x] == obj_id:
                return [x, y]
    return [(x1 + x2) // 2, (y1 + y2) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    videos = [l.strip() for l in VAL_LIST.read_text().splitlines() if l.strip()]
    if args.max_videos:
        videos = videos[:args.max_videos]
    # DAVIS val annotation ids -> contiguous object ids
    for vi, vid in enumerate(videos):
        out_path = OUT / f"{vid}.pkl"
        if out_path.exists() and not args.force:
            print(f"[l12davis] skip {vid}", flush=True)
            continue
        mask_dir = MASKS / vid
        frame_dir = FRAMES / vid
        if not mask_dir.is_dir():
            print(f"[l12davis] missing masks {vid}", flush=True)
            continue
        frame_files = sorted(frame_dir.glob("*.jpg"))
        # object ids from first frame
        m0 = load_mask(mask_dir / "00000.png")
        obj_ids = sorted(int(x) for x in np.unique(m0) if int(x) > 0)
        # map DAVIS ids (128,255,...) to contiguous 1..K
        id_map = {oid: i + 1 for i, oid in enumerate(obj_ids)}
        seeds = {}
        for oid in obj_ids:
            box = mask_bbox(m0, oid)
            if box is None:
                continue
            seeds[id_map[oid]] = {
                "mask": (m0 == oid).astype(np.uint8),
                "box": box,
                "point": interior_point(m0, oid, box),
            }
        frames = []
        for fi, fp in enumerate(frame_files):
            frame = fi
            det_p = DETS / vid / f"{frame:05d}.pth"
            if det_p.exists():
                det = pickle.load(open(det_p, "rb"))
                boxes = det["det_bboxes"].numpy().astype(np.float32)
                labs = det["det_labels"].numpy().astype(np.int64)
            else:
                boxes = np.zeros((0, 5), np.float32)
                labs = np.zeros((0,), np.int64)
            if len(boxes):
                keep = boxes[:, 4] >= args.score_thr
                boxes, labs = boxes[keep], labs[keep]
                if len(boxes) > args.topk:
                    idx = np.argsort(-boxes[:, 4])[:args.topk]
                    boxes, labs = boxes[idx], labs[idx]
            m = load_mask(mask_dir / f"{frame:05d}.png")
            gt_boxes = {}
            for oid in obj_ids:
                if oid in id_map:
                    b = mask_bbox(m, oid)
                    if b is not None:
                        gt_boxes[str(id_map[oid])] = b
            cand_gt = match_dets(boxes[:, :4], gt_boxes) if len(boxes) else []
            frames.append({
                "frame": frame,
                "boxes": boxes[:, :4].astype(np.float32),
                "gen": boxes[:, 4].astype(np.float32),
                "label": labs.astype(np.int32),
                "cand_gt": cand_gt,
                "gt_boxes": gt_boxes,
                "clip": np.zeros((len(boxes), 512), np.float16),
                "pbd": np.zeros((len(boxes), 2048), np.float16),
            })
        img = cv2.imread(str(frame_dir / frame_files[0].name))
        H, W = img.shape[:2]
        rec = {"video_id": vid, "image_size": [W, H], "frames": frames,
               "seeds": seeds, "obj_ids": obj_ids}
        with open(out_path, "wb") as f:
            pickle.dump(rec, f)
        print(f"[l12davis] {vid} frames={len(frames)} objs={len(seeds)} "
              f"{vi+1}/{len(videos)}", flush=True)


if __name__ == "__main__":
    main()
