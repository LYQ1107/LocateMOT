"""Lightweight feature sanity check on YouTube-VOS pairs.

GT is used only to (1) build reference prompts, (2) decide whether a
LocateAnything candidate corresponds to a GT object, and (3) form identity
pairs. GT is NOT used to select or filter current-frame candidates.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def load_youtube_vos_meta(root: str, split: str = "train") -> dict:
    meta_path = os.path.join(root, split, "meta.json")
    with open(meta_path) as f:
        return json.load(f)


def select_videos(meta: dict, n: int = 10, min_objects: int = 2) -> List[str]:
    candidates = []
    for vid, info in meta["videos"].items():
        objs = info.get("objects", {})
        if len(objs) >= min_objects:
            candidates.append(vid)
    rng = np.random.RandomState(20260806)
    rng.shuffle(candidates)
    return candidates[:n]


def mask_boxes_for_frame(mask_path: str) -> Dict[int, List[float]]:
    mask = np.asarray(Image.open(mask_path))
    boxes = {}
    for obj_id in np.unique(mask):
        obj_id = int(obj_id)
        if obj_id == 0:
            continue
        ys, xs = np.where(mask == obj_id)
        if len(xs) == 0:
            continue
        boxes[obj_id] = [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1]
    return boxes


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_tokens_to_gt(tokens, gt_boxes, image_size, iou_thresh=0.5):
    """Returns list of (token, gt_id) for candidates matching a GT box."""
    w, h = image_size
    matches = []
    used_gt = set()
    for tok in tokens:
        best_id, best_iou = None, 0.0
        for obj_id, gt in gt_boxes.items():
            if obj_id in used_gt:
                continue
            v = iou(
                [tok.box_xyxy[0], tok.box_xyxy[1], tok.box_xyxy[2], tok.box_xyxy[3]],
                gt,
            )
            if v > best_iou:
                best_id, best_iou = obj_id, v
        if best_id is not None and best_iou >= iou_thresh:
            matches.append((tok, best_id, best_iou))
            used_gt.add(best_id)
    return matches


def feature_vector(tok, name: str) -> Optional[np.ndarray]:
    v = getattr(tok, name, None)
    return np.asarray(v, dtype=np.float32) if v is not None else None


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def write_sanity_outputs(
    pairs: List[dict],
    metrics: dict,
    out_csv: str,
    out_json: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pair_type", "video_a", "frame_a", "obj_a", "video_b", "frame_b", "obj_b",
            "pbd_box_end", "pbd_coord_mean", "pbd_full_mean", "region", "fused",
        ])
        for p in pairs:
            w.writerow([
                p["pair_type"],
                p["video_a"], p["frame_a"], p["obj_a"],
                p["video_b"], p["frame_b"], p["obj_b"],
                p.get("pbd_box_end"), p.get("pbd_coord_mean"), p.get("pbd_full_mean"),
                p.get("region"), p.get("fused"),
            ])
    with open(out_json, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
