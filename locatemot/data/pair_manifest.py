"""Frozen split loading, subset selection and pair-manifest helpers."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

import numpy as np


def load_frozen_split(path: str) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["videos"]


def select_subset(entries: List[dict], n_youtube: int, n_mose: int, seed: int) -> List[dict]:
    youtube = [e for e in entries if "youtube" in e["dataset"]]
    mose = [e for e in entries if "mose" in e["dataset"]]
    rng = np.random.RandomState(seed)
    sel = []
    for pool, n in ((youtube, n_youtube), (mose, n_mose)):
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        sel.extend(pool[i] for i in idx)
    return sel


def split_hash(entries: List[dict]) -> str:
    payload = json.dumps(
        sorted((e["dataset"], e["video_id"]) for e in entries), sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def overlap(entries_a: List[dict], entries_b: List[dict]) -> List[tuple]:
    set_b = {(e["dataset"], e["video_id"]) for e in entries_b}
    return [(e["dataset"], e["video_id"]) for e in entries_a if (e["dataset"], e["video_id"]) in set_b]


def write_split_json(path: str, split_name: str, entries: List[dict], seed: int, source: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "schema_version": 2,
        "stage": "L0-C",
        "split": split_name,
        "seed": seed,
        "source_frozen_split": source,
        "hash": split_hash(entries),
        "summary": {
            "video_count": len(entries),
            "dataset_videos": {
                ds: sum(1 for e in entries if e["dataset"] == ds)
                for ds in sorted({e["dataset"] for e in entries})
            },
        },
        "videos": entries,
    }
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def choose_frames(video_len: int, n: int = 5) -> List[int]:
    if video_len <= 1:
        return [0]
    targets = [0, 3, 10, 30, 80][:n]
    frames = []
    for t in targets:
        if t < video_len:
            frames.append(t)
    if not frames:
        frames = [0]
    if len(frames) < min(n, video_len):
        extra = sorted(set(np.linspace(0, video_len - 1, min(n, video_len)).astype(int).tolist()))
        for f in extra:
            if f not in frames:
                frames.append(f)
        frames.sort()
    return frames[: min(n, video_len)]


def mask_boxes_for_frame(mask_path: str) -> Dict[int, List[float]]:
    from PIL import Image
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


def youtube_meta(root: str) -> dict:
    with open(os.path.join(root, "meta.json")) as f:
        return json.load(f)


def mose_meta(root: str) -> dict:
    with open(os.path.join(os.path.dirname(root), "meta_train.json")) as f:
        return json.load(f)


def video_frame_names(dataset: str, video_id: str, roots: dict) -> List[str]:
    root = roots["youtube" if "youtube" in dataset else "mose"]
    jpg_dir = os.path.join(root, "JPEGImages", video_id)
    if not os.path.isdir(jpg_dir):
        return []
    names = sorted(f for f in os.listdir(jpg_dir) if f.endswith(".jpg"))
    return [os.path.splitext(f)[0] for f in names]
