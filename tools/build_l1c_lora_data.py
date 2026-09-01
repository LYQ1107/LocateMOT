"""Stage L1-C: build LocateAnything LoRA grounding JSONL from train annotations.

Official format (Embodied/document/DATA_PREPARATION.md):
  {"conversations":[{"from":"human","value":"Detect all objects in <image-1> that match ..."},
   {"from":"gpt","value":"<ref>cat</ref><box>(x1,y1,x2,y2)</box>..."}],
   "image":"/abs/path.jpg"}
Coordinates are normalized integers in [0,1000].

Usage:
  python tools/build_l1c_lora_data.py --datasets dancetrack,bdd100k,tao_amodal,mot17,mot20 \
      --max-per-video 40
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
OUT = ROOT / "outputs" / "l1_c" / "lora"

DANCETRACK = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train")
MOT17 = Path("/data1/LWR/vranlee/DATASETS/JDE/MOT17/train")
MOT20 = Path("/data1/LWR/vranlee/M4FTMoveOut4Doing/ByteTrack-mbt/datasets/mix_mot20_ch/mot20_train")
BDD_LABELS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/train")
BDD_IMAGES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/bdd100k/images/track/train")
TAO_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal")


def _norm(v, n):
    return int(round(max(0.0, min(1.0, v / max(1, n))) * 1000))


def _box_str(b, w, h):
    x1, y1, x2, y2 = b
    return (f"<{_norm(x1, w)}><{_norm(y1, h)}>"
            f"<{_norm(x2, w)}><{_norm(y2, h)}>")


def build_dancetrack(max_per_video, out):
    rows = 0
    with open(out / "dancetrack.jsonl", "w") as f:
        for vid in sorted(os.listdir(DANCETRACK)):
            seq = DANCETRACK / vid
            gt_path = seq / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            frames = defaultdict(list)
            for line in gt_path.read_text().splitlines():
                p = line.split(",")
                if len(p) < 9 or int(p[7]) != 1:
                    continue
                fid, tid = int(p[0]), int(p[1])
                x, y, w, h = map(float, p[2:6])
                frames[fid].append((tid, x, y, x + w, y + h))
            fids = sorted(frames)
            step = max(1, len(fids) // max_per_video)
            img_dir = seq / "img1"
            for i, fid in enumerate(fids):
                if i % step:
                    continue
                img = img_dir / f"{fid:08d}.jpg"
                if not img.exists():
                    continue
                boxes = frames[fid]
                parts = "".join(f"<ref>person</ref><box>{_box_str(b[1:], 1920, 1080)}</box>"
                                for b in boxes)
                item = {
                    "conversations": [
                        {"from": "human",
                         "value": "Detect all objects in <image-1> that match the following description: person."},
                        {"from": "gpt", "value": parts},
                    ],
                    "image": str(img),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows += 1
    return rows


def build_mot(seq_root, ds_name, max_per_video, out):
    rows = 0
    with open(out / f"{ds_name}.jsonl", "w") as f:
        for seq in sorted(x for x in seq_root.iterdir() if x.is_dir()):
            gt_path = seq / "gt" / "gt.txt"
            if not gt_path.exists():
                continue
            frames = defaultdict(list)
            for line in gt_path.read_text().splitlines():
                p = line.split(",")
                if len(p) < 9 or int(p[6]) != 1 or int(p[7]) != 1:
                    continue
                fid, tid = int(p[0]), int(p[1])
                x, y, w, h = map(float, p[2:6])
                frames[fid].append((tid, x, y, x + w, y + h))
            fids = sorted(frames)
            step = max(1, len(fids) // max_per_video)
            img_dir = seq / "img1"
            for i, fid in enumerate(fids):
                if i % step:
                    continue
                img = img_dir / f"{fid:06d}.jpg"
                if not img.exists():
                    continue
                parts = "".join(f"<ref>person</ref><box>{_box_str(b[1:], 1920, 1080)}</box>"
                                for b in frames[fid])
                item = {
                    "conversations": [
                        {"from": "human",
                         "value": "Detect all objects in <image-1> that match the following description: person."},
                        {"from": "gpt", "value": parts},
                    ],
                    "image": str(img),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows += 1
    return rows


def build_bdd(max_per_video, out):
    rows = 0
    with open(out / "bdd100k.jsonl", "w") as f:
        for lab_path in sorted(BDD_LABELS.glob("*.json")):
            vid = lab_path.stem
            v = json.loads(lab_path.read_text())
            if len(v) > max_per_video:
                idx = set(round(i * (len(v) - 1) / (max_per_video - 1))
                          for i in range(max_per_video))
                v = [fr for i, fr in enumerate(v) if i in idx]
            for fr in v:
                img = BDD_IMAGES / vid / fr["name"]
                if not img.exists():
                    continue
                parts = []
                cats = set()
                for lab in fr.get("labels", []):
                    b = lab.get("box2d")
                    if b is None:
                        continue
                    cat = lab.get("category", "object")
                    cats.add(cat)
                    parts.append((cat, (b["x1"], b["y1"], b["x2"], b["y2"])))
                if not parts:
                    continue
                cat_text = ", ".join(sorted(cats)[:8])
                out_str = "".join(f"<ref>{c}</ref><box>{_box_str(b, 1280, 720)}</box>"
                                  for c, b in parts)
                item = {
                    "conversations": [
                        {"from": "human",
                         "value": (f"Detect all objects in <image-1> that match the "
                                   f"following description: {cat_text}.")},
                        {"from": "gpt", "value": out_str},
                    ],
                    "image": str(img),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows += 1
    return rows


def build_tao(max_per_video, out):
    ann = json.loads((TAO_ROOT / "annotations" / "train.json").read_text())
    cat_name = {c["id"]: c["name"] for c in ann["categories"]}
    ann_by_img = defaultdict(list)
    for a in ann["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    img_by_video = defaultdict(list)
    for im in ann["images"]:
        img_by_video[im["video"]].append(im)
    rows = 0
    with open(out / "tao_amodal.jsonl", "w") as f:
        for vid, ims in img_by_video.items():
            ims.sort(key=lambda x: x["frame_index"])
            if len(ims) > max_per_video:
                idx = set(round(i * (len(ims) - 1) / (max_per_video - 1))
                          for i in range(max_per_video))
                ims = [im for i, im in enumerate(ims) if i in idx]
            for im in ims:
                img = TAO_ROOT / "frames" / im["file_name"]
                if not img.exists():
                    continue
                boxes = []
                cats = set()
                for a in ann_by_img.get(im["id"], []):
                    b = a["bbox"]
                    cat = cat_name.get(a["category_id"], "object")
                    cats.add(cat)
                    boxes.append((cat, (b[0], b[1], b[0] + b[2], b[1] + b[3])))
                if not boxes:
                    continue
                cat_text = ", ".join(sorted(cats)[:8])
                out_str = "".join(f"<ref>{c}</ref><box>{_box_str(b, 1920, 1080)}</box>"
                                  for c, b in boxes)
                item = {
                    "conversations": [
                        {"from": "human",
                         "value": (f"Detect all objects in <image-1> that match the "
                                   f"following description: {cat_text}.")},
                        {"from": "gpt", "value": out_str},
                    ],
                    "image": str(img),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dancetrack,bdd100k,tao_amodal,mot17,mot20")
    ap.add_argument("--max-per-video", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        if ds == "dancetrack":
            total[ds] = build_dancetrack(args.max_per_video, out)
        elif ds == "bdd100k":
            total[ds] = build_bdd(args.max_per_video, out)
        elif ds == "tao_amodal":
            total[ds] = build_tao(args.max_per_video, out)
        elif ds in ("mot17", "mot20"):
            root = MOT17 if ds == "mot17" else MOT20
            total[ds] = build_mot(root, ds, args.max_per_video, out)
        print(f"[lora-data] {ds}: {total[ds]}", flush=True)
    (out / "stats.json").write_text(json.dumps(total, indent=2))
    print("[lora-data] done", flush=True)


if __name__ == "__main__":
    main()
