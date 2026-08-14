"""Stage L7: re-label OVMOT predictions with a different frozen perception
head (CLIP text cosine or GT-match oracle) without re-running the tracker.

Association (track ids) is untouched; only category ids change, so the
official TETA ClsA bottleneck can be attributed to perception vs identity.

Usage:
  python tools/l7_ovmot_relabel.py --pred pred.json --mode clip --out relabeled.json
  python tools/l7_ovmot_relabel.py --pred pred.json --mode oracle --out relabeled.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

GT_JSON = ("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/"
           "tao_val_lvis_v1_classes.json")
TEXT_EMB = ("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/metadata/"
            "lvis_v1_clip_a+cname.npy")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter) if ar + br - inter > 0 else 0.0


def load_candidates(data_dir):
    """{(video_name, frame): {'boxes','clip','label'}}"""
    store = {}
    for p in sorted(Path(data_dir).glob("*.pkl")):
        rec = pickle.load(open(p, "rb"))
        for fr in rec["frames"]:
            store[(rec["video_id"], int(fr["frame"]))] = fr
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data-dir", default="outputs/l7/data/tao_val")
    ap.add_argument("--mode", choices=["clip", "oracle"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    preds = json.load(open(args.pred))
    gt = json.load(open(GT_JSON))
    cands = load_candidates(args.data_dir)
    vid_name2id = {v["name"].replace("/", "-"): v["id"]
                   for v in gt["videos"]}
    vid_id2name = {v: k for k, v in vid_name2id.items()}
    img2vid = {i["id"]: i["video_id"] for i in gt["images"]}
    img2frame = {}
    for i in gt["images"]:
        stem = i["file_name"].rsplit("/", 1)[-1].replace(".jpg", "")
        fidx = int(stem[5:]) if stem.startswith("frame") \
            else int(i["frame_index"])
        img2frame[i["id"]] = fidx
    text = None
    if args.mode == "clip":
        text = np.load(TEXT_EMB).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
    else:
        img2anns = {}
        for a in gt["annotations"]:
            img2anns.setdefault(a["image_id"], []).append(a)
    from collections import defaultdict
    by_img = defaultdict(list)
    for i, x in enumerate(preds):
        by_img[x["image_id"]].append(i)
    n = 0

    def iou_matrix(ax, bx):
        ax = np.asarray(ax, np.float64)
        bx = np.asarray(bx, np.float64)
        ix1 = np.maximum(ax[:, None, 0], bx[None, :, 0])
        iy1 = np.maximum(ax[:, None, 1], bx[None, :, 1])
        ix2 = np.minimum(ax[:, None, 2], bx[None, :, 2])
        iy2 = np.minimum(ax[:, None, 3], bx[None, :, 3])
        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih
        ar = np.maximum(0.0, ax[:, 2] - ax[:, 0]) * np.maximum(
            0.0, ax[:, 3] - ax[:, 1])
        br = np.maximum(0.0, bx[:, 2] - bx[:, 0]) * np.maximum(
            0.0, bx[:, 3] - bx[:, 1])
        return inter / np.maximum(ar[:, None] + br[None, :] - inter, 1e-9)

    if args.mode == "clip":
        # group by (video_name, frame)
        key_of = {}
        for i, x in enumerate(preds):
            vid = img2vid[x["image_id"]]
            vname = vid_id2name.get(vid)
            key_of[i] = (vname, img2frame[x["image_id"]])
        frames = defaultdict(list)
        for i in range(len(preds)):
            frames[key_of[i]].append(i)
        for key, idxs in frames.items():
            fr = cands.get(key)
            if fr is None or len(idxs) == 0:
                continue
            pboxes = []
            for i in idxs:
                x = preds[i]
                pboxes.append([x["bbox"][0], x["bbox"][1],
                               x["bbox"][0] + x["bbox"][2],
                               x["bbox"][1] + x["bbox"][3]])
            m = iou_matrix(pboxes, fr["boxes"])
            best_j = m.argmax(1)
            best_v = m.max(1)
            clip = np.asarray(fr["clip"], np.float32)
            clip = clip / np.maximum(
                np.linalg.norm(clip, axis=1, keepdims=True), 1e-6)
            sims = text @ clip.T  # [C, N]
            for idx, j, v in zip(idxs, best_j, best_v):
                if v > 0.3:
                    preds[idx]["category_id"] = int(sims[:, j].argmax()) + 1
                    n += 1
    else:
        for image_id, idxs in by_img.items():
            anns = img2anns.get(image_id, [])
            if not anns:
                continue
            pboxes = []
            for i in idxs:
                x = preds[i]
                pboxes.append([x["bbox"][0], x["bbox"][1],
                               x["bbox"][0] + x["bbox"][2],
                               x["bbox"][1] + x["bbox"][3]])
            gboxes = [[a["bbox"][0], a["bbox"][1],
                       a["bbox"][0] + a["bbox"][2],
                       a["bbox"][1] + a["bbox"][3]] for a in anns]
            m = iou_matrix(pboxes, gboxes)
            best_j = m.argmax(1)
            best_v = m.max(1)
            for idx, j, v in zip(idxs, best_j, best_v):
                if v >= 0.5:
                    preds[idx]["category_id"] = anns[j]["category_id"]
                    n += 1
    with open(args.out, "w") as f:
        json.dump(preds, f)
    print(f"[relabel] mode={args.mode} changed={n} preds -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
