#!/usr/bin/env python
"""Stage L1-A D-CTRL: official OC-SORT/YOLOX-X DanceTrack detections.

Run with the OC-SORT conda env:
  /home/lwr/anaconda3/envs/OC-SORT/bin/python tools/cache_dancetrack_yolox.py --split val --gpu 9

Uses OC-SORT official vendored YOLOX (MIT/Apache-2.0) + ByteTrack official
DanceTrack YOLOX-X weights. Saves MOTChallenge-style detections per frame:
frame,x,y,w,h,score in {cache_root}/{video}/{frame:06d}.txt.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCSORT = os.path.join(ROOT, "references", "association_2025_2026", "OC-SORT")
sys.path.insert(0, OCSORT)

from yolox.data.data_augment import preproc  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

DANCETRACK = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"
WEIGHTS = "/data3/testdata/vranlee/code_previous/HybridSORT/pretrained/bytetrack_dance_model.pth.tar"


def image_size(vid, split):
    data_dir = "train" if split == "calibration" else split
    p = os.path.join(DANCETRACK, data_dir, vid, "seqinfo.ini")
    cfg = configparser.ConfigParser()
    cfg.read(p)
    return int(cfg["Sequence"]["imWidth"]), int(cfg["Sequence"]["imHeight"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "calibration", "val"], required=True)
    ap.add_argument("--gpu", type=int, default=9)
    ap.add_argument("--out", default="/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/detections_ctrl")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--nms", type=float, default=0.7)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True

    split_cfg = json.load(open(os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{args.split}.json")))
    vids = [v["video_id"] for v in split_cfg["videos"]]
    exp = get_exp(os.path.join(OCSORT, "exps/example/mot/yolox_dancetrack_val.py"), None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    model = exp.get_model()
    model.cuda().eval()
    ck = torch.load(WEIGHTS, map_location="cuda:0")
    model.load_state_dict(ck["model"])
    print(f"[dctrl] model loaded, videos={len(vids)}", flush=True)

    for vi, vid in enumerate(vids):
        img_w, img_h = image_size(vid, args.split)
        data_dir = "train" if args.split == "calibration" else args.split
        img_dir = os.path.join(DANCETRACK, data_dir, vid, "img1")
        out_dir = os.path.join(args.out, vid)
        os.makedirs(out_dir, exist_ok=True)
        names = sorted(os.listdir(img_dir))
        for n in names:
            fid = int(os.path.splitext(n)[0])
            out_path = os.path.join(out_dir, f"{fid:06d}.txt")
            if os.path.exists(out_path):
                continue
            img = cv2.imread(os.path.join(img_dir, n))
            if img is None:
                continue
            h, w = img.shape[:2]
            padded, r = preproc(img, (800, 1440), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            tensor = torch.from_numpy(padded).unsqueeze(0).cuda()
            with torch.no_grad():
                out = model(tensor)
                dets = postprocess(out, exp.num_classes, args.conf, args.nms)[0]
            rows = []
            if dets is not None and len(dets):
                dets = dets.cpu().numpy()
                boxes = dets[:, :4] / r
                scores = dets[:, 4] * dets[:, 5]
                for b, s in zip(boxes, scores):
                    x1, y1, x2, y2 = b
                    x1 = max(0.0, min(float(x1), w))
                    y1 = max(0.0, min(float(y1), h))
                    x2 = max(0.0, min(float(x2), w))
                    y2 = max(0.0, min(float(y2), h))
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
                    rows.append((fid, x1, y1, x2 - x1, y2 - y1, float(s)))
            with open(out_path, "w") as f:
                for row in rows:
                    f.write(f"{row[0]},{row[1]:.2f},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.4f}\n")
        print(f"[dctrl] {vid} done ({len(names)} frames)", flush=True)
    print("[dctrl] finished", flush=True)


if __name__ == "__main__":
    main()
