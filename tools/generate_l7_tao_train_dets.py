"""Stage L7: generate TAO train detections with the official Detic SwinB
checkpoint (local detic_masa.pth, verified key-compatible with the official
mmdet Detic config) for OVMOT joint training.

Run with the masaenv python:
  /home/lwr/anaconda3/envs/masaenv/bin/python tools/generate_l7_tao_train_dets.py \
      --gpus 4,5,6,7 --out outputs/l7/data/tao_train_dets

Output layout (same convention as the public val dets):
  <out>/train/<dataset>/<video_stem>/frameXXXX.pth  (or <stem>.pth)
  pickle {"det_bboxes": [N,5] xyxy+score, "det_labels": [N]} lvis v1 ids
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

MASA_ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa"
TAO_ROOT = ("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal")
DETIC_CFG = (MASA_ROOT + "/projects/Detic_new/configs/"
             "detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.py")
DETIC_CKPT = MASA_ROOT + "/saved_models/masa_models/detic_masa.pth"
TRAIN_GT = TAO_ROOT + "/annotations/train.json"


def worker(gpu, videos, out_root):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, MASA_ROOT)
    from mmdet.apis import init_detector, inference_detector
    model = init_detector(DETIC_CFG, None, device="cuda")
    ck = torch.load(DETIC_CKPT, map_location="cpu")
    sd = {k[9:]: v for k, v in ck["state_dict"].items()
          if k.startswith("detector.")}
    model.load_state_dict(sd, strict=True)
    model = model.eval()
    model.cfg.model.test_cfg.rcnn.score_thr = 0.05
    model.cfg.model.test_cfg.rcnn.max_per_img = 50
    for name, img_path in videos:
        parts = name.split("/")
        stem = parts[-1].replace(".jpg", "")
        if stem.startswith("frame"):
            fname = f"frame{int(stem[5:]):04d}.pth"
        else:
            fname = f"{stem}.pth"
        out_path = Path(out_root) / parts[0] / parts[1] / parts[2] / fname
        if out_path.exists():
            continue
        try:
            res = inference_detector(model, str(img_path))
            pred = res.pred_instances
            if len(pred) == 0:
                boxes = np.zeros((0, 5), np.float32)
                labels = np.zeros((0,), np.int64)
            else:
                b = pred.bboxes.detach().cpu().numpy()
                s = pred.scores.detach().cpu().numpy()
                l = pred.labels.detach().cpu().numpy()
                boxes = np.concatenate([b, s[:, None]], axis=1)
                labels = l.astype(np.int64)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump({"det_bboxes": torch.from_numpy(boxes),
                             "det_labels": torch.from_numpy(labels)}, f)
        except Exception as e:
            print(f"[dets:{gpu}] fail {name}: {e}", flush=True)
    print(f"[dets:{gpu}] done {len(videos)} frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--out", default="outputs/l7/data/tao_train_dets")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()
    gt = json.load(open(TRAIN_GT))
    imgs = sorted(gt["images"], key=lambda x: (x["video"], x["frame_index"]))
    if args.max_images:
        imgs = imgs[:args.max_images]
    items = [(i["file_name"], str(Path(TAO_ROOT) / "frames" / i["file_name"]))
             for i in imgs]
    gpus = [int(x) for x in args.gpus.split(",")]
    shards = [[] for _ in gpus]
    for i, item in enumerate(items):
        shards[i % len(gpus)].append(item)
    procs = []
    for gpu, shard in zip(gpus, shards):
        p = torch.multiprocessing.Process(
            target=worker, args=(gpu, shard, args.out))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[dets] all done", flush=True)


if __name__ == "__main__":
    main()
