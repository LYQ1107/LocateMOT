"""Stage L9: Detic dets for a subset of TAO train videos.

Same Detic SwinB checkpoint and output layout as the L7 script, but only
the videos listed in --video-names (e.g. the 105-video L6/L1B pilot set or
any chosen subset) are processed.

Usage (masaenv):
  python tools/generate_l9_tao_train_dets_subset.py --gpus 4,6 \
      --video-names outputs/l9/data/tao_train_videos.json \
      --out outputs/l9/data/tao_train_dets
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
TAO_ROOT = ("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
            "TAO-Amodal")
DETIC_CFG = (MASA_ROOT + "/projects/Detic_new/configs/"
             "detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.py")
DETIC_CKPT = MASA_ROOT + "/saved_models/masa_models/detic_masa.pth"
TRAIN_GT = TAO_ROOT + "/annotations/train.json"


def worker(gpu, videos, out_root):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import patch_adaptive_roi_align  # noqa: F401  (fixes torchvision adaptive ROIAlign OOM)
    os.chdir(MASA_ROOT)
    sys.path.insert(0, MASA_ROOT)
    from mmdet.apis import init_detector, inference_detector
    model = init_detector(DETIC_CFG, None, device="cuda")
    ck = torch.load(DETIC_CKPT, map_location="cpu")
    sd = {k[9:]: v for k, v in ck["state_dict"].items()
          if k.startswith("detector.")}
    model.load_state_dict(sd, strict=True)
    model = model.eval()
    model.cfg.test_dataloader.dataset.pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(480, 288), keep_ratio=True),
        dict(type="PackDetInputs"),
    ]
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
                keep = s >= 0.05
                b, s, l = b[keep], s[keep], l[keep]
                if len(b) > 50:
                    idx = np.argsort(-s)[:50]
                    b, s, l = b[idx], s[idx], l[idx]
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
    ap.add_argument("--gpus", default="4,6")
    ap.add_argument("--out", default="outputs/l9/data/tao_train_dets")
    ap.add_argument("--video-names", required=True)
    ap.add_argument("--max-frames-per-video", type=int, default=0)
    args = ap.parse_args()
    args.out = str(Path(args.out).resolve())
    names = set(json.load(open(args.video_names)))
    gt = json.load(open(TRAIN_GT))
    vids = {v["id"]: v["name"] for v in gt["videos"]}
    keep_ids = {vid for vid, name in vids.items() if name in names}
    imgs = [i for i in gt["images"] if i["video_id"] in keep_ids]
    if args.max_frames_per_video:
        by_vid = {}
        for i in imgs:
            by_vid.setdefault(i["video_id"], []).append(i)
        imgs = [i for vid in by_vid
                for i in by_vid[vid][:args.max_frames_per_video]]
    imgs.sort(key=lambda x: (x["video_id"], x["frame_index"]))
    items = [(i["file_name"],
              str(Path(TAO_ROOT) / "frames" / i["file_name"]))
             for i in imgs]
    print(f"[dets] videos={len(keep_ids)} frames={len(items)}", flush=True)
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
