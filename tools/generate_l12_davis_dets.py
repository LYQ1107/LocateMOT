"""Stage L12: DLA (Detic-SwinB) detections for DAVIS 2017 val frames.

Same detector/checkpoint/protocol as L10 KITTI/TAO runs (score>=0.05,
top-50).  Run with masaenv.

Usage:
  python tools/generate_l12_davis_dets.py --gpus 3 --max-videos 10
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
DAVIS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/DAVIS/DAVIS")
VAL_LIST = DAVIS / "ImageSets" / "2017" / "val.txt"
OUT = ROOT / "outputs" / "l12" / "cache" / "davis_dets"
MASA_ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa"
DETIC_CFG = (MASA_ROOT + "/projects/Detic_new/configs/"
             "detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.py")
DETIC_CKPT = MASA_ROOT + "/saved_models/masa_models/detic_masa.pth"


def worker(gpu, items):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import patch_adaptive_roi_align  # noqa: F401
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
    ok = fail = 0
    for vid, frame, img_path in items:
        out_path = OUT / vid / f"{frame:05d}.pth"
        if out_path.exists():
            ok += 1
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
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[l12davis-dets] fail {vid}/{frame}: {e}", flush=True)
    print(f"[l12davis-dets] gpu{gpu} ok={ok} fail={fail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--videos", nargs="*", default=None)
    args = ap.parse_args()
    videos = [l.strip() for l in VAL_LIST.read_text().splitlines() if l.strip()]
    if args.videos:
        videos = [v for v in videos if v in set(args.videos)]
    if args.max_videos:
        videos = videos[:args.max_videos]
    items = []
    for vid in videos:
        fd = DAVIS / "JPEGImages" / "480p" / vid
        for i, fp in enumerate(sorted(fd.glob("*.jpg"))):
            items.append((vid, i, fp))
    print(f"[l12davis-dets] videos={len(videos)} frames={len(items)}",
          flush=True)
    gpus = [int(x) for x in args.gpus.split(",")]
    shards = [[] for _ in gpus]
    for i, item in enumerate(items):
        shards[i % len(gpus)].append(item)
    procs = []
    for gpu, shard in zip(gpus, shards):
        p = torch.multiprocessing.Process(target=worker, args=(gpu, shard))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[l12davis-dets] all workers done", flush=True)


if __name__ == "__main__":
    main()
