"""Stage L10: Detic dets for Refer-KITTI-V2 frames.

Same Detic-SwinB checkpoint and protocol as the TAO-train DLA run
(score>=0.05, top-50), so the RMOT candidates use the same detector
family as the OVMOT train stream.

Usage (masaenv):
  python tools/generate_l10_kitti_dets.py --gpus 4,6,7 \
      --out outputs/l10/cache/kitti_dets
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
ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
KITTI_FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/KITTI_tracking"
                    "/training/image_02")
TRAIN_LIST = (Path("/data1/LWR/vranlee/SERVER_ONLY/avis/"
                   "LocateMOT_reference_repos") / "temp_rmot" /
              "datasets" / "data_path" / "refer-kitti-v2.train")
DETIC_CFG = (MASA_ROOT + "/projects/Detic_new/configs/"
             "detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.py")
DETIC_CKPT = MASA_ROOT + "/saved_models/masa_models/detic_masa.pth"
EVAL_SEQS = {"0005", "0011", "0013", "0019"}


def worker(gpu, items, out_root):
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
    n_ok = n_fail = 0
    for seq, frame, img_path in items:
        out_path = Path(out_root) / seq / f"{frame:06d}.pth"
        if out_path.exists():
            n_ok += 1
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
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[kitti-dets:{gpu}] fail {seq}/{frame}: {e}", flush=True)
    print(f"[kitti-dets:{gpu}] ok={n_ok} fail={n_fail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,6,7")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "l10" / "cache"
                                         / "kitti_dets"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true",
                    help="only the 4 official evaluation sequences")
    args = ap.parse_args()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    lines = [l.strip() for l in TRAIN_LIST.read_text().splitlines()
             if l.strip()]
    items = []
    if not args.eval_only:
        for line in lines:
            rel = line.replace("KITTI/training/image_02/", "")
            seq, fname = rel.split("/", 1)
            frame = int(fname.replace(".png", ""))
            items.append((seq, frame, KITTI_FRAMES / seq / fname))
    for seq in sorted(EVAL_SEQS):
        img_dir = KITTI_FRAMES / seq
        if img_dir.is_dir():
            for p in sorted(img_dir.glob("*.png")):
                items.append((seq, int(p.stem), p))
    if args.max_frames:
        items = items[:args.max_frames]
    print(f"[kitti-dets] frames={len(items)}", flush=True)
    gpus = [int(x) for x in args.gpus.split(",")]
    shards = [[] for _ in gpus]
    for i, item in enumerate(items):
        shards[i % len(gpus)].append(item)
    procs = []
    for gpu, shard in zip(gpus, shards):
        p = torch.multiprocessing.Process(
            target=worker, args=(gpu, shard, out_root))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[kitti-dets] all workers done", flush=True)


if __name__ == "__main__":
    main()
