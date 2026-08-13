"""Stage L7: OVMOT evaluation with the shared UIDM core on TAO (official TETA).

Protocol (verified against official code):
  - GT: official TAO val in LVIS v1 categories (tao_val_lvis_v1_classes.json)
  - candidates: official Detic public detections (same source as OVTrack/OVTR)
  - classification: frozen Detic label (perception), tracker provides association
  - metric: official TETA (LocA / AssocA / ClsA), Base (freq != r) / Novel (r)

The shared identity-dynamics core is frozen except the appearance projector;
CLIP ViT-B/32 crop embeddings are the open-vocabulary appearance token.

Usage:
  python tools/eval_l7_ovmot.py --data-dir outputs/l7/data/tao_val \
      --ckpt outputs/l7/checkpoints/ovmot_probe/latest.pt \
      --out outputs/l7/trackeval/ovmot_probe --gpu 5
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l6_uidm import UIDM  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
GT_JSON = ("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/"
           "tao_val_lvis_v1_classes.json")
TETA_RUN = str(ROOT / "references/l7/TETA/scripts/run_ovmot.py")

SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter) if ar + br - inter > 0 else 0.0


def build_gt_maps(gt):
    vid2imgs = {}
    for img in gt["images"]:
        stem = img["file_name"].rsplit("/", 1)[-1].replace(".jpg", "")
        if stem.startswith("frame"):
            fidx = int(stem[5:])
        else:
            fidx = int(img["frame_index"])
        vid2imgs.setdefault(img["video_id"], {})[fidx] = img["id"]
    return vid2imgs


def run_tracker(data_dir, ckpt_path, out_path, gpu, score_thr=0.05,
                new_margin=0.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    size = cfg.get("model", "large")
    model = UIDM(
        **SIZES[size], no_interaction=cfg.get("no_interaction", False),
        use_cue_rel=cfg.get("use_cue_rel", False),
        app_dim=cfg.get("app_dim", 512)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    preds = []
    gt = json.load(open(GT_JSON))
    vid2imgs = build_gt_maps(gt)
    vid_name2id = {v["name"].replace("/", "-"): v["id"] for v in gt["videos"]}
    t0 = time.time()
    files = sorted(Path(data_dir).glob("*.pkl"))
    for vi, pkl_path in enumerate(files):
        rec = pickle.load(open(pkl_path, "rb"))
        vname = rec["video_id"]
        vid = vid_name2id.get(vname)
        if vid is None:
            continue
        tracker = OnlineTracker(variant="UIDM", uidm=model,
                                device=str(device),
                                output_all_candidates=True)
        tracker.uidm_new_margin = new_margin
        tracker.l1d_weights = (0.4, 0.2, 0.4)
        tracker.l1d_threshold = 0.25
        for fr in rec["frames"]:
            frame = int(fr["frame"])
            boxes = fr["boxes"]
            cands = []
            for j in range(len(boxes)):
                if float(fr["gen"][j]) < score_thr:
                    continue
                x1, y1, x2, y2 = [float(v) for v in boxes[j]]
                cands.append({
                    "box": [x1, y1, x2, y2],
                    "features": {
                        "pbd_be": np.asarray(fr["clip"][j], np.float32),
                        "gen": float(fr["gen"][j]),
                    },
                    "label": int(fr["label"][j]),
                })
            tracker.image_size = rec["image_size"]
            outputs = tracker.process_frame(frame, cands)
            for o in outputs:
                x1, y1, x2, y2 = o["box"]
                best_l, best_v = None, -1.0
                for j, c in enumerate(cands):
                    v = iou(c["box"], o["box"])
                    if v > best_v:
                        best_v, best_l = v, c["label"]
                preds.append({
                    "image_id": vid2imgs[vid][frame],
                    "category_id": best_l if best_l is not None else 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(o.get("score", 1.0)),
                    "track_id": int(o["track_id"]),
                    "video_id": vid,
                })
        if (vi + 1) % 100 == 0 or vi + 1 == len(files):
            print(f"[ovmot] {vi+1}/{len(files)} elapsed={time.time()-t0:.0f}s",
                  flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(preds, f)
    print(f"[ovmot] wrote {len(preds)} predictions to {out_path}", flush=True)


def run_teta(tracker_name, trackers_root, gt_json, split="val"):
    cmd = [
        PY, TETA_RUN,
        "--GT_FOLDER", gt_json,
        "--TRACKERS_FOLDER", trackers_root,
        "--TRACKERS_TO_EVAL", tracker_name,
        "--TRACKER_SUB_FOLDER", "data",
        "--SPLIT_TO_EVAL", split,
        "--USE_PARALLEL", "False",
        "--PRINT_ONLY_COMBINED", "False",
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="outputs/l7/data/tao_val")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--new-margin", type=float, default=0.0)
    ap.add_argument("--gt-json", default=GT_JSON)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    tracker_dir = out / "trackers" / "UIDM" / "data"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    tmp_dir = data_dir
    if args.max_videos:
        tmp_dir = out / "subset_data"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(data_dir.glob("*.pkl"))[:args.max_videos]:
            (tmp_dir / p.name).symlink_to(p.resolve())
    run_tracker(str(tmp_dir), args.ckpt, tracker_dir / "pred.json",
                args.gpu, args.score_thr, args.new_margin)
    run_teta("UIDM", str(out / "trackers"), args.gt_json)


if __name__ == "__main__":
    main()
