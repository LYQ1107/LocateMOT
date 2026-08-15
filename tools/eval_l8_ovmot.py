"""Stage L8: TAO OVMOT evaluation with the shared L8 checkpoint.

Protocol is identical to L7's official TAO OVMOT evaluation (TETA, Base/
Novel/All), but the tracker is the L8 shared model: same UIDM core +
Unified Observation Adapter.  TAO val candidates have no cached PBD
identity tokens, so the observation uses the CLIP stream only (PBD zeros);
the model is trained with --pbd-dropout so the shared core handles this
missing-identity regime.

Usage:
  python tools/eval_l8_ovmot.py --ckpt outputs/l8/checkpoints/.../latest.pt \
      --out outputs/l8/trackeval/... --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
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

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402
from locatemot.models.l8_unified import L8UnifiedUIDM, load_l8_state  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402
from tools.train_l8_uidm import _specs  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
GT_JSON = ("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/"
           "tao_val_lvis_v1_classes.json")
TETA_RUN = str(ROOT / "references" / "l7" / "TETA" / "scripts"
               / "run_ovmot.py")
DATA_DIR = ROOT / "outputs" / "l7" / "data" / "tao_val"
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
        fidx = int(stem[5:]) if stem.startswith("frame") \
            else int(img["frame_index"])
        vid2imgs.setdefault(img["video_id"], {})[fidx] = img["id"]
    return vid2imgs


def run_tracker(model, data_dir, out_path, gpu, spec_emb, score_thr=0.05,
                shard=0, num_shards=1, pbd_cache=None):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "8")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    spec = torch.as_tensor(spec_emb[None], device=device)
    preds = []
    print("[l8ovmot] loading GT json", flush=True)
    gt = json.load(open(GT_JSON))
    print(f"[l8ovmot] GT loaded: {len(gt['images'])} imgs, "
          f"{len(gt['annotations'])} anns", flush=True)
    vid2imgs = build_gt_maps(gt)
    vid_name2id = {v["name"].replace("/", "-"): v["id"] for v in gt["videos"]}
    gt_img_anns = {}
    for a in gt["annotations"]:
        gt_img_anns.setdefault(a["image_id"], []).append(a)
    text_emb = np.load(
        "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/metadata/"
        "lvis_v1_clip_a+cname.npy").astype(np.float32)
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    t0 = time.time()
    files = sorted(Path(data_dir).glob("*.pkl"))
    print(f"[l8ovmot] files={len(files)} shard={shard}", flush=True)
    if num_shards > 1:
        files = [p for p in files
                 if int(hashlib.md5(p.stem.encode()).hexdigest(), 16)
                 % num_shards == shard]
    for vi, pkl_path in enumerate(files):
        rec = pickle.load(open(pkl_path, "rb"))
        vname = rec["video_id"]
        vid = vid_name2id.get(vname)
        if vid is None:
            continue
        tracker = OnlineTracker(
            variant="UIDM", uidm=model.uidm, device=str(device),
            output_all_candidates=True,
            uidm_adapter=model.adapter, uidm_spec=spec.cpu().numpy()[0])
        tracker.uidm_sem_in_core = model.sem_in_core
        tracker.uidm_new_margin = 0.0
        tracker.l1d_weights = (0.4, 0.2, 0.4)
        tracker.l1d_threshold = 0.25
        n_hit = n_miss = 0
        for fr in rec["frames"]:
            frame = int(fr["frame"])
            boxes = fr["boxes"]
            cache_feats = None
            if pbd_cache:
                ck = cache_key("tao", vname, frame, "pbd_full")
                cache_feats = read_frame_cache(pbd_cache, ck)
            if cache_feats is not None and \
                    int(cache_feats["meta"]["candidate_count"]) != len(boxes):
                cache_feats = None
            if cache_feats is not None:
                n_hit += 1
            else:
                n_miss += 1
            cands = []
            for j in range(len(boxes)):
                if float(fr["gen"][j]) < score_thr:
                    continue
                x1, y1, x2, y2 = [float(v) for v in boxes[j]]
                pbd_be = np.zeros(2048, np.float32)
                if cache_feats is not None:
                    pbd_be = np.asarray(
                        cache_feats["features"]["pbd_box_end_last"][j],
                        np.float32)
                cands.append({
                    "box": [x1, y1, x2, y2],
                    "features": {
                        "pbd": pbd_be,
                        "pbd_be": pbd_be,
                        "clip": np.asarray(fr["clip"][j], np.float32),
                        "gen": float(fr["gen"][j]),
                    },
                    "label": int(fr["label"][j]),
                })
            tracker.image_size = rec["image_size"]
            outputs = tracker.process_frame(frame, cands)
            cls_all = None
            if len(cands) and len(outputs):
                clip_all = np.stack([
                    np.asarray(c["features"]["clip"], np.float32)
                    for c in cands])
                clip_all /= np.maximum(
                    1e-6, np.linalg.norm(clip_all, axis=1, keepdims=True))
                cls_all = np.argmax(clip_all @ text_emb.T, axis=1) + 1
            for o in outputs:
                x1, y1, x2, y2 = o["box"]
                best_l, best_v, best_j = None, -1.0, -1
                for j, c in enumerate(cands):
                    v = iou(c["box"], o["box"])
                    if v > best_v:
                        best_v, best_l, best_j = v, c["label"], j
                cat_id = best_l if best_l is not None else 0
                if best_j >= 0:
                    cat_id = int(cls_all[best_j])
                preds.append({
                    "image_id": vid2imgs[vid][frame],
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(o.get("score", 1.0)),
                    "track_id": int(o["track_id"]),
                    "video_id": vid,
                })
        if (vi + 1) % 10 == 0 or vi + 1 == len(files):
            print(f"[l8ovmot] {vi+1}/{len(files)} cache_hit={n_hit} "
                  f"cache_miss={n_miss} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(preds, f)
    print(f"[l8ovmot] wrote {len(preds)} preds", flush=True)


def run_teta(tracker_name, trackers_root):
    subprocess.run([
        PY, TETA_RUN, "--GT_FOLDER", GT_JSON,
        "--TRACKERS_FOLDER", trackers_root,
        "--TRACKERS_TO_EVAL", tracker_name,
        "--TRACKER_SUB_FOLDER", "data",
        "--SPLIT_TO_EVAL", "val",
        "--USE_PARALLEL", "False",
        "--PRINT_ONLY_COMBINED", "False",
    ], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--pbd-cache", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    tracker_dir = out / "trackers" / "UIDM" / "data"
    if args.merge_only:
        preds = []
        data_dir = out / "trackers" / "UIDM" / "data"
        shard_files = sorted(data_dir.glob("pred_shard*.json"))
        if not shard_files:
            shard_files = sorted((data_dir / "shards")
                                 .glob("pred_shard*.json"))
        for p in shard_files:
            preds.extend(json.loads(p.read_text()))
        tracker_dir.mkdir(parents=True, exist_ok=True)
        (tracker_dir / "pred.json").write_text(json.dumps(preds))
        shard_dir = data_dir / "shards"
        shard_dir.mkdir(exist_ok=True)
        for p in shard_files:
            p.rename(shard_dir / p.name)
        print(f"[l8ovmot] merged {len(preds)} preds", flush=True)
        run_teta("UIDM", str(out / "trackers"))
        return
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model = L8UnifiedUIDM(**SIZES[cfg.get("model", "base")],
                          mode=cfg.get("mode", "unified"),
                          sem_in_core=cfg.get("sem_in_core", True),
                          cond_gated=cfg.get("cond_gated", False))
    load_l8_state(model, ck["model"])
    spec_emb = _specs(["all objects"])[0]
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    if args.max_videos and not args.data_dir:
        data_dir = Path(args.out) / "subset_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(DATA_DIR.glob("*.pkl"))[:args.max_videos]:
            (data_dir / p.name).symlink_to(p.resolve())
    run_tracker(model, str(data_dir),
                tracker_dir / f"pred_shard{args.shard}.json", args.gpu,
                spec_emb, args.score_thr, args.shard, args.num_shards,
                pbd_cache=args.pbd_cache)
    if args.num_shards == 1:
        run_teta("UIDM", str(out / "trackers"))


if __name__ == "__main__":
    main()
