"""Stage L10: crop-PBD cache for Refer-KITTI-V2 RMOT candidates.

Same protocol as `cache_l9_tao_pbd.py` (LocateAnything-3B crop,
generic object query, `pbd_box_end_last`), but reads the L10 RMOT-KITTI
pkls and KITTI tracking images.

Usage:
  python tools/cache_l10_kitti_pbd.py --gpu 0 --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import cache_key, exists, write_frame_cache  # noqa: E402
from locatemot.models.object_tokens.extractor import ObjectTokenExtractor  # noqa: E402

MODEL_COMMIT = "783f656d127ee498137b5ff52603ce36c292d317"
MODEL_DIR = ROOT / "models" / "LocateAnything-3B"
KITTI_FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/KITTI_tracking"
                    "/training/image_02")
QUERY = "Locate the main object in the image."


def _ckpt_hash(model_dir):
    return json.dumps({
        f: hashlib.sha256(open(os.path.join(model_dir, f), "rb").read())
        .hexdigest()[:12]
        for f in sorted(os.listdir(model_dir))
        if f.startswith("model-") and f.endswith(".safetensors")
    }, sort_keys=True)


def load_model(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS"):
        os.environ[var] = "4"
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to("cuda").eval()
    ckpt_hash = _ckpt_hash(MODEL_DIR)
    extractor = ObjectTokenExtractor(
        model, tok, proc, model_dir=str(MODEL_DIR),
        model_commit=MODEL_COMMIT, checkpoint_hash=ckpt_hash,
        seed=20260806)
    return extractor


def _pick_token(tokens):
    if not tokens:
        return None
    best = tokens[0]
    best_score = float(best.generation_score or 0.0)
    best_area = (best.box_xyxy[2] - best.box_xyxy[0]) * \
        (best.box_xyxy[3] - best.box_xyxy[1])
    for t in tokens[1:]:
        score = float(t.generation_score or 0.0)
        area = (t.box_xyxy[2] - t.box_xyxy[0]) * \
            (t.box_xyxy[3] - t.box_xyxy[1])
        if score > best_score or (score == best_score and area > best_area):
            best, best_score, best_area = t, score, area
    return best


def _extract_crop(extractor, image, box):
    W, H = image.size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
    x2, y2 = max(0, min(W, x2)), max(0, min(H, y2))
    if x2 - x1 < 2:
        x2 = min(W, x1 + 2)
    if y2 - y1 < 2:
        y2 = min(H, y1 + 2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = image.crop((x1, y1, x2, y2))
    res = extractor.extract(
        crop, question=QUERY, semantic_label="object",
        source_frame=f"kitti/{x1},{y1},{x2},{y2}",
        generation_mode="hybrid", max_new_tokens=64,
        temperature=0.7, top_p=0.9, repetition_penalty=1.1,
        in_token_limit=2048, need_region=False)
    t = _pick_token(res["object_tokens"])
    if t is None or t.pbd_box_end_feature is None:
        return None
    return {
        "pbd_box_end_last": np.asarray(t.pbd_box_end_feature, np.float32),
        "gen_score": float(t.generation_score or 0.0),
        "tokens": len(res["object_tokens"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--frame-shards", action="store_true",
                    help="shard by (video, frame), for balanced L16 rebuilds")
    ap.add_argument("--data-dir", default=str(ROOT / "outputs" / "l10"
                                              / "data" / "rmot_kitti"))
    ap.add_argument("--cache-root", default=str(ROOT / "outputs" / "l10"
                                                / "cache" / "kitti_pbd"))
    args = ap.parse_args()
    cache_root = args.cache_root
    os.makedirs(cache_root, exist_ok=True)
    files = sorted(Path(args.data_dir).glob("*.pkl"))
    if args.num_shards > 1 and not args.frame_shards:
        files = [p for p in files
                 if int(hashlib.md5(p.stem.encode()).hexdigest(), 16)
                 % args.num_shards == args.shard]
    print(f"[l10kitti shard {args.shard}] videos={len(files)} gpu={args.gpu}",
          flush=True)
    if not files:
        return
    extractor = load_model(args.gpu)
    ckpt_hash = _ckpt_hash(MODEL_DIR)
    n_frames = n_crops = n_fail = 0
    t_start = time.time()
    for vi, pkl_path in enumerate(files):
        rec = pickle.load(open(pkl_path, "rb"))
        seq = rec["video_id"]
        for fr in rec["frames"]:
            frame = int(fr["frame"])
            if args.frame_shards and args.num_shards > 1:
                unit = f"{seq}/{frame:06d}".encode()
                if int(hashlib.md5(unit).hexdigest(), 16) \
                        % args.num_shards != args.shard:
                    continue
            key = cache_key("kitti", seq, frame, "pbd_full")
            if exists(cache_root, key):
                n_frames += 1
                n_crops += len(fr["boxes"])
                continue
            img_path = KITTI_FRAMES / seq / f"{frame:06d}.png"
            if not img_path.exists():
                continue
            image = Image.open(img_path).convert("RGB")
            boxes = np.asarray(fr["boxes"], np.float32)
            feats = {"pbd_box_end_last": [], "gen_score": [],
                     "boxes": boxes,
                     "clip": np.asarray(fr["clip"], np.float32)}
            fails = []
            t0 = time.time()
            for j, box in enumerate(boxes):
                r = _extract_crop(extractor, image, box)
                n_crops += 1
                if r is None:
                    n_fail += 1
                    fails.append(j)
                    feats["pbd_box_end_last"].append(
                        np.zeros(2048, np.float32))
                    feats["gen_score"].append(0.0)
                    continue
                feats["pbd_box_end_last"].append(r["pbd_box_end_last"])
                feats["gen_score"].append(r["gen_score"])
            elapsed = time.time() - t0
            if feats["pbd_box_end_last"]:
                feats["pbd_box_end_last"] = np.stack(
                    feats["pbd_box_end_last"]).astype(np.float32)
            else:
                feats["pbd_box_end_last"] = np.zeros((0, 2048), np.float32)
            feats["gen_score"] = np.asarray(feats["gen_score"], np.float32)
            meta = {
                "dataset": "kitti", "video_id": seq, "frame": frame,
                "protocol": "pbd_full", "query": QUERY,
                "image_size": list(image.size),
                "candidate_count": len(boxes),
                "failed_candidates": fails,
                "model_commit": MODEL_COMMIT,
                "checkpoint_hash": ckpt_hash,
                "seconds": round(elapsed, 3),
            }
            write_frame_cache(cache_root, key, feats, meta)
            n_frames += 1
            if n_frames % 10 == 0:
                print(f"[l10kitti shard {args.shard}] frames={n_frames} "
                      f"crops={n_crops} fail={n_fail} "
                      f"elapsed={time.time()-t_start:.0f}s last={key}",
                      flush=True)
        if (vi + 1) % 5 == 0 or vi + 1 == len(files):
            print(f"[l10kitti shard {args.shard}] video {vi+1}/{len(files)} "
                  f"frames={n_frames} crops={n_crops} fail={n_fail}",
                  flush=True)
    with open(Path(cache_root) / f"shard{args.shard}.done", "w") as f:
        f.write(f"frames={n_frames} crops={n_crops} fail={n_fail}\n")
    print(f"[l10kitti shard {args.shard}] done frames={n_frames} "
          f"crops={n_crops} fail={n_fail} seconds={time.time()-t_start:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
