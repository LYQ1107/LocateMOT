"""Stage L9: TAO PBD identity-feature cache (crop-based, resumable).

For every Detic candidate box of every TAO frame, crop the box from the
original frame and run LocateAnything-3B on the crop with a generic object
query, then store the PBD box-end token (`pbd_box_end_last`, 2048-d) plus
the other PBD block features aligned with the candidate order.  The cache
is write-through per frame and resumable via the standard token-cache
`.complete` marker.

Why crop-based: Stage L1B showed that full-image generation on TAO returns
zero candidates for ~46% of frames and only ~3 boxes/frame otherwise,
which cannot cover the ~44 public Detic candidates per frame.

Usage (one process per shard; several processes may share one GPU):
  python tools/cache_l9_tao_pbd.py --gpu 1 --shard 0 --num-shards 8

The cache layout is `cache_root/tao/{video_id}/{frame:05d}/pbd_full` so it
can be read back with `locatemot.data.token_cache.cache_key`.
"""
from __future__ import annotations

import argparse
import csv
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
TAO_FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
                  "TAO-Amodal/frames")
VAL_GT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/"
              "annotations/tao_val_lvis_v1_classes.json")
DATA_DIR = ROOT / "outputs" / "l7" / "data" / "tao_val"
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
    """Choose the object token best representing the crop: highest
    generation score, then largest box."""
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
    """Run LocateAnything on one cropped candidate; return feature dict or
    None on failure."""
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
        source_frame=f"tao/{x1},{y1},{x2},{y2}",
        generation_mode="hybrid", max_new_tokens=64,
        temperature=0.7, top_p=0.9, repetition_penalty=1.1,
        in_token_limit=2048, need_region=False)
    t = _pick_token(res["object_tokens"])
    if t is None or t.pbd_box_end_feature is None:
        return None
    return {
        "pbd_box_end_last": np.asarray(t.pbd_box_end_feature, np.float32),
        "pbd_box_end_penultimate": np.asarray(
            t.pbd_box_end_penultimate_feature, np.float32)
        if t.pbd_box_end_penultimate_feature is not None else None,
        "pbd_coord_mean_last": np.asarray(
            t.pbd_coordinate_mean_feature, np.float32)
        if t.pbd_coordinate_mean_feature is not None else None,
        "pbd_full_block_mean_last": np.asarray(
            t.pbd_full_block_mean_feature, np.float32)
        if t.pbd_full_block_mean_feature is not None else None,
        "gen_score": float(t.generation_score or 0.0),
        "tokens": len(res["object_tokens"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--cache-root",
                    default=str(ROOT / "outputs" / "l9" / "cache"
                                / "tao_val_pbd"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--gt-json", default=str(VAL_GT))
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-frames-per-video", type=int, default=0)
    args = ap.parse_args()

    cache_root = args.cache_root
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(ROOT / "outputs" / "l9" / "cache", exist_ok=True)

    gt = json.load(open(args.gt_json))
    vid_name2id = {v["name"].replace("/", "-"): v["id"] for v in gt["videos"]}
    vid_id2imgs = {}
    for im in gt["images"]:
        vid_id2imgs.setdefault(im["video_id"], []).append(im)
    for v in vid_id2imgs.values():
        v.sort(key=lambda x: int(x["frame_index"]))

    files = sorted(Path(args.data_dir).glob("*.pkl"))
    if args.num_shards > 1:
        files = [p for p in files
                 if int(hashlib.md5(p.stem.encode()).hexdigest(), 16)
                 % args.num_shards == args.shard]
    if args.max_videos:
        files = files[:args.max_videos]
    print(f"[l9cache shard {args.shard}] videos={len(files)} gpu={args.gpu}",
          flush=True)
    if not files:
        return

    extractor = load_model(args.gpu)
    rows = []
    n_frames = 0
    n_crops = 0
    n_fail = 0
    t_start = time.time()

    for vi, pkl_path in enumerate(files):
        rec = pickle.load(open(pkl_path, "rb"))
        vname = rec["video_id"]
        vid = vid_name2id.get(vname)
        if vid is None:
            print(f"[l9cache] skip unknown video {vname}", flush=True)
            continue
        frame2file = {}
        for im in vid_id2imgs.get(vid, []):
            stem = im["file_name"].rsplit("/", 1)[-1].replace(".jpg", "")
            fidx = int(stem[5:]) if stem.startswith("frame") \
                else int(im["frame_index"])
            frame2file[fidx] = im["file_name"]
        frames = rec["frames"]
        if args.max_frames_per_video:
            frames = frames[:args.max_frames_per_video]
        for fi, fr in enumerate(frames):
            frame = int(fr["frame"])
            key = cache_key("tao", vname, frame, "pbd_full")
            if exists(cache_root, key):
                n_frames += 1
                n_crops += len(fr["boxes"])
                continue
            fname = frame2file.get(frame)
            if fname is None:
                print(f"[l9cache] no image for {vname} frame {frame}",
                      flush=True)
                continue
            img_path = TAO_FRAMES / fname
            image = Image.open(img_path).convert("RGB")
            boxes = np.asarray(fr["boxes"], np.float32)
            feats = {
                "pbd_box_end_last": [],
                "pbd_box_end_penultimate": [],
                "pbd_coord_mean_last": [],
                "pbd_full_block_mean_last": [],
                "gen_score": [],
                "boxes": boxes,
                "clip": np.asarray(fr["clip"], np.float32),
            }
            fails = []
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            for j, box in enumerate(boxes):
                r = _extract_crop(extractor, image, box)
                n_crops += 1
                if r is None:
                    n_fail += 1
                    fails.append(j)
                    for k in ("pbd_box_end_last",
                              "pbd_box_end_penultimate",
                              "pbd_coord_mean_last",
                              "pbd_full_block_mean_last"):
                        feats[k].append(np.zeros(2048, np.float32))
                    feats["gen_score"].append(0.0)
                    continue
                for k in ("pbd_box_end_last", "pbd_box_end_penultimate",
                          "pbd_coord_mean_last", "pbd_full_block_mean_last"):
                    v = r.get(k)
                    feats[k].append(v if v is not None
                                    else np.zeros(2048, np.float32))
                feats["gen_score"].append(r["gen_score"])
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            for k in ("pbd_box_end_last", "pbd_box_end_penultimate",
                      "pbd_coord_mean_last", "pbd_full_block_mean_last"):
                feats[k] = np.stack(feats[k]).astype(np.float32)
            feats["gen_score"] = np.asarray(feats["gen_score"],
                                            np.float32)
            meta = {
                "dataset": "tao", "video_id": vname, "frame": frame,
                "protocol": "pbd_full", "query": QUERY,
                "image_size": list(image.size),
                "candidate_count": len(boxes),
                "failed_candidates": fails,
                "model_commit": MODEL_COMMIT,
                "checkpoint_hash": _ckpt_hash(MODEL_DIR),
                "seconds": round(elapsed, 3),
                "peak_gpu_gb": round(peak, 3),
            }
            write_frame_cache(cache_root, key, feats, meta)
            n_frames += 1
            rows.append([key, round(elapsed, 3), round(peak, 3),
                         len(boxes), len(fails)])
            if n_frames % 10 == 0:
                print(f"[l9cache shard {args.shard}] frames={n_frames} "
                      f"crops={n_crops} fail={n_fail} "
                      f"elapsed={time.time()-t_start:.0f}s "
                      f"last={key}", flush=True)
        if (vi + 1) % 5 == 0 or vi + 1 == len(files):
            print(f"[l9cache shard {args.shard}] video {vi+1}/{len(files)} "
                  f"frames={n_frames} crops={n_crops} fail={n_fail}",
                  flush=True)

    csv_path = (ROOT / "outputs" / "l9" / "cache" /
                f"tao_val_pbd_shard{args.shard}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "seconds", "peak_gpu_gb", "candidates",
                    "failed"])
        w.writerows(rows)
    print(f"[l9cache shard {args.shard}] done frames={n_frames} "
          f"crops={n_crops} fail={n_fail} seconds={time.time()-t_start:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
