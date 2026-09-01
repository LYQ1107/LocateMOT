"""Stage L1-C: cache LocateAnything-LoRA PBD features for pilot datasets.

Reuses ObjectTokenExtractor with the LoRA-adapted checkpoint. Output format
matches the frozen cache (safetensors + meta + .complete) under
outputs/l1_c/cache_lora so existing runners can consume it.

Usage:
  python tools/cache_l1c_lora.py --split calibration --gpu 1 --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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

DANCETRACK = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
DEFAULT_CKPT = ROOT / "outputs/l1_c/checkpoints/lora/checkpoint-300"
QUERY = ("Locate all the instances that matches the following "
         "description: person.")


def frames_of(vid, split):
    data_dir = "train" if split == "calibration" else split
    img_dir = DANCETRACK / data_dir / vid / "img1"
    return sorted(int(p.stem) for p in img_dir.glob("*.jpg"))


def read_gt(vid, split):
    data_dir = "train" if split == "calibration" else split
    gt = {}
    p = DANCETRACK / data_dir / vid / "gt" / "gt.txt"
    if not p.exists():
        return gt
    for line in p.read_text().splitlines():
        q = line.split(",")
        if len(q) >= 9 and int(q[7]) == 1:
            fid = int(q[0])
            x, y, w, h = map(float, q[2:6])
            gt.setdefault(fid, []).append((x, y, x + w, y + h, int(q[1])))
    return gt


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _stack(tokens, attr):
    vals = [getattr(t, attr) for t in tokens]
    vals = [v for v in vals if v is not None]
    return np.asarray(vals, dtype=np.float32) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "calibration", "val"], required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(ROOT / "outputs/l1_c/cache_lora"))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split_cfg = json.load(open(ROOT / "configs/data"
                               f"/l1_a_dancetrack_{args.split}.json"))
    vids = [v["video_id"] for v in split_cfg["videos"]]
    jobs = []
    for vid in vids:
        for fid in frames_of(vid, args.split):
            jobs.append((vid, fid))
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(f"[lora-cache] split={args.split} shard={args.shard}/{args.num_shards} "
          f"jobs={len(jobs)}", flush=True)
    if not jobs:
        return

    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    # Merge LoRA into the base Qwen model so the object-token instrumented
    # generation (written for the frozen architecture) works unchanged.
    if hasattr(model, "language_model") and hasattr(
            model.language_model, "merge_and_unload"):
        model.language_model = model.language_model.merge_and_unload()
        model.language_model.to("cuda").eval()
    ckpt_hash = json.dumps({
        f: hashlib.sha256(open(os.path.join(args.ckpt, f), "rb").read())
        .hexdigest()[:12]
        for f in sorted(os.listdir(args.ckpt))
        if f.startswith("model-") and f.endswith(".safetensors")
    }, sort_keys=True)
    extractor = ObjectTokenExtractor(
        model, tok, proc, model_dir=args.ckpt,
        model_commit=f"lora-300-{Path(args.ckpt).name}",
        checkpoint_hash=ckpt_hash, seed=20260806)

    done = 0
    for vid, fid in jobs:
        key = cache_key("dancetrack", vid, fid, "lora")
        if exists(out, key):
            done += 1
            continue
        data_dir = "train" if args.split == "calibration" else args.split
        jpg = DANCETRACK / data_dir / vid / "img1" / f"{fid:08d}.jpg"
        if not jpg.exists():
            continue
        image = Image.open(jpg).convert("RGB")
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        result = extractor.extract(
            image, question=QUERY, semantic_label="person",
            source_frame=f"{vid}/{fid:05d}",
            generation_mode="hybrid", max_new_tokens=1024,
            temperature=0.7, top_p=0.9, top_k=None,
            repetition_penalty=1.1, in_token_limit=4096)
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        tokens = result["object_tokens"]
        features = {
            "pbd_coord_mean_last": _stack(tokens, "pbd_coordinate_mean_feature"),
            "pbd_box_end_last": _stack(tokens, "pbd_box_end_feature"),
            "region": _stack(tokens, "region_feature"),
            "geometry": _stack(tokens, "geometry_feature"),
            "gen_score": np.asarray([t.generation_score or 0.0 for t in tokens],
                                    dtype=np.float32),
            "boxes": np.asarray([t.box_xyxy for t in tokens], dtype=np.float32),
            "normalized_boxes": np.asarray([t.normalized_box for t in tokens],
                                           dtype=np.float32),
        }
        features = {k: v for k, v in features.items()
                    if v is not None and len(v) > 0}
        gt = read_gt(vid, args.split).get(fid, [])
        matched = {}
        for oid, gtb in [(g[4], g[:4]) for g in gt]:
            best_idx, best_iou = None, 0.0
            for i, tb in enumerate(tokens):
                iou = _iou(tb.box_xyxy, gtb)
                if iou > best_iou:
                    best_idx, best_iou = i, iou
            if best_idx is not None:
                matched[str(oid)] = {"candidate": best_idx,
                                     "iou": round(best_iou, 4)}
        meta = {
            "dataset": "dancetrack", "video_id": vid, "frame": fid,
            "protocol": "lora", "split": args.split, "query": QUERY,
            "image_size": list(image.size), "candidate_count": len(tokens),
            "gt_object_ids": [g[4] for g in gt],
            "gt_boxes": {str(g[4]): list(g[:4]) for g in gt},
            "matched_candidates": matched,
            "model_commit": f"lora-300-{Path(args.ckpt).name}",
            "checkpoint_hash": ckpt_hash,
            "seconds": round(elapsed, 3), "peak_gpu_gb": round(peak, 3),
        }
        write_frame_cache(out, key, features, meta)
        done += 1
        if done % 20 == 0:
            print(f"[lora-cache] shard={args.shard} done={done} "
                  f"last={key} {elapsed:.2f}s", flush=True)
    print(f"[lora-cache] shard={args.shard} finished done={done}", flush=True)


if __name__ == "__main__":
    main()
