#!/usr/bin/env python
"""Stage L0-C: build medium-scale ObjectToken cache for selected videos.

Usage (one process per GPU shard):
  python tools/build_l0_c_token_cache.py --gpu 1 --shard 0 --num-shards 2
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.data.pair_manifest import (  # noqa: E402
    choose_frames,
    mask_boxes_for_frame,
    video_frame_names,
)
from locatemot.data.token_cache import cache_key, exists, write_frame_cache  # noqa: E402
from locatemot.models.object_tokens.extractor import ObjectTokenExtractor  # noqa: E402

MODEL_COMMIT = "783f656d127ee498137b5ff52603ce36c292d317"
_YOUTUBE_META = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage_l0_c.yaml")
    ap.add_argument("--model", default="/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/models/LocateAnything-3B")
    ap.add_argument("--cache-root", default="/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", default="outputs/l0_c")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    generic_query = cfg["cache"]["generic_query"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.cache_root, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    ckpt_hash = json.dumps({
        f: hashlib.sha256(open(os.path.join(args.model, f), "rb").read()).hexdigest()[:12]
        for f in sorted(os.listdir(args.model))
        if f.startswith("model-") and f.endswith(".safetensors")
    }, sort_keys=True)
    extractor = ObjectTokenExtractor(
        model, tok, proc, model_dir=args.model,
        model_commit=MODEL_COMMIT, checkpoint_hash=ckpt_hash, seed=args.seed,
    )
    roots = {"youtube": cfg["data"]["youtube_vos_root"], "mose": cfg["data"]["mose_root"]}

    all_jobs = []
    for split_name in ["train", "calibration", "heldout"]:
        split_path = os.path.join("configs", "data", f"l0_c_{split_name}_videos.json")
        split = json.load(open(split_path))
        for entry in split["videos"]:
            dataset = entry["dataset"]
            vid = entry["video_id"]
            frames = video_frame_names(dataset, vid, roots)
            for frame in [int(frames[i]) for i in choose_frames(len(frames), cfg["data"]["frames_per_video"])]:
                protocols = ["category_guided"] if "youtube" in dataset else []
                protocols.append("generic")
                for proto in protocols:
                    all_jobs.append((dataset, vid, frame, proto))
    all_jobs = [j for i, j in enumerate(all_jobs) if i % args.num_shards == args.shard]
    if args.max_frames:
        all_jobs = all_jobs[: args.max_frames]
    print(f"[shard {args.shard}/{args.num_shards}] jobs: {len(all_jobs)}", flush=True)

    rows = []
    done = 0
    for dataset, vid, frame, proto in all_jobs:
        key = cache_key(dataset, vid, frame, proto)
        if exists(args.cache_root, key):
            done += 1
            continue
        jpg = os.path.join(roots["youtube" if "youtube" in dataset else "mose"], "JPEGImages", vid, f"{frame:05d}.jpg")
        mask_png = os.path.join(roots["youtube" if "youtube" in dataset else "mose"], "Annotations", vid, f"{frame:05d}.png")
        if not os.path.exists(jpg) or not os.path.exists(mask_png):
            print(f"missing files: {jpg} / {mask_png}", flush=True)
            continue
        gt_boxes = mask_boxes_for_frame(mask_png)
        image = Image.open(jpg).convert("RGB")
        if proto == "category_guided":
            categories = _frame_categories(dataset, vid, frame)
            if not categories:
                write_frame_cache(args.cache_root, key, {}, {
                    "dataset": dataset, "video_id": vid, "frame": int(frame), "protocol": proto,
                    "query": "", "image_size": list(image.size), "candidate_count": 0,
                    "gt_object_ids": [int(o) for o in gt_boxes.keys()],
                    "gt_boxes": {str(k): v for k, v in gt_boxes.items()},
                    "matched_candidates": {}, "categories": [],
                    "model_commit": MODEL_COMMIT, "checkpoint_hash": ckpt_hash,
                    "seconds": 0.0, "peak_gpu_gb": 0.0,
                })
                done += 1
                continue
            query = "Locate all the instances that matches the following description: " + "</c>".join(sorted(categories)) + "."
        else:
            query = generic_query
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        result = extractor.extract(
            image, question=query, semantic_label=proto, source_frame=f"{vid}/{frame:05d}",
            generation_mode="hybrid", max_new_tokens=args.max_new_tokens,
            temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.1,
            in_token_limit=4096,
        )
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        tokens = result["object_tokens"]
        raw_vision = result["trace"].get("raw_vision_features")
        # crop region features for GT boxes
        crop_region = None
        if raw_vision is not None and gt_boxes:
            region_results = extractor.region.extract(
                result["trace"]["pixel_values"], result["trace"]["image_grid_hws"], 0,
                [[b[0] / image.size[0] * 1000, b[1] / image.size[1] * 1000,
                  b[2] / image.size[0] * 1000, b[3] / image.size[1] * 1000] for b in gt_boxes.values()],
                vision_features=raw_vision,
            )
            crop_region = np.stack([
                r["region_feature"].cpu().float().numpy() if r and r["region_feature"] is not None else np.zeros(4608, dtype=np.float32)
                for r in region_results
            ]).astype(np.float32)
        features = {
            "pbd_coord_mean_last": _stack(tokens, "pbd_coordinate_mean_feature"),
            "pbd_coord_mean_penultimate": _stack(tokens, "pbd_coordinate_mean_penultimate_feature"),
            "pbd_box_end_last": _stack(tokens, "pbd_box_end_feature"),
            "region": _stack(tokens, "region_feature"),
            "geometry": _stack(tokens, "geometry_feature"),
            "gen_score": np.asarray([t.generation_score or 0.0 for t in tokens], dtype=np.float32),
            "boxes": np.asarray([t.box_xyxy for t in tokens], dtype=np.float32),
            "normalized_boxes": np.asarray([t.normalized_box for t in tokens], dtype=np.float32),
        }
        features = {k: v for k, v in features.items() if v is not None and len(v) > 0}
        if not features:
            print(f"no tokens: {key}", flush=True)
            write_frame_cache(args.cache_root, key, {}, {
                "dataset": dataset, "video_id": vid, "frame": int(frame), "protocol": proto,
                "query": query, "image_size": list(image.size), "candidate_count": 0,
                "gt_object_ids": [int(o) for o in gt_boxes.keys()],
                "gt_boxes": {str(k): v for k, v in gt_boxes.items()},
                "matched_candidates": {}, "categories": [],
                "model_commit": MODEL_COMMIT, "checkpoint_hash": ckpt_hash,
                "seconds": round(elapsed, 3), "peak_gpu_gb": round(peak, 3),
            })
            done += 1
            continue
        matched = {}
        w, h = image.size
        for obj_id, gtb in gt_boxes.items():
            best_idx, best_iou = None, 0.0
            for i, tb in enumerate(tokens):
                iou = _iou(tb.box_xyxy, gtb)
                if iou > best_iou:
                    best_idx, best_iou = i, iou
            if best_idx is not None and best_iou >= 0.5:
                matched[str(obj_id)] = best_idx
        meta = {
            "dataset": dataset, "video_id": vid, "frame": int(frame), "protocol": proto,
            "query": query, "image_size": list(image.size), "candidate_count": len(tokens),
            "gt_object_ids": [int(o) for o in gt_boxes.keys()],
            "gt_boxes": {str(k): v for k, v in gt_boxes.items()},
            "matched_candidates": matched,
            "categories": _frame_categories(dataset, vid, frame) if "youtube" in dataset else [],
            "model_commit": MODEL_COMMIT, "checkpoint_hash": ckpt_hash,
            "seconds": round(elapsed, 3), "peak_gpu_gb": round(peak, 3),
        }
        if crop_region is not None:
            features["crop_region"] = crop_region
            meta["crop_object_ids"] = [int(o) for o in gt_boxes.keys()]
        write_frame_cache(args.cache_root, key, features, meta)
        rows.append([key, round(elapsed, 3), round(peak, 3), len(tokens)])
        done += 1
        if done % 20 == 0:
            print(f"[shard {args.shard}] done={done}/{len(all_jobs)} last={key} {elapsed:.2f}s", flush=True)

    with open(os.path.join(args.out, f"cache_runtime_shard{args.shard}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "seconds", "peak_gpu_gb", "tokens"])
        w.writerows(rows)
    print(f"[shard {args.shard}] finished, done={done}", flush=True)


def _stack(tokens, attr):
    vals = [getattr(t, attr) for t in tokens]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return np.asarray(vals, dtype=np.float32)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _frame_categories(dataset, vid, frame):
    if "youtube" not in dataset:
        return []
    global _YOUTUBE_META
    if _YOUTUBE_META is None:
        meta_path = "/data3/testdata/vranlee/.MOTSynth.partial/YouTube-VOS-2019/train/meta.json"
        _YOUTUBE_META = json.load(open(meta_path))
    info = _YOUTUBE_META["videos"].get(vid)
    if not info:
        return []
    cats = set()
    fname = f"{int(frame):05d}"
    for oinfo in info["objects"].values():
        if fname in oinfo["frames"]:
            cats.add(oinfo["category"])
    return sorted(cats)


if __name__ == "__main__":
    main()
