"""Stage L1-B pilot ObjectToken cache (multi-dataset, resumable).

Usage:
  python tools/cache_l1b_locateanything.py --gpu 0 --shard 0 --num-shards 4

Datasets: dancetrack, mot17, mot20, tao_amodal, ytvos, mose.
Cache format matches locatemot.data.token_cache (safetensors + meta).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import cache_key, exists, write_frame_cache  # noqa: E402
from locatemot.models.object_tokens.extractor import ObjectTokenExtractor  # noqa: E402

MODEL_COMMIT = "783f656d127ee498137b5ff52603ce36c292d317"
DANCETRACK = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
MOT17 = Path("/data1/LWR/vranlee/DATASETS/JDE/MOT17")
MOT20 = Path("/data1/LWR/vranlee/M4FTMoveOut4Doing/ByteTrack-mbt/datasets/"
             "mix_mot20_ch/mot20_train")
TAO_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
                "TAO-Amodal")
YTVOS = Path("/data3/testdata/vranlee/.MOTSynth.partial/YouTube-VOS-2019")
MOSE = Path("/data3/testdata/vranlee/.MOTSynth.partial/MOSEv2")
BDD_LABELS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
                  "annotations/box_track_20")
BDD_IMAGES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
                  "bdd100k/images/track")

GENERIC_QUERY = "Locate all the objects in the image."


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _bbox_of_mask(arr, obj_id):
    ys, xs = np.where(arr == obj_id)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
            int(ys.max()) + 1]


class Loaders:
    def __init__(self):
        self._tao = None
        self._tao_ann = None
        self._yt = None
        self._mo = None

    def _mot_frames(self, seq_root):
        return sorted(int(p.stem) for p in (seq_root / "img1").glob("*.jpg"))

    def _mot_gt(self, seq_root):
        out = defaultdict(list)
        gt = seq_root / "gt" / "gt.txt"
        if not gt.exists():
            return out
        for line in gt.read_text().splitlines():
            p = line.split(",")
            if len(p) < 9:
                continue
            fid, tid = int(p[0]), int(p[1])
            cls, ignore = int(p[7]), int(p[6])
            if cls != 1 or ignore == 0:
                continue
            x, y, w, h = map(float, p[2:6])
            if w <= 0 or h <= 0:
                continue
            out[fid].append((x, y, x + w, y + h, tid))
        return out

    def frame_and_gt(self, ds, vid, fid):
        if ds == "dancetrack":
            seq = DANCETRACK / "train" / vid
            jpg = seq / "img1" / f"{int(fid):08d}.jpg"
            gt = {}
            fid_i = int(fid)
            for line in (seq / "gt" / "gt.txt").read_text().splitlines():
                p = line.split(",")
                if len(p) >= 9 and int(p[0]) == fid_i and int(p[7]) == 1:
                    x, y, w, h = map(float, p[2:6])
                    gt.setdefault(int(p[1]), [x, y, x + w, y + h])
            return jpg, [(v[0], v[1], v[2], v[3], k)
                         for k, v in gt.items()]
        if ds in ("mot17", "mot20"):
            base = MOT17 / "train" if ds == "mot17" else MOT20
            seq = base / vid
            jpg = seq / "img1" / f"{int(fid):06d}.jpg"
            return jpg, self._mot_gt(seq).get(int(fid), [])
        if ds == "tao_amodal":
            return self._tao_frame_gt(vid, fid)
        if ds == "ytvos":
            jpg = YTVOS / "train" / "JPEGImages" / vid / f"{fid}.jpg"
            return jpg, self._vos_frame_gt(
                YTVOS / "train" / "Annotations" / vid, fid)
        if ds == "mose":
            jpg = MOSE / "train" / "JPEGImages" / vid / f"{fid}.jpg"
            return jpg, self._vos_frame_gt(
                MOSE / "train" / "Annotations" / vid, fid)
        if ds == "bdd100k":
            return self._bdd_frame_gt(vid, int(fid))
        raise ValueError(ds)

    def _bdd_frame_gt(self, vid, frame_index):
        if not hasattr(self, "_bdd"):
            self._bdd = {}
        if vid not in self._bdd:
            lab = BDD_LABELS / "train" / f"{vid}.json"
            self._bdd[vid] = json.loads(lab.read_text())
        frames = self._bdd[vid]
        fr = next((f for f in frames
                   if int(f["frameIndex"]) == frame_index), None)
        if fr is None:
            return None, []
        jpg = BDD_IMAGES / "train" / vid / fr["name"]
        gt = []
        for lab in fr.get("labels", []):
            b = lab.get("box2d")
            if b is None:
                continue
            gt.append((b["x1"], b["y1"], b["x2"], b["y2"],
                       str(lab["id"])))
        return jpg, gt

    def _tao_frame_gt(self, vid, frame_index):
        if self._tao is None:
            self._tao = json.loads(
                (TAO_ROOT / "annotations" / "train.json").read_text())
            self._tao_ann = defaultdict(list)
            for a in self._tao["annotations"]:
                self._tao_ann[a["image_id"]].append(a)
            self._tao_img = {}
            for im in self._tao["images"]:
                self._tao_img[(im["video"], int(im["frame_index"]))] = im
        im = self._tao_img.get((vid, int(frame_index)))
        if im is None:
            return None, []
        jpg = TAO_ROOT / "frames" / im["file_name"]
        gt = []
        for a in self._tao_ann.get(im["id"], []):
            b = a["bbox"]
            gt.append((b[0], b[1], b[0] + b[2], b[1] + b[3],
                       int(a["track_id"])))
        return jpg, gt

    def _vos_frame_gt(self, ann_dir, fid):
        p = ann_dir / f"{Path(str(fid)).stem}.png"
        if not p.exists():
            return []
        arr = np.asarray(Image.open(p))
        ids = np.unique(arr)
        ids = ids[ids != 0]
        gt = []
        for oid in ids:
            box = _bbox_of_mask(arr, int(oid))
            if box is not None:
                gt.append((box[0], box[1], box[2], box[3], int(oid)))
        return gt

    def query_for(self, ds, vid):
        if ds in ("dancetrack", "mot17", "mot20"):
            return ("Locate all the instances that matches the following "
                    "description: person.")
        if ds == "mose":
            return GENERIC_QUERY
        if ds == "ytvos":
            if self._yt is None:
                self._yt = json.loads(
                    (YTVOS / "train" / "meta.json").read_text())
            objs = self._yt["videos"][vid].get("objects", {})
            cats = sorted({o.get("category", "object")
                           for o in objs.values()})
            return ("Locate all the instances that matches the following "
                    "description: " + ", ".join(cats[:6]) + ".")
        if ds == "tao_amodal":
            if self._tao is None:
                self._tao = json.loads(
                    (TAO_ROOT / "annotations" / "train.json").read_text())
            cat_name = {c["id"]: c["name"] for c in self._tao["categories"]}
            cats = set()
            for a in self._tao["annotations"]:
                im = next((x for x in self._tao["images"]
                           if x["id"] == a["image_id"]), None)
                if im and im["video"] == vid:
                    cats.add(cat_name.get(a["category_id"], "object"))
            return ("Locate all the instances that matches the following "
                    "description: " + ", ".join(sorted(cats)[:6]) + ".")
        if ds == "bdd100k":
            if not hasattr(self, "_bdd_q"):
                self._bdd_q = {}
            if vid not in self._bdd_q:
                lab = BDD_LABELS / "train" / f"{vid}.json"
                cnt = Counter()
                for f in json.loads(lab.read_text()):
                    for l in f.get("labels", []):
                        cnt[l.get("category", "object")] += 1
                cats = [c for c, _ in cnt.most_common(6)]
                self._bdd_q[vid] = (
                    "Locate all the instances that matches the following "
                    "description: " + ", ".join(cats) + ".")
            return self._bdd_q[vid]
        raise ValueError(ds)


def load_jobs(cfg, shard, num_shards):
    jobs = []
    for ds, videos in cfg.items():
        ds = ds.rsplit("_", 1)[0]
        for vid, frames in videos.items():
            if isinstance(frames, dict) and "frames" in frames:
                frames = frames["frames"]
            frames = [f["frame_index"] if isinstance(f, dict) else f
                      for f in frames]
            for fid in frames:
                fid = str(fid).rsplit(".", 1)[0]
                jobs.append((ds, vid, fid))
    jobs.sort()
    return [j for i, j in enumerate(jobs)
            if i % num_shards == shard]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=str(ROOT / "models/LocateAnything-3B"))
    ap.add_argument("--cache-root",
                    default="/data3/testdata/vranlee/.MOTSynth.partial/"
                            "LocateMOT_L1B/cache_dla")
    ap.add_argument("--split-config",
                    default=str(ROOT / "configs/l1_b/pilot_videos.json"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--protocol", default="pilot")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.cache_root, exist_ok=True)
    os.makedirs(ROOT / "outputs" / "l1_b", exist_ok=True)
    cfg = json.loads(Path(args.split_config).read_text())
    jobs = load_jobs(cfg, args.shard, args.num_shards)
    print(f"[shard {args.shard}/{args.num_shards}] jobs: {len(jobs)}",
          flush=True)
    if not jobs:
        return
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to("cuda").eval()
    ckpt_hash = json.dumps({
        f: hashlib.sha256(open(os.path.join(args.model, f), "rb").read())
        .hexdigest()[:12]
        for f in sorted(os.listdir(args.model))
        if f.startswith("model-") and f.endswith(".safetensors")
    }, sort_keys=True)
    extractor = ObjectTokenExtractor(
        model, tok, proc, model_dir=args.model,
        model_commit=MODEL_COMMIT, checkpoint_hash=ckpt_hash,
        seed=20260806)
    loaders = Loaders()
    rows = []
    done = 0
    for ds, vid, fid in jobs:
        key = cache_key(ds, vid, int(fid), args.protocol)
        if exists(args.cache_root, key):
            done += 1
            continue
        jpg, gt = loaders.frame_and_gt(ds, vid, fid)
        if jpg is None or not Path(jpg).exists():
            print(f"[warn] missing frame {ds}/{vid}/{fid}", flush=True)
            continue
        query = loaders.query_for(ds, vid)
        image = Image.open(jpg).convert("RGB")
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        result = extractor.extract(
            image, question=query, semantic_label="object",
            source_frame=f"{ds}/{vid}/{fid}",
            generation_mode="hybrid", max_new_tokens=args.max_new_tokens,
            temperature=0.7, top_p=0.9, top_k=None,
            repetition_penalty=1.1, in_token_limit=4096)
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        tokens = result["object_tokens"]

        def _stack(attr):
            vals = [getattr(t, attr) for t in tokens]
            vals = [v for v in vals if v is not None]
            return np.asarray(vals, dtype=np.float32) if vals else None

        features = {
            "pbd_coord_mean_last": _stack("pbd_coordinate_mean_feature"),
            "pbd_coord_mean_penultimate": _stack(
                "pbd_coordinate_mean_penultimate_feature"),
            "pbd_box_end_last": _stack("pbd_box_end_feature"),
            "pbd_box_end_penultimate": _stack(
                "pbd_box_end_penultimate_feature"),
            "pbd_full_block_mean_last": _stack(
                "pbd_full_block_mean_feature"),
            "region": _stack("region_feature"),
            "geometry": _stack("geometry_feature"),
            "fused": _stack("fused_feature"),
            "gen_score": np.asarray(
                [t.generation_score or 0.0 for t in tokens],
                dtype=np.float32),
            "boxes": np.asarray([t.box_xyxy for t in tokens],
                                dtype=np.float32),
            "normalized_boxes": np.asarray(
                [t.normalized_box for t in tokens], dtype=np.float32),
        }
        features = {k: v for k, v in features.items()
                    if v is not None and len(v) > 0}
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
            "dataset": ds, "video_id": str(vid), "frame": int(fid),
            "protocol": args.protocol, "query": query,
            "image_size": list(image.size),
            "candidate_count": len(tokens),
            "gt_object_ids": [g[4] for g in gt],
            "gt_boxes": {str(g[4]): list(g[:4]) for g in gt},
            "matched_candidates": matched,
            "model_commit": MODEL_COMMIT, "checkpoint_hash": ckpt_hash,
            "seconds": round(elapsed, 3), "peak_gpu_gb": round(peak, 3),
        }
        write_frame_cache(args.cache_root, key, features, meta)
        rows.append([key, round(elapsed, 3), round(peak, 3), len(tokens)])
        done += 1
        if done % 20 == 0:
            print(f"[shard {args.shard}] done={done}/{len(jobs)} "
                  f"last={key} {elapsed:.2f}s", flush=True)
    out_csv = ROOT / "outputs" / "l1_b" / \
        f"dla_cache_runtime_pilot_shard{args.shard}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "seconds", "peak_gpu_gb", "tokens"])
        w.writerows(rows)
    print(f"[shard {args.shard}] finished, done={done}", flush=True)


if __name__ == "__main__":
    main()
