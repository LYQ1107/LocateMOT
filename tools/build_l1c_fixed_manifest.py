"""Stage L1-C: build fixed candidate manifests from existing frozen caches.

The manifest is a lightweight index: it does NOT copy PBD/region features.
It stores per-frame candidate boxes/scores/GT/matching + sha256 so that every
association method consumes exactly the same candidate set.

Usage:
  python tools/build_l1c_fixed_manifest.py --datasets dancetrack,bdd100k,tao_amodal,mot17,mot20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
OUT = ROOT / "outputs" / "l1_c" / "fixed_candidate_manifest"

L1A_CACHE = Path("/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla")
L1B_CACHE = Path("/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1B/cache_dla")

BDD_LABELS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/box_track_20/train")
TAO_ANN = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json")


def _hash_line(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _protocol(stem: str) -> str:
    return stem[:-5] if stem.endswith(".meta") else stem


def _load_boxes_scores(safe_path: Path):
    with safe_open(str(safe_path), framework="numpy") as f:
        boxes = f.get_tensor("boxes") if "boxes" in f.keys() else np.zeros((0, 4))
        scores = f.get_tensor("gen_score") if "gen_score" in f.keys() else np.zeros(0)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return boxes, scores


class CategoryLoader:
    def __init__(self):
        self._bdd = {}
        self._tao = None

    def bdd_categories(self, vid):
        if vid not in self._bdd:
            lab = BDD_LABELS / f"{vid}.json"
            cats = {}
            for fr in json.loads(lab.read_text()):
                for lab_ in fr.get("labels", []):
                    if lab_.get("box2d") is None:
                        continue
                    cats[str(lab_["id"])] = lab_.get("category", "object")
            self._bdd[vid] = cats
        return self._bdd[vid]

    def tao_categories(self):
        if self._tao is None:
            ann = json.loads(TAO_ANN.read_text())
            cat_name = {c["id"]: c["name"] for c in ann["categories"]}
            track_to_cat = {}
            for a in ann["annotations"]:
                track_to_cat.setdefault(a["track_id"], cat_name.get(a["category_id"], "object"))
            self._tao = track_to_cat
        return self._tao


def enumerate_l1a(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    rel = meta_path.relative_to(L1A_CACHE).parts
    # dancetrack/{vid}/{frame}/{protocol}.meta.json
    vid, frame, proto = rel[1], rel[2], _protocol(meta_path.stem)
    safe = meta_path.parent / f"{meta_path.name[:-len('.meta.json')]}.safetensors"
    boxes, scores = _load_boxes_scores(safe)
    return {
        "dataset": "dancetrack", "video_id": vid, "frame": int(frame),
        "protocol": proto, "split": meta.get("split", "unknown"),
        "image_size": list(meta.get("image_size", [1280, 720])),
        "candidate_count": int(meta.get("candidate_count", len(boxes))),
        "boxes": boxes.tolist(), "scores": scores.tolist(),
        "gt_ids": [str(g) for g in meta.get("gt_object_ids", [])],
        "gt_boxes": {str(k): list(v) for k, v in meta.get("gt_boxes", {}).items()},
        "matched": {str(k): v for k, v in meta.get("matched_candidates", {}).items()},
        "cache_root": str(L1A_CACHE),
    }


def enumerate_l1b(meta_path: Path, cats: CategoryLoader) -> dict:
    meta = json.loads(meta_path.read_text())
    rel = meta_path.relative_to(L1B_CACHE).parts
    # {ds}[/train]/{vid}/{frame}/{protocol}.meta.json
    if rel[0] == "tao_amodal":
        ds, vid, frame, proto = rel[0], rel[3], int(rel[4]), _protocol(meta_path.stem)
    else:
        ds, vid, frame, proto = rel[0], rel[1], rel[2], _protocol(meta_path.stem)
    safe = meta_path.parent / f"{meta_path.name[:-len('.meta.json')]}.safetensors"
    boxes, scores = _load_boxes_scores(safe)
    entry = {
        "dataset": ds, "video_id": vid, "frame": int(frame),
        "protocol": proto, "split": "train",
        "image_size": list(meta.get("image_size", [1280, 720])),
        "candidate_count": int(meta.get("candidate_count", len(boxes))),
        "boxes": boxes.tolist(), "scores": scores.tolist(),
        "gt_ids": [str(g) for g in meta.get("gt_object_ids", [])],
        "gt_boxes": {str(k): list(v) for k, v in meta.get("gt_boxes", {}).items()},
        "matched": {str(k): v for k, v in meta.get("matched_candidates", {}).items()},
        "cache_root": str(L1B_CACHE),
    }
    if ds == "bdd100k":
        cats_map = cats.bdd_categories(vid)
        entry["gt_categories"] = {gid: cats_map.get(gid, "object") for gid in entry["gt_ids"]}
    elif ds == "tao_amodal":
        cats_map = cats.tao_categories()
        entry["gt_categories"] = {gid: cats_map.get(int(gid), "object") for gid in entry["gt_ids"]}
    else:
        entry["gt_categories"] = {gid: "person" for gid in entry["gt_ids"]}
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dancetrack,bdd100k,tao_amodal,mot17,mot20")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cats = CategoryLoader()

    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        if ds == "dancetrack":
            roots = [(L1A_CACHE, "train"), (L1A_CACHE, "calibration"), (L1A_CACHE, "val")]
        else:
            roots = [(L1B_CACHE, "train")]
        for cache_root, split in roots:
            metas = []
            base = cache_root / ds
            for dirpath, _, files in os.walk(base):
                for fn in files:
                    if fn.endswith(".meta.json"):
                        metas.append(Path(dirpath) / fn)
            metas.sort()
            entries = []
            total_hash = hashlib.sha256()
            for i, mp in enumerate(metas):
                entry = enumerate_l1a(mp) if ds == "dancetrack" else enumerate_l1b(mp, cats)
                if ds == "dancetrack" and entry["split"] != split:
                    continue
                payload = json.dumps(
                    {"video": entry["video_id"], "frame": entry["frame"],
                     "boxes": entry["boxes"], "scores": entry["scores"]},
                    sort_keys=True, separators=(",", ":"))
                h = _hash_line(payload)
                entry["entry_sha256"] = h
                total_hash.update(h.encode("utf-8"))
                entries.append(entry)
                if (i + 1) % 2000 == 0:
                    print(f"[{ds}/{split}] {i + 1}/{len(metas)}", flush=True)
            stem = f"{ds}_{split}"
            with open(out / f"{stem}.jsonl", "w") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            manifest = {
                "dataset": ds, "split": split,
                "frames": len(entries),
                "videos": len({e["video_id"] for e in entries}),
                "candidates": sum(e["candidate_count"] for e in entries),
                "total_sha256": total_hash.hexdigest(),
                "model_commit": "783f656d127ee498137b5ff52603ce36c292d317",
                "generator": "LocateAnything-3B frozen",
                "protocols": sorted({e["protocol"] for e in entries}),
                "path": str(out / f"{stem}.jsonl"),
            }
            (out / f"{stem}.manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False))
            print(f"[{ds}/{split}] done frames={len(entries)} "
                  f"videos={manifest['videos']} sha={manifest['total_sha256'][:16]}", flush=True)


if __name__ == "__main__":
    main()
