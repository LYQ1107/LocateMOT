"""Stage L1-D: TrackEval on custom AC candidate sets (DanceTrack/BDD/MOT17/MOT20).

Usage:
  python tools/run_l1d_trackeval.py --split dancetrack_val \
      --variants L1D,L1DB_w0.70.30.0_t0.30
  python tools/run_l1d_trackeval.py --split bdd \
      --manifest outputs/l1_d/manifests/bdd100k_eval.jsonl \
      --tracker-root outputs/l1_d/trackeval_bdd \
      --variants L1D,L1DB_w0.70.30.0_t0.30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "references", "TrackEval-official"))

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

from trackeval.eval import Evaluator  # noqa: E402
from trackeval.metrics.hota import HOTA  # noqa: E402
from trackeval.metrics.clear import CLEAR  # noqa: E402
from trackeval.metrics.identity import Identity  # noqa: E402

DANCETRACK = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"


def metric_value(m):
    if hasattr(m, "tolist"):
        arr = m.tolist()
        if isinstance(arr, list) and arr:
            return float(arr[0])
        return float(m)
    return float(m)


def load_manifest(path):
    by_video = defaultdict(list)
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])
    return by_video


def write_gt(gt_dir, vid, entries, fps=30):
    seq_dir = os.path.join(gt_dir, vid)
    os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)
    img_w, img_h = entries[0].get("image_size", [1280, 720])
    seq_len = max(int(e["frame"]) for e in entries) + 1
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={vid}\n")
        f.write(f"imDir=img1\n")
        f.write(f"frameRate={fps}\n")
        f.write(f"seqLength={seq_len}\n")
        f.write(f"imWidth={img_w}\n")
        f.write(f"imHeight={img_h}\n")
    with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as f:
        for e in entries:
            fr = int(e["frame"]) + 1
            for gid, box in e.get("gt_boxes", {}).items():
                x1, y1, x2, y2 = box
                f.write(f"{fr},{gid},{x1:.2f},{y1:.2f},"
                        f"{x2-x1:.2f},{y2-y1:.2f},1,1,-1,-1\n")


def build_data(manifest, tracker_root, variants, split, fps=30):
    by_video = load_manifest(manifest)
    data_root = os.path.join(ROOT, "outputs", "l1_d", f"trackeval_data_{split}")
    gt_base = os.path.join(data_root, "gt", "mot_challenge")
    trk_base = os.path.join(data_root, "trackers", "mot_challenge")
    label = f"custom-{split}"
    gt_dir = os.path.join(gt_base, label)
    trk_dir = os.path.join(trk_base, label)
    vids = sorted(by_video)
    for vid in vids:
        write_gt(gt_dir, vid, by_video[vid], fps=fps)
        for variant in variants:
            src = os.path.join(tracker_root, variant, f"{vid}.txt")
            if not os.path.exists(src):
                continue
            dst = os.path.join(trk_dir, variant, "data", f"{vid}.txt")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                # MOTChallenge timesteps start at 1; manifests may be 0-based
                lines = [l for l in open(src) if l.strip()]
                shifted = []
                for l in lines:
                    p = l.strip().split(",")
                    p[0] = str(int(float(p[0])) + 1)
                    shifted.append(",".join(p) + "\n")
                with open(dst, "w") as f:
                    f.writelines(shifted)
    seqmap_dir = os.path.join(gt_base, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    text = "name\n" + "\n".join(vids) + "\n"
    with open(os.path.join(seqmap_dir, f"{label}.txt"), "w") as f:
        f.write(text)
    inner = os.path.join(gt_dir, "seqmaps")
    os.makedirs(inner, exist_ok=True)
    with open(os.path.join(inner, f"{label}.txt"), "w") as f:
        f.write(text)
    return gt_dir, trk_dir


def build_dancetrack(split, variants):
    data_root = os.path.join(ROOT, "outputs", "l1_c", f"trackeval_data_{split}")
    gt_base = os.path.join(data_root, "gt", "mot_challenge")
    trk_base = os.path.join(data_root, "trackers", "mot_challenge")
    gt_dir = os.path.join(gt_base, f"DanceTrack-{split}")
    trk_dir = os.path.join(trk_base, f"DanceTrack-{split}")
    return gt_dir, trk_dir


def run_eval(gt_dir, trk_dir, variants, split):
    dataset_cfg = {
        "GT_FOLDER": gt_dir,
        "TRACKERS_FOLDER": trk_dir,
        "OUTPUT_FOLDER": os.path.join(ROOT, "outputs", "l1_d", "trackeval"),
        "TRACKERS_TO_EVAL": variants,
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "DanceTrack" if split.startswith("dancetrack") else "custom",
        "SPLIT_TO_EVAL": split if split.startswith("dancetrack") else split,
        "PRINT_CONFIG": False,
        "DO_PREPROC": True,
        "TRACKER_SUB_FOLDER": "data",
        "SEQMAP_FOLDER": None,
    }
    evaluator = Evaluator({
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    })
    metrics = [HOTA(), CLEAR(), Identity()]
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    res, _ = evaluator.evaluate(
        [MotChallenge2DBox(dataset_cfg)], metrics, show_progressbar=False)
    return res


def flatten(res):
    dataset = list(res.keys())[0]
    out = {}
    for tracker, seqs in res[dataset].items():
        combined = seqs.get("COMBINED_SEQ", {})
        per_cls = combined.get("pedestrian", {})
        row = {}
        for metric_name, vals in per_cls.items():
            for k, v in vals.items():
                if k.startswith("CLR_"):
                    k = k[4:]
                prefix = {"CLEAR": "CLEAR", "Identity": "Identity"}.get(
                    metric_name, metric_name)
                mv = metric_value(v)
                row[f"{prefix}_{k}"] = mv
                if prefix == "HOTA" and k in ("HOTA", "DetA", "AssA", "LocA"):
                    row[f"{prefix}_{k}(0)"] = mv
        out[tracker] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--tracker-root", default="")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    variants = args.variants.split(",")
    if args.split.startswith("dancetrack"):
        gt_dir, trk_dir = build_dancetrack(args.split, variants)
    else:
        gt_dir, trk_dir = build_data(args.manifest, args.tracker_root,
                                     variants, args.split, fps=args.fps)
    present = [v for v in variants if os.path.exists(os.path.join(trk_dir, v, "data"))]
    if not present:
        print("[trackeval] no outputs found")
        return
    res = run_eval(os.path.dirname(gt_dir), os.path.dirname(trk_dir),
                   present, args.split)
    flat = flatten(res)
    fields = ["HOTA_HOTA(0)", "HOTA_DetA(0)", "HOTA_AssA(0)", "HOTA_LocA(0)",
              "CLEAR_MOTA", "Identity_IDF1", "CLEAR_IDSW", "CLEAR_FP",
              "CLEAR_FN", "CLEAR_Frag"]
    out_path = os.path.join(ROOT, "outputs", "l1_d",
                            f"ac_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2, default=str)
    rows = []
    for v in present:
        row = {"variant": v}
        for fld in fields:
            row[fld] = flat[v].get(fld, "")
        rows.append(row)
        print(v, {k: round(float(row[k]), 4) if row[k] != "" else "" for k in
                  ["HOTA_HOTA(0)", "HOTA_AssA(0)", "Identity_IDF1", "CLEAR_IDSW"]})
    with open(os.path.join(ROOT, "outputs", "l1_d", f"ac_{args.split}.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant"] + fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
