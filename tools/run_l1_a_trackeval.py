#!/usr/bin/env python
"""Stage L1-A: official TrackEval on T0-T6 full-video results."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

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
VARIANTS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]


def build_data_dirs(protocol, split, force=False):
    data_root = os.path.join(ROOT, "outputs", "l1_a", f"trackeval_data_{protocol}_{split}")
    gt_base = os.path.join(data_root, "gt", "mot_challenge")
    trk_base = os.path.join(data_root, "trackers", "mot_challenge")
    gt_dir = os.path.join(gt_base, f"DanceTrack-{split}")
    trk_dir = os.path.join(trk_base, f"DanceTrack-{split}")
    split_cfg = json.load(open(os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{split}.json")))
    vids = [v["video_id"] for v in split_cfg["videos"]]
    for vid in vids:
        data_dir = "train" if split == "calibration" else split
        src_gt = os.path.join(DANCETRACK, data_dir, vid, "gt", "gt.txt")
        dst_gt = os.path.join(gt_dir, vid, "gt", "gt.txt")
        os.makedirs(os.path.dirname(dst_gt), exist_ok=True)
        if not os.path.exists(dst_gt):
            os.symlink(src_gt, dst_gt)
        src_ini = os.path.join(DANCETRACK, data_dir, vid, "seqinfo.ini")
        dst_ini = os.path.join(gt_dir, vid, "seqinfo.ini")
        if not os.path.exists(dst_ini):
            os.symlink(src_ini, dst_ini)
        for variant in VARIANTS:
            src_tr = os.path.join(ROOT, "outputs", "l1_a", "trackeval", protocol, variant, f"{vid}.txt")
            if not os.path.exists(src_tr):
                continue
            dst_tr = os.path.join(trk_dir, variant, "data", f"{vid}.txt")
            os.makedirs(os.path.dirname(dst_tr), exist_ok=True)
            if not os.path.exists(dst_tr):
                os.symlink(src_tr, dst_tr)
    seqmap_dir = os.path.join(gt_base, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(os.path.join(seqmap_dir, f"DanceTrack-{split}.txt"), "w") as f:
        f.write("name\n")
        for vid in vids:
            f.write(vid + "\n")
    return gt_base, trk_base, vids


def metric_value(m):
    """TrackEval stores arrays for HOTA metrics; take alpha=0.05 scalar."""
    if hasattr(m, "tolist"):
        arr = m.tolist()
        if isinstance(arr, list) and arr:
            return float(arr[0])
        return float(m)
    return float(m)


def run_eval(gt_dir, trk_dir, variants, split):
    dataset_cfg = {
        "GT_FOLDER": gt_dir,
        "TRACKERS_FOLDER": trk_dir,
        "OUTPUT_FOLDER": os.path.join(ROOT, "outputs", "l1_a", "trackeval"),
        "TRACKERS_TO_EVAL": variants,
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "DanceTrack",
        "SPLIT_TO_EVAL": split,
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
    res, _ = evaluator.evaluate(
        [_build_dataset(dataset_cfg)], metrics, show_progressbar=False)
    return res


def _build_dataset(cfg):
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    return MotChallenge2DBox(cfg)


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
                prefix = {"CLEAR": "CLEAR", "Identity": "Identity"}.get(metric_name, metric_name)
                row[f"{prefix}_{k}"] = metric_value(v)
        out[tracker] = {"combined": row, "per_seq": {}}
        for seq, cls_data in seqs.items():
            if seq == "COMBINED_SEQ":
                continue
            p = cls_data.get("pedestrian", {})
            per = {}
            for metric_name, vals in p.items():
                for k, v in vals.items():
                    if k.startswith("CLR_"):
                        k = k[4:]
                    prefix = {"CLEAR": "CLEAR", "Identity": "Identity"}.get(metric_name, metric_name)
                    per[f"{prefix}_{k}"] = metric_value(v)
            out[tracker]["per_seq"][seq] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["dla", "ctrl"], default="dla")
    ap.add_argument("--split", default="val")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    variants = args.variants.split(",")
    gt_dir, trk_dir, vids = build_data_dirs(args.protocol, args.split)
    # only eval variants that have outputs
    split_dir = os.path.join(trk_dir, f"DanceTrack-{args.split}")
    present = [v for v in variants if os.path.exists(os.path.join(split_dir, v, "data"))]
    if not present:
        print("[trackeval] no tracker outputs found")
        return
    res = run_eval(gt_dir, trk_dir, present, args.split)
    flat = flatten(res)
    out_dir = os.path.join(ROOT, "outputs", "l1_a")
    with open(os.path.join(out_dir, f"trackeval_{args.protocol}_{args.split}.json"), "w") as f:
        json.dump(flat, f, indent=2, default=str)
    # main_results.csv
    fields = ["HOTA_HOTA(0)", "HOTA_DetA(0)", "HOTA_AssA(0)", "HOTA_LocA(0)",
              "CLEAR_MOTA", "CLEAR_MOTP", "Identity_IDF1", "Identity_IDP", "Identity_IDR",
              "CLEAR_IDSW", "CLEAR_FP", "CLEAR_FN", "CLEAR_Frag", "CLEAR_MT", "CLEAR_PT", "CLEAR_ML"]
    rows = []
    for tracker in present:
        row = {"variant": tracker}
        for fld in fields:
            row[fld] = flat[tracker]["combined"].get(fld, "")
        rows.append(row)
    with open(os.path.join(out_dir, f"main_results_{args.protocol}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant"] + fields)
        w.writeheader()
        w.writerows(rows)
    # per-sequence
    with open(os.path.join(out_dir, f"per_sequence_results_{args.protocol}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "sequence"] + fields)
        for tracker in present:
            for seq, row in flat[tracker]["per_seq"].items():
                w.writerow([tracker, seq] + [row.get(fld, "") for fld in fields])
    print(json.dumps(rows, indent=2))
    print("[trackeval] done")


if __name__ == "__main__":
    main()
