"""Stage L1-C: Frozen vs LoRA PBD representation diagnostics.

Uses the same DanceTrack calibration frames and GT matching for both
representations; reports same-ID/different-ID cosine, ROC/PR-AUC, same-category
retrieval R@1/mAP, and candidate-quality metrics.

Usage:
  python tools/compare_frozen_lora_pbd.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402

SEED = 20260806
FROZEN_ROOT = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla"
LORA_ROOT = os.path.join(ROOT, "outputs/l1_c/cache_lora")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-6)


def load_feats(root, ds, vid, fid, protocol):
    fr = read_frame_cache(root, cache_key(ds, vid, fid, protocol))
    if fr is None:
        return None
    f = fr["features"]
    be = np.asarray(f["pbd_box_end_last"], dtype=np.float32)
    co = np.asarray(f["pbd_coord_mean_last"], dtype=np.float32)
    boxes = np.asarray(f["boxes"], dtype=np.float64)
    return {"be": be, "co": co, "boxes": boxes,
            "meta": fr["meta"]}


def roc_pr(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    tpr = tp / max(1, pos)
    fpr = fp / max(1, neg)
    # AUC via trapezoid
    auc = float(np.trapezoid(tpr, fpr)) if pos and neg else float("nan")
    precision = tp / np.maximum(1, tp + fp)
    recall = tpr
    # PR-AUC via average precision
    ap = float(np.sum(precision * labels) / max(1, pos)) if pos else float("nan")
    return auc, ap


def main():
    rng = random.Random(SEED)
    manifest = os.path.join(ROOT, "outputs/l1_c/fixed_candidate_manifest",
                            "dancetrack_calibration.jsonl")
    by_video = defaultdict(list)
    with open(manifest) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])

    stats = {"frames": 0, "gt": 0,
             "frozen_recall50": [], "lora_recall50": [],
             "frozen_cand": [], "lora_cand": [],
             "frozen_dup": [], "lora_dup": [],
             "frozen_small_recall": [], "lora_small_recall": []}
    same = {"frozen": [], "lora": []}
    diff = {"frozen": [], "lora": []}
    labels, scores_f, scores_l = [], [], []
    r1 = {"frozen": {"n": 0, "hit": 0}, "lora": {"n": 0, "hit": 0}}
    aps = {"frozen": [], "lora": []}

    for vid, entries in by_video.items():
        # pick frame pairs with gap 1..8
        for idx in range(1, len(entries)):
            a = entries[idx - 1]
            b = entries[idx]
            if b["frame"] - a["frame"] > 40:
                continue
            fa = load_feats(FROZEN_ROOT, "dancetrack", vid, a["frame"], "person")
            fb = load_feats(FROZEN_ROOT, "dancetrack", vid, b["frame"], "person")
            la = load_feats(LORA_ROOT, "dancetrack", vid, a["frame"], "lora")
            lb = load_feats(LORA_ROOT, "dancetrack", vid, b["frame"], "lora")
            if not all([fa, fb, la, lb]):
                continue
            stats["frames"] += 1
            stats["gt"] += len(b["gt_boxes"])
            # candidate quality per representation
            for name, fr, rp in [("frozen", fa, FROZEN_ROOT),
                                 ("lora", la, LORA_ROOT)]:
                boxes = fr["boxes"]
                stats[f"{name}_cand"].append(len(boxes))
                # duplicates
                dup = 0
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        if iou(boxes[i], boxes[j]) > 0.9:
                            dup += 1
                            break
                stats[f"{name}_dup"].append(dup)
                recall = 0
                small_recall = 0
                small_n = 0
                for gid, gt in b["gt_boxes"].items():
                    gw, gh = gt[2]-gt[0], gt[3]-gt[1]
                    small = min(gw, gh) < 32
                    hit = any(iou(gt, box) >= 0.5 for box in boxes)
                    recall += int(hit)
                    if small:
                        small_n += 1
                        small_recall += int(hit)
                ngt = max(1, len(b["gt_boxes"]))
                stats[f"{name}_recall50"].append(recall / ngt)
                stats[f"{name}_small_recall"].append(
                    small_recall / max(1, small_n))
            # pair diagnostics: for each GT present in both frames
            common = set(a["gt_boxes"]) & set(b["gt_boxes"])
            for gid in common:
                if gid not in a.get("matched", {}) or gid not in b.get("matched", {}):
                    continue
                ai = int(a["matched"][gid]["candidate"])
                bi = int(b["matched"][gid]["candidate"])
                if ai >= len(fa["be"]) or bi >= len(fb["be"]):
                    continue
                if ai >= len(la["be"]) or bi >= len(lb["be"]):
                    continue
                for name, x, y in [("frozen", fa, fb), ("lora", la, lb)]:
                    sim = float(np.dot(norm(x["be"][ai]), norm(y["be"][bi])))
                    same[name].append(sim)
                    labels.append(1)
                    scores_f.append(sim if name == "frozen" else 0.0)
                    scores_l.append(sim if name == "lora" else 0.0)
                # negatives: other GT candidates in frame b (same category)
                negs = []
                for og, om in b.get("matched", {}).items():
                    if og == gid:
                        continue
                    oi = int(om["candidate"])
                    negs.append(oi)
                rng.shuffle(negs)
                for oi in negs[:5]:
                    for name, x, y in [("frozen", fa, fb), ("lora", la, lb)]:
                        if oi >= len(y["be"]):
                            continue
                        sim = float(np.dot(norm(x["be"][ai]), norm(y["be"][oi])))
                        diff[name].append(sim)
                        labels.append(0)
                        scores_f.append(sim if name == "frozen" else 0.0)
                        scores_l.append(sim if name == "lora" else 0.0)
                # retrieval R@1 / mAP
                for name, x, y in [("frozen", fa, fb), ("lora", la, lb)]:
                    gallery = [bi]
                    for og, om in b.get("matched", {}).items():
                        if og != gid:
                            gallery.append(int(om["candidate"]))
                    gallery = sorted(set(gallery))
                    gallery = [j for j in gallery if j < len(y["be"])]
                    if bi not in gallery:
                        continue
                    sims = [float(np.dot(norm(x["be"][ai]), norm(y["be"][j])))
                            for j in gallery]
                    correct_pos = gallery.index(bi)
                    rank = sorted(range(len(gallery)),
                                  key=lambda k: -sims[k]).index(correct_pos) + 1
                    r1[name]["n"] += 1
                    r1[name]["hit"] += int(rank == 1)
                    aps[name].append(1.0 / rank)

    auc_f, ap_f = roc_pr([l for l, s in zip(labels, scores_f) if s > 0 or l],
                         [s for l, s in zip(labels, scores_f) if s > 0 or l])
    mask_l = [i for i, s in enumerate(scores_l) if s > 0 or labels[i]]
    auc_l, ap_l = roc_pr([labels[i] for i in mask_l],
                         [scores_l[i] for i in mask_l])
    out = {
        "frames": stats["frames"], "gt_detections": stats["gt"],
        "frozen": {
            "same_id_cos_mean": round(float(np.mean(same["frozen"])), 4),
            "diff_id_cos_mean": round(float(np.mean(diff["frozen"])), 4),
            "roc_auc": round(auc_f, 4), "pr_auc": round(ap_f, 4),
            "r1": round(r1["frozen"]["hit"] / max(1, r1["frozen"]["n"]), 4),
            "mAP": round(float(np.mean(aps["frozen"])), 4),
            "recall50": round(float(np.mean(stats["frozen_recall50"])), 4),
            "cand_per_frame": round(float(np.mean(stats["frozen_cand"])), 2),
            "dup_per_frame": round(float(np.mean(stats["frozen_dup"])), 2),
            "small_recall": round(float(np.mean(stats["frozen_small_recall"])), 4),
        },
        "lora": {
            "same_id_cos_mean": round(float(np.mean(same["lora"])), 4),
            "diff_id_cos_mean": round(float(np.mean(diff["lora"])), 4),
            "roc_auc": round(auc_l, 4), "pr_auc": round(ap_l, 4),
            "r1": round(r1["lora"]["hit"] / max(1, r1["lora"]["n"]), 4),
            "mAP": round(float(np.mean(aps["lora"])), 4),
            "recall50": round(float(np.mean(stats["lora_recall50"])), 4),
            "cand_per_frame": round(float(np.mean(stats["lora_cand"])), 2),
            "dup_per_frame": round(float(np.mean(stats["lora_dup"])), 2),
            "small_recall": round(float(np.mean(stats["lora_small_recall"])), 4),
        },
        "n_pairs": len(same["frozen"]),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(ROOT, "outputs/l1_c/frozen_vs_lora_pbd.json"),
              "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
