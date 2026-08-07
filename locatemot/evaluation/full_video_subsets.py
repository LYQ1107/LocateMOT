"""Stratified full-video analysis: low-IoU, density, ambiguity, reactivation."""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(video_dir):
    """{frame: [(x1,y1,x2,y2,oid)]} class=1."""
    gt = defaultdict(list)
    for line in open(os.path.join(video_dir, "gt", "gt.txt")):
        p = line.strip().split(",")
        if len(p) < 9 or int(p[7]) != 1:
            continue
        fid, oid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        if w <= 0 or h <= 0:
            continue
        gt[fid].append((x, y, x + w, y + h, oid))
    return gt


def load_tracker_txt(path):
    """{frame: [(x1,y1,x2,y2,tid)]}."""
    out = defaultdict(list)
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        fid = int(float(p[0]))
        tid = int(float(p[1]))
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        out[fid].append((x, y, x + w, y + h, tid))
    return out


def match_gt_to_tracker(gt_boxes, tr_boxes, thr=0.5):
    """For each GT box, best tracker box with IoU>=thr -> (tid, iou) or None."""
    out = {}
    for oid, g in gt_boxes:
        best, best_iou = None, 0.0
        for t in tr_boxes:
            iou = _iou(g[:4], t[:4])
            if iou > best_iou:
                best, best_iou = t[4], iou
        out[oid] = (best, best_iou) if best is not None and best_iou >= thr else None
    return out


def iou_bucket(iou):
    if iou < 0.1:
        return "<0.1"
    if iou < 0.3:
        return "0.1-0.3"
    if iou < 0.5:
        return "0.3-0.5"
    return ">=0.5"


def density_bucket(n):
    if n <= 5:
        return "low"
    if n <= 10:
        return "medium"
    return "high"


def analyze_video(gt, tr, video_id):
    """Returns per-continuation events with attributes and reactivation events."""
    frames = sorted(gt)
    cont = []
    react = []
    last_tid = {}
    last_frame = {}
    lost_since = {}
    for t in frames:
        match = match_gt_to_tracker(gt[t], tr.get(t, []))
        n_gt = len(gt[t])
        boxes = {oid: g[:4] for oid, g in gt[t]}
        density = density_bucket(n_gt)
        # ambiguity: GT-level top2 IoU overlap
        amb = False
        for oid, b in boxes.items():
            for oid2, b2 in boxes.items():
                if oid2 <= oid:
                    continue
                if _iou(b, b2) >= 0.1:
                    amb = True
        for oid, g in gt[t]:
            prev_box = None
            if oid in last_frame and t - 1 == last_frame[oid]:
                prev_box = last_box.get(oid)
            if prev_box is not None:
                iou = _iou(prev_box, g[:4])
                m_now = match.get(oid)
                m_prev = last_match.get(oid)
                if m_now is not None and m_prev is not None:
                    same = m_now[0] == m_prev[0]
                    cont.append({
                        "video": video_id, "frame": t, "oid": oid,
                        "iou_bucket": iou_bucket(iou), "density": density,
                        "ambiguous": amb, "same_id": same,
                    })
                    if not same:
                        # reactivation candidate if there was a >=2-frame gap
                        pass
            # reactivation tracking
            m = match.get(oid)
            if m is None:
                lost_since[oid] = lost_since.get(oid, 0) + 1
            else:
                if lost_since.get(oid, 0) >= 2 and oid in last_tid:
                    react.append({
                        "video": video_id, "frame": t, "oid": oid,
                        "gap": lost_since[oid],
                        "prev_tid": last_tid[oid], "new_tid": m[0],
                        "id_kept": m[0] == last_tid[oid],
                    })
                lost_since[oid] = 0
                last_tid[oid] = m[0]
            last_frame[oid] = t
        # for objects that disappear from GT, keep last_box for consecutive check
        last_box = dict(boxes)
        last_match = match
    return cont, react


def analyze_splits(gt_root, tracker_root, variants, split_videos, protocols=("dla",)):
    """Run subset + reactivation analysis; returns per-variant aggregates."""
    out = {}
    for protocol in protocols:
        for variant in variants:
            cont_all = []
            react_all = []
            for vid in split_videos:
                gt = load_gt(os.path.join(gt_root, vid))
                tr = load_tracker_txt(os.path.join(tracker_root, protocol, variant, f"{vid}.txt"))
                cont, react = analyze_video(gt, tr, vid)
                cont_all.extend(cont)
                react_all.extend(react)
            out[f"{protocol}|{variant}"] = {"continuations": cont_all, "reactivations": react_all}
    return out


def aggregate(analyzed):
    rows = []
    for key, data in analyzed.items():
        cont = data["continuations"]
        react = data["reactivations"]
        total = len(cont)
        same = sum(1 for c in cont if c["same_id"])
        row = {"variant": key, "continuations": total,
               "association_accuracy": same / total if total else 0.0,
               "idsw": total - same}
        for bucket in ("<0.1", "0.1-0.3", "0.3-0.5", ">=0.5"):
            sub = [c for c in cont if c["iou_bucket"] == bucket]
            n = len(sub)
            row[f"iou_{bucket}_n"] = n
            row[f"iou_{bucket}_acc"] = sum(1 for c in sub if c["same_id"]) / n if n else 0.0
        for bucket in ("low", "medium", "high"):
            sub = [c for c in cont if c["density"] == bucket]
            n = len(sub)
            row[f"density_{bucket}_n"] = n
            row[f"density_{bucket}_acc"] = sum(1 for c in sub if c["same_id"]) / n if n else 0.0
        amb = [c for c in cont if c["ambiguous"]]
        n = len(amb)
        row["ambiguous_n"] = n
        row["ambiguous_acc"] = sum(1 for c in amb if c["same_id"]) / n if n else 0.0
        row["reactivation_events"] = len(react)
        row["reactivation_id_kept"] = sum(1 for r in react if r["id_kept"])
        row["reactivation_accuracy"] = row["reactivation_id_kept"] / len(react) if react else 0.0
        row["reactivation_mean_gap"] = np.mean([r["gap"] for r in react]) if react else 0.0
        rows.append(row)
    return rows
