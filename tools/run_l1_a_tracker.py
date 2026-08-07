#!/usr/bin/env python
"""Stage L1-A: run T0-T6 full-video trackers on a DanceTrack split.

Usage:
  python tools/run_l1_a_tracker.py --variants T0,T1,T2 --split calibration \
      --gpu 3 --protocol dla
  python tools/run_l1_a_tracker.py --variants T0,T1 --split val \
      --gpu 3 --protocol ctrl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import read_frame_cache  # noqa: E402
from locatemot.models.track_decoder.relation_track_decoder import RelationTrackDecoderModel  # noqa: E402
from locatemot.models.trajectory import (  # noqa: E402
    MemoryFusion,
    MotionPredictor,
    MotionResidualHead,
    ReactivationResidualHead,
    TrajectoryEncoder,
)
from locatemot.tracking.online_tracker import OnlineTracker, _feat_dict  # noqa: E402

DANCETRACK = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"
DLA_CACHE = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla"
CTRL_CACHE = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/detections_ctrl"


def load_b6(device, ckpt="outputs/l0_d/checkpoints/b6/best.pt"):
    model = RelationTrackDecoderModel(use_pbd_base=True, use_region_geom=True, residual=True)
    ck = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model


def load_temporal(ckpt, device):
    if not ckpt or not os.path.exists(ckpt):
        return None, None, None, None, None, 0.0
    ck = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    traj = TrajectoryEncoder()
    motion = MotionPredictor()
    mem = MemoryFusion()
    mhead = MotionResidualHead()
    rhead = ReactivationResidualHead()
    state = ck["model"] if "model" in ck else ck
    missing = []
    for m, name in [(traj, "trajectory_encoder"), (motion, "motion_predictor"),
                    (mem, "memory_fusion"), (mhead, "motion_residual_head"),
                    (rhead, "reactivation_head")]:
        pref = name + "."
        sub = {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}
        if sub:
            m.load_state_dict(sub)
        else:
            missing.append(name)
    if missing:
        print(f"[load_temporal] missing modules in checkpoint: {missing}", flush=True)
    for m in (traj, motion, mem, mhead, rhead):
        m.to(device).eval()
    nm_bias = float(state.get("nm_bias", 0.0)) if "nm_bias" in state else 0.0
    return traj, motion, mem, mhead, rhead, nm_bias


def _video_frames(vid, split):
    data_dir = "train" if split == "calibration" else split
    img_dir = os.path.join(DANCETRACK, data_dir, vid, "img1")
    names = sorted(os.listdir(img_dir))
    return [int(os.path.splitext(n)[0]) for n in names]


def _image_size(vid, split):
    import configparser
    data_dir = "train" if split == "calibration" else split
    p = os.path.join(DANCETRACK, data_dir, vid, "seqinfo.ini")
    cfg = configparser.ConfigParser()
    cfg.read(p)
    return (int(cfg["Sequence"]["imWidth"]), int(cfg["Sequence"]["imHeight"]))


def _load_dla_frame(cache_root, vid, fid):
    from locatemot.data.token_cache import cache_key
    return read_frame_cache(cache_root, cache_key("dancetrack", vid, fid, "person"))


def _load_ctrl_frame(cache_root, vid, fid):
    txt = os.path.join(cache_root, vid, f"{fid:06d}.txt")
    if not os.path.exists(txt):
        return None
    rows = np.loadtxt(txt, delimiter=",").reshape(-1, 5)
    return rows


def build_candidates_dla(frame):
    f = frame["features"]
    meta = frame["meta"]
    boxes = np.asarray(f["boxes"], dtype=np.float64)
    cands = []
    for i in range(len(boxes)):
        feats = _feat_dict(f, i, boxes[i], meta["image_size"])
        cands.append({"box": boxes[i], "features": feats, "index": i})
    return cands, meta


def build_candidates_ctrl(rows, vid):
    cands = []
    for r in rows:
        x1, y1, w, h, s = r
        box = np.asarray([x1, y1, x1 + w, y1 + h], dtype=np.float64)
        feats = {
            "pbd": np.zeros(2048, dtype=np.float32),
            "pbd_be": np.zeros(2048, dtype=np.float32),
            "region": np.zeros(4608, dtype=np.float32),
            "geom": np.asarray([0, 0, 0, 0, 0], dtype=np.float32),
            "gen": float(s),
        }
        cands.append({"box": box, "features": feats, "index": len(cands)})
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="T0,T1,T2,T3,T4,T5,T6")
    ap.add_argument("--split", choices=["train", "calibration", "val"], required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--protocol", choices=["dla", "ctrl"], default="dla")
    ap.add_argument("--temporal-ckpt", default="")
    ap.add_argument("--out", default="outputs/l1_a")
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    b6 = None
    variants = args.variants.split(",")
    if any(v in ("T2", "T3", "T4", "T5", "T6") for v in variants):
        b6 = load_b6(device)
    traj, motion, mem, mhead, rhead, nm_bias = load_temporal(args.temporal_ckpt, device)

    split_cfg = json.load(open(os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{args.split}.json")))
    vids = [v["video_id"] for v in split_cfg["videos"]]
    if args.max_videos:
        vids = vids[: args.max_videos]

    cache_root = DLA_CACHE if args.protocol == "dla" else CTRL_CACHE
    out_root = os.path.join(ROOT, args.out, "trackeval", args.protocol)
    os.makedirs(out_root, exist_ok=True)
    stats = {}
    for variant in variants:
        t0 = time.time()
        tracker = OnlineTracker(
            variant=variant,
            b6=b6,
            trajectory_encoder=traj if variant in ("T3", "T4", "T5", "T6") else None,
            motion_predictor=motion if variant in ("T4", "T5", "T6") else None,
            memory_fusion=mem if variant in ("T5", "T6") else None,
            motion_residual_head=mhead if variant in ("T4", "T5", "T6") else None,
            reactivation_head=rhead if variant == "T6" else None,
            device=device,
            iou_threshold=0.3,
            no_match_theta=-3.5,
            no_match_bias=nm_bias if variant in ("T2", "T3", "T4", "T5", "T6") else 0.0,
            max_age=30,
            min_hits=3,
            memory_conf_threshold=0.5,
        )
        total_frames = 0
        for vid in vids:
            tracker.reset()
            tracker.image_size = _image_size(vid, args.split)
            rows = []
            for fid in _video_frames(vid, args.split):
                if args.protocol == "dla":
                    frame = _load_dla_frame(cache_root, vid, fid)
                    if frame is None:
                        continue
                    cands, meta = build_candidates_dla(frame)
                    tracker.image_size = meta["image_size"]
                else:
                    dets = _load_ctrl_frame(cache_root, vid, fid)
                    cands = build_candidates_ctrl(dets, vid) if dets is not None else []
                outputs = tracker.process_frame(fid, cands)
                for o in outputs:
                    x1, y1, x2, y2 = o["box"]
                    rows.append([fid, o["track_id"], x1, y1, x2 - x1, y2 - y1, 1.0, -1, -1, -1])
                total_frames += 1
            vid_out = os.path.join(out_root, variant, f"{vid}.txt")
            os.makedirs(os.path.dirname(vid_out), exist_ok=True)
            with open(vid_out, "w") as f:
                for r in rows:
                    f.write(",".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + "\n")
        elapsed = time.time() - t0
        stats[variant] = {"videos": len(vids), "frames": total_frames,
                          "seconds": round(elapsed, 2),
                          "fps": round(total_frames / elapsed, 2) if elapsed > 0 else 0.0}
        print(f"[run] {variant}: {stats[variant]}", flush=True)
    with open(os.path.join(ROOT, args.out, f"tracker_runtime_{args.protocol}_{args.split}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[run] done")


if __name__ == "__main__":
    main()
