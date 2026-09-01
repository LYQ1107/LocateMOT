#!/usr/bin/env python
"""Minimal smoke tests for Stage L1-A tracker variants (CPU)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.models.track_decoder.relation_track_decoder import RelationTrackDecoderModel  # noqa: E402
from locatemot.models.trajectory import (  # noqa: E402
    MemoryFusion,
    MotionPredictor,
    MotionResidualHead,
    ReactivationResidualHead,
    TrajectoryEncoder,
)
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402


def fake_candidates(n, seed=0):
    rng = np.random.RandomState(seed)
    cands = []
    for i in range(n):
        x1, y1 = rng.uniform(0, 500, 2)
        w, h = rng.uniform(40, 120, 2)
        box = np.asarray([x1, y1, x1 + w, y1 + h])
        feats = {
            "pbd": rng.randn(2048).astype(np.float32) * 0.1,
            "pbd_be": rng.randn(2048).astype(np.float32) * 0.1,
            "region": rng.randn(4608).astype(np.float32) * 0.1,
            "geom": np.asarray([x1 / 640, y1 / 480, (x1 + w) / 640, (y1 + h) / 480,
                                (w * h) / (640 * 480)], dtype=np.float32),
            "gen": 0.8,
        }
        cands.append({"box": box, "features": feats})
    return cands


def main():
    torch.manual_seed(0)
    b6 = RelationTrackDecoderModel(use_pbd_base=True, use_region_geom=True, residual=True)
    ck_path = os.path.join(ROOT, "outputs/l0_d/checkpoints/b6/best.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        b6.load_state_dict(ck["model"])
    b6.eval()
    traj = TrajectoryEncoder().eval()
    motion = MotionPredictor().eval()
    mem = MemoryFusion().eval()
    mhead = MotionResidualHead().eval()
    rhead = ReactivationResidualHead().eval()
    base_cands = fake_candidates(5, seed=1)
    for variant in ("T0", "T1", "T2", "T3", "T4", "T5", "T6"):
        tracker = OnlineTracker(
            variant=variant,
            b6=b6,
            trajectory_encoder=traj if variant in ("T3", "T4", "T5", "T6") else None,
            motion_predictor=motion if variant in ("T4", "T5", "T6") else None,
            memory_fusion=mem if variant in ("T5", "T6") else None,
            motion_residual_head=mhead if variant in ("T4", "T5", "T6") else None,
            reactivation_head=rhead if variant == "T6" else None,
            device="cpu",
            image_size=(640, 480),
        )
        counts = []
        for fid in range(1, 8):
            n = 3 + fid % 3
            cands = []
            for c in base_cands[:n]:
                box = c["box"].copy()
                box[0] += fid * 8
                box[2] += fid * 8
                feats = {k: (v.copy() if hasattr(v, "copy") else v)
                         for k, v in c["features"].items()}
                cands.append({"box": box, "features": feats})
            out = tracker.process_frame(fid, cands)
            counts.append(len(out))
        ids = [o["track_id"] for o in out]
        assert len(set(ids)) == len(ids), "duplicate track ids in one frame"
        print(f"[smoke] {variant} ok outputs={counts}")
    # one-to-one and NO_MATCH behavior
    tracker = OnlineTracker(variant="T2", b6=b6, device="cpu", image_size=(640, 480))
    for fid in (1, 2, 3):
        out = tracker.process_frame(fid, fake_candidates(4, seed=fid))
        assert len({o["track_id"] for o in out}) == len(out)
    print("[smoke] all variants passed")


if __name__ == "__main__":
    main()
