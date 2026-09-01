"""History-window helpers shared by T3-T6."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .track_state import Obs


def recent_observations(track, k: int, cur_frame: int) -> List[Obs]:
    """Last up-to-k accepted observations with frame < cur_frame."""
    return [o for o in track.history if o.frame < cur_frame][-k:]


def build_window_tensors(
    tracks,
    k: int,
    cur_frame: int,
    device="cpu",
) -> Dict[str, torch.Tensor]:
    """Build [B,K] window tensors for a list of tracks.

    History entries older than the current frame are used; missing entries are
    padded with zeros and mask=True. geom is normalized xyxy + area.
    """
    B = len(tracks)
    K = k
    pbd = torch.zeros(B, K, 2048, device=device)
    region = torch.zeros(B, K, 4608, device=device)
    geom = torch.zeros(B, K, 5, device=device)
    gen = torch.zeros(B, K, device=device)
    gaps = torch.zeros(B, K, device=device)
    mask = torch.ones(B, K, dtype=torch.bool, device=device)
    boxes = torch.zeros(B, K, 4, device=device)
    for b, trk in enumerate(tracks):
        obs = recent_observations(trk, k, cur_frame)
        if not obs:
            continue
        start = K - len(obs)
        for j, o in enumerate(obs):
            idx = start + j
            mask[b, idx] = False
            gaps[b, idx] = float(max(1, cur_frame - o.frame))
            boxes[b, idx] = torch.as_tensor(o.box, dtype=torch.float32, device=device)
            f = o.features
            if f.get("pbd") is not None:
                pbd[b, idx] = torch.as_tensor(f["pbd"], dtype=torch.float32, device=device)
            if f.get("region") is not None:
                region[b, idx] = torch.as_tensor(f["region"], dtype=torch.float32, device=device)
            if f.get("geom") is not None:
                geom[b, idx] = torch.as_tensor(f["geom"], dtype=torch.float32, device=device)
            gen[b, idx] = float(o.gen_score or 0.0)
    return {"pbd": pbd, "region": region, "geom": geom, "gen": gen,
            "gaps": gaps, "mask": mask, "boxes": boxes}


def last_features(track) -> Optional[Dict[str, np.ndarray]]:
    if track.history:
        return track.history[-1].features
    return track.last_features


def normalize_geom(box_xyxy, image_size):
    w, h = float(image_size[0]), float(image_size[1])
    x1, y1, x2, y2 = box_xyxy
    bw = max(1e-3, x2 - x1)
    bh = max(1e-3, y2 - y1)
    n = [x1 / w, y1 / h, x2 / w, y2 / h]
    area = min(1.0, (bw * bh) / (w * h))
    return np.asarray([n[0], n[1], n[2], n[3], area], dtype=np.float32)
