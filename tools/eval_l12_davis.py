"""Stage L12: frozen shared UIDM prompt-seeded identity persistence.

Controlled prompt-type evaluation on DAVIS 2017 val (multi-object):
  - mask: seed token from mask-masked crop PBD;
  - box:  seed token from tight bbox crop PBD;
  - point:seed token from point-centered square crop PBD.

The shared UIDM is frozen; only the seed identity token differs.
Seeded-only policy: NEW is disabled (uidm_new_margin large), seeds are
injected as birth states at frame 0.

Output: results/l12/davis_<prompt>.json with identity metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402
from locatemot.models.l8_unified import L8UnifiedUIDM, load_l8_state  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402
from locatemot.models.l1d_association import CAND_FEATURES  # noqa: E402
from tools.train_l9_uidm import _specs  # noqa: E402

DATA = ROOT / "outputs" / "l12" / "data" / "davis"
PBD_CACHE = ROOT / "outputs" / "l12" / "cache" / "davis_pbd"
SEED_PBD = ROOT / "outputs" / "l12" / "data" / "davis_seed_pbd.json"
SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    den = ar + br - inter
    return inter / den if den > 1e-9 else 0.0


def seed_h(model, sp, clip, spec, device):
    with torch.no_grad():
        t = torch.as_tensor(sp, dtype=torch.float32, device=device)
        tok = model.uidm.pbd_encoder(t.unsqueeze(0))
        cand_feat = torch.zeros(1, len(CAND_FEATURES), device=device)
        tok = tok + model.uidm.cand_mlp(cand_feat)
        if model.adapter is not None:
            sem, _ = model.adapter(
                t.unsqueeze(0),
                torch.as_tensor(clip, dtype=torch.float32,
                                device=device).unsqueeze(0),
                torch.as_tensor(spec, dtype=torch.float32,
                                device=device).unsqueeze(0))
            tok = tok + sem
        h = model.uidm.memory.init(tok).squeeze(0).cpu().numpy()
    return np.asarray(h, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", choices=["mask", "box", "point"],
                    required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match-thr", type=float, default=0.0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model = L8UnifiedUIDM(
        **SIZES[cfg.get("model", "base")],
        mode=cfg.get("mode", "unified"),
        sem_in_core=cfg.get("sem_in_core", True),
        cond_gated=cfg.get("cond_gated", False)).to(device)
    load_l8_state(model, ck["model"])
    model.eval()
    spec = _specs(["all objects"], device=device)[0]
    seed_pbd = json.loads(SEED_PBD.read_text()) if SEED_PBD.exists() else {}
    videos = sorted(p.stem for p in DATA.glob("*.pkl"))
    if args.videos:
        videos = [v for v in videos if v in set(args.videos)]
    if args.max_videos:
        videos = videos[:args.max_videos]
    rows = []
    d_model = model.uidm.d_model
    for vi, vid in enumerate(videos):
        rec = pickle.load(open(DATA / f"{vid}.pkl", "rb"))
        frames = {fr["frame"]: fr for fr in rec["frames"]}
        seeds = rec.get("seeds", {})
        tracker = OnlineTracker(
            variant="UIDM", uidm=model.uidm, device=str(device),
            output_all_candidates=True,
            uidm_adapter=model.adapter, uidm_spec=spec)
        tracker.uidm_sem_in_core = model.sem_in_core
        tracker.uidm_new_margin = 1e6  # seeded-only policy
        tracker.uidm_seeded_only = True
        tracker.uidm_seeded_match_thr = args.match_thr
        tracker.l1d_weights = (0.4, 0.2, 0.4)
        tracker.l1d_threshold = 0.25
        # frame 0: inject seed births
        fr0 = frames[0]
        n0 = len(fr0["boxes"])
        pbd0 = np.zeros((n0, 2048), np.float32)
        d = read_frame_cache(str(PBD_CACHE),
                             cache_key("davis", vid, 0, "pbd_full"))
        if d is not None:
            p = np.asarray(d["features"]["pbd_box_end_last"], np.float32)
            if len(p) == n0:
                pbd0 = p
        cands = []
        for j in range(n0):
            cands.append({
                "box": fr0["boxes"][j],
                "features": {
                    "pbd": pbd0[j], "pbd_be": pbd0[j],
                    "region": np.zeros(4608, np.float32),
                    "geom": np.zeros(5, np.float32),
                    "gen": float(fr0["gen"][j]),
                },
                "index": j,
            })
        tracker.image_size = rec["image_size"]
        used_idx = set()
        for oid, seed in seeds.items():
            s = seed_pbd.get(vid, {}).get(str(oid), {}).get(args.prompt)
            if s is None:
                continue
            sp = np.asarray(s, np.float32)
            sclip = seed_pbd.get(vid, {}).get(str(oid),
                                              {}).get(args.prompt + "_clip")
            if sclip is None:
                sclip = np.zeros(512, np.float32)
            else:
                sclip = np.asarray(sclip, np.float32)
            best_j, best_iou = -1, 0.0
            for j in range(n0):
                if j in used_idx:
                    continue
                v = iou(fr0["boxes"][j], seed["box"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j < 0 or best_iou < 0.3:
                # inject synthetic candidate so the seed always persists
                cands.append({
                    "box": seed["box"],
                    "features": {
                        "pbd": sp, "pbd_be": sp,
                        "region": np.zeros(4608, np.float32),
                        "geom": np.zeros(5, np.float32),
                        "gen": 0.9,
                    },
                    "index": len(cands) - 1,
                })
                best_j = len(cands) - 1
            else:
                used_idx.add(best_j)
            tracker._uidm_forced_birth[best_j] = {
                "h": seed_h(model, sp, sclip, spec, device),
                "anchor": sp, "ref_pbd": sp, "anchor_pbd": sp,
                "alive": 5.0,
            }
        outs0 = tracker.process_frame(0, cands)
        seed_track = {}
        for oid, seed in seeds.items():
            best_o, best_iou = None, 0.0
            for o in outs0:
                v = iou(o["box"], seed["box"])
                if v > best_iou:
                    best_iou, best_o = v, o
            if best_o is not None and best_iou >= 0.3:
                seed_track[str(oid)] = best_o["track_id"]
        # subsequent frames
        obj_stats = {str(oid): {"frames": 0, "matched": 0,
                                "switched": 0, "ids": set()}
                     for oid in seeds}
        for frame in sorted(frames):
            if frame == 0:
                continue
            fr = frames[frame]
            n = len(fr["boxes"])
            if n == 0:
                continue
            pbd = np.zeros((n, 2048), np.float32)
            d = read_frame_cache(str(PBD_CACHE),
                                 cache_key("davis", vid, frame, "pbd_full"))
            if d is not None:
                p = np.asarray(d["features"]["pbd_box_end_last"], np.float32)
                if len(p) == n:
                    pbd = p
            cands = []
            for j in range(n):
                cands.append({
                    "box": fr["boxes"][j],
                    "features": {
                        "pbd": pbd[j], "pbd_be": pbd[j],
                        "region": np.zeros(4608, np.float32),
                        "geom": np.zeros(5, np.float32),
                        "gen": float(fr["gen"][j]),
                    },
                    "index": j,
                })
            outs = tracker.process_frame(frame, cands)
            for oid, seed in seeds.items():
                gb = fr["gt_boxes"].get(str(oid))
                if gb is None:
                    continue
                st = seed_track.get(str(oid))
                if st is None:
                    continue
                best_o, best_iou = None, 0.0
                for o in outs:
                    v = iou(o["box"], gb)
                    if v > best_iou:
                        best_iou, best_o = v, o
                stat = obj_stats[str(oid)]
                stat["frames"] += 1
                if best_o is not None and best_iou >= 0.5:
                    stat["matched"] += 1
                    stat["ids"].add(best_o["track_id"])
                    if best_o["track_id"] != st:
                        stat["switched"] += 1
        for oid, st in obj_stats.items():
            rows.append({
                "video": vid, "prompt": args.prompt, "object": oid,
                "frames": st["frames"], "matched": st["matched"],
                "switched": st["switched"],
                "distinct_ids": len(st["ids"]),
            })
        print(f"[l12davis] {vid} {args.prompt} {vi+1}/{len(videos)}",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    nf = sum(r["frames"] for r in rows)
    nm = sum(r["matched"] for r in rows)
    nsw = sum(r["switched"] for r in rows)
    print(f"[l12davis] {args.prompt} frames={nf} matched={nm} "
          f"persistence={nm/max(1,nf):.3f} switch={nsw/max(1,nm):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
