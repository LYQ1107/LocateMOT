"""Stage L1-C: train Unified Association Decoder (UAF, frozen LocateAnything).

Clip-based, dataset-balanced training on fixed candidate manifests.

Usage:
  python tools/train_l1c_uaf.py --gpu 1 --steps 100000 --smoke 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import read_frame_cache  # noqa: E402
from locatemot.models.ua_decoder import (  # noqa: E402
    UnifiedAssociationDecoder, association_loss, motion_loss,
)

SEED = 20260806
MANIFEST_DIR = ROOT / "outputs" / "l1_c" / "fixed_candidate_manifest"
CLIP_LEN = 8
MAX_K = 8
MAX_TRACKS = 24
MAX_CANDS = 64


def load_manifest(path):
    by_video = defaultdict(dict)
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]][int(e["frame"])] = e
    return {v: dict(sorted(frames.items())) for v, frames in by_video.items()}


def read_frame(entry):
    if entry.get("_frame") is not None:
        return entry["_frame"]
    root = entry["cache_root"]
    key = f"{entry['dataset']}/{entry['video_id']}/{int(entry['frame']):05d}/{entry['protocol']}"
    fr = read_frame_cache(root, key)
    entry["_frame"] = fr
    return fr


def frame_candidates(entry):
    fr = read_frame(entry)
    if fr is None:
        return []
    feats = fr["features"]
    n = int(entry["candidate_count"])
    boxes = np.asarray(feats.get("boxes", np.zeros((0, 4))), dtype=np.float64)
    if len(boxes) != n:
        n = len(boxes)
    out = []
    for i in range(n):
        def _f(key, shape):
            if key in feats and len(feats[key]) > i:
                v = np.asarray(feats[key][i], dtype=np.float32)
                if v.shape == shape:
                    return np.nan_to_num(v)
            return np.zeros(shape, dtype=np.float32)
        f = {
            "pbd": _f("pbd_coord_mean_last", (2048,)),
            "pbd_be": _f("pbd_box_end_last", (2048,)),
            "region": _f("region", (4608,)),
            "geom": _f("geometry", (5,)),
            "gen": float(np.nan_to_num(feats["gen_score"][i])) if "gen_score" in feats
            and len(feats["gen_score"]) > i else 0.0,
            "box": np.asarray(boxes[i], dtype=np.float64),
        }
        out.append(f)
    return out


def norm_feat(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-6 else x


class ClipDataset:
    def __init__(self, manifests, seed=SEED):
        self.manifests = manifests  # list of (name, by_video)
        self.rng = random.Random(seed)
        self.videos = []
        for name, by_video in manifests:
            for vid, frames in by_video.items():
                if len(frames) >= 2:
                    self.videos.append((name, vid, list(frames.items())))

    def __len__(self):
        return max(1, len(self.videos))

    def sample_clip(self):
        name, vid, frames = self.rng.choice(self.videos)
        L = min(CLIP_LEN, len(frames))
        start = self.rng.randint(0, len(frames) - L)
        clip = frames[start:start + L]
        return name, vid, clip


def _obs_features(f):
    return {
        "pbd": f["pbd"], "pbd_be": f["pbd_be"], "region": f["region"],
        "geom": f["geom"], "gen": f["gen"], "box": f["box"],
    }


def build_clip(name, vid, clip, max_tracks=MAX_TRACKS, max_cands=MAX_CANDS):
    """Return one training sample dict (tensors ready for collate)."""
    entries = [e for _, e in clip]
    # per-frame candidate lists
    cands_by_frame = [frame_candidates(e) for e in entries]
    # gt-object -> per-frame matched candidate index
    object_frames = defaultdict(list)
    for fi, e in enumerate(entries):
        for gid, m in e.get("matched", {}).items():
            object_frames[gid].append((fi, int(m["candidate"])))

    # object observation list: (frame_idx, candidate_idx, features, box)
    obs_by_obj = {gid: [] for gid in object_frames}
    samples = []
    for t in range(1, len(entries)):
        e = entries[t]
        cands = cands_by_frame[t]
        if not cands:
            continue
        # update observation lists with frames < t
        for gid, lst in object_frames.items():
            for fi, ci in lst:
                if fi >= t:
                    continue
                f = cands_by_frame[fi][ci]
                obs = (fi, ci, _obs_features(f), f["box"])
                if not obs_by_obj[gid] or obs_by_obj[gid][-1][0] != fi:
                    obs_by_obj[gid].append(obs)
        active = {gid: obs for gid, obs in obs_by_obj.items() if obs}
        # cap tracks (most recently active first)
        tracks = sorted(active.items(), key=lambda kv: -kv[1][-1][0])
        tracks = tracks[:max_tracks]
        if not tracks:
            continue
        track_index = {gid: i for i, (gid, _) in enumerate(tracks)}
        # cap candidates by gen score, keep all matched first
        matched_flags = [any(m.get("candidate") == i
                             for m in e.get("matched", {}).values())
                         for i in range(len(cands))]
        matched_idx = [i for i, mf in enumerate(matched_flags) if mf]
        unmatched_idx = sorted(
            (i for i, mf in enumerate(matched_flags) if not mf),
            key=lambda i: -cands[i]["gen"])
        keep = (matched_idx + unmatched_idx[:8])[:max_cands]
        keep_set = set(keep)
        labels = []
        valid = []
        for i in range(len(cands)):
            if i not in keep_set:
                continue
            gid = next((g for g, m in e.get("matched", {}).items()
                        if m.get("candidate") == i), None)
            if gid is not None and gid in track_index:
                labels.append(track_index[gid])
            else:
                labels.append(len(tracks))  # NEW
            valid.append(1)
        if not valid:
            continue
        cand_idx = keep
        sample = {
            "name": name, "video": vid, "frame": e["frame"], "t": t,
            "tracks": tracks, "cands": [cands[i] for i in cand_idx],
            "labels": labels, "valid": valid,
            "image_size": e.get("image_size", [1280, 720]),
        }
        samples.append(sample)
    return samples


def _arr(v, shape, dtype=np.float32):
    a = np.zeros(shape, dtype=dtype)
    if v is not None:
        a[:] = np.asarray(v, dtype=dtype)
    return a


def collate(samples, device, image_scale=None):
    """Samples from one frame -> padded batch tensors (batch dim=1 or stacked)."""
    B = len(samples)
    max_t = max(len(s["tracks"]) for s in samples) if samples else 0
    max_n = max(len(s["cands"]) for s in samples) if samples else 0
    D = 2048
    R = 4608
    cur_pbd = np.zeros((B, max_n, D), np.float32)
    cur_pbd_be = np.zeros((B, max_n, D), np.float32)
    cur_reg = np.zeros((B, max_n, R), np.float32)
    cur_geom = np.zeros((B, max_n, 5), np.float32)
    cur_norm_geom = np.zeros((B, max_n, 4), np.float32)
    cur_gen = np.zeros((B, max_n, 1), np.float32)
    trk_pbd = np.zeros((B, max_t, MAX_K, D), np.float32)
    trk_reg = np.zeros((B, max_t, MAX_K, R), np.float32)
    trk_geom = np.zeros((B, max_t, MAX_K, 5), np.float32)
    trk_gen = np.zeros((B, max_t, MAX_K), np.float32)
    trk_times = np.zeros((B, max_t, MAX_K), np.int64)
    trk_mask = np.zeros((B, max_t, MAX_K), np.bool_)
    trk_last_geom = np.zeros((B, max_t, 4), np.float32)
    trk_valid = np.zeros((B, max_t), np.bool_)
    gap = np.zeros((B, max_t), np.float32)
    labels = np.full((B, max_n), max_t, np.int64)
    valid = np.zeros((B, max_n), np.bool_)
    motion_target = np.zeros((B, max_n, 4), np.float32)
    motion_mask = np.zeros((B, max_n), np.bool_)
    cur_box_px = np.zeros((B, max_n, 4), np.float32)

    for b, s in enumerate(samples):
        iw, ih = (image_scale or s["image_size"])
        diag = math.hypot(iw, ih) + 1e-6
        for n, c in enumerate(s["cands"]):
            cur_pbd[b, n] = c["pbd"]
            cur_pbd_be[b, n] = c["pbd_be"]
            cur_reg[b, n] = c["region"]
            cur_geom[b, n] = c["geom"]
            bx = c["box"]
            cur_box_px[b, n] = bx
            cur_norm_geom[b, n] = [bx[0] / iw, bx[1] / ih, bx[2] / iw, bx[3] / ih]
            cur_gen[b, n, 0] = c["gen"]
            labels[b, n] = s["labels"][n]
            valid[b, n] = bool(s["valid"][n])
        for ti, (gid, obs) in enumerate(s["tracks"]):
            obs = obs[-MAX_K:]
            start = MAX_K - len(obs)
            last = obs[-1]
            trk_last_geom[b, ti] = [last[3][0] / iw, last[3][1] / ih,
                                    last[3][2] / iw, last[3][3] / ih]
            trk_valid[b, ti] = True
            gap[b, ti] = max(1, s["frame"] - entries_frame_of(s, obs[-1][0]))
            for j, (fi, ci, f, box) in enumerate(obs):
                idx = start + j
                trk_mask[b, ti, idx] = True
                trk_times[b, ti, idx] = s["frame"] - int(entries_frame_of(s, fi))
                trk_pbd[b, ti, idx] = f["pbd"]
                trk_reg[b, ti, idx] = f["region"]
                trk_geom[b, ti, idx] = f["geom"]
                trk_gen[b, ti, idx] = f["gen"]
        # motion target for candidates matched to tracks
        for n, c in enumerate(s["cands"]):
            lab = labels[b, n]
            if lab < len(s["tracks"]):
                last_box = s["tracks"][lab][1][-1][3]
                motion_target[b, n] = (cur_box_px[b, n] - last_box) / diag
                motion_mask[b, n] = True

    def to_t(x):
        return torch.as_tensor(x, device=device)

    pbd_cos = np.zeros((B, max_t, max_n), np.float32)
    region_cos = np.zeros((B, max_t, max_n), np.float32)
    for b in range(B):
        for t in range(max_t):
            if not trk_valid[b, t]:
                continue
            ref_pbd = trk_pbd[b, t, trk_mask[b, t]][-1]
            ref_reg = trk_reg[b, t, trk_mask[b, t]][-1]
            for n in range(max_n):
                pbd_cos[b, t, n] = float(np.dot(norm_feat(ref_pbd), norm_feat(cur_pbd[b, n])))
                region_cos[b, t, n] = float(np.dot(norm_feat(ref_reg), norm_feat(cur_reg[b, n])))

    return {
        "cur_pbd": to_t(cur_pbd), "cur_pbd_be": to_t(cur_pbd_be),
        "cur_region": to_t(cur_reg), "cur_geom": to_t(cur_geom),
        "cur_norm_geom": to_t(cur_norm_geom), "cur_gen": to_t(cur_gen),
        "trk_pbd": to_t(trk_pbd), "trk_region": to_t(trk_reg),
        "trk_geom": to_t(trk_geom), "trk_gen": to_t(trk_gen),
        "trk_times": to_t(trk_times), "trk_mask": to_t(trk_mask),
        "trk_last_geom": to_t(trk_last_geom), "trk_valid": to_t(trk_valid),
        "gap": to_t(gap), "pbd_cos": to_t(pbd_cos), "region_cos": to_t(region_cos),
        "assign_targets": to_t(labels), "assign_valid": to_t(valid),
        "motion_target": to_t(motion_target), "motion_mask": to_t(motion_mask),
        "cur_mask": torch.ones(B, max_n, dtype=torch.bool, device=device),
    }


def entries_frame_of(s, fi):
    return s["frame"] - s["t"] + fi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs/l1_c/checkpoints/uaf"))
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--datasets", default="dancetrack,bdd100k,tao_amodal,mot17,mot20")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifests = []
    for ds in args.datasets.split(","):
        p = MANIFEST_DIR / f"{ds}_train.jsonl"
        if not p.exists():
            print(f"[skip] missing manifest {p}", flush=True)
            continue
        manifests.append((ds, load_manifest(p)))
        print(f"[data] {ds}: {sum(len(v) for v in manifests[-1][1].values())} frames", flush=True)
    if not manifests:
        raise SystemExit("no manifests")
    dataset = ClipDataset(manifests)
    model = UnifiedAssociationDecoder().to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        global_step = int(ck.get("step", 0))
        print(f"[resume] step={global_step}", flush=True)
    steps_total = args.smoke if args.smoke else args.steps
    run_steps = max(1, steps_total - global_step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (global_step + s + 1) / max(1, args.warmup))
        * 0.5 * (1 + math.cos(math.pi * min(global_step + s, steps_total)
                              / steps_total)))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    t0 = time.time()
    model.train()
    for step in range(run_steps):
        samples = []
        tries = 0
        while len(samples) < args.batch and tries < 64:
            name, vid, clip = dataset.sample_clip()
            samples.extend(build_clip(name, vid, clip))
            tries += 1
        if not samples:
            raise RuntimeError("no samples generated (manifest/cache issue)")
        batch = collate(samples[:args.batch], "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(batch)
            loss_main, cnt = association_loss(batch, pred)
            loss_motion, mcnt = motion_loss(batch, pred)
            loss = loss_main + 0.1 * loss_motion
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        global_step += 1
        if step % 20 == 0 or step == run_steps - 1:
            lr = opt.param_groups[0]["lr"]
            print(f"[uaf] step={global_step} loss={loss.item():.4f} "
                  f"main={loss_main.item():.4f} mot={loss_motion.item():.4f} "
                  f"cnt={cnt} lr={lr:.2e} "
                  f"elapsed={(time.time() - t0):.1f}s", flush=True)
        if args.save_every and global_step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "step": global_step,
                        "config": {"d_model": 256, "num_layers": 4, "num_heads": 8,
                                   "max_k": MAX_K}},
                       out / f"step{global_step}.pt")
    torch.save({"model": model.state_dict(), "step": global_step,
                "config": {"d_model": 256, "num_layers": 4, "num_heads": 8,
                           "max_k": MAX_K}},
               out / "final.pt")
    print(f"[uaf] done steps={global_step} seconds={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
