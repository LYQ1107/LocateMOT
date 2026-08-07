#!/usr/bin/env python
"""Stage L1-A: train T3-T6 temporal modules on DanceTrack train (D-LA cache).

Only TrajectoryEncoder / MotionPredictor / MemoryFusion / residual heads and a
scalar no-match bias are trained; L0-D B6 stays frozen. Official val is never
read here. Each video is processed in one forward pass to build samples with
real histories; sample tensors are kept per video (~1-2GB RAM), then training
iterates videos sequentially with within-video shuffling.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402
from locatemot.models.track_decoder.features import category_hash_embedding  # noqa: E402
from locatemot.models.track_decoder.relation_track_decoder import RelationTrackDecoderModel  # noqa: E402
from locatemot.models.trajectory.temporal_bundle import TemporalBundle, quantize_gap  # noqa: E402

DANCETRACK = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"
DLA_CACHE = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla"
K = 8
MAX_REF = 8
MAX_CUR = 48


def read_gt(vid):
    per_frame = {}
    for line in open(os.path.join(DANCETRACK, "train", vid, "gt", "gt.txt")):
        p = line.strip().split(",")
        if len(p) < 9 or int(p[7]) != 1:
            continue
        fid, oid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        if w <= 0 or h <= 0:
            continue
        per_frame.setdefault(fid, {})[oid] = (x, y, x + w, y + h)
    return per_frame


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def normalize_geom(box, image_size):
    w, h = float(image_size[0]), float(image_size[1])
    x1, y1, x2, y2 = box
    area = max(0.0, (x2 - x1) * (y2 - y1)) / (w * h)
    return np.asarray([x1 / w, y1 / h, x2 / w, y2 / h, area], dtype=np.float32)


def _candidate_feats(f, i, meta):
    return {
        "pbd": np.asarray(f["pbd_coord_mean_last"][i], dtype=np.float32),
        "pbd_be": np.asarray(f["pbd_box_end_last"][i], dtype=np.float32),
        "region": np.asarray(f["region"][i], dtype=np.float32),
        "geom": normalize_geom(f["boxes"][i], meta["image_size"]),
        "gen": float(f["gen_score"][i]) if "gen_score" in f else 0.0,
        "box": np.asarray(f["boxes"][i], dtype=np.float32),
    }


class SampleBuilder:
    def __init__(self, seed=20260806, max_ref=MAX_REF, max_cur=MAX_CUR, per_video_cap=250):
        self.rng = random.Random(seed)
        self.max_ref = max_ref
        self.max_cur = max_cur
        self.per_video_cap = per_video_cap

    def build_video_samples(self, vid):
        gt = read_gt(vid)
        frames = sorted(gt)
        hist = {}
        samples = []
        for t in frames:
            frame_cache = read_frame_cache(DLA_CACHE, cache_key("dancetrack", vid, t, "person"))
            if frame_cache is None:
                continue
            f = frame_cache["features"]
            meta = frame_cache["meta"]
            cands = [_candidate_feats(f, i, meta) for i in range(len(f["boxes"]))]
            matched = {}
            for oid, gtb in gt.get(t, {}).items():
                best_idx, best_iou = None, 0.0
                for i, c in enumerate(cands):
                    iou = _iou(c["box"], gtb)
                    if iou > best_iou:
                        best_idx, best_iou = i, iou
                if best_idx is not None and best_iou >= 0.5:
                    matched[oid] = best_idx
            track_oids = [oid for oid, h in hist.items() if h]
            if len(track_oids) >= 2 and cands:
                # prefer tracks whose last box moved far / reappeared
                def hard_score(oid):
                    gtb = gt.get(t, {}).get(oid)
                    last_box = hist[oid][-1][2]
                    if gtb is not None:
                        return 1.0 - _iou(last_box, gtb)
                    return 0.8
                track_oids.sort(key=hard_score, reverse=True)
                selected = track_oids[: self.max_ref]
                score = float(np.mean([hard_score(o) for o in selected]))
                sample = self._tensors(selected, hist, cands, matched, gt.get(t, {}), t)
                if sample is not None:
                    sample["score"] = score
                    samples.append((t, sample))
            for oid, cidx in matched.items():
                if oid not in hist:
                    hist[oid] = deque(maxlen=K)
                hist[oid].append((t, cands[cidx], cands[cidx]["box"]))
            for oid in list(hist):
                hist[oid] = deque([h for h in hist[oid] if t - h[0] <= 128], maxlen=K)
                if not hist[oid]:
                    del hist[oid]
        samples.sort(key=lambda x: x[1]["score"], reverse=True)
        return samples[: self.per_video_cap]

    def _tensors(self, oids, hist, cands, matched, gt_frame, t):
        M = len(oids)
        N = min(len(cands), self.max_cur)
        cands = cands[:N]
        if M == 0 or N == 0:
            return None
        win = {
            "pbd": np.zeros((M, K, 2048), dtype=np.float32),
            "pbd_be": np.zeros((M, K, 2048), dtype=np.float32),
            "region": np.zeros((M, K, 4608), dtype=np.float32),
            "geom": np.zeros((M, K, 5), dtype=np.float32),
            "gen": np.zeros((M, K), dtype=np.float32),
            "gaps": np.zeros((M, K), dtype=np.float32),
            "mask": np.ones((M, K), dtype=bool),
            "boxes": np.zeros((M, K, 4), dtype=np.float32),
        }
        anchors = {"pbd": np.zeros((M, 2048), dtype=np.float32),
                   "region": np.zeros((M, 4608), dtype=np.float32)}
        emas = {"pbd": np.zeros((M, 2048), dtype=np.float32),
                "region": np.zeros((M, 4608), dtype=np.float32)}
        geom_last = np.zeros((M, 5), dtype=np.float32)
        gen_last = np.zeros((M,), dtype=np.float32)
        conf = np.zeros((M,), dtype=np.float32)
        ref_boxes = np.zeros((M, 4), dtype=np.float32)
        gaps = np.zeros((M,), dtype=np.float32)
        motion_boxes = np.zeros((M, 4, 4), dtype=np.float32)
        motion_gaps = np.zeros((M, 4), dtype=np.float32)
        labels = np.zeros((M, N), dtype=np.float32)
        nm_targets = np.ones((M,), dtype=np.float32)
        gt_motion = np.zeros((M, 4), dtype=np.float32)
        has_motion = np.zeros((M,), dtype=bool)
        for m, oid in enumerate(oids):
            h = list(hist[oid])
            start = K - len(h)
            for j, (fr, feats, box) in enumerate(h):
                idx = start + j
                win["mask"][m, idx] = False
                win["gaps"][m, idx] = max(1, t - fr)
                win["pbd"][m, idx] = feats["pbd"]
                win["pbd_be"][m, idx] = feats["pbd_be"]
                win["region"][m, idx] = feats["region"]
                win["geom"][m, idx] = feats["geom"]
                win["gen"][m, idx] = feats["gen"]
                win["boxes"][m, idx] = feats["box"]
            last_feats = h[-1][1]
            anchors["pbd"][m] = h[0][1]["pbd"]
            anchors["region"][m] = h[0][1]["region"]
            ema_pbd = np.zeros(2048, dtype=np.float32)
            ema_region = np.zeros(4608, dtype=np.float32)
            alpha = 0.5
            for _, ff, _ in h:
                ema_pbd = (1 - alpha) * ema_pbd + alpha * ff["pbd"]
                ema_region = (1 - alpha) * ema_region + alpha * ff["region"]
            emas["pbd"][m] = ema_pbd
            emas["region"][m] = ema_region
            geom_last[m] = last_feats["geom"]
            gen_last[m] = last_feats["gen"]
            conf[m] = last_feats["gen"]
            ref_boxes[m] = last_feats["box"]
            gaps[m] = max(1, t - h[-1][0])
            obs = h[-4:]
            for j, (fr, ff, box) in enumerate(obs):
                motion_boxes[m, j] = box
                motion_gaps[m, j] = max(1, t - fr)
            # pad leading motion slots with the first observation (velocity 0)
            if len(obs) < 4:
                pad_box = obs[0][2]
                for j in range(4 - len(obs)):
                    motion_boxes[m, j] = pad_box
                    motion_gaps[m, j] = 1.0
            gtb = gt_frame.get(oid)
            if gtb is not None:
                gt_motion[m] = gtb
                has_motion[m] = True
                if oid in matched:
                    ci = matched[oid]
                    if ci < N:
                        labels[m, ci] = 1.0
                        nm_targets[m] = 0.0
                    else:
                        nm_targets[m] = 1.0
            else:
                nm_targets[m] = 1.0
        cur = {
            "pbd": np.stack([c["pbd"] for c in cands]).astype(np.float32),
            "pbd_be": np.stack([c["pbd_be"] for c in cands]).astype(np.float32),
            "region": np.stack([c["region"] for c in cands]).astype(np.float32),
            "geom": np.stack([c["geom"] for c in cands]).astype(np.float32),
            "gen": np.asarray([c["gen"] for c in cands], dtype=np.float32),
            "boxes": np.stack([c["box"] for c in cands]).astype(np.float32),
        }
        return {
            "M": M, "N": N, "win": win, "anchors": anchors, "emas": emas,
            "geom_last": geom_last, "gen_last": gen_last, "conf": conf,
            "ref_boxes": ref_boxes, "gaps": gaps, "motion_boxes": motion_boxes,
            "motion_gaps": motion_gaps, "labels": labels, "nm_targets": nm_targets,
            "gt_motion": gt_motion, "has_motion": has_motion, "cur": cur,
        }


def build_b6_batch(refs, curs, ref_boxes, cur_boxes, gap, device):
    cat = category_hash_embedding("person", 32)
    M = len(refs)
    N = len(curs)

    def st(feats, key, dim):
        parts = []
        for f in feats:
            v = f.get(key)
            if v is None:
                v = torch.zeros(dim, dtype=torch.float32, device=device)
            elif not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=torch.float32, device=device)
            else:
                v = v.to(dtype=torch.float32, device=device)
            parts.append(v)
        return torch.stack(parts)

    def gen(feats):
        vals = [f.get("gen", 0.0) for f in feats]
        return torch.as_tensor([vals], dtype=torch.float32, device=device)

    return {
        "ref_pbd": st(refs, "pbd", 2048).unsqueeze(0),
        "ref_pbd_be": st(refs, "pbd_be", 2048).unsqueeze(0),
        "ref_region": st(refs, "region", 4608).unsqueeze(0),
        "ref_geom": st(refs, "geom", 5).unsqueeze(0),
        "ref_gen": gen(refs),
        "ref_cat": cat.unsqueeze(0).unsqueeze(0).expand(1, M, 32).clone(),
        "ref_mask": torch.ones(1, M, dtype=torch.bool, device=device),
        "ref_boxes": torch.as_tensor(np.asarray(ref_boxes, dtype=np.float32).reshape(1, M, 4), device=device),
        "cur_pbd": st(curs, "pbd", 2048).unsqueeze(0),
        "cur_pbd_be": st(curs, "pbd_be", 2048).unsqueeze(0),
        "cur_region": st(curs, "region", 4608).unsqueeze(0),
        "cur_geom": st(curs, "geom", 5).unsqueeze(0),
        "cur_gen": gen(curs),
        "cur_cat": cat.unsqueeze(0).unsqueeze(0).expand(1, N, 32).clone(),
        "cur_mask": torch.ones(1, N, dtype=torch.bool, device=device),
        "cur_boxes": torch.as_tensor(np.asarray(cur_boxes, dtype=np.float32).reshape(1, N, 4), device=device),
        "gap": torch.as_tensor([[float(gap)]], dtype=torch.float32, device=device),
    }


def iou_matrix(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    ix1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    ar = torch.clamp(a[:, 2] - a[:, 0], min=0) * torch.clamp(a[:, 3] - a[:, 1], min=0)
    ac = torch.clamp(b[:, 2] - b[:, 0], min=0) * torch.clamp(b[:, 3] - b[:, 1], min=0)
    return inter / (ar[:, None] + ac[None, :] - inter + 1e-6)


def center_dist(a, b):
    ac = (a[:, :2] + a[:, 2:]) / 2
    bc = (b[:, :2] + b[:, 2:]) / 2
    return torch.sqrt(((ac[:, None] - bc[None]) ** 2).sum(-1) + 1e-6)


def forward_sample(bundle, b6, sample, device, image_size=(1280, 720)):
    M, N = sample["M"], sample["N"]
    if M == 0 or N == 0:
        return None
    win = {k: torch.as_tensor(sample["win"][k], device=device) for k in sample["win"]}
    traj = bundle.trajectory_refs(win)
    anchors = {k: torch.as_tensor(sample["anchors"][k], device=device) for k in ("pbd", "region")}
    emas = {k: torch.as_tensor(sample["emas"][k], device=device) for k in ("pbd", "region")}
    geom_last = torch.as_tensor(sample["geom_last"], device=device)
    gen_last = torch.as_tensor(sample["gen_last"], device=device)
    conf = torch.as_tensor(sample["conf"], device=device)
    mem = bundle.memory_refs(anchors, emas, traj, geom_last, gen_last, conf)
    refs = []
    for m in range(M):
        last_valid = (~win["mask"][m]).nonzero()[-1].item()
        refs.append({
            "pbd": mem["pbd"][m],
            "pbd_be": win["pbd_be"][m, last_valid],
            "region": mem["region"][m],
            "geom": mem["geom"][m],
            "gen": float(mem["gen"][m]),
        })
    cur = {k: torch.as_tensor(sample["cur"][k], device=device) for k in sample["cur"]}
    cur_dicts = [{"pbd": cur["pbd"][j], "pbd_be": cur["pbd_be"][j],
                  "region": cur["region"][j], "geom": cur["geom"][j],
                  "gen": float(cur["gen"][j])} for j in range(N)]
    ref_boxes = sample["ref_boxes"]
    cur_boxes = sample["cur"]["boxes"]
    qgaps = quantize_gap([int(g) for g in sample["gaps"]])
    labels = torch.as_tensor(sample["labels"], device=device)
    nm_targets = torch.as_tensor(sample["nm_targets"], device=device)
    motion_boxes = torch.as_tensor(sample["motion_boxes"], device=device)
    motion_gaps = torch.as_tensor(sample["motion_gaps"], device=device)
    pred_delta = bundle.motion_predictor(motion_boxes, motion_gaps)
    total = torch.zeros((), device=device)
    n_groups = 0
    cur_boxes_t = torch.as_tensor(cur_boxes, device=device)
    for g in sorted(set(qgaps)):
        idxs = [i for i in range(M) if qgaps[i] == g]
        refs_g = [refs[i] for i in idxs]
        batch = build_b6_batch(refs_g, cur_dicts, [ref_boxes[i] for i in idxs],
                               cur_boxes, g, device)
        pred = b6(batch)
        match = pred["match_logits"][0]
        pred_boxes_g = bundle.motion_predictor.predict_box(
            motion_boxes[idxs], motion_gaps[idxs])
        ref_boxes_t = torch.as_tensor(ref_boxes[idxs], device=device)
        iou_last = iou_matrix(ref_boxes_t, cur_boxes_t)
        iou_pred = iou_matrix(pred_boxes_g, cur_boxes_t)
        cd_pred = center_dist(pred_boxes_g, cur_boxes_t)
        cd_last = center_dist(ref_boxes_t, cur_boxes_t)
        diag = float(np.hypot(*image_size))
        gap_arr = torch.as_tensor([sample["gaps"][i] for i in idxs], device=device).float()
        feats = torch.stack([
            match, iou_last, iou_pred, cd_pred / diag,
            (cd_pred - cd_last).abs() / diag,
            torch.log1p(gap_arr)[:, None].expand(-1, N),
        ], dim=-1).reshape(-1, 6)
        match = match + bundle.motion_residual(feats).reshape(len(idxs), N)
        lost = torch.tensor([bool(sample["gaps"][i] >= 2) for i in idxs],
                            dtype=torch.bool, device=device)
        if lost.any():
            ref_t = F.normalize(pred["ref_feats"][0], dim=-1)
            cur_t = F.normalize(pred["cur_feats"][0], dim=-1)
            traj_cos = ref_t @ cur_t.T
            ref_be = torch.stack([refs[i]["pbd_be"] for i in idxs])
            pbd_cos = F.normalize(ref_be, dim=-1) @ F.normalize(cur["pbd_be"], dim=-1).T
            feats_r = torch.stack([
                match, traj_cos, pbd_cos, iou_pred, cd_pred / diag,
                torch.log1p(gap_arr)[:, None].expand(-1, N),
            ], dim=-1).reshape(-1, 6)
            resid = bundle.reactivation_residual(feats_r).reshape(len(idxs), N)
            match = match + resid * lost.float()[:, None]
        nm = pred["no_match_logits"][0] - bundle.nm_bias
        total = total + F.binary_cross_entropy_with_logits(match, labels[idxs])
        total = total + F.binary_cross_entropy_with_logits(nm, nm_targets[idxs])
        n_groups += 1
    total = total / max(1, n_groups)
    hm = sample["has_motion"]
    if hm.any():
        gt_b = torch.as_tensor(sample["gt_motion"][hm], device=device)
        last_b = torch.as_tensor(sample["motion_boxes"][hm, -1], device=device)
        total = total + 0.1 * bundle.motion_predictor.motion_loss(
            pred_delta[hm], gt_b, last_b)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="outputs/l1_a/checkpoints/temporal")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    split = json.load(open(os.path.join(ROOT, "configs/data/l1_a_dancetrack_train.json")))
    videos = [v["video_id"] for v in split["videos"]]

    b6 = RelationTrackDecoderModel(use_pbd_base=True, use_region_geom=True, residual=True)
    ck = torch.load(os.path.join(ROOT, "outputs/l0_d/checkpoints/b6/best.pt"),
                    map_location="cpu", weights_only=False)
    b6.load_state_dict(ck["model"])
    b6.to(device).eval()
    for p in b6.parameters():
        p.requires_grad_(False)

    bundle = TemporalBundle()
    bundle.to(device)
    opt = torch.optim.AdamW(bundle.parameters(), lr=args.lr, weight_decay=1e-4)
    builder = SampleBuilder()
    rng = random.Random(20260806)
    step = 0
    t0 = time.time()
    all_losses = []
    for epoch in range(args.epochs):
        vid_order = list(videos)
        rng.shuffle(vid_order)
        for vid in vid_order:
            samples = builder.build_video_samples(vid)
            order = list(range(len(samples)))
            rng.shuffle(order)
            for si in order:
                t, sample = samples[si]
                opt.zero_grad()
                loss = forward_sample(bundle, b6, sample, device)
                if loss is None:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bundle.parameters(), 1.0)
                opt.step()
                all_losses.append(float(loss))
                step += 1
                if step % args.log_every == 0:
                    print(f"[train] epoch={epoch} step={step} "
                          f"loss={np.mean(all_losses[-args.log_every:]):.4f} "
                          f"time={time.time()-t0:.0f}s", flush=True)
            print(f"[train] video {vid} samples={len(samples)} step={step} "
                  f"loss={np.mean(all_losses[-200:]):.4f}", flush=True)
        os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
        torch.save({"model": bundle.state_dict(), "step": step, "epoch": epoch,
                    "samples": len(videos), "seed": 20260806},
                   os.path.join(ROOT, args.out, f"epoch{epoch}.pt"))
    torch.save({"model": bundle.state_dict(), "step": step, "epoch": args.epochs,
                "samples": len(videos), "seed": 20260806},
               os.path.join(ROOT, args.out, "best.pt"))
    print(f"[train] done, steps={step}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
