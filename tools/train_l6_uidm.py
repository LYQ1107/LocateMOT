"""Stage L6 UIDM: model-in-the-loop training on heterogeneous MOT domains.

Usage (single GPU pilot):
  python tools/train_l6_uidm.py \
      --data-dir outputs/l6/data \
      --domains bdd100k_train dancetrack_calibration mot17_train mot20_train \
      --out outputs/l6/checkpoints/uidm_base --model base --gpu 7 \
      --epochs 40 --batch 8 --val-domains dancetrack_val

Multi-GPU (after pilot): --gpu 7,8,9 with torchrun-style DDP via --ddp.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l6_uidm import (  # noqa: E402
    UIDM,
    decode_lsa,
    uidm_frame_loss,
    uidm_total_loss,
)

MAX_AGE = 30
H = 16
MAX_SLOTS = 64


def load_video(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class L6ClipDataset(Dataset):
    """Random H-frame windows from per-video sequence files."""

    def __init__(self, data_dir, domains, seed=20260806, max_videos=0):
        self.rng = random.Random(seed)
        self.pools = {}
        self.all = []
        for dom in domains:
            index_path = os.path.join(data_dir, dom, "index.json")
            if not os.path.exists(index_path):
                print(f"[l6data] skip missing domain {dom}", flush=True)
                continue
            with open(index_path) as f:
                index = json.load(f)
            vids = sorted(index["videos"].keys())
            if max_videos:
                vids = vids[:max_videos]
            self.pools[dom] = [{
                "path": index["videos"][v]["path"],
                "n": int(index["videos"][v]["frames"]),
            } for v in vids]
            for v in vids:
                self.all.append((dom, index["videos"][v]["path"],
                                 int(index["videos"][v]["frames"])))
        self.cache = OrderedDict()
        self.cache_max = 8

    def __len__(self):
        return max(1000, len(self.all) * 16)

    def _get_video(self, dom, path):
        key = (dom, path)
        if key not in self.cache:
            self.cache[key] = load_video(path)
            if len(self.cache) > self.cache_max:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, idx):
        r = random.Random((idx * 1000003 + os.getpid()) % (2 ** 31))
        dom = r.choice(list(self.pools.keys()))
        pool = self.pools[dom]
        v = r.choice(pool)
        rec = self._get_video(dom, v["path"])
        n = len(rec["frames"])
        if n <= H:
            start = 0
            frames = rec["frames"]
        else:
            start = r.randrange(0, n - H + 1)
            frames = rec["frames"][start:start + H]
        return {
            "video": rec["video_id"],
            "domain": dom,
            "image_size": tuple(rec["image_size"]),
            "frames": frames,
        }


def _box_norm(box, image_size):
    w, h = float(image_size[0]), float(image_size[1])
    b = np.asarray(box, np.float64)
    return np.asarray([b[0] / w, b[1] / h, b[2] / w, b[3] / h], np.float32)


class UIDMRollout:
    """Batched model-in-the-loop rollout over one batch of clips.

    Slot table is fixed-size (S slots).  All state updates are functional
    (no in-place writes into graph-carrying tensors), so BPTT flows from
    later frames back through the memory cells and encoders.  The
    assignment *selection* (teacher labels or Hungarian on detached logits)
    is non-differentiable, exactly as in the straight-through design.
    """

    def __init__(self, model, clips, device, teacher=True,
                 max_age=MAX_AGE, max_slots=MAX_SLOTS, raw=None,
                 stateless=False, no_lifecycle=False, app_key="pbd",
                 new_score_thr=0.0):
        self.model = model
        self.raw = raw if raw is not None else model
        self.stateless = stateless
        self.no_lifecycle = no_lifecycle
        self.new_score_thr = new_score_thr
        self.device = device
        self.B = len(clips)
        self.H = H
        self.teacher = teacher
        self.app_key = app_key
        self.unified = getattr(raw, "adapter", None) is not None
        self.max_age = max_age
        d = self.raw.d_model
        # identity maps per clip
        self.gid_maps = []
        self.observed = []  # per clip: dict gid_int -> np bool [H]
        self.gt_cand_at = []  # per clip: gid_int -> np int [H] (cand idx or -1)
        for c in clips:
            gid_map = {}
            obs = defaultdict(lambda: np.zeros(self.H, bool))
            cand_at = defaultdict(lambda: np.full(self.H, -1, np.int64))
            for f, fr in enumerate(c["frames"]):
                for gid in fr["gt_boxes"]:
                    if gid not in gid_map:
                        gid_map[gid] = len(gid_map)
                    obs[gid_map[gid]][f] = True
                for j, gid in enumerate(fr["cand_gt"]):
                    if gid is not None:
                        if gid not in gid_map:
                            gid_map[gid] = len(gid_map)
                        cand_at[gid_map[gid]][f] = j
            self.gid_maps.append(gid_map)
            self.observed.append({k: v for k, v in obs.items()})
            self.gt_cand_at.append({k: v for k, v in cand_at.items()})
        S = min(max_slots, max(8, max(len(m) + 8 for m in self.gid_maps)))
        self.S = S
        self.h = torch.zeros(self.B, S, d, device=device)
        self.anchor = torch.zeros_like(self.h)
        app_dim = getattr(raw, "app_dim", 2048)
        self.ref_pbd = torch.zeros(self.B, S, app_dim, device=device)
        self.anchor_pbd = torch.zeros_like(self.ref_pbd)
        self.alive_logit = torch.full((self.B, S), -5.0, device=device)
        self.active = torch.zeros(self.B, S, dtype=torch.bool, device=device)
        self.last_box = torch.zeros(self.B, S, 4, device=device)
        self.prev_box = torch.zeros_like(self.last_box)
        self.last_seen = torch.zeros(self.B, S, dtype=torch.long, device=device)
        self.age = torch.zeros(self.B, S, dtype=torch.long, device=device)
        self.hits = torch.zeros(self.B, S, dtype=torch.long, device=device)
        self.slot_gt = torch.full((self.B, S), -1, dtype=torch.long,
                                  device=device)
        self.loss_acc = {}
        self.n_frames = 0

    def run(self, clips):
        for f in range(self.H):
            self._step(clips, f)
        return dict(self.loss_acc), self.n_frames

    def _step(self, clips, f):
        if self.stateless:
            self.h = torch.zeros_like(self.h)
        model = self.model
        dev = self.device
        B = self.B
        d = self.raw.d_model
        # candidate tensors
        Nmax = max(len(c["frames"][f]["boxes"]) for c in clips)
        Nmax = max(1, Nmax)
        app_dim = getattr(self.raw, "app_dim", 2048)
        cand_pbd = torch.zeros(B, Nmax, app_dim, device=dev)
        cand_feat = torch.zeros(B, Nmax, len(
            __import__("locatemot.models.l1d_association", fromlist=["CAND_FEATURES"]).CAND_FEATURES),
            device=dev)
        cand_mask = torch.zeros(B, Nmax, dtype=torch.bool, device=dev)
        cand_boxes = torch.zeros(B, Nmax, 4, device=dev)
        cand_gen = torch.zeros(B, Nmax, device=dev)
        cand_gt_int = torch.full((B, Nmax), -1, dtype=torch.long, device=dev)
        clip_dim = getattr(getattr(self.raw, "adapter", None), "clip_proj", None)
        clip_dim = clip_dim.mlp[0].in_features if clip_dim is not None else 512
        cand_clip = torch.zeros(B, Nmax, clip_dim, device=dev)
        cand_rel_t = torch.zeros(B, Nmax, device=dev)
        cand_w = torch.ones(B, Nmax, device=dev)
        cand_nw_t = torch.ones(B, Nmax, device=dev)
        no_unmatched_new = [False] * B
        S = self.S
        trk_tok = self.h
        trk_mask = self.active
        pair_feats = torch.zeros(B, S, Nmax, 19, device=dev)
        track_feats = torch.zeros(B, S, 16, device=dev)
        gap = torch.zeros(B, S, device=dev)
        for b, c in enumerate(clips):
            fr = c["frames"][f]
            n = len(fr["boxes"])
            for j in range(n):
                cand_mask[b, j] = True
                cand_boxes[b, j] = torch.as_tensor(
                    fr["boxes"][j], dtype=torch.float32, device=dev)
                cand_pbd[b, j] = torch.as_tensor(
                    np.asarray(fr[self.app_key][j], np.float32), device=dev)
                cand_gen[b, j] = float(fr["gen"][j])
                if self.unified and "clip" in fr:
                    cand_clip[b, j] = torch.as_tensor(
                        np.asarray(fr["clip"][j], np.float32), device=dev)
                if "target" in fr and len(fr["target"]) > j:
                    cand_rel_t[b, j] = float(fr["target"][j])
                if "cand_w" in fr and len(fr["cand_w"]) > j:
                    cand_w[b, j] = float(fr["cand_w"][j])
                if "cand_nw" in fr and len(fr["cand_nw"]) > j:
                    cand_nw_t[b, j] = float(fr["cand_nw"][j])
                if fr.get("no_unmatched_new"):
                    no_unmatched_new[b] = True
                gid = fr["cand_gt"][j]
                if gid is not None and gid in self.gid_maps[b]:
                    cand_gt_int[b, j] = self.gid_maps[b][gid]
        if self.unified:
            spec_dim = getattr(self.raw.adapter, "spec_proj", None)
            spec_dim = spec_dim.mlp[0].in_features if spec_dim is not None \
                else 512
            spec = torch.zeros(B, spec_dim, device=dev)
            for b, c in enumerate(clips):
                if c.get("spec") is not None:
                    spec[b] = torch.as_tensor(c["spec"], dtype=torch.float32,
                                              device=dev)
            if self.raw.adapter.mode == "semantic":
                cand_pbd = torch.zeros_like(cand_pbd)
            cand_sem, _ = self.raw.adapter(cand_pbd, cand_clip, spec)
        # batched GPU evidence features (same schema as L1DK)
        from locatemot.models.l6_uidm import compute_affinity_features_torch
        cur_gap = torch.clamp(
            torch.as_tensor([c["frames"][f]["frame"] for c in clips],
                            device=dev)[:, None] - self.last_seen, min=1)
        gap = cur_gap.float()
        img_size = torch.as_tensor(
            [c["image_size"] for c in clips], dtype=torch.float32, device=dev)
        pair_feats, track_feats, cand_feat, _ = \
            compute_affinity_features_torch(
                self.last_box, self.prev_box, cand_boxes,
                self.ref_pbd, self.anchor_pbd, cand_pbd, cand_gen,
                gap, self.age.float(), self.hits.float(), img_size)
        frame = {
            "cand_pbd": cand_pbd,
            "cand_feat": cand_feat,
            "pair_feats": pair_feats, "track_feats": track_feats,
            "cand_mask": cand_mask, "trk_mask": trk_mask,
            "gap": gap, "trk_tok": trk_tok,
        }
        if self.unified:
            frame["cand_clip"] = cand_clip
            frame["spec"] = spec
            frame["cand_sem"] = cand_sem
        pred = model(frame)
        # ---- GT-based loss targets (never overwritten by student) ----
        row_target = torch.full((B, S), Nmax, dtype=torch.long, device=dev)
        col_target = torch.full((B, Nmax), S, dtype=torch.long, device=dev)
        row_valid = torch.zeros(B, S, dtype=torch.bool, device=dev)
        col_valid = torch.zeros(B, Nmax, dtype=torch.bool, device=dev)
        alive_target = torch.zeros(B, S, dtype=torch.bool, device=dev)
        no_match_target = torch.zeros(B, S, dtype=torch.bool, device=dev)
        new_target = torch.zeros(B, Nmax, dtype=torch.bool, device=dev)
        row_w = torch.ones(B, S, device=dev)
        col_w = cand_w.clone()
        new_w = cand_nw_t.clone()
        match_box = torch.zeros(B, S, 4, device=dev)
        row_box_valid = torch.zeros(B, S, dtype=torch.bool, device=dev)
        # assignment for state update (teacher labels or student decode)
        update_j = torch.full((B, S), -1, dtype=torch.long, device=dev)
        birth_mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
        birth_cand = torch.full((B, S), -1, dtype=torch.long, device=dev)
        if not self.teacher:
            pair_all = pred["pair_logits"].detach().cpu().numpy()
            nm_all = pred["no_match"].detach().cpu().numpy()
            nw_all = pred["new"].detach().cpu().numpy()
        for b, c in enumerate(clips):
            fr = c["frames"][f]
            active = self.active[b]
            si_list = active.nonzero().flatten().tolist()
            for si in si_list:
                g = int(self.slot_gt[b, si])
                if g < 0:
                    continue
                row_valid[b, si] = True
                # alive: observed within last MAX_AGE frames
                last = -1e9
                for ff in range(max(0, f - self.max_age), f + 1):
                    if self.observed[b].get(g, np.zeros(self.H, bool))[ff]:
                        last = ff
                alive_target[b, si] = last >= f - self.max_age
                cand_at = self.gt_cand_at[b].get(g)
                if cand_at is not None and cand_at[f] >= 0:
                    row_target[b, si] = int(cand_at[f])
                    j = int(cand_at[f])
                    row_w[b, si] = cand_w[b, j]
                    match_box[b, si] = torch.as_tensor(
                        _box_norm(fr["boxes"][j], c["image_size"]),
                        dtype=torch.float32, device=dev)
                    row_box_valid[b, si] = True
                    if self.teacher:
                        update_j[b, si] = j
                else:
                    no_match_target[b, si] = True
            free = (~self.active[b]).nonzero().flatten()
            free_ptr = 0
            for j in range(len(fr["boxes"])):
                col_valid[b, j] = True
                g = int(cand_gt_int[b, j])
                if g >= 0:
                    # find active slot with this gid
                    found = -1
                    for si in si_list:
                        if int(self.slot_gt[b, si]) == g:
                            found = si
                            break
                    if found >= 0:
                        col_target[b, j] = found
                    else:
                        new_target[b, j] = True
                        if self.teacher and free_ptr < len(free):
                            si = int(free[free_ptr])
                            free_ptr += 1
                            birth_mask[b, si] = True
                            birth_cand[b, si] = j
                            self.slot_gt[b, si] = g
                else:
                    if (not no_unmatched_new[b]
                            and float(fr["gen"][j]) >= self.new_score_thr):
                        new_target[b, j] = True
            # student births
            if not self.teacher:
                N = len(fr["boxes"])
                if N > 0:
                    pair = pair_all[b, :, :N]
                    nm = nm_all[b]
                    nw = nw_all[b, :N]
                    matches, births = decode_lsa(pair, nm, nw)
                    for t, j, _ in matches:
                        if self.active[b, t]:
                            update_j[b, t] = j
                            g = int(cand_gt_int[b, j])
                            self.slot_gt[b, t] = g if g >= 0 else -1
                    for j in births:
                        free = (~self.active[b]).nonzero().flatten()
                        if len(free):
                            si = int(free[0])
                            birth_mask[b, si] = True
                            birth_cand[b, si] = j
                            self.slot_gt[b, si] = int(cand_gt_int[b, j])
        tgt = {
            "row_target": row_target, "col_target": col_target,
            "row_valid": row_valid, "col_valid": col_valid,
            "alive_target": alive_target, "no_match_target": no_match_target,
            "new_target": new_target, "match_box": match_box,
            "row_box_valid": row_box_valid,
            "col_w": col_w, "row_w": row_w,
            "new_w": new_w,
        }
        losses = uidm_frame_loss(frame, pred, tgt)
        if "relevance" in pred:
            rel_mask = cand_mask & (cand_rel_t >= 0)
            if rel_mask.any():
                losses["loss_relevance"] = F.binary_cross_entropy_with_logits(
                    pred["relevance"][rel_mask], cand_rel_t[rel_mask])
            else:
                losses["loss_relevance"] = pred["relevance"].sum() * 0.0
        if self.no_lifecycle:
            for k in ("loss_nm", "loss_new", "loss_alive"):
                losses[k] = losses[k].detach() * 0.0
        for k, v in losses.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self.loss_acc[k] = self.loss_acc.get(k, v.detach() * 0) + v
        # diagnostic row accuracy (not part of the loss)
        if row_valid.any():
            row_logits = torch.cat(
                [pred["pair_logits"],
                 pred["no_match"].unsqueeze(-1)], dim=-1)
            acc = (row_logits[row_valid].argmax(-1)
                   == row_target[row_valid]).float().sum()
            self.loss_acc["acc_row"] = self.loss_acc.get(
                "acc_row", 0.0) + float(acc)
            self.loss_acc["n_row"] = self.loss_acc.get(
                "n_row", 0.0) + int(row_valid.sum())
        self.n_frames += 1

        # ---- state update (functional, BPTT-friendly) ----
        match_mask = update_j >= 0
        match_idx = match_mask.nonzero()
        ctx_slot = pred["trk_tok"]
        match_onehot = torch.zeros(B, S, Nmax, device=dev)
        if len(match_idx):
            bb = match_idx[:, 0]
            ss = match_idx[:, 1]
            jj = update_j[bb, ss].clamp(max=Nmax - 1)
            match_onehot[bb, ss, jj.clamp(max=Nmax - 1)] = 1.0
        obs_slot = torch.einsum("bsj,bjd->bsd", match_onehot, pred["cand_tok"])
        h_matched = self.raw.memory.update(self.h, obs_slot, ctx_slot)
        h_decayed = self.raw.memory.decay(self.h, gap)
        birth_onehot = torch.zeros(B, S, Nmax, device=dev)
        birth_idx = birth_mask.nonzero()
        if len(birth_idx):
            bb = birth_idx[:, 0]
            ss = birth_idx[:, 1]
            jj = birth_cand[bb, ss].clamp(max=Nmax - 1)
            birth_onehot[bb, ss, jj] = 1.0
        h_init = self.raw.memory.init(
            torch.einsum("bsj,bjd->bsd", birth_onehot, pred["cand_tok"]))
        upd = match_mask.float().unsqueeze(-1)
        dec = (self.active & ~match_mask).float().unsqueeze(-1)
        bir = birth_mask.float().unsqueeze(-1)
        h_next = self.h * (1 - (upd + dec + bir)) + \
            h_matched * upd + h_decayed * dec + h_init * bir

        with torch.no_grad():
            alive_next = pred["alive_pre"].detach()
            if self.no_lifecycle:
                alive_next = torch.full_like(alive_next, 2.0)
                alive_next = torch.where(birth_mask,
                                         torch.full_like(alive_next, 1.0),
                                         alive_next)
                alive_next = torch.where(
                    self.active & ~match_mask & ~birth_mask,
                    torch.full_like(alive_next, 1.0), alive_next)
            alive_next = torch.where(match_mask, alive_next + 2.0, alive_next)
            alive_next = torch.where(self.active & ~match_mask & ~birth_mask,
                                     alive_next - 1.0, alive_next)
            alive_next = torch.where(birth_mask,
                                     torch.full_like(alive_next, 1.0),
                                     alive_next)
            self.alive_logit = alive_next
            # termination (constant mask, also used in differentiable path)
            gap_ok = (frame["gap"] <= self.max_age) | ~self.active
            active_next = (self.alive_logit > 0.0) & gap_ok
            self.active = active_next
            # boxes / age / hits
            for b in range(self.B):
                fr = clips[b]["frames"][f]
                matched = match_mask[b]
                if matched.any():
                    jj_full = update_j[b].clamp(max=Nmax - 1)
                    cb_sel = cand_boxes[b][jj_full]
                    new_box = torch.where(matched.unsqueeze(-1),
                                          cb_sel, self.last_box[b])
                    self.prev_box[b] = torch.where(
                        matched.unsqueeze(-1), self.last_box[b],
                        self.prev_box[b])
                    self.last_box[b] = new_box
                    self.last_seen[b] = torch.where(matched,
                                                    torch.tensor(
                                                        fr["frame"],
                                                        device=dev),
                                                    self.last_seen[b])
                    self.age[b] = torch.where(matched, self.age[b] + 1,
                                              self.age[b])
                    self.hits[b] = torch.where(matched, self.hits[b] + 1,
                                               self.hits[b])
                    ref_sel = cand_pbd[b][jj_full]
                    self.ref_pbd[b] = torch.where(matched.unsqueeze(-1),
                                                  ref_sel, self.ref_pbd[b])
                if birth_mask[b].any():
                    jj_full = birth_cand[b].clamp(max=Nmax - 1)
                    cb_sel = cand_boxes[b][jj_full]
                    self.last_box[b] = torch.where(birth_mask[b].unsqueeze(-1),
                                                   cb_sel, self.last_box[b])
                    self.prev_box[b] = torch.where(
                        birth_mask[b].unsqueeze(-1), cb_sel, self.prev_box[b])
                    self.last_seen[b] = torch.where(
                        birth_mask[b], torch.tensor(fr["frame"], device=dev),
                        self.last_seen[b])
                    self.age[b] = torch.where(birth_mask[b],
                                              torch.ones_like(self.age[b]),
                                              self.age[b])
                    self.hits[b] = torch.where(birth_mask[b],
                                               torch.ones_like(self.hits[b]),
                                               self.hits[b])
                    ref_sel = cand_pbd[b][jj_full]
                    self.ref_pbd[b] = torch.where(birth_mask[b].unsqueeze(-1),
                                                  ref_sel, self.ref_pbd[b])
                    self.anchor_pbd[b] = self.ref_pbd[b].clone()
                    self.anchor[b] = torch.where(
                        birth_mask[b].unsqueeze(-1),
                        self.h[b].detach(), self.anchor[b])
        # zero inactive slots (differentiable path)
        self.h = h_next * active_next.float().unsqueeze(-1)
        self.ref_pbd = self.ref_pbd * self.active.float().unsqueeze(-1)
        self.anchor_pbd = self.anchor_pbd * \
            self.active.float().unsqueeze(-1)
        self.anchor = self.anchor * self.active.float().unsqueeze(-1)
        return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="outputs/l6/data")
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="base", choices=["small", "base", "large"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gpu", type=str, default="7")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--teacher-steps", type=int, default=1000)
    ap.add_argument("--teacher-final", type=float, default=0.4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--stateless", action="store_true")
    ap.add_argument("--no-interaction", action="store_true")
    ap.add_argument("--no-trackloss", action="store_true")
    ap.add_argument("--no-lifecycle", action="store_true")
    ap.add_argument("--no-cue-rel", action="store_true")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--w-rel", type=float, default=0.1)
    ap.add_argument("--app-key", default="pbd",
                    help="frame field holding the appearance token")
    ap.add_argument("--freeze-core", action="store_true",
                    help="train only the appearance projector (frozen UIDM core)")
    args = ap.parse_args()
    rank = 0
    if args.ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sizes = {
        "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
        "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
        "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
    }
    app_dim = 512 if args.app_key == "clip" else 2048
    model = UIDM(**sizes[args.model], no_interaction=args.no_interaction,
                 use_cue_rel=not args.no_cue_rel,
                 app_dim=app_dim).to(device)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model_sd = model.state_dict()
        ck_sd = ck["model"]
        filtered = {
            k: v for k, v in ck_sd.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if rank == 0:
            print(f"[l6] init-ckpt {args.init_ckpt} missing={len(missing)} "
                  f"unexpected={len(unexpected)}", flush=True)
    if args.freeze_core:
        for name, p in model.named_parameters():
            if not name.startswith("pbd_encoder."):
                p.requires_grad = False
    raw_model = model
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[l6] model={args.model} params={n_params/1e6:.2f}M device={device}",
          flush=True)
    ds = L6ClipDataset(args.data_dir, args.domains, seed=args.seed,
                       max_videos=args.max_videos)
    sampler = None
    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=True, seed=args.seed)
    print(f"[l6] rank={rank} "
          f"domains={list(ds.pools.keys())} "
          f"videos={sum(len(p) for p in ds.pools.values())}", flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=(sampler is None),
        sampler=sampler, num_workers=4,
        collate_fn=lambda x: x, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * max(1, len(loader))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05,
        anneal_strategy="cos")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args)
    cfg["n_params"] = n_params
    cfg["use_cue_rel"] = not args.no_cue_rel
    cfg["app_dim"] = raw_model.app_dim
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    curve = []
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        ep = defaultdict(float)
        ep_n = 0
        for bi, batch in enumerate(loader):
            teacher = step < args.teacher_steps or (
                random.random() < args.teacher_final)
            rollout = UIDMRollout(model, batch, device, teacher=teacher,
                                  raw=raw_model, stateless=args.stateless,
                                  no_lifecycle=args.no_lifecycle,
                                  app_key=args.app_key)
            losses, nf = rollout.run(batch)
            row_acc = losses.get("acc_row", 0.0) / max(
                1.0, losses.get("n_row", 0.0))
            if args.no_trackloss:
                loss = (losses["loss_row"] + losses["loss_col"]) if nf \
                    else torch.zeros((), device=device)
            else:
                loss = uidm_total_loss(losses, w_rel=args.w_rel) if nf \
                    else torch.zeros((), device=device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in losses.items():
                ep[k] += float(v.detach()) if isinstance(v, torch.Tensor) \
                    else float(v)
            ep_n += 1
            if step % 10 == 0:
                lr = sched.get_last_lr()[0]
                print(f"[l6] ep={epoch} step={step} loss={float(loss):.4f} "
                      f"row={losses.get('loss_row',0):.4f} "
                      f"col={losses.get('loss_col',0):.4f} "
                      f"sw={losses.get('loss_switch',0):.4f} "
                      f"rel={losses.get('loss_rel',0):.4f} "
                      f"rowacc={row_acc:.3f} "
                      f"lr={lr:.2e} elapsed={time.time()-t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        row = {"epoch": epoch, "step": step}
        for k, v in ep.items():
            row[k] = v / max(1, ep_n)
        curve.append(row)
        print("[l6] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                 else f"{k}={v}" for k, v in row.items()),
              flush=True)
        if rank == 0:
            torch.save({"model": raw_model.state_dict(), "epoch": epoch,
                        "cfg": cfg, "curve": curve},
                       out_dir / f"epoch{epoch}.pt")
            torch.save({"model": raw_model.state_dict(), "epoch": epoch,
                        "cfg": cfg, "curve": curve}, out_dir / "latest.pt")
            with open(out_dir / "learning_curve.json", "w") as f:
                json.dump(curve, f, indent=2)
    if args.ddp:
        torch.distributed.destroy_process_group()
    print(f"[l6] done seconds={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
