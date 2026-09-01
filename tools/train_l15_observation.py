"""Train the Stage L15 query-conditioned observation frontend.

Only train-split RMOT expression labels are read here.  Official KITTI-V2
test sequences (0005/0011/0013/0019) are excluded explicitly.  The resulting
head is used only to rank an existing proposal pool; the L11 UIDM checkpoint
is never loaded or modified by this script.

Example:
  /home/lwr/anaconda3/envs/locatemot/bin/python tools/train_l15_observation.py \
    --out outputs/l15/checkpoints/observation_head.pt --gpu 0 \
    --max-steps 1200
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l15_observation import L15ObservationHead  # noqa: E402


KITTI_ROOT = ROOT / "outputs" / "l11" / "data" / "rmot_kitti"
DANCE_ROOT = ROOT / "outputs" / "l8" / "data" / "rmot_train"
KITTI_TEST = {"0005", "0011", "0013", "0019"}


def _as_float(x, shape):
    a = np.asarray(x, np.float32)
    if a.shape != shape:
        return np.zeros(shape, np.float32)
    return np.nan_to_num(a, copy=False)


def _geometry(boxes, image_size):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    w, h = [max(1.0, float(x)) for x in image_size]
    if len(boxes):
        x1, y1, x2, y2 = boxes.T
    else:
        x1 = y1 = x2 = y2 = np.zeros(0, np.float32)
    bw = np.maximum(0.0, x2 - x1)
    bh = np.maximum(0.0, y2 - y1)
    cx = (x1 + x2) * 0.5 / w
    cy = (y1 + y2) * 0.5 / h
    nw, nh = bw / w, bh / h
    area = nw * nh
    aspect = np.clip(bw / np.maximum(bh, 1.0), 0.0, 20.0) / 20.0
    bottom = y2 / h
    return np.stack((cx, cy, nw, nh, area, aspect, bottom), axis=1).astype(
        np.float32)


class ExpressionPool:
    """Random-access expression/frame sampler with bounded video cache."""

    def __init__(self, root, include_videos=None):
        self.root = Path(root)
        self.meta = json.loads((self.root / "expressions.json").read_text())
        allow = None if include_videos is None else set(include_videos)
        self.items = []
        for video, expressions in sorted(self.meta.items()):
            if allow is not None and video not in allow:
                continue
            pkl = self.root / f"{video}.pkl"
            if not pkl.exists():
                continue
            for ei, entry in enumerate(expressions):
                if entry.get("label", {}):
                    self.items.append((video, ei, pkl))
        self.cache = OrderedDict()
        self.cache_max = 4
        if not self.items:
            raise RuntimeError(f"no training expressions in {self.root}")

    def __len__(self):
        return len(self.items)

    def _get(self, path):
        key = str(path)
        if key not in self.cache:
            with path.open("rb") as f:
                self.cache[key] = pickle.load(f)
            if len(self.cache) > self.cache_max:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def sample(self, rng, max_pos=4, max_neg=12):
        video, ei, path = self.items[rng.randrange(len(self.items))]
        rec = self._get(path)
        entry = self.meta[video][ei]
        labels = entry.get("label", {})
        label_frames = [int(k) for k, v in labels.items() if v]
        if not label_frames:
            return None
        if rng.random() < 0.80:
            frame_no = rng.choice(label_frames)
            frame = next((x for x in rec["frames"]
                          if int(x["frame"]) == frame_no), None)
        else:
            frame = rng.choice(rec["frames"])
        if frame is None:
            return None
        boxes = np.asarray(frame.get("boxes", []), np.float32).reshape(-1, 4)
        n = len(boxes)
        if n == 0:
            return None
        cand_gt = [None] * n
        raw_gt = frame.get("cand_gt", [])
        for i in range(min(n, len(raw_gt))):
            value = raw_gt[i]
            cand_gt[i] = None if value is None else str(value)
        current_ids = labels.get(str(int(frame["frame"])),
                                 labels.get(int(frame["frame"]), []))
        target_ids = {str(x) for x in current_ids}
        y = np.asarray([1.0 if x is not None and x in target_ids else 0.0
                        for x in cand_gt], np.float32)
        pos = np.flatnonzero(y > 0.5).tolist()
        neg = np.flatnonzero(y <= 0.5).tolist()
        if not pos and rng.random() > 0.25:
            return None
        rng.shuffle(pos)
        rng.shuffle(neg)
        keep = pos[:max_pos] + neg[:max_neg]
        if not keep:
            return None
        clip = _as_float(frame.get("clip"), (n, 512))[keep]
        spec = _as_float(entry.get("spec"), (512,))
        geom = _geometry(boxes, rec["image_size"])[keep]
        gen = _as_float(frame.get("gen"), (n,))[keep]
        return clip, spec, geom, gen, y[keep]


def sample_batch(pools, rng, batch_size):
    rows = []
    attempts = 0
    while len(rows) < batch_size and attempts < batch_size * 20:
        attempts += 1
        pool = pools[0] if rng.random() < 0.65 else pools[1]
        item = pool.sample(rng)
        if item is not None:
            rows.append(item)
    if not rows:
        raise RuntimeError("sampler produced no valid expression/frame pairs")
    clip = np.concatenate([x[0] for x in rows], axis=0)
    spec = np.concatenate([
        np.repeat(x[1][None, :], len(x[0]), axis=0) for x in rows], axis=0)
    geom = np.concatenate([x[2] for x in rows], axis=0)
    gen = np.concatenate([x[3] for x in rows], axis=0)
    target = np.concatenate([x[4] for x in rows], axis=0)
    return clip, spec, geom, gen, target


def ranking_loss(logits, target):
    pos = logits[target > 0.5]
    neg = logits[target <= 0.5]
    if len(pos) and len(neg):
        pairs = (0.25 - pos[:, None] + neg[None, :]).clamp_min(0.0)
        pair = pairs.mean()
    else:
        pair = logits.new_zeros(())
    weights = torch.where(target > 0.5,
                          torch.full_like(target, 2.5),
                          torch.ones_like(target))
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights)
    return bce + 0.5 * pair, bce.detach(), pair.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--save-every", type=int, default=300)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_kitti = json.loads((KITTI_ROOT / "expressions.json").read_text())
    train_videos = sorted(set(all_kitti) - KITTI_TEST)
    kitti = ExpressionPool(KITTI_ROOT, train_videos)
    dance = ExpressionPool(DANCE_ROOT)
    model = L15ObservationHead(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for step in range(1, args.max_steps + 1):
        clip, spec, geom, gen, target = sample_batch(
            (kitti, dance), rng, args.batch)
        clip, spec, geom, gen, target = [torch.from_numpy(x).to(device)
                                         for x in (clip, spec, geom, gen,
                                                   target)]
        logits = model(clip, spec, geom, gen)
        loss, bce, pair = ranking_loss(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        value = float(loss.detach().cpu())
        history.append({"step": step, "loss": value,
                        "bce": float(bce.cpu()), "pair": float(pair.cpu()),
                        "positive_rate": float(target.mean().cpu())})
        if step == 1 or step % 100 == 0:
            print(f"[l15obs] step={step}/{args.max_steps} loss={value:.4f} "
                  f"bce={float(bce):.4f} pair={float(pair):.4f} "
                  f"pos={float(target.mean()):.3f}", flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            if value <= best or step == args.max_steps:
                best = min(best, value)
                torch.save({"model": model.state_dict(),
                            "cfg": {"hidden": args.hidden,
                                    "feature_dim": model.feature_dim,
                                    "seed": args.seed},
                            "train_videos": train_videos,
                            "history_tail": history[-100:]}, out)
    log_path = out.with_suffix(out.suffix + ".json")
    log_path.write_text(json.dumps({"args": vars(args),
                                    "train_videos": train_videos,
                                    "kitti_expressions": len(kitti),
                                    "dance_expressions": len(dance),
                                    "history": history}, indent=2))
    print(f"[l15obs] saved={out} log={log_path} device={device}", flush=True)


if __name__ == "__main__":
    main()

