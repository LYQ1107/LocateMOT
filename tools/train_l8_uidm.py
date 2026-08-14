"""Stage L8: train the Unified Observation Adapter + shared UIDM.

Data: task-balanced mixture of
  - RMOT (Refer-Dance train, expression spec)
  - ordinary closed-set MOT (L7 CLIP caches, category spec)

Usage (single GPU smoke):
  python tools/train_l8_uidm.py --out outputs/l8/checkpoints/smoke \
      --mode unified --epochs 1 --max-steps 20 --gpu 0

Full 4-GPU:
  torchrun --nproc_per_node=4 tools/train_l8_uidm.py \
      --out outputs/l8/checkpoints/uidm_l8_unified --mode unified \
      --epochs 40 --batch 4 --gpu 0,1,2,3 --ddp --freeze-core
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
from torch.utils.data import Dataset

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from tools.train_l6_uidm import H, MAX_AGE, MAX_SLOTS, UIDMRollout  # noqa: E402
from locatemot.models.l6_uidm import uidm_total_loss  # noqa: E402
from locatemot.models.l8_unified import L8UnifiedUIDM, clip_text_embed  # noqa: E402

ORDINARY_DOMAINS = [
    ("bdd100k_train", "outputs/l7/data/clip_closed/bdd100k_train",
     "person, car, truck, bus, rider, bicycle, motorcycle, train"),
    ("dancetrack_calibration", "outputs/l7/data/clip_closed/dancetrack_calibration",
     "person"),
    ("dancetrack_train", "outputs/l7/data/clip_closed/dancetrack_train",
     "person"),
    ("mot17_train", "outputs/l7/data/clip_closed/mot17_train", "person"),
    ("mot20_train", "outputs/l7/data/clip_closed/mot20_train", "person"),
]

RMOT_PKL = ROOT / "outputs" / "l8" / "data" / "rmot_train"
SPEC_CACHE = ROOT / "outputs" / "l8" / "data" / "specs.json"


def _specs(texts, device="cuda"):
    cache = {}
    if SPEC_CACHE.exists():
        cache = json.loads(SPEC_CACHE.read_text())
    todo = [t for t in texts if t not in cache]
    if todo:
        embs = clip_text_embed(todo, device=device)
        for t, e in zip(todo, embs.cpu().numpy()):
            cache[t] = e.astype(np.float32).tolist()
        SPEC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SPEC_CACHE.write_text(json.dumps(cache))
    return np.asarray([cache[t] for t in texts], np.float32)


class L8Dataset(Dataset):
    """Task-balanced mixture of ordinary MOT clips and RMOT clips."""

    def __init__(self, seed=20260806, p_rmot=0.5, rmot_only=False,
                 max_rmot_expr=0, pbd_dropout=0.15):
        self.rng = random.Random(seed)
        self.p_rmot = p_rmot
        self.rmot_only = rmot_only
        self.pbd_dropout = pbd_dropout
        self.ordinary = []
        self.ordinary_by_domain = {}
        if not rmot_only:
            for name, data_dir, spec_text in ORDINARY_DOMAINS:
                idx = Path(data_dir) / "index.json"
                if not idx.exists():
                    print(f"[l8data] skip ordinary domain {name}", flush=True)
                    continue
                index = json.loads(idx.read_text())
                vids = sorted(index["videos"].keys())
                group = []
                for v in vids:
                    item = {
                        "domain": name, "spec": spec_text,
                        "path": index["videos"][v]["path"],
                        "n": int(index["videos"][v]["frames"]),
                    }
                    self.ordinary.append(item)
                    group.append(item)
                if group:
                    self.ordinary_by_domain[name] = group
        # RMOT pool
        self.rmot = []
        self.rmot_meta = {}
        exp_path = RMOT_PKL / "expressions.json"
        if exp_path.exists():
            self.rmot_meta = json.loads(exp_path.read_text())
        for vid, exps in self.rmot_meta.items():
            pkl = RMOT_PKL / f"{vid}.pkl"
            if not pkl.exists():
                continue
            if max_rmot_expr:
                exps = exps[:max_rmot_expr]
            for ei, e in enumerate(exps):
                self.rmot.append({"video": vid, "path": str(pkl),
                                  "expr_idx": ei})
        self.cache = OrderedDict()
        self.cache_max = 4
        self.spec_cache = {}
        all_spec_texts = sorted({x["spec"] for x in self.ordinary}
                                | {self.rmot_meta[x["video"]][x["expr_idx"]]
                                   ["sentence"] for x in self.rmot})
        if all_spec_texts:
            self.spec_cache = dict(zip(
                all_spec_texts,
                _specs(all_spec_texts)))
        print(f"[l8data] ordinary_videos={len(self.ordinary)} "
              f"rmot_clips={len(self.rmot)}", flush=True)

    def __len__(self):
        return max(1000, (len(self.ordinary) + len(self.rmot)) * 16)

    def _get_video(self, path):
        if path not in self.cache:
            self.cache[path] = pickle.load(open(path, "rb"))
            if len(self.cache) > self.cache_max:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(path)
        return self.cache[path]

    def __getitem__(self, idx):
        r = random.Random((idx * 1000003 + os.getpid()) % (2 ** 31))
        drop_pbd = r.random() < self.pbd_dropout
        if self.rmot and (self.rmot_only or r.random() < self.p_rmot):
            item = r.choice(self.rmot)
            rec = self._get_video(item["path"])
            meta = self.rmot_meta[item["video"]][item["expr_idx"]]
            label = meta["label"]
            frames = rec["frames"]
            n = len(frames)
            if n <= H:
                start, win = 0, frames
            else:
                start = r.randrange(0, n - H + 1)
                win = frames[start:start + H]
            out_frames = []
            for fr in win:
                cand_gt = fr["cand_gt"]
                ids = label.get(str(fr["frame"]), [])
                target = np.zeros(len(fr["boxes"]), np.float32)
                for j, gid in enumerate(cand_gt):
                    if gid is not None and gid in ids:
                        target[j] = 1.0
                out_frames.append({
                    "frame": fr["frame"], "boxes": fr["boxes"],
                    "pbd": np.zeros_like(np.asarray(fr["pbd"], np.float32))
                    if drop_pbd else np.asarray(fr["pbd"], np.float32),
                    "clip": np.asarray(fr["clip"], np.float32),
                    "gen": fr["gen"], "cand_gt": cand_gt,
                    "gt_boxes": fr["gt_boxes"], "target": target,
                })
            return {
                "video": item["video"], "domain": "rmot",
                "image_size": tuple(rec["image_size"]),
                "spec": self.spec_cache[meta["sentence"]],
                "frames": out_frames,
            }
        dom = r.choice(list(self.ordinary_by_domain.keys()))
        item = r.choice(self.ordinary_by_domain[dom])
        rec = self._get_video(item["path"])
        n = len(rec["frames"])
        if n <= H:
            win = rec["frames"]
        else:
            start = r.randrange(0, n - H + 1)
            win = rec["frames"][start:start + H]
        out_frames = []
        for fr in win:
            n_c = len(fr["boxes"])
            out_frames.append({
                "frame": fr["frame"], "boxes": fr["boxes"],
                "pbd": np.zeros_like(np.asarray(fr["pbd"], np.float32))
                if drop_pbd else np.asarray(fr["pbd"], np.float32),
                "clip": np.asarray(fr["clip"], np.float32),
                "gen": fr["gen"], "cand_gt": fr["cand_gt"],
                "gt_boxes": fr["gt_boxes"],
                "target": np.ones(n_c, np.float32),
            })
        return {
            "video": rec["video_id"], "domain": item["domain"],
            "image_size": tuple(rec["image_size"]),
            "spec": self.spec_cache[item["spec"]],
            "frames": out_frames,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="unified",
                    choices=["unified", "identity", "semantic"])
    ap.add_argument("--model", default="base",
                    choices=["small", "base", "large"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-core", type=float, default=5e-5)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--teacher-steps", type=int, default=1000)
    ap.add_argument("--teacher-final", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--freeze-core", action="store_true")
    ap.add_argument("--rmot-only", action="store_true")
    ap.add_argument("--p-rmot", type=float, default=0.5)
    ap.add_argument("--w-relevance", type=float, default=0.2)
    ap.add_argument("--max-rmot-expr", type=int, default=0)
    ap.add_argument("--pbd-dropout", type=float, default=0.15)
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
    model = L8UnifiedUIDM(**sizes[args.model], mode=args.mode).to(device)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        full_sd = model.state_dict()
        ck_sd = ck["model"]
        filtered = {k: v for k, v in ck_sd.items()
                    if k in full_sd and full_sd[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if rank == 0:
            print(f"[l8] init {args.init_ckpt} missing={len(missing)} "
                  f"unexpected={len(unexpected)} "
                  f"adapter_loaded={'adapter.clip_proj.mlp.0.weight' in filtered}",
                  flush=True)
    if args.freeze_core:
        for p in model.uidm.parameters():
            p.requires_grad = False
    raw_model = model
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    if rank == 0:
        print(f"[l8] mode={args.mode} model={args.model} "
              f"trainable={n_params/1e6:.2f}M device={device}", flush=True)

    ds = L8Dataset(seed=args.seed, p_rmot=args.p_rmot,
                   rmot_only=args.rmot_only,
                   max_rmot_expr=args.max_rmot_expr,
                   pbd_dropout=args.pbd_dropout)
    sampler = None
    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=(sampler is None),
        sampler=sampler, num_workers=2, collate_fn=lambda x: x,
        drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW([
        {"params": raw_model.uidm.parameters(), "lr": args.lr_core},
        {"params": raw_model.adapter.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    total_steps = args.epochs * max(1, len(loader))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_core, args.lr], total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args)
    cfg["n_params"] = n_params
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
                                  raw=raw_model, app_key="pbd")
            losses, nf = rollout.run(batch)
            loss = uidm_total_loss(
                losses, w_rel=0.1, w_relevance=args.w_relevance) if nf \
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
            if step % 10 == 0 and rank == 0:
                print(f"[l8] ep={epoch} step={step} loss={float(loss):.4f} "
                      f"row={losses.get('loss_row',0):.4f} "
                      f"col={losses.get('loss_col',0):.4f} "
                      f"rel={losses.get('loss_rel',0):.4f} "
                      f"relv={losses.get('loss_relevance',0):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        row = {"epoch": epoch, "step": step}
        for k, v in ep.items():
            row[k] = v / max(1, ep_n)
        curve.append(row)
        if rank == 0:
            print("[l8] " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()), flush=True)
            torch.save({"model": raw_model.state_dict(), "epoch": epoch,
                        "cfg": cfg, "curve": curve},
                       out_dir / f"epoch{epoch}.pt")
            torch.save({"model": raw_model.state_dict(), "epoch": epoch,
                        "cfg": cfg, "curve": curve}, out_dir / "latest.pt")
            with open(out_dir / "learning_curve.json", "w") as f:
                json.dump(curve, f, indent=2)
    if args.max_steps and step >= args.max_steps and rank == 0:
        torch.save({"model": raw_model.state_dict(), "epoch": epoch,
                    "cfg": cfg, "curve": curve}, out_dir / "latest.pt")
        torch.save({"model": raw_model.state_dict(), "epoch": epoch,
                    "cfg": cfg, "curve": curve}, out_dir / f"epoch{epoch}.pt")
    if args.ddp:
        torch.distributed.destroy_process_group()
    print(f"[l8] done seconds={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
