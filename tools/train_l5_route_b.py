"""Stage L5 Route B: sequence-local dynamic identity prediction.

Usage:
  python tools/train_l5_route_b.py \
      --data outputs/l5/clips/small_bdd_train.pkl \
             outputs/l5/clips/small_dance_train.pkl \
      --val-data outputs/l5/clips/small_bdd_val.pkl \
                outputs/l5/clips/small_dance_val.pkl \
      --out outputs/l5/checkpoints/route_b_small --model small \
      --epochs 60 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l5_route_b import L5IdentityPredictor, l5b_loss  # noqa: E402
from tools.train_l5_route_a import (  # noqa: E402
    L5ClipDataset,
    collate,
    prepare_samples,
)


def load_flat(paths, source="u0"):
    flat = []
    clips = {}
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        clips.update(d["videos"])
        for vid, rec in d["videos"].items():
            for spec, views in rec["views"].items():
                for s in views.get(source, []):
                    s = dict(s)
                    s.update({"video": vid, "spec": spec,
                              "dataset": d["domain"],
                              "image_size": rec["image_size"],
                              "source": source})
                    flat.append(s)
    return flat, clips


def build_slot_maps(flat):
    """Per-video GT id -> sequence-local slot (sorted for stability)."""
    per_video = defaultdict(set)
    for s in flat:
        for g in s.get("track_cur_gt", s.get("track_dom_gt", [])):
            if g is not None:
                per_video[s["video"]].add(str(g))
        for g in s.get("_cand_gt", []):
            if g is not None:
                per_video[s["video"]].add(str(g))
    return {vid: {g: i for i, g in enumerate(sorted(gids))}
            for vid, gids in per_video.items()}


def add_slot_targets(batch, slot_maps, max_slots=128):
    B = len(batch["meta"])
    Nmax = max(len(c) for c in batch["cand_gt_str"])
    slot_target = np.full((B, Nmax), -1, np.int64)
    n_slots = np.zeros(B, np.int64)
    for b, m in enumerate(batch["meta"]):
        smap = slot_maps[m["video"]]
        G = min(len(smap), max_slots)
        n_slots[b] = G
        for j, gid in enumerate(batch["cand_gt_str"][b]):
            if gid is not None:
                slot_target[b, j] = min(smap.get(str(gid), G), G)
            else:
                slot_target[b, j] = G
    batch["slot_target"] = torch.from_numpy(slot_target)
    batch["n_slots"] = torch.from_numpy(n_slots)
    return batch


def slot_accuracy(batch, pred):
    logits = pred["slot_logits"]
    target = batch["slot_target"]
    valid = target >= 0
    if not valid.any():
        return 0.0, 0
    acc = int((logits[valid].argmax(-1) == target[valid]).sum())
    return acc / int(valid.sum()), int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val-data", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="base",
                    choices=["small", "base", "large"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-slots", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[l5b] device={device}", flush=True)

    flat, clips = load_flat(args.data, source="u0")
    flat = prepare_samples(flat, clips)
    groups = defaultdict(list)
    for s in flat:
        groups[(s["dataset"], s["video"], int(s["frame_id"]),
                s["source"])].append(s)
    groups = list(groups.values())
    val_flat, val_clips = load_flat(args.val_data, source="u0")
    val_flat = prepare_samples(val_flat, val_clips)
    slot_maps = build_slot_maps(flat + val_flat)
    print(f"[l5b] train_groups={len(groups)} train_samples={len(flat)} "
          f"val={len(val_flat)} slots_per_video="
          f"[{min(len(m) for m in slot_maps.values())}.."
          f"{max(len(m) for m in slot_maps.values())}]", flush=True)

    sizes = {
        "small": dict(d_model=128, temporal_layers=2, set_layers=2,
                      n_heads=4, ffn_dim=512),
        "base": dict(d_model=256, temporal_layers=4, set_layers=4,
                     n_heads=8, ffn_dim=1024),
        "large": dict(d_model=384, temporal_layers=6, set_layers=6,
                      n_heads=8, ffn_dim=1536),
    }
    model = L5IdentityPredictor(**sizes[args.model],
                                max_slots=args.max_slots).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[l5b] model={args.model} params={n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = L5ClipDataset(groups, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=4, persistent_workers=True, drop_last=False)
    total_steps = args.epochs * max(1, len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05,
        anneal_strategy="cos")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args)
    cfg["n_params"] = n_params
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    curve = []
    t0 = time.time()
    save_epochs = {1, 5, 10, 20, 40, 60}
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_acc = 0
        ep_n = 0
        ep_steps = 0
        for batch in loader:
            batch = add_slot_targets(batch, slot_maps, args.max_slots)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = l5b_loss(batch, pred, new_weight=model.new_weight)
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ep_loss += float(loss["loss"])
            ep_steps += 1
            a, n = slot_accuracy(batch, pred)
            ep_acc += int(a * n)
            ep_n += n
            if ep_steps % 50 == 0:
                print(f"[l5b] ep={epoch} step={ep_steps} "
                      f"loss={float(loss['loss']):.4f} "
                      f"slot_acc={ep_acc/max(1,ep_n):.3f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        val_acc = 0.0
        val_n = 0
        if val_flat:
            model.eval()
            with torch.no_grad():
                for i in range(0, len(val_flat), 64):
                    sub = val_flat[i:i + 64]
                    batch = collate([[s] for s in sub])
                    batch = add_slot_targets(batch, slot_maps, args.max_slots)
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor)
                             else v for k, v in batch.items()}
                    pred = model(batch)
                    a, n = slot_accuracy(batch, pred)
                    val_acc += a * n
                    val_n += n
            val_acc = val_acc / max(1, val_n)
        row = {"epoch": epoch, "loss": ep_loss / max(1, ep_steps),
               "train_slot_acc": ep_acc / max(1, ep_n),
               "val_slot_acc": val_acc}
        curve.append(row)
        print(f"[l5b] epoch={epoch} " +
              " ".join(f"{k}={v:.4f}" if isinstance(v, float)
                       else f"{k}={v}" for k, v in row.items()), flush=True)
        if epoch in save_epochs or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "cfg": cfg, "curve": curve},
                       out_dir / f"epoch{epoch}.pt")
        with open(out_dir / "learning_curve.json", "w") as f:
            json.dump(curve, f, indent=2)
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "cfg": cfg, "curve": curve}, out_dir / "final.pt")
    print(f"[l5b] done seconds={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
