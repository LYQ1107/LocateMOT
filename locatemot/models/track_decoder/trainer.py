"""Shared training loop for B3/B4 with calibration early stopping."""
from __future__ import annotations

import hashlib
import json
import math
import os

import torch
from torch.utils.data import DataLoader

from locatemot.data.collate import collate_track_batch


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def train_loop(
    model,
    train_ds,
    calib_ds,
    out_dir,
    seed=20260806,
    lr=2e-4,
    weight_decay=1e-4,
    warmup_ratio=0.05,
    max_epochs=60,
    patience=8,
    batch_size=16,
    grad_clip=1.0,
    eval_every=200,
    precision="bf16",
    cfg=None,
    collate=collate_track_batch,
    loss_fn=None,
    eval_fn=None,
    device="cuda",
):
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    total_steps = steps_per_epoch * max_epochs
    warmup = int(total_steps * warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, s / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * s / total_steps)),
    )
    precomputed = hasattr(train_ds, "batch") and hasattr(train_ds, "metas")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              num_workers=0, drop_last=True) if not precomputed else None
    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, collate_fn=collate) \
        if not precomputed else None
    best_score, best_step, bad_epochs = -1e9, -1, 0
    curves = []
    ckpt_paths = {}
    use_amp = precision == "bf16" and device == "cuda"
    step = 0
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        if precomputed:
            perm = torch.randperm(len(train_ds))
            batch_iter = (
                _to_device(train_ds.batch(perm[i:i + batch_size]), device)
                for i in range(0, len(train_ds) - batch_size + 1, batch_size)
            )
        else:
            batch_iter = train_loader
        for batch in batch_iter:
            batch = _to_device(batch, device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(batch)
                losses = loss_fn(pred, batch)
                loss = losses["loss"]
            if not torch.isfinite(loss):
                if not getattr(train_loop, "_saved_bad_batch", False):
                    torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                               "/tmp/bad_batch.pt")
                    train_loop._saved_bad_batch = True
                stats = {k: (float(v) if torch.isfinite(v) else "nan") for k, v in losses.items()}
                pred_stats = {
                    k: (float(v.detach().abs().max()) if isinstance(v, torch.Tensor) and v.numel() else None)
                    for k, v in pred.items() if isinstance(v, torch.Tensor)
                }
                print(
                    f"[trainer] step {step} non-finite loss; components={stats} "
                    f"pred_absmax={pred_stats} "
                    f"targets=[{batch['match_targets'].min().item()},{batch['match_targets'].max().item()}] "
                    f"cur_mask_sum={batch['cur_mask'].sum().item()} "
                    f"ref_mask_sum={batch['ref_mask'].sum().item()}",
                    flush=True,
                )
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            step += 1
            epoch_loss += float(loss.detach())
            if step % eval_every == 0:
                model.eval()
                if precomputed:
                    score, metrics = eval_fn(model, _PrecomputedLoader(calib_ds, batch_size, device), device)
                else:
                    score, metrics = eval_fn(model, calib_loader, device)
                model.train()
                curves.append({
                    "step": step, "epoch": epoch, "train_loss": round(epoch_loss / max(1, step), 4),
                    "calib_score": round(score, 4), **{k: round(v, 4) for k, v in metrics.items()},
                })
                if score > best_score:
                    best_score = score
                    best_step = step
                    bad_epochs = 0
                    torch.save(_ckpt(model, opt, sched, step, seed, cfg), os.path.join(out_dir, "best.pt"))
                else:
                    bad_epochs += 1
        torch.save(_ckpt(model, opt, sched, step, seed, cfg), os.path.join(out_dir, "latest.pt"))
        if bad_epochs >= patience:
            break
    ckpt_paths["best"] = os.path.join(out_dir, "best.pt")
    ckpt_paths["latest"] = os.path.join(out_dir, "latest.pt")
    with open(os.path.join(out_dir, "training_curves.csv"), "w") as f:
        if curves:
            keys = list(curves[0].keys())
            f.write(",".join(keys) + "\n")
            for c in curves:
                f.write(",".join(str(c[k]) for k in keys) + "\n")
    with open(os.path.join(out_dir, "checkpoint_manifest.json"), "w") as f:
        json.dump({
            "best_step": best_step, "best_score": best_score, "seed": seed,
            "config_hash": config_hash(cfg or {}), "paths": ckpt_paths,
        }, f, ensure_ascii=False, indent=2)
    return {"best_score": best_score, "best_step": best_step, "curves": curves}


class _PrecomputedLoader:
    def __init__(self, ds, batch_size, device):
        self.ds = ds
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        perm = torch.arange(len(self.ds))
        for i in range(0, len(self.ds), self.batch_size):
            yield _to_device(self.ds.batch(perm[i:i + self.batch_size]), self.device)


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _ckpt(model, opt, sched, step, seed, cfg):
    return {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "seed": seed,
        "config_hash": config_hash(cfg or {}),
    }
