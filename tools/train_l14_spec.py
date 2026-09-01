"""Stage L14: frozen-UIDM specification and trajectory-language training.

The script keeps the L11 UIDM core and its lifecycle/transition modules
frozen.  Only the new Specification Encoder/Adapter, the existing relevance
head, and the optional TrajectoryLanguageMemory are trainable.  It uses the
train splits of Refer-Dance and Refer-KITTI-V2; the KITTI-V2 training videos
are also the clean visual-domain source for the related Refer-KITTI protocol.

Example (four A100s):
  torchrun --nproc_per_node=4 tools/train_l14_spec.py \
    --init-ckpt outputs/l11/checkpoints/uidm_l11_main/step11000.pt \
    --out outputs/l14/checkpoints/spec_stage1 --batch 2 --workers 2 \
    --max-steps 2000 --gpu-list 6,7,8,9
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l6_uidm import uidm_total_loss  # noqa: E402
from locatemot.models.l8_unified import L8UnifiedUIDM, load_l8_state  # noqa: E402
from tools.train_l6_uidm import H, MAX_AGE, MAX_SLOTS, UIDMRollout  # noqa: E402


SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def _gid(x):
    return None if x is None else str(x)


class RMOTExpressionDataset(Dataset):
    """Windowed expression clips from one leakage-controlled train split."""

    def __init__(self, root, seed=20260823, exclude=(), max_expr=0):
        self.root = Path(root)
        self.seed = int(seed)
        self.exclude = {str(x) for x in exclude}
        exp_path = self.root / "expressions.json"
        if not exp_path.exists():
            raise FileNotFoundError(exp_path)
        self.meta = json.loads(exp_path.read_text())
        self.items = []
        for vid, expressions in sorted(self.meta.items()):
            if vid in self.exclude:
                continue
            pkl = self.root / f"{vid}.pkl"
            if not pkl.exists():
                continue
            for ei, expression in enumerate(expressions):
                if max_expr and ei >= int(max_expr):
                    break
                # Empty training expressions do not provide a positive
                # trajectory for the contrastive objective.
                if not expression.get("label"):
                    continue
                self.items.append((vid, ei, str(pkl)))
        if not self.items:
            raise RuntimeError(f"no RMOT train items in {self.root}")
        self.cache = OrderedDict()
        self.cache_max = 4
        print(f"[l14data] root={self.root} videos="
              f"{len({x[0] for x in self.items})} expressions={len(self.items)}",
              flush=True)

    def __len__(self):
        # Sampling is random inside __getitem__; multiplying a large KITTI
        # expression index by eight only delays checkpoints and makes a
        # nominal epoch misleading.  One pass over expressions is enough for
        # a sampler epoch, with the explicit --max-steps controlling budget.
        return max(1000, len(self.items))

    def _get(self, path):
        if path not in self.cache:
            with open(path, "rb") as f:
                self.cache[path] = pickle.load(f)
            if len(self.cache) > self.cache_max:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(path)
        return self.cache[path]

    def __getitem__(self, idx):
        rng = random.Random(self.seed + int(idx) * 1000003 + os.getpid())
        vid, ei, path = self.items[rng.randrange(len(self.items))]
        rec = self._get(path)
        meta = self.meta[vid][ei]
        all_frames = rec["frames"]
        if len(all_frames) > H:
            start = rng.randrange(0, len(all_frames) - H + 1)
            frames = all_frames[start:start + H]
        else:
            frames = all_frames
        packed = []
        label = meta.get("label", {})
        for fr in frames:
            n = len(fr["boxes"])
            ids = [_gid(x) for x in label.get(str(fr["frame"]), [])]
            ids_set = set(ids)
            cand_gt = [_gid(x) for x in fr["cand_gt"]]
            target = np.asarray(
                [1.0 if x is not None and x in ids_set else 0.0
                 for x in cand_gt], np.float32)
            gt_boxes = {str(k): np.asarray(v, np.float32)
                        for k, v in fr["gt_boxes"].items()}
            packed.append({
                "frame": int(fr["frame"]),
                "boxes": np.asarray(fr["boxes"], np.float32),
                "pbd": np.asarray(fr["pbd"], np.float32),
                "clip": np.asarray(fr["clip"], np.float32),
                "gen": np.asarray(fr["gen"], np.float32),
                "cand_gt": cand_gt,
                "gt_boxes": gt_boxes,
                "target": target,
                "target_gids": ids,
                "cand_w": np.ones(n, np.float32),
                "cand_nw": np.ones(n, np.float32),
            })
        return {
            "video": vid,
            "expression": meta.get("sentence", meta.get("expression", "")),
            "domain": "rmot",
            "image_size": tuple(rec["image_size"]),
            "spec": np.asarray(meta["spec"], np.float32),
            "spec_valid": 1.0,
            "frames": packed,
        }


class RMOTMixture(Dataset):
    """Balanced Dance/KITTI expression sampler without duplicating frames."""

    def __init__(self, dance, kitti, p_dance=0.5, seed=20260823):
        self.dance = dance
        self.kitti = kitti
        self.p_dance = float(p_dance)
        self.seed = int(seed)

    def __len__(self):
        return max(len(self.dance), len(self.kitti))

    def __getitem__(self, idx):
        r = random.Random(self.seed + int(idx) * 9176 + os.getpid())
        if r.random() < self.p_dance:
            return self.dance[idx % len(self.dance)]
        return self.kitti[idx % len(self.kitti)]


def _trainable_adapter(model, train_relevance=True):
    """Freeze L11 and unfreeze only the new L14 adapter/language paths."""
    for p in model.parameters():
        p.requires_grad = False
    names = []
    prefixes = (
        "adapter.spec_encoder.", "adapter.spec_adapter.",
        "adapter.spec_track.", "adapter.trajectory_language.",
    )
    if train_relevance:
        prefixes += ("adapter.relevance.",)
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = True
            names.append(name)
    if not names:
        raise RuntimeError("no L14 adapter parameters were selected")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=sorted(SIZES), default="large")
    ap.add_argument("--gpu-list", default="6,7,8,9")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", choices=["bfloat16", "float16"],
                    default="bfloat16")
    ap.add_argument("--p-dance", type=float, default=0.5)
    ap.add_argument("--w-traj", type=float, default=0.25)
    ap.add_argument("--w-relevance", type=float, default=0.2)
    ap.add_argument("--teacher-steps", type=int, default=400)
    ap.add_argument("--teacher-final", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--no-trajectory-memory", action="store_true")
    ap.add_argument(
        "--train-relevance", action="store_true",
        help="also fine-tune the legacy acquisition relevance head; "
             "off by default for the controlled Stage1 protocol")
    args = ap.parse_args()

    rank = 0
    world = 1
    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L14 training requires CUDA")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    model = L8UnifiedUIDM(
        **SIZES[args.model], mode="unified", cond_gated=True,
        spec_conditioned=True,
        trajectory_memory=not args.no_trajectory_memory).to(device)
    ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = load_l8_state(model, ck["model"])
    train_names = _trainable_adapter(
        model, train_relevance=args.train_relevance)
    if rank == 0:
        print(f"[l14] init={args.init_ckpt} missing={len(missing)} "
              f"unexpected={len(unexpected)} trainable={len(train_names)}",
              flush=True)

    # V2's 17 train videos are disjoint from its official four-video eval
    # split.  Dance is already materialized as the 40-video train cache; its
    # 25-video validation cache is never sampled here.
    dance = RMOTExpressionDataset(ROOT / "outputs/l8/data/rmot_train",
                                  seed=args.seed)
    kitti = RMOTExpressionDataset(ROOT / "outputs/l11/data/rmot_kitti",
                                  seed=args.seed + 17,
                                  exclude=("0005", "0011", "0013", "0019"))
    ds = RMOTMixture(dance, kitti, p_dance=args.p_dance, seed=args.seed)
    sampler = None
    if world > 1:
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
    kwargs = dict(batch_size=args.batch, shuffle=sampler is None,
                  sampler=sampler, collate_fn=lambda x: x, drop_last=True,
                  num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch)
    loader = DataLoader(ds, **kwargs)

    raw_model = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=False)
    params = [p for p in raw_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = int(args.max_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_steps), eta_min=args.lr * 0.05)
    amp_dtype = (torch.bfloat16 if args.amp_dtype == "bfloat16"
                 else torch.float16)
    use_scaler = bool(args.amp and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        cfg = vars(args).copy()
        cfg.update({"world_size": world, "trainable_names": train_names,
                    "dance_expressions": len(dance.items),
                    "kitti_expressions": len(kitti.items),
                    "spec_conditioned": True,
                    "trajectory_memory": not args.no_trajectory_memory,
                    "sem_in_core": True, "cond_gated": True,
                    "amp_dtype": args.amp_dtype})
        (out / "train_config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False))
    else:
        cfg = None

    step = 0
    opt_step = 0
    curve = []
    totals = {}
    count = 0
    t0 = time.time()
    raw_model.train()
    # Keep the frozen L11 pathway deterministic while the new adapter and
    # language memory remain in train mode.  This makes the Stage-1 delta
    # attributable to learned specification paths rather than dropout in the
    # frozen transition core.
    raw_model.uidm.eval()
    for name in ("pbd_proj", "clip_proj", "spec_proj", "gate",
                 "cond_gate", "sem_transform"):
        getattr(raw_model.adapter, name).eval()
    opt.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            teacher = step < args.teacher_steps or (
                random.random() < args.teacher_final)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=args.amp):
                rollout = UIDMRollout(
                    model, batch, device, teacher=teacher, raw=raw_model,
                    max_age=MAX_AGE, max_slots=MAX_SLOTS,
                    app_key="pbd", new_score_thr=0.30)
                losses, nf = rollout.run(batch)
                if nf:
                    loss = uidm_total_loss(
                        losses, w_rel=0.1, w_relevance=args.w_relevance)
                    loss = loss + args.w_traj * losses.get(
                        "loss_traj", loss.new_zeros(()))
                else:
                    loss = torch.zeros((), device=device)
                scaled_loss = loss / max(1, args.grad_accum)
            scaler.scale(scaled_loss).backward()
            step += 1
            count += 1
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    totals[key] = totals.get(key, 0.0) + float(value.detach())
            if step % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                opt_step += 1
            if rank == 0 and step % 20 == 0:
                print(f"[l14] epoch={epoch} step={step} "
                      f"loss={float(loss):.4f} "
                      f"traj={float(losses.get('loss_traj', 0.0)):.4f} "
                      f"row={float(losses.get('loss_row', 0.0)):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if rank == 0 and args.save_every and step % args.save_every == 0:
                payload_cfg = cfg if cfg is not None else vars(args)
                torch.save({"model": raw_model.state_dict(),
                            "epoch": epoch, "step": step,
                            "cfg": payload_cfg, "curve": curve,
                            "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict()},
                           out / f"step{step}.pt")
            if step >= args.max_steps:
                break
        if rank == 0:
            row = {"epoch": epoch, "step": step,
                   "batches": count,
                   "loss": sum(totals.values()) / max(1, count),
                   "loss_traj": totals.get("loss_traj", 0.0) / max(1, count),
                   "loss_row": totals.get("loss_row", 0.0) / max(1, count),
                   "loss_col": totals.get("loss_col", 0.0) / max(1, count)}
            curve.append(row)
            print("[l14] " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()), flush=True)
            payload_cfg = cfg if cfg is not None else vars(args)
            torch.save({"model": raw_model.state_dict(), "epoch": epoch,
                        "step": step, "cfg": payload_cfg, "curve": curve,
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict()},
                       out / "latest.pt")
            (out / "learning_curve.json").write_text(
                json.dumps(curve, indent=2))
        if step >= args.max_steps:
            break
    if rank == 0:
        print(f"[l14] done steps={step} optimizer_steps={opt_step} "
              f"seconds={time.time()-t0:.1f}", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
