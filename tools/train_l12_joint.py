"""Stage L12 joint fine-tuning on prompt and discovery streams.

The prompt branch uses only the 20 DAVIS videos excluded from the frozen
10-video controlled evaluation.  Prompt seeds are initialized through the
same PBD + adapter + UIDM memory path as ``eval_l12_davis.py``.  The other
three quarters of each training batch come from the existing MOT, OVMOT,
and RMOT mixture, so the resulting checkpoint remains one shared model.

This is a deliberately small continuation experiment: the core receives a
much lower learning rate than the observation adapter to limit catastrophic
forgetting while still testing whether prompt adaptation transfers through
the shared identity dynamics.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402
from locatemot.models.l6_uidm import uidm_total_loss  # noqa: E402
from locatemot.models.l8_unified import (  # noqa: E402
    L8UnifiedUIDM,
    load_l8_state,
)
from tools.train_l6_uidm import H, MAX_AGE, MAX_SLOTS, UIDMRollout  # noqa: E402
from tools.train_l9_uidm import ALL_OBJECTS_SPEC, L9Dataset  # noqa: E402


DATA = ROOT / "outputs" / "l12" / "data" / "davis"
PBD_CACHE = ROOT / "outputs" / "l12" / "cache" / "davis_pbd"
SEED_PBD = ROOT / "outputs" / "l12" / "data" / "davis_seed_pbd.json"

# These are the exact videos used by the frozen controlled comparison.
# Keeping them out of prompt training makes the final prompt evaluation a
# held-out test rather than a reconstruction of the reported numbers.
EVAL_VIDEOS = {
    "dogs-jump", "gold-fish", "india", "lab-coat", "loading", "pigs",
    "shooting", "soapbox", "paragliding-launch", "kite-surf",
}

SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def _norm_gid(gid):
    return None if gid is None else str(gid)


class DavisPromptDataset(Dataset):
    """Held-out-safe prompt-seeded DAVIS training clips."""

    def __init__(self, spec, seed=20260806, prompt_types=None):
        self.spec = np.asarray(spec, np.float32)
        self.seed = int(seed)
        self.prompt_types = tuple(prompt_types or ("mask", "box", "point"))
        self.videos = sorted(
            p.stem for p in DATA.glob("*.pkl") if p.stem not in EVAL_VIDEOS)
        if not self.videos:
            raise RuntimeError("no held-out-safe DAVIS prompt training videos")
        self.seed_pbd = json.loads(SEED_PBD.read_text())
        self.cache = OrderedDict()
        self.cache_max = 4
        print(f"[l12prompt] train_videos={len(self.videos)} "
              f"types={self.prompt_types}", flush=True)

    def __len__(self):
        return max(640, len(self.videos) * 32)

    def _get(self, vid):
        if vid not in self.cache:
            with open(DATA / f"{vid}.pkl", "rb") as f:
                self.cache[vid] = pickle.load(f)
            if len(self.cache) > self.cache_max:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(vid)
        return self.cache[vid]

    def _pbd(self, vid, frame, n):
        rec = read_frame_cache(
            str(PBD_CACHE), cache_key("davis", vid, frame, "pbd_full"))
        if rec is None:
            raise RuntimeError(
                f"missing DAVIS prompt-training PBD cache: {vid}/{frame}")
        pbd = np.asarray(rec["features"]["pbd_box_end_last"], np.float32)
        if pbd.shape != (n, 2048):
            raise RuntimeError(
                f"misaligned PBD cache {vid}/{frame}: {pbd.shape} != {(n, 2048)}")
        return pbd

    def __getitem__(self, idx):
        rng = random.Random((self.seed + idx * 1000003 + os.getpid())
                            % (2 ** 31))
        vid = self.videos[rng.randrange(len(self.videos))]
        prompt_type = self.prompt_types[rng.randrange(len(self.prompt_types))]
        rec = self._get(vid)
        # The seed is injected on frame 0 by the evaluation protocol.  The
        # learned rollout must therefore begin with the first association
        # frame, not replay the birth frame as an already-active match.
        frames = rec["frames"][1:H + 1]
        if len(frames) < H:
            # Very short videos are valid; UIDMRollout consumes the available
            # prefix only when the dataset itself returns it.
            frames = list(frames)
        seed_meta = self.seed_pbd.get(vid, {})
        seeds = []
        for oid, seed in rec.get("seeds", {}).items():
            info = seed_meta.get(str(oid), {})
            pbd = info.get(prompt_type)
            clip = info.get(prompt_type + "_clip")
            if pbd is None or clip is None:
                continue
            seeds.append({
                "gid": str(oid),
                "box": np.asarray(seed["box"], np.float32),
                "pbd": np.asarray(pbd, np.float32),
                "clip": np.asarray(clip, np.float32),
            })
        if not seeds:
            raise RuntimeError(f"no {prompt_type} seed tokens for {vid}")

        out_frames = []
        for fr in frames:
            n = len(fr["boxes"])
            cand_gt = [_norm_gid(g) for g in fr["cand_gt"]]
            gt_boxes = {str(k): np.asarray(v, np.float32)
                        for k, v in fr["gt_boxes"].items()}
            out_frames.append({
                "frame": int(fr["frame"]),
                "boxes": np.asarray(fr["boxes"], np.float32),
                "pbd": self._pbd(vid, int(fr["frame"]), n),
                "clip": np.asarray(fr["clip"], np.float32),
                "gen": np.asarray(fr["gen"], np.float32),
                "cand_gt": cand_gt,
                "gt_boxes": gt_boxes,
                "target": np.asarray(
                    [1.0 if g is not None else 0.0 for g in cand_gt],
                    np.float32),
                "cand_w": np.ones(n, np.float32),
                "cand_nw": np.ones(n, np.float32),
                "no_unmatched_new": True,
            })
        return {
            "video": vid,
            "domain": "prompt_" + prompt_type,
            "image_size": tuple(rec["image_size"]),
            "spec": self.spec,
            "seeded": True,
            "seeds": seeds,
            "frames": out_frames,
        }


class JointDataset(Dataset):
    """Prompt branch mixed with the existing three discovery streams."""

    def __init__(self, base, prompt, p_prompt=0.25, seed=20260806):
        self.base = base
        self.prompt = prompt
        self.p_prompt = float(p_prompt)
        self.seed = int(seed)

    def __len__(self):
        return max(len(self.base), len(self.prompt), 1000)

    def __getitem__(self, idx):
        rng = random.Random((self.seed + idx * 9176 + os.getpid())
                            % (2 ** 31))
        if rng.random() < self.p_prompt:
            return self.prompt[idx]
        return self.base[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--model", choices=sorted(SIZES), default="large")
    ap.add_argument("--mode", choices=["unified", "identity", "semantic"],
                    default="unified")
    ap.add_argument("--cond-gated", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="prompt adapter maximum learning rate")
    ap.add_argument("--lr-core", type=float, default=1e-6)
    ap.add_argument("--p-prompt", type=float, default=0.25)
    ap.add_argument("--p-rmot", type=float, default=0.30)
    ap.add_argument("--p-ovmot", type=float, default=0.30)
    ap.add_argument("--pbd-dropout", type=float, default=0.15)
    ap.add_argument("--teacher-steps", type=int, default=200)
    ap.add_argument("--teacher-final", type=float, default=0.5)
    ap.add_argument("--new-score-thr", type=float, default=0.30)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--freeze-core", action="store_true")
    ap.add_argument("--ovmot-dir", default="outputs/l10/data/tao_train")
    ap.add_argument("--rmot-dir", default=None)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L12 joint training requires a CUDA device")

    model = L8UnifiedUIDM(
        **SIZES[args.model], mode=args.mode,
        cond_gated=args.cond_gated).to(device)
    model.sem_in_core = True
    ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = load_l8_state(model, ck["model"])
    print(f"[l12joint] init={args.init_ckpt} missing={len(missing)} "
          f"unexpected={len(unexpected)} device={device}", flush=True)
    if args.freeze_core:
        for p in model.uidm.parameters():
            p.requires_grad = False

    base = L9Dataset(
        seed=args.seed, p_rmot=args.p_rmot, p_ovmot=args.p_ovmot,
        pbd_dropout=args.pbd_dropout, ovmot_dir=args.ovmot_dir,
        rmot_dir=args.rmot_dir)
    prompt = DavisPromptDataset(
        base.spec_cache[ALL_OBJECTS_SPEC], seed=args.seed)
    ds = JointDataset(base, prompt, p_prompt=args.p_prompt, seed=args.seed)
    loader_kwargs = dict(
        batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=lambda x: x, drop_last=True,
        pin_memory=True)
    if args.workers > 0:
        loader_kwargs.update(
            persistent_workers=True, prefetch_factor=args.prefetch)
    loader = DataLoader(ds, **loader_kwargs)

    raw_model = model
    core_params = [p for p in raw_model.uidm.parameters() if p.requires_grad]
    adapter_params = [p for p in raw_model.adapter.parameters()
                      if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": core_params, "lr": args.lr_core},
        {"params": adapter_params, "lr": args.lr},
    ], weight_decay=1e-4)
    total_steps = args.max_steps or args.epochs * max(1, len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_core, args.lr], total_steps=total_steps,
        pct_start=0.10, anneal_strategy="cos")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg.update({"n_params": sum(p.numel() for p in raw_model.parameters()
                                if p.requires_grad),
               "eval_videos_excluded": sorted(EVAL_VIDEOS),
               "prompt_train_videos": prompt.videos})
    (out_dir / "train_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False))

    step = 0
    curve = []
    t0 = time.time()

    def save(epoch, tag="latest"):
        torch.save({
            "model": raw_model.state_dict(), "epoch": epoch, "step": step,
            "cfg": cfg, "curve": curve, "opt_state": opt.state_dict(),
            "sched_state": sched.state_dict(),
        }, out_dir / f"{tag}.pt")
        (out_dir / "learning_curve.json").write_text(
            json.dumps(curve, indent=2))

    for epoch in range(1, args.epochs + 1):
        raw_model.train()
        totals = defaultdict(float)
        n_batch = 0
        for batch in loader:
            teacher = step < args.teacher_steps or (
                random.random() < args.teacher_final)
            rollout = UIDMRollout(
                raw_model, batch, device, teacher=teacher,
                raw=raw_model, max_age=MAX_AGE, max_slots=MAX_SLOTS,
                app_key="pbd", new_score_thr=args.new_score_thr)
            losses, nf = rollout.run(batch)
            if not nf:
                continue
            loss = uidm_total_loss(
                losses, w_rel=0.1, w_relevance=0.2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            n_batch += 1
            for k, value in losses.items():
                totals[k] += float(value.detach()) if isinstance(
                    value, torch.Tensor) else float(value)
            if step % 10 == 0:
                print(f"[l12joint] epoch={epoch} step={step} "
                      f"loss={float(loss):.4f} "
                      f"row={float(losses.get('loss_row', 0)):.4f} "
                      f"new={float(losses.get('loss_new', 0)):.4f} "
                      f"prompt_mix={args.p_prompt:.2f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if step % args.save_every == 0:
                save(epoch, tag=f"step{step}")
            if step >= args.max_steps:
                break
        row = {"epoch": epoch, "step": step}
        row.update({k: v / max(1, n_batch) for k, v in totals.items()})
        curve.append(row)
        print("[l12joint] " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()), flush=True)
        save(epoch)
        if step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
