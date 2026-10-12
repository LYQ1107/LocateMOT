#!/usr/bin/env python3
"""Guarded eight-GPU U2-P training, with full evaluation between its two epochs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# MMEngine's native loader omits weights_only. Our verified local training
# checkpoints include HistoryBuffer and optimizer state, including on resume.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/mmdetection"))
from tools.unified.build_u2_portability_artifact import FOUNDATION_SHA

DIRECTORY = ROOT / "outputs/unified/u2/perception_adaptation"
CONFIG = ROOT / "tools/unified/configs/u2_perception_adaptation_fp32.py"
RECIPE = ROOT / "outputs/unified/u2/protocol/perception_adaptation_recipe.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def preflight(args: argparse.Namespace) -> dict:
    from mmengine.config import Config
    cfg = Config.fromfile(str(CONFIG))
    recipe = json.loads(RECIPE.read_text())
    if (cfg.train_cfg.max_epochs != 2 or cfg.train_dataloader.batch_size != 2
            or cfg.train_dataloader.batch_sampler is not None
            or cfg.optim_wrapper.type != "OptimWrapper" or cfg.optim_wrapper.optimizer.lr != 1e-5
            or cfg.randomness.seed != recipe["seed"] or recipe["rows_per_epoch"] != 20000
            or recipe["maximum_epochs"] != 2 or recipe["starting_checkpoint_sha256"] != FOUNDATION_SHA):
        raise ValueError("Configuration differs from the registered bounded FP32 replay")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    portability_path = DIRECTORY / "f0_preflight_portability.json"
    subprocess.run([sys.executable, str(ROOT / "tools/unified/build_u2_portability_artifact.py"),
                    "--out", str(portability_path)], check=True, cwd=ROOT)
    portability = json.loads(portability_path.read_text())
    reasons = []
    if portability["ordinary_missing_local_data"] or portability["tao_missing_local_data"]:
        reasons.append("incomplete required original source-image coverage")
    if portability["ordinary_unmeasured"] or portability["tao_unmeasured"]:
        reasons.append("full F0 portability audits are still incomplete")
    if not reasons and portability["status"] == "PASS":
        reasons.append("F0 already passes; optional perception adaptation is unnecessary")
    replay_path = DIRECTORY / "replay_manifest.json"
    if not replay_path.is_file():
        reasons.append("complete registered 20,000-row replay is not ready")
    else:
        replay = json.loads(replay_path.read_text())
        if (replay["rows_per_epoch"] != 20000 or replay["seed"] != recipe["seed"]
                or replay["allocations"] != recipe["allocations"]
                or replay["replay_sha256"] != digest(Path(replay["replay_path"]))):
            raise ValueError("Replay differs from the registered mixture")
    if args.epoch == 2:
        if not args.epoch1_selection or not args.resume:
            raise ValueError("Epoch 2 requires complete epoch-1 selection and its original checkpoint")
        selection = json.loads(args.epoch1_selection.read_text())
        if (selection["epoch"] != 1 or not selection["next_epoch_eligible"]
                or selection["foundation_sha256"] != digest(args.resume)
                or any(digest(Path(source["path"])) != source["sha256"] for source in selection["sources"].values())):
            raise ValueError("Epoch 2 requires an unchanged, fully evaluated epoch-1 checkpoint")
    return {"state": "READY" if not reasons else "WAITING_PREREQUISITES", "reasons": reasons,
            "epoch": args.epoch, "starting_foundation_sha256": FOUNDATION_SHA,
            "recipe_sha256": digest(RECIPE), "config_sha256": digest(CONFIG),
            "world_size": 8, "global_batch": 16, "steps_per_epoch": 1250,
            "maximum_total_optimizer_steps": 2500, "optimization_launched": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=DIRECTORY / "training")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epoch1-selection", type=Path)
    args = parser.parse_args()
    if args.check_only:
        result = preflight(args)
        (DIRECTORY / "training_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        parser.error("Use .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8; registered global batch is 16")
    import torch
    import torch.distributed as dist
    from mmengine.config import Config
    from mmengine.runner import Runner

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    eligibility = [None]
    if dist.get_rank() == 0:
        eligibility[0] = preflight(args)
    dist.broadcast_object_list(eligibility, src=0)
    if eligibility[0]["state"] != "READY":
        raise RuntimeError(json.dumps(eligibility[0]))
    cfg = Config.fromfile(str(CONFIG))
    cfg.launcher = "pytorch"
    cfg.work_dir = str(args.work_dir.resolve())
    # Pause at epoch 1 for full audits. Epoch 2 resumes the same optimizer,
    # sampler epoch and scheduler, retaining the absolute two-epoch cap.
    # Native MMEngine restores the fixed seed, not worker RNG streams; each
    # separately launched epoch uses the same registered base seed 20261012.
    cfg.train_cfg.max_epochs = args.epoch
    cfg.max_epochs = args.epoch
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not 0 <= saved["meta"]["epoch"] < args.epoch or saved["meta"]["iter"] >= args.epoch * 1250:
            raise ValueError("Resume checkpoint is outside the registered unfinished epoch")
        cfg.resume = True
        cfg.load_from = str(args.resume.resolve())
        del saved
    else:
        if (args.work_dir / "epoch_1.pth").exists():
            raise ValueError("Epoch 1 already completed; evaluate it before any further optimization")
        if digest(Path(cfg.load_from)) != FOUNDATION_SHA:
            raise ValueError("Initial checkpoint differs from immutable U1R2")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    runner = Runner.from_cfg(cfg)
    if len(runner.train_dataloader) != 1250:
        raise ValueError("Registered 20k replay must produce exactly 1,250 batch-2 steps per rank")
    runner.train()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
