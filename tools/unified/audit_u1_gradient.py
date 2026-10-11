#!/usr/bin/env python3
"""One-step FP32/TF32 gradient audit from the frozen U1-A epoch4 checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MMDET_ROOT = PROJECT_ROOT / "third_party/mmdetection"
for path in (PROJECT_ROOT, MMDET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from mmengine.config import Config  # noqa: E402
from mmengine.runner import load_checkpoint  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmdet.registry import DATASETS, MODELS  # noqa: E402
from mmdet.utils import register_all_modules  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "tools/unified/configs/u1r2_generic_refinement_fp32.py")
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/user/reference_envs/u1_generic_adaptation_20261001/valid_work_bf16_batch2_frozen01_noacc/epoch_4.pth"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/unified/u1r2/gradient_audit.json")
    args = parser.parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(args.config))
    cfg.model.language_model.name = str((PROJECT_ROOT / "weights/bert-base-uncased").resolve())
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    loaded = load_checkpoint(model, str(args.checkpoint), map_location="cpu", strict=False)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    if not (0 <= args.index < len(dataset)):
        raise IndexError(args.index)
    batch = pseudo_collate([dataset[args.index]])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model.to(device).train()
    batch = model.data_preprocessor(batch, training=True)
    # The preprocessor returns device tensors only after the model is moved;
    # explicitly move data samples in case the installed MMEngine version does
    # not infer the target device from the module.
    if isinstance(batch.get("inputs"), torch.Tensor):
        batch["inputs"] = batch["inputs"].to(device)
    if "data_samples" in batch:
        batch["data_samples"] = [sample.to(device) for sample in batch["data_samples"]]
    losses = model(**batch, mode="loss")
    scalar_losses = []
    for name, value in losses.items():
        if torch.is_tensor(value):
            scalar_losses.append(value.float().mean())
        elif isinstance(value, (list, tuple)):
            scalar_losses.extend(item.float().mean() for item in value if torch.is_tensor(item))
    if not scalar_losses:
        raise RuntimeError(f"model returned no scalar losses: {losses.keys()}")
    total_loss = torch.stack(scalar_losses).sum()
    total_loss.backward()
    parameters = []
    nonfinite = []
    no_grad = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None:
            no_grad.append(name)
            continue
        finite = bool(torch.isfinite(grad).all())
        norm = float(torch.linalg.vector_norm(grad.float()).item())
        max_abs = float(grad.float().abs().max().item())
        row = {"name": name, "finite": finite, "norm": norm, "max_abs": max_abs}
        parameters.append(row)
        if not finite:
            nonfinite.append(row)
    payload = {
        "format": "locatemot-u-u1r2-fp32-gradient-audit-v1",
        "status": "PASS" if parameters and not nonfinite else "FAIL",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_load": {"missing_keys": loaded.get("missing_keys", []) if isinstance(loaded, dict) else [], "unexpected_keys": loaded.get("unexpected_keys", []) if isinstance(loaded, dict) else []},
        "device": str(device),
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "sample_index": args.index,
        "total_loss": float(total_loss.detach().cpu()),
        "trainable_parameter_count": len(parameters),
        "nonfinite_gradient_count": len(nonfinite),
        "no_gradient_parameter_count": len(no_grad),
        "nonfinite_gradients": nonfinite,
        "no_gradient_parameters": no_grad,
        "parameters": parameters,
        "official_test_labels_read": False,
        "screening_gt_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "device", "total_loss", "trainable_parameter_count", "nonfinite_gradient_count", "no_gradient_parameter_count")}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
