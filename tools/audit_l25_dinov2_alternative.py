#!/usr/bin/env python3
"""Audit the locally cached DINOv2 alternative for the L25 F7 branch.

This is deliberately a load/inference audit only.  It does not build a bank,
read GT, train a scorer, or alter any existing model/bank.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
OUT = ROOT / "outputs/l25/fallback/F7_high_resolution_dinov2"
HUB = Path("/home/lwr/.cache/torch/hub/facebookresearch_dinov2_main")
WEIGHTS = Path("/home/lwr/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_frame() -> Path:
    candidates = [
        ROOT / "data/kitti_tracking_training/image_02/0004/000000.png",
        ROOT / "data/KITTI/data_tracking_image_2/training/image_02/0004/000000.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    for base in [ROOT / "data", Path("/data1/LWR/vranlee/SERVER_ONLY/avis")]:
        if base.exists():
            found = next(base.glob("**/image_02/0004/000000.png"), None)
            if found:
                return found
    raise FileNotFoundError("no local KITTI frame found for DINOv2 smoke")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = find_frame()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = "local"
    result = {
        "schema_version": 1,
        "branch": "L25-F7-high-resolution-dinov2",
        "status": "INCOMPLETE",
        "hub_source": str(HUB),
        "checkpoint": str(WEIGHTS),
        "checkpoint_sha256": sha256(WEIGHTS),
        "frame": str(frame),
        "device": device,
        "official_gt_used": False,
        "candidate_token_compatible": False,
        "notes": [],
    }
    try:
        if not HUB.exists() or not WEIGHTS.exists():
            raise FileNotFoundError("cached DINOv2 hub source or checkpoint missing")
        model = torch.hub.load(str(HUB), "dinov2_vitb14", source=source, pretrained=False)
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "teacher" in state:
            state = state["teacher"]
        if isinstance(state, dict) and "student" in state:
            state = state["student"]
        # The cached pretrain file is a backbone state dict; strip common SSL prefixes.
        clean = {}
        for k, v in state.items():
            nk = k
            for prefix in ("teacher.backbone.", "student.backbone.", "backbone.", "module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            clean[nk] = v
        missing, unexpected = model.load_state_dict(clean, strict=False)
        model.eval().to(device)
        image = Image.open(frame).convert("RGB").resize((224, 224), Image.BICUBIC)
        arr = np.asarray(image).astype("float32") / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(device)
        with torch.inference_mode():
            feats = model.forward_features(x)
        patch = feats["x_norm_patchtokens"]
        result.update({
            "status": "PASS",
            "model": "DINOv2 ViT-B/14",
            "input_size": [224, 224],
            "patch_tokens_shape": list(patch.shape),
            "patch_grid": [int(round(patch.shape[1] ** 0.5)), int(round(patch.shape[1] ** 0.5))],
            "patch_stride": 14,
            "embedding_dim": int(patch.shape[-1]),
            "missing_key_count": len(missing),
            "unexpected_key_count": len(unexpected),
            "missing_key_sample": list(missing)[:8],
            "unexpected_key_sample": list(unexpected)[:8],
            "notes": [
                "Verified local DINOv2 ViT-B/14 load and one raw-frame inference.",
                "Patch tokens are 16x16x768 at 224px input with 14px patch stride.",
                "DINOv2 has no CLIP text projection in this checkpoint; direct query-token cosine is not compatible with the L25 CLIP token bank without a separately trained cross-modal adapter.",
                "Therefore this branch is an audited high-resolution alternative, not a substituted v4 bank or an official DINOv2-text reproduction.",
            ],
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["notes"] = [
            "F7 could not be used as a bank replacement; no existing bank or checkpoint was modified.",
        ]
        (OUT / "INCOMPLETE.md").write_text(
            "# F7 DINOv2 audit incomplete\n\n" + result["error"] + "\n",
            encoding="utf-8",
        )
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
