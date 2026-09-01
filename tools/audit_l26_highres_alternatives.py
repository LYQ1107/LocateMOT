#!/usr/bin/env python3
"""Record verifiable high-resolution cross-modal alternatives for L26."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
CACHE = Path("/home/lwr/.cache")
OUT = ROOT / "outputs/l26/fallback/F5_high_resolution_alternative"


def digest(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    dino = CACHE / "torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
    clip_dir = CACHE / "clip"
    clip_l14 = sorted(clip_dir.glob("*ViT-L-14*") if clip_dir.exists() else [])
    grounding = sorted(CACHE.rglob("*grounding*dino*.pth")) + sorted(CACHE.rglob("*GroundingDINO*.pth"))
    report = {
        "format": "locatemot-l26-high-resolution-alternative-audit-v1",
        "selection_or_fitting": False,
        "alternatives": {
            "dinov2_vitb14": {
                "status": "verified_and_used_by_v5",
                "checkpoint": str(dino),
                "sha256": digest(dino),
                "patch_grid": "16x16",
                "patch_stride": 14,
                "dimension": 768,
            },
            "clip_vitl14": {
                "status": "available" if clip_l14 else "unavailable_local",
                "files": [str(x) for x in clip_l14],
                "cross_modal_projection_verified": False,
            },
            "groundingdino_region_feature": {
                "status": "available" if grounding else "unavailable_local",
                "files": [str(x) for x in grounding],
                "region_feature_extraction_verified": False,
            },
        },
        "decision": "v5 DINOv2 dense features retained; no unavailable weight was substituted or fabricated",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "audit.md").write_text(
        "# L26 F5 high-resolution alternatives\n\n"
        "DINOv2 ViT-B/14 is the only verified dense backbone used by v5.\n"
        "Unavailable CLIP ViT-L/14 or GroundingDINO weights are not represented as results.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
