#!/usr/bin/env python3
"""Record local L52 model/dependency availability without downloading models."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
OUT = ROOT / "outputs/l52/audit/availability.json"


def sha256(path: Path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(url: str):
    try:
        value = subprocess.check_output(["git", "ls-remote", url, "HEAD"], text=True, timeout=20)
        return value.split()[0] if value.split() else None
    except Exception:
        return None


def package(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"installed": False, "origin": None, "version": None}
    version = None
    error = None
    try:
        module = __import__(name)
        version = getattr(module, "__version__", None)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {"installed": True, "origin": str(spec.origin), "version": version, "import_error": error}


def weight(path: str, role: str, usable: str, dims: str):
    p = Path(path)
    return {"path": str(p), "exists": p.is_file(), "size_bytes": p.stat().st_size if p.is_file() else None,
            "sha256": sha256(p), "role": role, "usable_for_l52": usable, "input_output": dims}


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    weights = [
        weight("/home/lwr/.cache/clip/ViT-B-16.pt", "OpenAI CLIP ViT-B/16", "actual reuse", "RGB image -> 512-D projected image; visual patch path -> 196x768 tokens"),
        weight("/home/lwr/.cache/clip/ViT-B-32.pt", "OpenAI CLIP ViT-B/32", "available but not selected", "RGB image -> 512-D projected image"),
        weight("/home/lwr/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth", "DINOv2 ViT-B/14", "visual-only audited alternative; no text projection", "RGB image -> 256x768 patch tokens"),
        weight("/data1/LWR/vranlee/SERVER_ONLY/avis/dinov3-main/weights/backbones/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth", "DINOv3 ConvNeXt-B", "not used: local visual checkpoint, no L52 text interface", "RGB image -> visual features (architecture-specific)"),
        weight("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth", "GroundingDINO Swin-T", "weight exists outside project, but package/config are not installed in L52 env; not reused", "image + text prompt -> open-set boxes/scores"),
        weight("/data1/LWR/vranlee/SERVER_ONLY/avis/samurai/sam2/checkpoints/sam2.1_hiera_tiny.pt", "SAM2.1 Hiera-T", "visual segmentation only; not cross-modal grounding; not reused", "image/video + prompt -> masks"),
        weight("/home/lwr/.cache/clip/open_clip_pytorch_model.bin", "OpenCLIP local artifact", "not selected: open_clip package absent and artifact architecture/provenance not verified", "unknown until matching config is verified"),
    ]
    repos = [
        {"name": "OpenAI CLIP", "url": "https://github.com/openai/CLIP", "paper_url": "https://arxiv.org/abs/2103.00020", "head": git_head("https://github.com/openai/CLIP.git"), "license": "MIT", "reuse": "actual local CLIP ViT-B/16 implementation/weights"},
        {"name": "DKGTrack", "url": "https://github.com/acyddl/DKGTrack", "paper_url": "https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.pdf", "head": git_head("https://github.com/acyddl/DKGTrack.git"), "license": "repository research-use notice", "reuse": "structure reference only; no local checkout/weight reused"},
        {"name": "FlexHook", "url": "https://github.com/buptLwz/FlexHook", "paper_url": "https://arxiv.org/abs/2503.10617", "head": git_head("https://github.com/buptLwz/FlexHook.git"), "license": "MIT (local mirror LICENSE)", "reuse": "structure reference only; local mirror not imported into L52"},
        {"name": "iKUN", "url": "https://github.com/dyhBUPT/iKUN", "paper_url": "https://arxiv.org/abs/2312.16245", "head": git_head("https://github.com/dyhBUPT/iKUN.git"), "license": "MIT", "reuse": "structure/reference only; no L52 tracker or checkpoint use"},
        {"name": "GroundingDINO", "url": "https://github.com/IDEA-Research/GroundingDINO", "paper_url": "https://arxiv.org/abs/2303.05499", "head": git_head("https://github.com/IDEA-Research/GroundingDINO.git"), "license": "Apache-2.0 repository license", "reuse": "not directly usable in current environment; no silent fallback"},
        {"name": "DINOv2", "url": "https://github.com/facebookresearch/dinov2", "paper_url": "https://arxiv.org/abs/2304.07193", "head": git_head("https://github.com/facebookresearch/dinov2.git"), "license": "Apache-2.0", "reuse": "previously verified visual-only alternative; no text projection"},
        {"name": "DINOv3", "url": "https://github.com/facebookresearch/dinov3", "paper_url": "https://arxiv.org/abs/2508.10104", "head": git_head("https://github.com/facebookresearch/dinov3.git"), "license": "Meta DINOv3 license", "reuse": "local ConvNeXt checkpoint recorded, not used for L52"},
        {"name": "SAM", "url": "https://github.com/facebookresearch/segment-anything", "paper_url": "https://arxiv.org/abs/2304.02643", "head": git_head("https://github.com/facebookresearch/segment-anything.git"), "license": "Apache-2.0", "reuse": "segmentation-only, not L52 grounding"},
        {"name": "OpenCLIP", "url": "https://github.com/mlfoundations/open_clip", "paper_url": "https://arxiv.org/abs/2212.07143", "head": git_head("https://github.com/mlfoundations/open_clip.git"), "license": "MIT", "reuse": "not selected: package/config match absent"},
    ]
    result = {
        "format": "locatemot-l52-availability-v1", "stage": "L52-A1",
        "project_root": str(ROOT), "cwd_verified": True, "download_attempted": False,
        "disk_free_bytes": shutil.disk_usage("/data1").free,
        "authorized_gpu_policy": "GPU0 only if needed; GPUs4-7 observed occupied; no parallel L52 jobs",
        "packages": {name: package(name) for name in ("torch", "torchvision", "transformers", "open_clip", "clip", "dinov2", "segment_anything", "groundingdino", "PIL", "cv2")},
        "weights": weights, "public_repositories": repos,
        "actual_l52_visual_reuse": {"name": "OpenAI CLIP ViT-B/16", "path": "/home/lwr/.cache/clip/ViT-B-16.pt", "sha256": sha256(Path("/home/lwr/.cache/clip/ViT-B-16.pt")), "mode": "streaming candidate crop and context visual patch tokens; no persistent cache"},
        "weak_baseline_limit": "CLIP is cross-modal, but L52 candidate/context patch extraction uses frozen visual tokens and expression tokens; no verified token-to-region annotation or static/motion mask exists.",
        "unavailable_or_not_reused": ["GroundingDINO package/config integration", "GLIP/WeDetect/LocateAnything verified L52 weight+API", "OpenCLIP matching package/config", "DINOv2/DINOv3 text projection", "SAM cross-modal grounding"],
        "screening_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(OUT), "clip_sha256": result["actual_l52_visual_reuse"]["sha256"], "download_attempted": False}, indent=2))


if __name__ == "__main__":
    main()
