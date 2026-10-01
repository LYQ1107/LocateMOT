#!/usr/bin/env python3
"""Load the U1 MMDetection Swin-B asset and run one legal, label-free frame."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MMDET_ROOT = PROJECT_ROOT / "third_party/mmdetection"
for path in (PROJECT_ROOT, MMDET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from mmengine.config import Config  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
from mmdet.structures import DetDataSample  # noqa: E402
from mmdet.utils import register_all_modules  # noqa: E402

from locatemot.unified.data.legal_scope import assert_legal_path, assert_legal_video  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    register_all_modules(init_default_scope=True)
    video = assert_legal_video("0004", dataset="refer_kitti_v1")
    image_path = assert_legal_path(
        PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video / "000063.png"
    )
    checkpoint = PROJECT_ROOT / "weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
    bert_path = PROJECT_ROOT / "weights/bert-base-uncased"
    config_path = MMDET_ROOT / "configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py"

    # MMDetection's config expects OpenCV-style BGR input and its data
    # preprocessor performs the BGR->RGB conversion.  Keep that contract
    # explicit when reading with PIL so the smoke does not swap channels twice.
    image = np.asarray(Image.open(image_path).convert("RGB"))[..., ::-1].copy()
    sample = DetDataSample()
    sample.set_metainfo(
        dict(img_id=0, img_shape=image.shape[:2], ori_shape=image.shape[:2], scale_factor=(1.0, 1.0))
    )
    sample.text = "car . pedestrian . cyclist ."
    sample.custom_entities = True

    cfg = Config.fromfile(str(config_path))
    cfg.model.language_model.name = str(bert_path)
    model = MODELS.build(cfg.model)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = loaded.get("state_dict", loaded.get("model", loaded))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    inputs = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    with torch.no_grad():
        outputs = model.test_step({"inputs": [inputs], "data_samples": [sample]})
    predictions = outputs[0].pred_instances
    boxes = predictions.bboxes.tensor if hasattr(predictions.bboxes, "tensor") else predictions.bboxes
    payload = {
        "format": "locatemot-u-u1-foundation-smoke-v1",
        "status": "pass",
        "model": "MMDetection GroundingDINO Swin-B CogCoOR",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "device": str(device),
        "video": video,
        "image": str(image_path.relative_to(PROJECT_ROOT)),
        "image_shape": list(image.shape),
        "prediction_count": int(len(predictions)),
        "finite_boxes": bool(torch.isfinite(boxes).all()),
        "finite_scores": bool(torch.isfinite(predictions.scores).all()),
        "missing_checkpoint_keys": len(missing),
        "unexpected_checkpoint_keys": len(unexpected),
        "official_test_labels_read": False,
        "screening_gt_used": False,
    }
    out = PROJECT_ROOT / "outputs/unified/u1_foundation_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
