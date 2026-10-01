"""Shared MMDetection GroundingDINO inference contract for LocateMOT-U.

The wrapper deliberately delegates image loading, resize, packing and
``DetDataPreprocessor`` handling to MMDetection's registered test pipeline.
Callers pass a path (not a hand-built tensor) so the returned boxes are in the
original image coordinate system used by the evaluator.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MMDET_ROOT = PROJECT_ROOT / "third_party/mmdetection"
for _path in (PROJECT_ROOT, MMDET_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from mmcv.transforms import Compose  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmdet.apis import inference_detector  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
from mmdet.utils import register_all_modules  # noqa: E402

from locatemot.unified.data.legal_scope import assert_legal_path  # noqa: E402


DEFAULT_CONFIG = MMDET_ROOT / (
    "configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
DEFAULT_BERT = PROJECT_ROOT / "weights/bert-base-uncased"
PREPROCESSING_CONTRACT = {
    "loader": "LoadImageFromFile",
    "resize": {"type": "FixScaleResize", "scale": [800, 1333], "keep_ratio": True},
    "pack": "PackDetInputs",
    "data_preprocessor": "DetDataPreprocessor",
    "bgr_to_rgb": True,
    "box_coordinate_contract": "model postprocess restores boxes to ori_shape",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GroundingRuntime:
    model: torch.nn.Module
    config: Config
    test_pipeline: Compose
    checkpoint: Path
    checkpoint_sha256: str
    device: torch.device
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def load_grounding_runtime(
    *,
    config_path: Path = DEFAULT_CONFIG,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    bert_path: Path = DEFAULT_BERT,
    device: str | torch.device | None = None,
    max_per_img: int | None = None,
    training_contract: bool = False,
) -> GroundingRuntime:
    """Build the checkpoint-compatible Swin-B runtime and official pipeline.

    The released MMDetection checkpoint predates two inference-irrelevant
    configuration parameters: contrastive ``log_scale`` and the denoising
    label embedding.  ``log_scale=None`` removes eight uninitialized contrastive
    parameters.  MMDetection's DINO implementation requires a denoising
    generator even for evaluation, so its single missing embedding is retained
    and explicitly audited as training-only.
    """
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    cfg.model.language_model.name = str(Path(bert_path).resolve())
    cfg.model.bbox_head.contrastive_cfg.log_scale = None
    if max_per_img is not None:
        cfg.model.test_cfg.max_per_img = int(max_per_img)
    model = MODELS.build(cfg.model)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = loaded.get("state_dict", loaded.get("model", loaded))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    runtime_device = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model.cfg = cfg
    model.to(runtime_device).eval()
    test_pipeline = Compose(cfg.test_pipeline)
    return GroundingRuntime(
        model=model,
        config=cfg,
        test_pipeline=test_pipeline,
        checkpoint=Path(checkpoint),
        checkpoint_sha256=sha256_file(Path(checkpoint)),
        device=runtime_device,
        missing_keys=tuple(sorted(str(key) for key in missing)),
        unexpected_keys=tuple(sorted(str(key) for key in unexpected)),
    )


def run_grounding(
    runtime: GroundingRuntime,
    image_path: str | Path,
    prompt: str,
    *,
    custom_entities: bool = True,
) -> Any:
    """Run one image through the official MMDetection test pipeline."""
    path = assert_legal_path(Path(image_path).resolve())
    with torch.inference_mode():
        return inference_detector(
            runtime.model,
            str(path),
            test_pipeline=runtime.test_pipeline,
            text_prompt=str(prompt),
            custom_entities=bool(custom_entities),
        )


def prediction_arrays(result: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract finite-safe box/score tensors without changing coordinates."""
    predictions = result.pred_instances
    boxes = predictions.bboxes.tensor if hasattr(predictions.bboxes, "tensor") else predictions.bboxes
    boxes = boxes.detach().float().cpu()
    scores = predictions.scores.detach().float().cpu()
    return boxes, scores


def runtime_metadata(runtime: GroundingRuntime, *, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return {
        "config": str(Path(config_path).resolve().relative_to(PROJECT_ROOT)),
        "checkpoint": str(runtime.checkpoint.resolve().relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "device": str(runtime.device),
        "missing_keys": list(runtime.missing_keys),
        "unexpected_keys": list(runtime.unexpected_keys),
        "preprocessing_contract": PREPROCESSING_CONTRACT,
    }


__all__ = [
    "DEFAULT_BERT",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_CONFIG",
    "GroundingRuntime",
    "PREPROCESSING_CONTRACT",
    "load_grounding_runtime",
    "prediction_arrays",
    "run_grounding",
    "runtime_metadata",
]
