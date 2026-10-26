"""Coordinate geometry and the future shared temporal-decoder interface."""
from dataclasses import dataclass
from typing import Any

import torch

from .hungarian_criterion import cxcywh_to_xyxy


def valid_ratio(metadata):
    r = metadata["resize_and_padding"]
    return (r["new_width"] / (r["new_width"] + r["pad_width"]),
            r["new_height"] / (r["new_height"] + r["pad_height"]))


def targets_on_grid(boxes, metadata, device):
    width, height = metadata["image_size"]
    ratio = valid_ratio(metadata)
    return torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4) * torch.tensor(
        [ratio[0] / width, ratio[1] / height] * 2, device=device)


def boxes_to_original(predicted, metadata):
    width, height = metadata["image_size"]
    ratio = valid_ratio(metadata)
    xyxy = cxcywh_to_xyxy(predicted).clamp_min(0)
    xyxy = xyxy * xyxy.new_tensor([width / ratio[0], height / ratio[1]] * 2)
    xyxy[..., 0::2].clamp_(0, width)
    xyxy[..., 1::2].clamp_(0, height)
    return xyxy


@dataclass
class ObjectState:
    frame_id: int
    candidate_boxes: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_visual_tokens: torch.Tensor
    candidate_geometry: torch.Tensor
    task_text_tokens: Any = None
    category_embeddings: Any = None
    historical_identity_tokens: Any = None
    causal_track_state: Any = None


def object_state(frame_id, prediction, metadata):
    return ObjectState(frame_id, boxes_to_original(prediction["boxes"][0], metadata),
                       prediction["logits"][0].sigmoid(), prediction["visual_tokens"][0],
                       prediction["boxes"][0])
