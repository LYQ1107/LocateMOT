"""Label-free U2 features aligned with the native GroundingDINO candidates.

The detector still owns preprocessing, proposals, vocabulary replay and NMS.
Only query provenance and features are attached before its native chunk concat.
Identity features come from the pre-text neck maps, sampled in padded image
coordinates. Semantic features come from the selected final decoder queries.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch

from .grounding_inference import GroundingRuntime, run_grounding, run_grounding_chunked
from ..models.u2.perception_bridge import PerceptionBridge


class _FeatureCapture:
    def __init__(self) -> None:
        self.visual_maps: tuple[torch.Tensor, ...] = ()
        self.decoder: torch.Tensor | None = None
        self.backbone_calls = 0
        self.chunk_index = -1

    def backbone(self, module, inputs, output) -> None:
        self.backbone_calls += 1

    def neck(self, module, inputs, output) -> None:
        # Keep an independent pre-text snapshot even if a future detector
        # implementation modifies visual tensors inside its fusion encoder.
        self.visual_maps = tuple(level.detach().clone() for level in output)

    def head(self, module, inputs, output) -> None:
        hidden_states = inputs[0]
        if hidden_states.shape[1] != 1:
            raise ValueError("U2 candidate capture operates on one current frame")
        self.decoder = hidden_states[-1, 0].detach()
        self.chunk_index += 1


@contextmanager
def _capture_features(runtime: GroundingRuntime) -> Iterator[_FeatureCapture]:
    from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

    model = runtime.model
    if model.training:
        raise ValueError("U2 candidate features require the frozen evaluation runtime")
    capture = _FeatureCapture()
    head = model.bbox_head
    method_name = "_predict_by_feat_single"
    original = getattr(head, method_name)
    had_override = method_name in head.__dict__

    def predict_with_features(self, cls_score, bbox_pred, token_positive_maps, img_meta, rescale=True):
        predictions = original(cls_score, bbox_pred, token_positive_maps, img_meta, rescale)
        k = self.test_cfg.get("max_per_img", len(cls_score))
        # Reproduce the native query selection exactly, including repeated
        # queries selected for different predicted categories. Box matching
        # would lose that provenance for coincident/repeated proposals.
        if token_positive_maps is not None:
            class_scores = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None], positive_maps=[token_positive_maps])[0]
            _, flat_indices = class_scores.reshape(-1).topk(k)
            query_indices = flat_indices // class_scores.shape[-1]
        else:
            _, query_indices = cls_score.sigmoid().max(-1).values.topk(k)
        if capture.decoder is None:
            raise RuntimeError("The native head did not expose its decoder features")
        predictions.decoder_query = capture.decoder[query_indices]
        predictions.semantic_visual = predictions.decoder_query
        predictions.query_index = query_indices
        predictions.chunk_index = torch.full_like(query_indices, capture.chunk_index)
        return predictions

    hooks = [model.backbone.register_forward_hook(capture.backbone),
             model.neck.register_forward_hook(capture.neck),
             head.register_forward_hook(capture.head)]
    setattr(head, method_name, MethodType(predict_with_features, head))
    try:
        yield capture
    finally:
        if had_override:
            setattr(head, method_name, original)
        else:
            delattr(head, method_name)
        for hook in hooks:
            hook.remove()


def _identity_features(capture: _FeatureCapture, result: Any, boxes: torch.Tensor) -> torch.Tensor:
    if capture.backbone_calls != 1 or not capture.visual_maps:
        raise RuntimeError("Candidates must use exactly one pre-text visual extraction")
    # Neck maps include the data preprocessor's right/bottom padding. Original
    # image normalization alone shifts samples near image edges into padding.
    pad_h, pad_w = result.metainfo["batch_input_shape"]
    scale = boxes.new_tensor(result.metainfo["scale_factor"]).repeat(2)
    normalizer = boxes.new_tensor([pad_w, pad_h, pad_w, pad_h])
    normalized_boxes = boxes * scale / normalizer
    return PerceptionBridge.sample_id_visual(capture.visual_maps, normalized_boxes)


def run_u2_candidates(
    runtime: GroundingRuntime,
    image_path,
    vocabulary: str | list[str],
    *,
    max_candidates: int = 150,
    chunked_size: int = 32,
    merge_iou: float = 0.8,
) -> dict[str, Any]:
    """Extract one label-free frame; this function never opens annotations.

    Dataset callers must supply their already frozen vocabulary and candidate
    cap. Full cache generation remains conditional on the U2-P selection gate.
    """
    with _capture_features(runtime) as capture:
        if isinstance(vocabulary, list):
            result = run_grounding_chunked(runtime, image_path, vocabulary,
                                           chunked_size=chunked_size, merge_iou=merge_iou,
                                           top_k=max_candidates)
        else:
            result = run_grounding(runtime, image_path, vocabulary, custom_entities=True)
            predictions = result.pred_instances
            # Match the ordinary portability audit's registered score sort.
            keep = torch.argsort(predictions.scores, descending=True)[:max_candidates]
            result.pred_instances = predictions[keep]
        predictions = result.pred_instances
        boxes = predictions.bboxes
        if hasattr(boxes, "tensor"):
            boxes = boxes.tensor
        identity = _identity_features(capture, result, boxes)
        tensors = {"boxes": boxes, "scores": predictions.scores,
                   "id_visual": identity, "semantic_visual": predictions.semantic_visual,
                   "decoder_query": predictions.decoder_query,
                   "predicted_class": predictions.labels,
                   "query_index": predictions.query_index, "chunk_index": predictions.chunk_index}
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise RuntimeError("Nonfinite U2 candidate tensor")
        return {**{name: value.detach() for name, value in tensors.items()},
                "source": "frozen_groundingdino",
                "source_metadata": {"foundation_sha256": runtime.checkpoint_sha256,
                                    "identity_source": "pre_text_neck_multiscale_center_grid_mean",
                                    "identity_coordinate_system": "resized_xyxy_over_padded_batch_shape",
                                    "semantic_source": "final_decoder_selected_query",
                                    "backbone_calls": capture.backbone_calls,
                                    "chunk_count": capture.chunk_index + 1,
                                    "max_candidates": max_candidates,
                                    "vocabulary": vocabulary,
                                    "chunked_size": chunked_size if isinstance(vocabulary, list) else None,
                                    "merge_iou": merge_iou if isinstance(vocabulary, list) else None,
                                    "ori_shape": result.metainfo["ori_shape"],
                                    "img_shape": result.metainfo["img_shape"],
                                    "batch_input_shape": result.metainfo["batch_input_shape"],
                                    "scale_factor": result.metainfo["scale_factor"],
                                    "annotations_read": False}}


__all__ = ["run_u2_candidates"]
