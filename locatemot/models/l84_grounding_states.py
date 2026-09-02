"""L84-selected GroundingDINO decoder state extraction.

This module contains no detector construction and no candidate selection.  It
only exposes the fixed-reference Z states and the native iterative-refinement
R states from the already audited local GroundingDINO runtime.  L69 row
identity, order, boxes, and count are never changed.
"""
from __future__ import annotations

from typing import Any

import torch


SELECTED_STAGE_NAMES = ("Z0", "Z1", "Z4", "Z6", "R1", "R4", "R6")


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.float()).all()):
        raise FloatingPointError(f"nonfinite {name}")


def capture_l84_states(
    model: Any,
    event: dict[str, Any],
    boxes: torch.Tensor,
    image_shape: tuple[int, int] | list[int],
    scale_factor: Any,
    *,
    no_refpe_in_content: bool = False,
    selected_name: str | None = None,
) -> dict[str, Any]:
    """Return the seven registered states for one complete candidate row set.

    ``no_refpe_in_content`` is the sole L84 structural variant.  Reference
    points and decoder query positional encoding remain unchanged; only the
    content seed changes from ``visual_seed + reference_position`` to
    ``visual_seed``.
    """
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding
    from tools.l82_audit_grounding_interface import fixed_reference_decoder
    from locatemot.models.l82_grounding_reference import (
        boxes_to_reference_points,
        boxes_xyxy_to_normalized,
        candidate_seed_with_reference,
        pool_memory_by_box,
    )

    memory = event["memory"]
    if memory.ndim != 3 or tuple(memory.shape[:1]) != (1,) or int(memory.shape[-1]) != 256:
        raise AssertionError(f"unexpected encoder memory: {tuple(memory.shape)}")
    if boxes.ndim != 2 or tuple(boxes.shape[-1:]) != (4,):
        raise AssertionError(f"unexpected candidate boxes: {tuple(boxes.shape)}")
    count = int(boxes.shape[0])
    boxes_norm = boxes_xyxy_to_normalized(boxes.to(memory.device), image_shape, scale_factor)
    references = boxes_to_reference_points(boxes_norm)
    visual_seed, roi_audit = pool_memory_by_box(
        memory, event["spatial_shapes"], event["level_start_index"], boxes_norm,
        event["memory_mask"], grid_size=4,
    )
    seed_with_refpe, reference_position = candidate_seed_with_reference(
        visual_seed, references, model.decoder.ref_point_head, coordinate_to_encoding,
    )
    content_seed = visual_seed if no_refpe_in_content else seed_with_refpe
    fixed_hidden = fixed_reference_decoder(model, content_seed, references, event)
    native_result = model.decoder(
        query=content_seed.unsqueeze(0), value=event["memory"],
        key_padding_mask=event["memory_mask"], self_attn_mask=None,
        reference_points=references.unsqueeze(0),
        spatial_shapes=event["spatial_shapes"],
        level_start_index=event["level_start_index"],
        valid_ratios=event["valid_ratios"],
        reg_branches=model.bbox_head.reg_branches,
        memory_text=event["memory_text"],
        text_attention_mask=event["text_attention_mask"],
    )
    if not isinstance(native_result, (tuple, list)) or len(native_result) != 2:
        raise AssertionError("native decoder did not return (hidden,references)")
    native_hidden, native_references = native_result
    if fixed_hidden.ndim != 4 or native_hidden.ndim != 4:
        raise AssertionError("decoder hidden orientation drift")
    if tuple(fixed_hidden.shape[1:2]) != (1,) or tuple(native_hidden.shape[1:2]) != (1,):
        raise AssertionError("decoder hidden batch orientation drift")
    if int(fixed_hidden.shape[2]) != count or int(native_hidden.shape[2]) != count:
        raise AssertionError("decoder changed candidate row count")

    fixed_by_name = {
        "Z0": visual_seed.float().detach().clone(),
        "Z1": fixed_hidden[0, 0].float().detach().clone(),
        "Z4": fixed_hidden[3, 0].float().detach().clone(),
        "Z6": fixed_hidden[5, 0].float().detach().clone(),
    }
    native_by_name = {
        "R1": native_hidden[0, 0].float().detach().clone(),
        "R4": native_hidden[3, 0].float().detach().clone(),
        "R6": native_hidden[5, 0].float().detach().clone(),
    }
    states = {**fixed_by_name, **native_by_name}
    for name, value in states.items():
        if value.shape != (count, 256):
            raise AssertionError(f"{name} state shape drift: {tuple(value.shape)}")
        _finite(value, name)
    _finite(reference_position, "reference positional encoding")
    _finite(native_references, "native references")
    if selected_name is not None:
        if selected_name not in states:
            raise KeyError(f"unknown selected state {selected_name}")
        states = {f"{selected_name}_no_refpe" if no_refpe_in_content else selected_name: states[selected_name]}
    return {
        "states": states,
        "all_registered_states": {key: value for key, value in {**fixed_by_name, **native_by_name}.items()},
        "boxes_norm": boxes_norm.float().detach().clone(),
        "references": references.float().detach().clone(),
        "visual_seed": visual_seed.float().detach().clone(),
        "reference_position": reference_position.float().detach().clone(),
        "native_references": native_references.float().detach().clone(),
        "fixed_hidden_shape": list(fixed_hidden.shape),
        "native_hidden_shape": list(native_hidden.shape),
        "native_reference_shape": list(native_references.shape),
        "candidate_count": count,
        "content_seed": "visual_seed" if no_refpe_in_content else "visual_seed_plus_reference_position",
        "reference_position_route": "decoder_query_pos_and_reference_points",
        "roi_audit": roi_audit,
        "all_rows_retained": True,
        "candidate_deletion": False,
        "candidate_truncation": False,
    }


def no_refpe_state_name(selected_name: str) -> str:
    if selected_name not in SELECTED_STAGE_NAMES or selected_name == "Z0":
        raise ValueError(f"no-refPE only applies to a decoder stage, got {selected_name}")
    return f"{selected_name}_no_refpe"


__all__ = ["SELECTED_STAGE_NAMES", "capture_l84_states", "no_refpe_state_name"]
