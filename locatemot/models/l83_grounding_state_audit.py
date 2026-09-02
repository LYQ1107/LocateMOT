"""L83 decoder-state capture and distribution diagnostics.

The functions in this module are read-only wrappers around the already
audited local GroundingDINO runtime.  They expose the fixed candidate seed,
the seed after the pretrained reference positional encoding, every fixed-
reference decoder layer, and a separately labelled native-refinement control.
No candidate is selected, deleted, or re-boxed.
"""
from __future__ import annotations

from typing import Any

import torch


def capture_grounding_stages(
    model: Any,
    event: dict[str, Any],
    boxes: torch.Tensor,
    image_shape: tuple[int, int] | list[int],
    scale_factor: Any,
) -> dict[str, Any]:
    """Capture Z0/Zp/Z1..ZL plus a frozen native-refinement control."""
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding
    from tools.l82_audit_grounding_interface import fixed_reference_decoder
    from locatemot.models.l82_grounding_reference import (
        boxes_to_reference_points,
        boxes_xyxy_to_normalized,
        candidate_seed_with_reference,
        pool_memory_by_box,
    )

    memory = event["memory"]
    if memory.ndim != 3 or int(memory.shape[0]) != 1:
        raise AssertionError(f"unexpected encoder memory shape: {tuple(memory.shape)}")
    boxes_norm = boxes_xyxy_to_normalized(boxes.to(memory.device), image_shape, scale_factor)
    if boxes_norm.ndim != 2 or boxes_norm.shape[-1] != 4:
        raise AssertionError("candidate normalized box shape drift")
    visual_seed, roi_audit = pool_memory_by_box(
        memory,
        event["spatial_shapes"],
        event["level_start_index"],
        boxes_norm,
        event["memory_mask"],
        grid_size=4,
    )
    references = boxes_to_reference_points(boxes_norm)
    seed, reference_position = candidate_seed_with_reference(
        visual_seed,
        references,
        model.decoder.ref_point_head,
        coordinate_to_encoding,
    )
    fixed_hidden = fixed_reference_decoder(model, seed, references, event)
    if fixed_hidden.ndim != 4 or fixed_hidden.shape[1] != 1:
        raise AssertionError(f"fixed decoder layer shape drift: {tuple(fixed_hidden.shape)}")

    # This is diagnostic only.  It uses the native regression branches inside
    # the frozen decoder, but its refined references never replace L69 rows.
    native_result = model.decoder(
        query=seed.unsqueeze(0),
        value=event["memory"],
        key_padding_mask=event["memory_mask"],
        self_attn_mask=None,
        reference_points=references.unsqueeze(0),
        spatial_shapes=event["spatial_shapes"],
        level_start_index=event["level_start_index"],
        valid_ratios=event["valid_ratios"],
        reg_branches=model.bbox_head.reg_branches,
        memory_text=event["memory_text"],
        text_attention_mask=event["text_attention_mask"],
    )
    if not isinstance(native_result, (tuple, list)) or len(native_result) != 2:
        raise AssertionError("native decoder did not return states and references")
    native_hidden, native_references = native_result
    if native_hidden.ndim != 4 or native_hidden.shape[1] != 1:
        raise AssertionError(f"native decoder layer shape drift: {tuple(native_hidden.shape)}")
    for name, value in {
        "visual_seed": visual_seed,
        "reference_position": reference_position,
        "seed": seed,
        "fixed_hidden": fixed_hidden,
        "native_hidden": native_hidden,
        "native_references": native_references,
    }.items():
        if not bool(torch.isfinite(value.float()).all()):
            raise FloatingPointError(f"nonfinite decoder audit tensor: {name}")

    stages: dict[str, torch.Tensor] = {
        "Z0": visual_seed.float().detach().clone(),
        "Zp": seed.float().detach().clone(),
    }
    for layer_index in range(int(fixed_hidden.shape[0])):
        stages[f"Z{layer_index + 1}"] = fixed_hidden[layer_index, 0].float().detach().clone()
    native_stages = {
        f"R{layer_index + 1}": native_hidden[layer_index, 0].float().detach().clone()
        for layer_index in range(int(native_hidden.shape[0]))
    }
    native_stages["R0"] = seed.float().detach().clone()
    if any(value.shape != (int(boxes.shape[0]), int(memory.shape[-1])) for value in stages.values()):
        raise AssertionError("decoder audit changed candidate row count or dimension")
    return {
        "stages": stages,
        "native_stages": native_stages,
        "native_references": native_references.float().detach().clone(),
        "boxes_norm": boxes_norm.float().detach().clone(),
        "references": references.float().detach().clone(),
        "reference_position": reference_position.float().detach().clone(),
        "roi_audit": roi_audit,
        "fixed_hidden_shape": list(fixed_hidden.shape),
        "native_hidden_shape": list(native_hidden.shape),
        "native_reference_shape": list(native_references.shape),
        "candidate_count": int(boxes.shape[0]),
        "all_rows_retained": True,
        "candidate_deletion": False,
        "candidate_truncation": False,
    }


def compare_state_vectors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Compact finite difference summary for two same-key state tensors."""
    if left.shape != right.shape:
        raise ValueError(f"state shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    difference = left.float() - right.float()
    row_norm = difference.norm(dim=-1)
    return {
        "shape": list(left.shape),
        "mean_l2": float(row_norm.mean()),
        "p50_l2": float(row_norm.median()),
        "max_l2": float(row_norm.max()),
        "nonzero_fraction": float((row_norm > 1e-6).float().mean()),
        "finite": bool(torch.isfinite(difference).all()),
    }


__all__ = ["capture_grounding_stages", "compare_state_vectors"]
