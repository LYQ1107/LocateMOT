#!/usr/bin/env python3
"""Launch the U1-A MMDetection trainer with the registered BF16 workaround.

The installed MMCV CUDA deformable-attention kernel accepts FP16/FP32 but not
BF16.  During BF16 autocast, keep only that custom operation in FP32; the
autograd cast returns gradients to the BF16 caller.  This wrapper leaves the
MMDetection model, data pipeline, and optimizer configuration unchanged.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import torch
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction


_ORIGINAL_FORWARD = MultiScaleDeformableAttnFunction.forward


def _bf16_safe_forward(
    ctx,
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: torch.Tensor,
) -> torch.Tensor:
    if value.dtype == torch.bfloat16:
        return _ORIGINAL_FORWARD(
            ctx,
            value.float(),
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations.float(),
            attention_weights.float(),
            im2col_step,
        )
    return _ORIGINAL_FORWARD(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )


MultiScaleDeformableAttnFunction.forward = staticmethod(_bf16_safe_forward)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(
    str(PROJECT_ROOT / "third_party/mmdetection/tools/train.py"),
    run_name="__main__",
)
