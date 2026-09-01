"""Efficient adaptive ROIAlign drop-in for torchvision 0.16.

torchvision 0.16's pure-Python roi_align with sampling_ratio=0 (adaptive
sampling) materialises [K, C, PH, PW, H, W] interpolation tensors, which
explodes memory for normal feature maps (e.g. 216x256x7x7x100x136 -> ~274
GiB).  This module implements the same adaptive-sampling math (the
standard ROIAlign CUDA kernel algorithm) with per-ROI grids, so results
are numerically equivalent while memory stays small.

Usage: import patch_adaptive_roi_align  (patches torchvision.ops.roi_align)
"""
from __future__ import annotations

import torch


def _adaptive_roi_align(input, rois, output_size, spatial_scale,
                        sampling_ratio, aligned):
    if torch.is_autocast_enabled() and input.is_cuda \
            and input.dtype != torch.double:
        input = input.float()
        rois = rois.float()
    orig_dtype = input.dtype
    N, C, H, W = input.shape
    PH, PW = output_size
    K = rois.shape[0]
    if K == 0:
        return torch.zeros((0, C, PH, PW), dtype=input.dtype,
                           device=input.device)
    batch = rois[:, 0].int()
    offset = 0.5 if aligned else 0.0
    xs1 = rois[:, 1] * spatial_scale - offset
    ys1 = rois[:, 2] * spatial_scale - offset
    xs2 = rois[:, 3] * spatial_scale - offset
    ys2 = rois[:, 4] * spatial_scale - offset
    if not aligned:
        ws = torch.clamp(xs2 - xs1, min=1.0)
        hs = torch.clamp(ys2 - ys1, min=1.0)
    else:
        ws = xs2 - xs1
        hs = ys2 - ys1
    bin_h = hs / PH
    bin_w = ws / PW
    grid_h = torch.ceil(hs / PH).int()
    grid_w = torch.ceil(ws / PW).int()
    out = torch.zeros((K, C, PH, PW), dtype=input.dtype, device=input.device)
    feat_flat = input.reshape(N, C, H * W)
    ph = torch.arange(PH, device=input.device)
    pw = torch.arange(PW, device=input.device)
    for k in range(K):
        gh = max(int(grid_h[k]), 1)
        gw = max(int(grid_w[k]), 1)
        iy = torch.arange(gh, device=input.device)
        ix = torch.arange(gw, device=input.device)
        y = (ys1[k] + ph[:, None] * bin_h[k]
             + (iy[None, :] + 0.5) * (bin_h[k] / gh))  # [PH, gh]
        x = (xs1[k] + pw[:, None] * bin_w[k]
             + (ix[None, :] + 0.5) * (bin_w[k] / gw))  # [PW, gw]
        yl = torch.clamp(torch.floor(y).long(), 0, H - 1)
        xl = torch.clamp(torch.floor(x).long(), 0, W - 1)
        yh = torch.clamp(yl + 1, 0, H - 1)
        xh = torch.clamp(xl + 1, 0, W - 1)
        ly = (y - yl.float()).clamp(min=0)
        lx = (x - xl.float()).clamp(min=0)
        hy = 1.0 - ly
        hx = 1.0 - lx
        # indices [PH, gh, PW, gw]
        i11 = yl[:, :, None, None] * W + xl[None, None, :, :]
        i12 = yl[:, :, None, None] * W + xh[None, None, :, :]
        i21 = yh[:, :, None, None] * W + xl[None, None, :, :]
        i22 = yh[:, :, None, None] * W + xh[None, None, :, :]
        w11 = hy[:, :, None, None] * hx[None, None, :, :]
        w12 = hy[:, :, None, None] * (1.0 - hx[None, None, :, :])
        w21 = (1.0 - hy[:, :, None, None]) * hx[None, None, :, :]
        w22 = (1.0 - hy[:, :, None, None]) * (1.0 - hx[None, None, :, :])
        f = feat_flat[batch[k]]  # [C, H*W]
        v = (f[:, i11] * w11[None] + f[:, i12] * w12[None]
             + f[:, i21] * w21[None] + f[:, i22] * w22[None])  # [C,PH,gh,PW,gw]
        out[k] = v.sum(dim=(2, 4)) / float(gh * gw)
    return out.to(orig_dtype)


_orig_roi_align = None


def _patched_roi_align(input, rois, output_size, spatial_scale=1.0,
                       sampling_ratio=0, aligned=True):
    if int(sampling_ratio) == 0 and rois.shape[0] > 0:
        return _adaptive_roi_align(input, rois, output_size, spatial_scale,
                                   int(sampling_ratio), aligned)
    return _orig_roi_align(input, rois, output_size, spatial_scale,
                           sampling_ratio, aligned)


def apply_patch():
    global _orig_roi_align
    import torchvision.ops as tv_ops
    if _orig_roi_align is None:
        _orig_roi_align = tv_ops.roi_align
    tv_ops.roi_align = _patched_roi_align
    return True


apply_patch()

