"""MoonViT region feature extraction with explicit grid mapping."""
from __future__ import annotations

import math
from typing import List, Optional

import torch


class MoonViTRegionExtractor:
    """Mean-pools MoonViT raw tokens inside a normalized box.

    Mapping verified against official code:
    - image processor produces a pre-merge patch grid ``grid_hws`` (H/14, W/14);
    - MoonVitPretrainedModel applies a 2x2 patch merger, so the feature grid is
      (H/28, W/28) and each token has 4 * hidden_size channels.
    """

    def __init__(self, model):
        self.model = model
        self.merge_kernel = tuple(getattr(model.vision_model, "merge_kernel_size", (2, 2)))

    def extract(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws,
        image_index: int = 0,
        normalized_boxes: Optional[List[List[float]]] = None,
        vision_features: Optional[list] = None,
    ) -> List[dict]:
        if normalized_boxes is None:
            normalized_boxes = []
        raw = vision_features
        if raw is None:
            raw = self.model.extract_feature(pixel_values, image_grid_hws)
        if isinstance(raw, (list, tuple)):
            feat = raw[image_index]
        else:
            feat = raw
        grid_hw = image_grid_hws[image_index]
        h_feat = int(grid_hw[0] // self.merge_kernel[0])
        w_feat = int(grid_hw[1] // self.merge_kernel[1])
        expected = h_feat * w_feat
        if feat.shape[0] != expected:
            raise ValueError(
                f"feature token count {feat.shape[0]} != expected grid {expected} "
                f"(grid_hws={grid_hw.tolist()}, merge={self.merge_kernel})"
            )
        feat2d = feat.reshape(h_feat, w_feat, -1)
        results = []
        for box in normalized_boxes:
            if len(box) != 4:
                results.append(None)
                continue
            x1n, y1n, x2n, y2n = (float(v) / 1000 for v in box)
            c0 = max(0, min(w_feat - 1, int(x1n * w_feat)))
            c1 = max(0, min(w_feat, math.ceil(x2n * w_feat)))
            r0 = max(0, min(h_feat - 1, int(y1n * h_feat)))
            r1 = max(0, min(h_feat, math.ceil(y2n * h_feat)))
            region = feat2d[r0:r1, c0:c1]
            token_count = region.shape[0] * region.shape[1]
            if token_count == 0:
                results.append({
                    "region_feature": None,
                    "region_token_count": 0,
                    "feature_grid_shape": [h_feat, w_feat],
                    "box_in_feature_coordinates": [c0, r0, c1, r1],
                })
                continue
            results.append({
                "region_feature": region.reshape(-1, feat2d.shape[-1]).mean(dim=0),
                "region_token_count": token_count,
                "feature_grid_shape": [h_feat, w_feat],
                "box_in_feature_coordinates": [c0, r0, c1, r1],
            })
        return results
