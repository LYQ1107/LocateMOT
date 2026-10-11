"""Convert frozen U1 perception outputs into task-independent candidates."""
from __future__ import annotations

from typing import Any

import torch


class PerceptionBridge:
    """Keep query-independent identity visual features separate from semantic features."""

    def __init__(self, hidden_dim: int = 256) -> None:
        self.hidden_dim = int(hidden_dim)

    def __call__(
        self,
        *,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        id_visual: torch.Tensor,
        semantic_visual: torch.Tensor,
        decoder_query: torch.Tensor | None = None,
        source: str = "u1",
    ) -> dict[str, torch.Tensor | str]:
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("boxes must be [N,4] xyxy")
        n = boxes.shape[0]
        if scores.shape != (n,):
            raise ValueError("scores must be [N]")
        for name, value in (("id_visual", id_visual), ("semantic_visual", semantic_visual)):
            if value.shape != (n, self.hidden_dim):
                raise ValueError(f"{name} must be [N,{self.hidden_dim}]")
        if decoder_query is None:
            decoder_query = semantic_visual
        if decoder_query.shape != (n, self.hidden_dim):
            raise ValueError(f"decoder_query must be [N,{self.hidden_dim}]")
        return {
            "boxes": boxes,
            "scores": scores,
            "id_visual": id_visual,
            "semantic_visual": semantic_visual,
            "decoder_query": decoder_query,
            "source": source,
        }

    @staticmethod
    def sample_id_visual(feature_map: torch.Tensor | tuple[torch.Tensor, ...], boxes: torch.Tensor) -> torch.Tensor:
        """Grid-sample query-independent multi-scale maps at box centers.

        Each level is sampled independently and averaged after ROI sampling,
        preserving the original visual channel dimension while avoiding any
        text-conditioned decoder query in the identity feature.
        """
        if isinstance(feature_map, (tuple, list)):
            if not feature_map:
                raise ValueError("feature_map sequence cannot be empty")
            samples = [PerceptionBridge.sample_id_visual(level, boxes) for level in feature_map]
            if len({sample.shape[-1] for sample in samples}) != 1:
                raise ValueError("multi-scale identity maps must share channel dimension")
            return torch.stack(samples, dim=0).mean(dim=0)
        if feature_map.ndim == 3:
            feature_map = feature_map.unsqueeze(0)
        if feature_map.ndim != 4 or boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("feature_map must be [1,C,H,W] and boxes [N,4]")
        if feature_map.shape[0] != 1:
            raise ValueError("sample_id_visual expects one frame")
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        points = centers.mul(2.0).sub(1.0).view(1, -1, 1, 2)
        sampled = torch.nn.functional.grid_sample(
            feature_map, points, mode="bilinear", align_corners=False, padding_mode="border"
        )
        return sampled[0, :, :, 0].transpose(0, 1)

    @classmethod
    def from_feature_maps(
        cls,
        *,
        visual_map: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        semantic_visual: torch.Tensor,
        decoder_query: torch.Tensor | None = None,
        source: str = "u1",
    ) -> dict[str, torch.Tensor | str]:
        id_visual = cls.sample_id_visual(visual_map, boxes)
        bridge = cls(hidden_dim=id_visual.shape[-1])
        return bridge(
            boxes=boxes,
            scores=scores,
            id_visual=id_visual,
            semantic_visual=semantic_visual,
            decoder_query=decoder_query,
            source=source,
        )
