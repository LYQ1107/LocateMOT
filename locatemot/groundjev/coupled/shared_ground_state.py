"""Auditable state shared by GroundAnything and the structured JEV readout.

The state is deliberately a value object.  It binds every region token to one
image/frame and one model identity; consumers cannot silently mix features
from another frame or checkpoint.  ``contextual_tokens`` is optional because
the first Phase-A bridge can expose native GroundAnything visual tokens before
the full DLM language hidden-state hook is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class SharedGroundState:
    """One causal GroundAnything forward and its spatial evidence."""

    frame_id: int | str
    image_key: str
    model_sha256: str
    boxes: tuple[tuple[float, float, float, float], ...]
    vision_tokens: torch.Tensor
    region_tokens: torch.Tensor
    image_grid_thw: tuple[int, int, int]
    spatial_layout: tuple[int, int]
    image_size: tuple[int, int]
    task_query: str = ""
    contextual_tokens: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_key:
            raise ValueError("image_key must identify the current frame")
        if not self.model_sha256:
            raise ValueError("model_sha256 is required to bind shared evidence")
        if self.vision_tokens.ndim not in (2, 3, 4):
            raise ValueError("vision_tokens must be token or spatial tensor")
        if self.region_tokens.ndim != 2:
            raise ValueError("region_tokens must have shape [num_boxes, hidden]")
        if self.region_tokens.shape[0] != len(self.boxes):
            raise ValueError("region token count does not match boxes")
        if self.contextual_tokens is not None and self.contextual_tokens.ndim not in (2, 3):
            raise ValueError("contextual_tokens must have shape [tokens, hidden]")
        for name, tensor in (("vision_tokens", self.vision_tokens),
                             ("region_tokens", self.region_tokens),
                             ("contextual_tokens", self.contextual_tokens)):
            if tensor is not None and not torch.isfinite(tensor.detach()).all():
                raise ValueError(f"{name} contains non-finite values")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive")
        gh, gw = self.spatial_layout
        if gh <= 0 or gw <= 0:
            raise ValueError("spatial_layout must be positive")

    @property
    def feature_dim(self) -> int:
        return int(self.region_tokens.shape[-1])

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)

    @property
    def has_language_context(self) -> bool:
        return self.contextual_tokens is not None

    def detached(self) -> "SharedGroundState":
        """Return a serializable inference copy without breaking provenance."""
        return SharedGroundState(
            frame_id=self.frame_id,
            image_key=self.image_key,
            model_sha256=self.model_sha256,
            boxes=self.boxes,
            vision_tokens=self.vision_tokens.detach(),
            region_tokens=self.region_tokens.detach(),
            image_grid_thw=self.image_grid_thw,
            spatial_layout=self.spatial_layout,
            image_size=self.image_size,
            task_query=self.task_query,
            contextual_tokens=(self.contextual_tokens.detach()
                               if self.contextual_tokens is not None else None),
            metadata=dict(self.metadata),
        )

    def audit_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "image_key": self.image_key,
            "model_sha256": self.model_sha256,
            "num_boxes": self.num_boxes,
            "feature_dim": self.feature_dim,
            "vision_tokens_shape": list(self.vision_tokens.shape),
            "region_tokens_shape": list(self.region_tokens.shape),
            "contextual_tokens_shape": (list(self.contextual_tokens.shape)
                                        if self.contextual_tokens is not None else None),
            "image_grid_thw": list(self.image_grid_thw),
            "spatial_layout": list(self.spatial_layout),
            "image_size": list(self.image_size),
            "task_query": self.task_query,
            "metadata": dict(self.metadata),
        }
