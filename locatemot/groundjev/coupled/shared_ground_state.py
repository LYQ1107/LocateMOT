"""Causal state produced by one GroundAnything frame encoding.

The state is deliberately explicit about frame and model provenance.  A
region feature from another frame or checkpoint must fail validation instead
of silently entering identity matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch


@dataclass
class SharedCandidate:
    index: int
    box: tuple[float, float, float, float]
    label: str
    region_token: torch.Tensor
    local_token: Optional[torch.Tensor] = None
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def detached(self) -> "SharedCandidate":
        return SharedCandidate(
            index=self.index,
            box=tuple(float(v) for v in self.box),
            label=self.label,
            region_token=self.region_token.detach().to("cpu").clone(),
            local_token=None if self.local_token is None else self.local_token.detach().to("cpu").clone(),
            score=self.score,
            metadata=dict(self.metadata),
        )


@dataclass
class SharedGroundState:
    frame_id: int
    image_key: str
    model_sha256: str
    boxes: torch.Tensor
    labels: list[str]
    vision_tokens: torch.Tensor
    contextual_tokens: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    spatial_layout: dict[str, Any]
    task_query: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    candidates: list[SharedCandidate] = field(default_factory=list)

    def validate(self) -> None:
        if self.boxes.ndim != 2 or self.boxes.shape[-1] != 4:
            raise ValueError(f"boxes must have shape [N,4], got {tuple(self.boxes.shape)}")
        if len(self.labels) != self.boxes.shape[0]:
            raise ValueError("labels and boxes have different lengths")
        if self.vision_tokens.ndim != 2 or self.vision_tokens.shape[0] != self.boxes.shape[0]:
            raise ValueError("vision_tokens must be one region token per candidate")
        if len(self.candidates) != self.boxes.shape[0]:
            raise ValueError("candidates and boxes have different lengths")
        for i, candidate in enumerate(self.candidates):
            if candidate.index != i:
                raise ValueError("candidate indices must be contiguous and frame-local")
            if candidate.region_token.ndim != 1:
                raise ValueError("candidate region tokens must be vectors")
            if candidate.region_token.shape[0] != self.vision_tokens.shape[1]:
                raise ValueError("candidate region token dimension disagrees with state")
            if candidate.metadata.get("frame_id", self.frame_id) != self.frame_id:
                raise ValueError("cross-frame candidate feature mixed into state")
            if candidate.metadata.get("model_sha256", self.model_sha256) != self.model_sha256:
                raise ValueError("candidate feature came from another model checkpoint")

    @property
    def feature_dim(self) -> int:
        return int(self.vision_tokens.shape[-1])

    @classmethod
    def from_region_tokens(
        cls,
        *,
        frame_id: int,
        image_key: str,
        model_sha256: str,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        region_tokens: torch.Tensor,
        local_tokens: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        spatial_layout: Optional[dict[str, Any]] = None,
        task_query: Optional[str] = None,
        scores: Optional[Sequence[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "SharedGroundState":
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=region_tokens.device)
        if box_tensor.numel() == 0:
            box_tensor = box_tensor.reshape(0, 4)
        if region_tokens.ndim != 2 or region_tokens.shape[0] != box_tensor.shape[0]:
            raise ValueError("region_tokens must have shape [number_of_boxes, feature_dim]")
        local = None if local_tokens is None else local_tokens
        candidates: list[SharedCandidate] = []
        for i, (box, label) in enumerate(zip(box_tensor.detach().cpu().tolist(), labels)):
            candidates.append(SharedCandidate(
                index=i,
                box=tuple(float(v) for v in box),
                label=str(label),
                region_token=region_tokens[i],
                local_token=None if local is None else local[i],
                score=None if scores is None else float(scores[i]),
                metadata={"frame_id": int(frame_id), "model_sha256": str(model_sha256)},
            ))
        state = cls(
            frame_id=int(frame_id), image_key=str(image_key), model_sha256=str(model_sha256),
            boxes=box_tensor, labels=[str(x) for x in labels], vision_tokens=region_tokens,
            contextual_tokens=None, image_grid_thw=image_grid_thw,
            spatial_layout=dict(spatial_layout or {}), task_query=task_query,
            metadata=dict(metadata or {}), candidates=candidates,
        )
        state.validate()
        return state

    def memory_snapshot(self) -> list[SharedCandidate]:
        """Return detached finite state suitable for persistent memory."""
        return [candidate.detached() for candidate in self.candidates]
