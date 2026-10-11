"""Causal online state shared by MOT and OVMOT."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class UnifiedTrackState:
    """Persistent per-video state; no task specification is stored in identity fields."""

    track_ids: torch.Tensor
    id_memory: torch.Tensor
    identity_anchor: torch.Tensor
    appearance_ema: torch.Tensor
    semantic_memory: torch.Tensor
    boxes: torch.Tensor
    # Normalized [dcx/scale, dcy/scale, dlogw, dlogh] motion state.
    velocity: torch.Tensor
    age: torch.Tensor
    hits: torch.Tensor
    lost_age: torch.Tensor
    last_frame: torch.Tensor
    confidence: torch.Tensor
    alive: torch.Tensor
    next_track_id: int = 0
    frame_index: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, hidden_dim: int = 256, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> "UnifiedTrackState":
        empty_float = torch.empty((0, hidden_dim), device=device, dtype=dtype)
        empty_box = torch.empty((0, 4), device=device, dtype=dtype)
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        return cls(
            track_ids=empty_long,
            id_memory=empty_float.clone(),
            identity_anchor=empty_float.clone(),
            appearance_ema=empty_float.clone(),
            semantic_memory=empty_float.clone(),
            boxes=empty_box,
            velocity=empty_box.clone(),
            age=empty_long.clone(),
            hits=empty_long.clone(),
            lost_age=empty_long.clone(),
            last_frame=empty_long.clone(),
            confidence=torch.empty((0,), device=device, dtype=dtype),
            alive=torch.empty((0,), device=device, dtype=torch.bool),
        )

    @property
    def num_tracks(self) -> int:
        return int(self.track_ids.numel())

    @property
    def hidden_dim(self) -> int:
        return int(self.id_memory.shape[-1])

    def clone(self) -> "UnifiedTrackState":
        fields = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        fields["metadata"] = dict(self.metadata)
        return UnifiedTrackState(**fields)

    def to(self, device: torch.device | str) -> "UnifiedTrackState":
        fields = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        fields["metadata"] = dict(self.metadata)
        return UnifiedTrackState(**fields)

    def detach(self) -> "UnifiedTrackState":
        fields = {
            name: value.detach() if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        fields["metadata"] = dict(self.metadata)
        return UnifiedTrackState(**fields)

    def as_dict(self) -> dict[str, Any]:
        return {name: value for name, value in self.__dict__.items()}
