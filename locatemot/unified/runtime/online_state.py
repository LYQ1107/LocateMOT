"""One online state format shared by MOT, OVMOT, RMOT and prompt tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class OnlineState:
    frame_index: int = 0
    track_ids: torch.Tensor | None = None
    track_state: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def advance(self, *, track_ids: torch.Tensor, track_state: torch.Tensor) -> "OnlineState":
        self.frame_index += 1
        self.track_ids = track_ids
        self.track_state = track_state
        return self
