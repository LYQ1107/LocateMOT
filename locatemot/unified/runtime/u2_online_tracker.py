"""Causal online runtime for the shared U2 MOT/OVMOT core."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models.u2.shared_track_core import SharedTrackCore
from ..models.u2.track_state import UnifiedTrackState


class U2OnlineTracker:
    """Run one ``SharedTrackCore`` state transition per current frame."""

    state_format = "locatemot-u2-unified-track-state-v1"

    def __init__(self, core: SharedTrackCore | None = None, *, device: torch.device | str = "cpu") -> None:
        self.core = core if core is not None else SharedTrackCore()
        self.core.to(device).eval()
        self.state: UnifiedTrackState | None = None
        self.frame_index = 0

    def reset(self) -> None:
        self.state = None
        self.frame_index = 0

    @torch.no_grad()
    def step(
        self,
        candidates: dict[str, torch.Tensor],
        *,
        task_type: str = "mot",
        class_text: torch.Tensor | None = None,
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        index = self.frame_index if frame_index is None else int(frame_index)
        output = self.core(
            candidates,
            self.state,
            task_type=task_type,
            class_text=class_text,
            frame_index=index,
            training=False,
        )
        self.state = output["state"]
        self.frame_index = index + 1
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.state_format,
            "frame_index": self.frame_index,
            "state": None if self.state is None else self.state.as_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != self.state_format:
            raise ValueError("unknown U2 online-state format")
        raw = payload.get("state")
        self.frame_index = int(payload.get("frame_index", 0))
        if raw is None:
            self.state = None
            return
        self.state = UnifiedTrackState(**raw)

    def save_state(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination)

    def load_state(self, path: str | Path, *, map_location: torch.device | str = "cpu") -> None:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        self.load_state_dict(payload)


__all__ = ["U2OnlineTracker"]
