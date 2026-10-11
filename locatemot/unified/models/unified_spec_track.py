"""LocateMOT-U U2 composition root.

Only ordinary MOT and OVMOT are enabled in U2.  Both call the same
``SharedTrackCore`` and ``UnifiedTrackState`` transition.  Text is accepted
only as precomputed floating-point features from the frozen U1 language
backbone; integer vocabulary IDs and a second random embedding table are
intentionally rejected.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .foundation import SharedVisualFoundation
from .u2.perception_bridge import PerceptionBridge
from .u2.specification_bridge import SpecificationBridge
from .u2.shared_track_core import SharedTrackCore
from .u2.track_state import UnifiedTrackState


class UnifiedSpecTrack(nn.Module):
    """One U1 foundation plus one persistent track core for MOT and OVMOT."""

    task_to_id = {"mot": 0, "ovmot": 1}

    def __init__(self, hidden_dim: int = 256, max_age: int = 30) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.foundation = SharedVisualFoundation()
        for parameter in self.foundation.parameters():
            parameter.requires_grad_(False)
        foundation_dim = int(self.foundation.config.feature_dim)
        self.frame_feature_projection = (
            nn.Identity() if foundation_dim == hidden_dim else nn.Linear(foundation_dim, hidden_dim)
        )
        self.perception_bridge = PerceptionBridge(hidden_dim=hidden_dim)
        self.specification_bridge = SpecificationBridge(hidden_dim=hidden_dim)
        self.track_core = SharedTrackCore(hidden_dim=hidden_dim, max_age=max_age)

    @property
    def shared_track_core(self) -> SharedTrackCore:
        """Stable descriptive alias without registering a second module copy."""
        return self.track_core

    @property
    def spec_encoder(self) -> SpecificationBridge:
        """Compatibility alias for the U1-shared language bridge."""
        return self.specification_bridge

    def _frame_candidates(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Create a shape-valid candidate stream for integration smoke tests.

        Production U2 consumes detector candidates through ``PerceptionBridge``.
        The fallback emits one full-frame candidate from the shared visual map;
        it is deliberately not used as a tracking benchmark protocol.
        """
        if frames.ndim != 5 or frames.shape[0] != 1:
            raise ValueError("frame fallback expects [1,T,3,H,W]")
        visual = self.foundation(frames)["frame_features"][:, -1]
        pooled = self.frame_feature_projection(visual.mean(dim=(-1, -2))[0])
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=frames.device, dtype=frames.dtype)
        scores = torch.ones((1,), device=frames.device, dtype=frames.dtype)
        feature = pooled.unsqueeze(0)
        return {
            "boxes": boxes,
            "scores": scores,
            "id_visual": feature,
            "semantic_visual": feature,
            "decoder_query": feature,
        }

    def _specification(self, specification: torch.Tensor | dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        if specification is None:
            return None
        if isinstance(specification, dict):
            if "spec_tokens" not in specification:
                raise KeyError("specification dict must contain precomputed spec_tokens")
            tokens = specification["spec_tokens"]
        else:
            tokens = specification
        if not torch.is_floating_point(tokens):
            raise TypeError("U2 specification must be frozen U1 BERT features, not integer token IDs")
        if tokens.ndim == 3:
            if tokens.shape[0] != 1:
                raise ValueError("single online frame accepts one specification sequence")
            tokens = tokens[0]
        return self.specification_bridge(tokens)

    def forward(
        self,
        candidates: dict[str, torch.Tensor] | torch.Tensor | None = None,
        task_type: str = "mot",
        specification: torch.Tensor | dict[str, torch.Tensor] | None = None,
        online_state: UnifiedTrackState | None = None,
        *,
        frames: torch.Tensor | None = None,
        frame_index: int | None = None,
        training: bool = False,
    ) -> dict[str, Any]:
        if task_type not in self.task_to_id:
            raise KeyError(f"U2 supports only {sorted(self.task_to_id)}; got {task_type!r}")
        if isinstance(candidates, torch.Tensor):
            # Preserve the convenient positional ``model(frames, task_type, …)``
            # form while keeping candidate dictionaries as the production API.
            if frames is not None:
                raise ValueError("pass either candidates or frames, not both")
            frames = candidates
            candidates = None
        if candidates is None:
            if frames is None:
                raise ValueError("U2 forward requires candidates or frames")
            candidates = self._frame_candidates(frames)
        spec = self._specification(specification)
        class_text = None if spec is None else spec["spec_tokens"]
        output = self.track_core(
            candidates,
            online_state,
            task_type=task_type,
            class_text=class_text,
            frame_index=frame_index,
            training=training,
        )
        return {
            "tracks": output,
            "task_output": output["track_ids"],
            "online_state": output["state"],
            "spec": spec,
            "foundation_frozen": True,
            "task_type": task_type,
        }
