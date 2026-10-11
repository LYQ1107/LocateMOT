"""Compatibility name for the frozen U1 language feature bridge."""
from __future__ import annotations

import torch

from .u2.specification_bridge import SpecificationBridge


class SpecificationEncoder(SpecificationBridge):
    """Reject token IDs; consume precomputed U1 BERT features only."""

    def forward(self, spec_tokens: torch.Tensor, task_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if not torch.is_floating_point(spec_tokens):
            raise TypeError("SpecificationEncoder expects frozen U1 BERT features, not integer IDs")
        output = super().forward(spec_tokens)
        if task_id is not None:
            output["task_id"] = task_id
        return output
