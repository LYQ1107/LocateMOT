"""Joint GroundJEV losses with explicit validity masks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class JointLossWeights:
    grounding: float = 1.0
    identity_choice: float = 1.0
    yesno: float = 0.5
    birth: float = 0.5
    lifecycle: float = 0.25
    memory: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor,
                valid: torch.Tensor) -> Optional[torch.Tensor]:
    if not bool(valid.any()):
        return None
    return F.binary_cross_entropy_with_logits(logits[valid], targets[valid].float())


def identity_choice_loss(logits: Sequence[torch.Tensor], targets: Sequence[Optional[int]]) -> Optional[torch.Tensor]:
    values = []
    for row, target in zip(logits, targets):
        if target is None:
            continue
        if row.ndim != 1 or not 0 <= int(target) < row.numel():
            raise ValueError("identity target is outside its dynamic slot list")
        values.append(F.cross_entropy(row.unsqueeze(0), torch.tensor([int(target)], device=row.device)))
    return torch.stack(values).mean() if values else None


def joint_losses(*, identity_logits: Sequence[torch.Tensor], identity_targets: Sequence[Optional[int]],
                 admit_logits: Optional[torch.Tensor] = None,
                 admit_targets: Optional[torch.Tensor] = None,
                 admit_valid: Optional[torch.Tensor] = None,
                 birth_logits: Optional[torch.Tensor] = None,
                 birth_targets: Optional[torch.Tensor] = None,
                 birth_valid: Optional[torch.Tensor] = None,
                 lifecycle_logits: Optional[torch.Tensor] = None,
                 lifecycle_targets: Optional[torch.Tensor] = None,
                 lifecycle_valid: Optional[torch.Tensor] = None,
                 memory_logits: Optional[torch.Tensor] = None,
                 memory_targets: Optional[torch.Tensor] = None,
                 memory_valid: Optional[torch.Tensor] = None,
                 yesno_logits: Optional[torch.Tensor] = None,
                 yesno_targets: Optional[torch.Tensor] = None,
                 yesno_valid: Optional[torch.Tensor] = None,
                 grounding_loss: Optional[torch.Tensor] = None,
                 weights: JointLossWeights = JointLossWeights()) -> dict[str, torch.Tensor | None]:
    """Return every component separately and the weighted total.

    ``identity_logits`` are a list because every candidate can have a distinct
    dynamic shortlist.  All dense auxiliary heads require an explicit validity
    tensor; unknown/partial annotations contribute no loss.
    """
    losses: dict[str, torch.Tensor | None] = {}
    losses["grounding"] = grounding_loss
    losses["identity_choice"] = identity_choice_loss(identity_logits, identity_targets)
    losses["admit"] = (_masked_bce(admit_logits.reshape(-1), admit_targets.reshape(-1), admit_valid.reshape(-1))
                       if admit_logits is not None and admit_targets is not None and admit_valid is not None else None)
    losses["birth"] = (_masked_bce(birth_logits.reshape(-1), birth_targets.reshape(-1), birth_valid.reshape(-1))
                       if birth_logits is not None and birth_targets is not None and birth_valid is not None else None)
    losses["lifecycle"] = (F.cross_entropy(lifecycle_logits[lifecycle_valid], lifecycle_targets[lifecycle_valid])
                           if lifecycle_logits is not None and lifecycle_targets is not None and lifecycle_valid is not None
                           and bool(lifecycle_valid.any()) else None)
    losses["memory"] = (_masked_bce(memory_logits[..., 1], memory_targets, memory_valid)
                        if memory_logits is not None and memory_targets is not None and memory_valid is not None else None)
    losses["yesno"] = (_masked_bce(yesno_logits.reshape(-1), yesno_targets.reshape(-1), yesno_valid.reshape(-1))
                       if yesno_logits is not None and yesno_targets is not None and yesno_valid is not None else None)
    total = None
    scale = {
        "grounding": weights.grounding,
        "identity_choice": weights.identity_choice,
        "yesno": weights.yesno,
        "birth": weights.birth,
        "lifecycle": weights.lifecycle,
        "memory": weights.memory,
        "admit": weights.yesno,
    }
    for name, value in losses.items():
        if value is None:
            continue
        total = value * float(scale.get(name, 1.0)) if total is None else total + value * float(scale.get(name, 1.0))
    losses["total"] = total
    return losses
