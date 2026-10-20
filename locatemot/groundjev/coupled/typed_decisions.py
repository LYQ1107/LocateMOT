"""Typed JEV vocabulary and causal decision containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from ..data_v2.schema import IdentityDecision
from .dynamic_identity_slots import IdentitySlot

LIFECYCLE_CHOICES = ("ACTIVE", "LOST", "REACTIVATE", "TERMINATE")
MEMORY_CHOICES = ("HOLD", "UPDATE")
ADMISSION_CHOICES = ("REJECT", "ADMIT")


@dataclass
class TypedDecision:
    """One candidate's structured output before global assignment."""

    candidate_index: int
    identity_logits: torch.Tensor
    identity_probabilities: torch.Tensor
    slots: tuple[IdentitySlot, ...]
    admission_logits: torch.Tensor
    lifecycle_logits: torch.Tensor
    memory_logits: torch.Tensor
    yesno_logit: torch.Tensor
    chosen_token: str
    chosen_probability: float
    decision: str
    track_id: Optional[int]
    admission_probability: float
    lifecycle_probabilities: dict[str, float]
    memory_update_probability: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def probability_dict(self) -> dict[str, float]:
        return {slot.token: float(prob) for slot, prob in
                zip(self.slots, self.identity_probabilities.detach().float().tolist())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "slots": [slot.__dict__ for slot in self.slots],
            "identity_logits": self.identity_logits.detach().float().tolist(),
            "identity_probabilities": self.probability_dict(),
            "chosen_token": self.chosen_token,
            "chosen_probability": self.chosen_probability,
            "decision": self.decision,
            "track_id": self.track_id,
            "admission_probability": self.admission_probability,
            "lifecycle_probabilities": self.lifecycle_probabilities,
            "memory_update_probability": self.memory_update_probability,
            "yesno_logit": float(self.yesno_logit.detach().float().item()),
            "metadata": self.metadata,
        }


@dataclass
class TypedDecisionBatch:
    """Batch of per-candidate outputs plus provenance for attribution."""

    decisions: list[TypedDecision]
    frame_id: int | str
    image_key: str
    model_sha256: str
    visual_forward_calls: int
    assignment: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "image_key": self.image_key,
            "model_sha256": self.model_sha256,
            "visual_forward_calls": self.visual_forward_calls,
            "decisions": [item.to_dict() for item in self.decisions],
            "assignment": {str(k): v for k, v in self.assignment.items()},
            "metadata": self.metadata,
        }


def decode_identity(
    probabilities: torch.Tensor,
    slots: Sequence[IdentitySlot],
) -> tuple[str, Optional[int], str, float]:
    if probabilities.ndim != 1 or probabilities.shape[0] != len(slots):
        raise ValueError("identity probabilities and dynamic slots disagree")
    index = int(probabilities.argmax().item())
    slot = slots[index]
    token = slot.token
    if slot.kind == "existing":
        decision = (IdentityDecision.REACTIVATE.value if slot.lost_age > 0
                    else IdentityDecision.MATCH.value)
        track_id = slot.track_id
    elif slot.kind == "new":
        decision, track_id = IdentityDecision.NEW.value, None
    elif slot.kind == "no_match":
        decision, track_id = IdentityDecision.NO_MATCH.value, None
    else:
        decision, track_id = IdentityDecision.REJECT.value, None
    return token, track_id, decision, float(probabilities[index].item())
