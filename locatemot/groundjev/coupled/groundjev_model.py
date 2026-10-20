"""Unified GroundJEV model shell around one shared GroundAnything state."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import nn

from .dynamic_identity_slots import IdentitySlot
from .g2_jev_readout import G2JEVReadout
from .identity_memory import IdentityMemory
from .shared_ground_state import SharedGroundState
from .typed_decisions import TypedDecision, TypedDecisionBatch


class GroundJEVCoupledModel(nn.Module):
    """Typed JEV decisions from a single GroundAnything feature computation.

    The class intentionally accepts a ``SharedGroundState`` rather than an
    image.  Image encoding belongs to :class:`G2FeatureBridge`, ensuring that
    the runtime cannot accidentally encode the same frame a second time for
    identity matching.
    """

    def __init__(self, input_dim: int, *, hidden_dim: int = 512,
                 contextual_dim: Optional[int] = None) -> None:
        super().__init__()
        self.readout = G2JEVReadout(input_dim, hidden_dim=hidden_dim,
                                    contextual_dim=contextual_dim)

    @staticmethod
    def memory_features(memories: Mapping[int, IdentityMemory], *,
                        device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
        features = {}
        for track_id, memory in memories.items():
            feature = memory.feature(device=device, dtype=dtype)
            if feature is not None:
                features[int(track_id)] = feature
        return features

    def forward(self, state: SharedGroundState,
                slots_per_candidate: Sequence[Sequence[IdentitySlot]], *,
                memories: Optional[Mapping[int, IdentityMemory]] = None,
                memory_enabled: bool = True,
                candidate_indices: Optional[Sequence[int]] = None) -> TypedDecisionBatch:
        memories = memories or {}
        features = self.memory_features(memories, device=state.region_tokens.device,
                                        dtype=state.region_tokens.dtype)
        decisions = self.readout(
            state.region_tokens,
            slots_per_candidate,
            contextual_tokens=state.contextual_tokens,
            memory_features=features,
            memory_enabled=memory_enabled,
            candidate_indices=candidate_indices,
        )
        return TypedDecisionBatch(
            decisions=decisions,
            frame_id=state.frame_id,
            image_key=state.image_key,
            model_sha256=state.model_sha256,
            visual_forward_calls=int(state.metadata.get("visual_forward_calls_for_frame", 1)),
            metadata={"memory_enabled": bool(memory_enabled),
                      "single_backbone": True,
                      "second_vlm_loaded": False,
                      "contextual_tokens_present": state.has_language_context},
        )
