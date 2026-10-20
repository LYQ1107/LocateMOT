"""GroundJEV single-backbone model wrapper."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .dynamic_identity_slots import DynamicIdentitySlots
from .g2_jev_readout import G2JEVReadout
from .identity_memory import IdentityMemoryState
from .memory_tokens import MemoryTokenProjector
from .shared_ground_state import SharedGroundState


class GroundJEVModel(nn.Module):
    """Typed JEV readout attached to one shared G2 state.

    The `backbone` is external and supplies `SharedGroundState`; this module
    never instantiates a second visual or language model.
    """

    def __init__(self, region_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.region_dim = int(region_dim)
        self.hidden_dim = int(hidden_dim)
        self.readout = G2JEVReadout(region_dim, hidden_dim)
        self.memory_projector = MemoryTokenProjector(region_dim, hidden_dim)

    def forward(self, state: SharedGroundState, memory: IdentityMemoryState,
                slots: DynamicIdentitySlots) -> dict[str, Any]:
        state.validate()
        if state.feature_dim != self.region_dim:
            raise ValueError("G2 state feature dimension disagrees with coupled model")
        track_tokens = [track.token().to(state.vision_tokens.device) for track in memory.ordered()
                        if track.state != "TERMINATED"]
        if track_tokens:
            raw = torch.stack(track_tokens, dim=0)
            projected = self.memory_projector(raw)
        else:
            projected = state.vision_tokens.new_empty((0, self.hidden_dim), dtype=torch.float32)
        existing_ids = [slot.track_id for slot in slots.slots if slot.track_id is not None]
        if len(existing_ids) != projected.shape[0]:
            # A track can be omitted from the shortlist.  Keep option order
            # explicit and represent omitted tokens by a zero vector; this is
            # visible to the evaluator and never turns a retrieval miss into NEW.
            by_id = {track.persistent_track_id: track for track in memory.ordered()
                     if track.state != "TERMINATED"}
            vectors = []
            for track_id in existing_ids:
                track = by_id.get(int(track_id))
                if track is None:
                    vectors.append(state.vision_tokens.new_zeros((self.hidden_dim,), dtype=torch.float32))
                else:
                    vectors.append(self.memory_projector(track.token().to(state.vision_tokens.device).unsqueeze(0))[0])
            projected = torch.stack(vectors) if vectors else state.vision_tokens.new_empty((0, self.hidden_dim), dtype=torch.float32)
        action_tokens = [slot.token for slot in slots.slots if slot.track_id is None]
        slot_tokens = self.readout.make_slot_tokens(projected, action_tokens)
        outputs = self.readout(state.vision_tokens, slot_tokens,
                               action_start=len(existing_ids))
        outputs.update({"option_tokens": slots.tokens, "model_sha256": state.model_sha256,
                        "frame_id": state.frame_id, "image_key": state.image_key,
                        "single_backbone": True})
        return outputs
