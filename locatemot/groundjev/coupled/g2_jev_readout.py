"""Typed JEV-style readouts over shared GroundAnything tokens."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .region_tokens import RegionTokenProjector


LIFECYCLE_ACTIONS = ("ACTIVE", "LOST", "REACTIVATE", "TERMINATE")


class G2JEVReadout(nn.Module):
    """One typed readout module, with dynamic identity options.

    Identity matching is a dot-product readout between a candidate token and
    the current identity-slot tokens.  It has no image encoder, proposal
    network, or independent temporal Transformer.
    """

    def __init__(self, region_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.region_dim = int(region_dim)
        self.hidden_dim = int(hidden_dim)
        self.region_projector = RegionTokenProjector(region_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.slot_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.admit_head = nn.Linear(hidden_dim, 1)
        self.yesno_head = nn.Linear(hidden_dim, 1)
        self.lifecycle_head = nn.Linear(hidden_dim, len(LIFECYCLE_ACTIONS))
        self.memory_head = nn.Linear(hidden_dim, 1)
        # NEW/NO_MATCH/REJECT are action slots in the same readout space.
        self.action_slots = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.action_slots, std=0.02)

    def encode_candidates(self, region_tokens: torch.Tensor) -> torch.Tensor:
        return self.region_projector(region_tokens)

    def make_slot_tokens(self, memory_tokens: torch.Tensor,
                         action_order: Sequence[str]) -> torch.Tensor:
        """Build dynamic slots; memory_tokens contains only causal track state."""
        if memory_tokens.ndim != 2 or memory_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("memory_tokens must be [number_of_tracks, hidden_dim]")
        pieces: list[torch.Tensor] = [memory_tokens]
        action_index = {"NEW": 0, "NO_MATCH": 1, "REJECT": 2}
        for token in action_order:
            if token not in action_index:
                raise ValueError(f"unknown action slot {token}")
            pieces.append(self.action_slots[action_index[token]].unsqueeze(0))
        return torch.cat(pieces, dim=0) if pieces else memory_tokens.new_empty((0, self.hidden_dim))

    def forward(self, region_tokens: torch.Tensor, slot_tokens: torch.Tensor,
                *, action_start: int = 0) -> dict[str, torch.Tensor | int]:
        candidate = self.encode_candidates(region_tokens)
        if slot_tokens.ndim != 2 or slot_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("slot_tokens must be [options, hidden_dim]")
        q = self.query(candidate)
        k = self.slot_key(slot_tokens)
        identity_logits = q @ k.transpose(0, 1) / (self.hidden_dim ** 0.5)
        return {
            "candidate_hidden": candidate,
            "slot_hidden": slot_tokens,
            "identity_logits": identity_logits,
            "admit_logits": self.admit_head(candidate).squeeze(-1),
            "yesno_logits": (self.yesno_head(candidate).squeeze(-1)),
            "lifecycle_logits": self.lifecycle_head(candidate),
            "memory_logits": self.memory_head(candidate).squeeze(-1),
            "action_start": int(action_start),
        }
