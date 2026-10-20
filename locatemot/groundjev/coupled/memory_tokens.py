"""Projection and causal injection of identity state into an opt-in G2 input space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class MemoryConditionedPrompt:
    """Result of inserting causal identity tokens into a DLM prompt.

    The wrapper deliberately stops at the tensor contract consumed by the
    GroundAnything DLM.  The caller remains responsible for shifting labels
    and answer-token positions using the returned insertion offset.  Keeping
    this boundary explicit prevents a memory ablation from silently changing
    the original G2 prompt or its loss mask.
    """

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    inserted_tokens: int
    insertion_index: int
    memory_enabled: bool

    @property
    def sequence_length(self) -> int:
        return int(self.inputs_embeds.shape[1])

    def audit_dict(self) -> dict[str, int | bool]:
        return {
            "inserted_tokens": self.inserted_tokens,
            "insertion_index": self.insertion_index,
            "sequence_length": self.sequence_length,
            "memory_enabled": self.memory_enabled,
        }


class IdentityMemoryTokenProjector(nn.Module):
    """Lightweight memory-token projection for the future C2 path.

    The module is intentionally standalone until the official GroundAnything
    DLM input hook is enabled.  It never reads future labels and returns one
    token per causal identity in caller-provided order.
    """

    def __init__(self, input_dim: int, g2_embedding_dim: int, *, hidden_dim: int = 512) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.g2_embedding_dim = int(g2_embedding_dim)
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, g2_embedding_dim),
        )

    def forward(self, memories: Mapping[int, torch.Tensor], *,
                ordered_ids: Optional[list[int]] = None) -> torch.Tensor:
        ids = ordered_ids if ordered_ids is not None else sorted(memories)
        if not ids:
            return torch.empty((0, self.g2_embedding_dim),
                               device=next(self.parameters()).device,
                               dtype=next(self.parameters()).dtype)
        values = []
        for track_id in ids:
            value = memories[int(track_id)]
            if value.ndim != 1 or value.shape[0] != self.input_dim:
                raise ValueError(f"memory {track_id} has invalid shape {tuple(value.shape)}")
            values.append(value)
        return self.project(torch.stack(values).to(next(self.parameters()).device,
                                                     dtype=next(self.parameters()).dtype))


def inject_identity_memory(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    memory_tokens: torch.Tensor,
    *,
    answer_start: Optional[int] = None,
    enabled: bool = True,
) -> MemoryConditionedPrompt:
    """Insert causal identity tokens before the current answer region.

    ``inputs_embeds`` is the already prepared GroundAnything prompt and must
    have shape ``[batch, sequence, hidden]``.  ``memory_tokens`` may be
    ``[memory, hidden]`` (broadcast to every sample) or ``[batch, memory,
    hidden]``.  The disabled/empty path returns the original tensors without
    cloning, which makes the G2 baseline contract exact.  Memory is inserted
    at ``answer_start``; it can therefore influence answer generation but can
    never be placed after the answer region by this helper.

    This function does not fabricate labels or logits.  The DLM caller must
    shift its answer labels by ``inserted_tokens`` and record the returned
    insertion index in the training receipt.
    """
    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2 or attention_mask.shape[:2] != inputs_embeds.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if memory_tokens.ndim not in (2, 3):
        raise ValueError("memory_tokens must have shape [memory, hidden] or [batch, memory, hidden]")
    batch, sequence, hidden = inputs_embeds.shape
    if answer_start is None:
        answer_start = sequence
    answer_start = int(answer_start)
    if not 0 <= answer_start <= sequence:
        raise ValueError("answer_start must lie inside the prompt sequence")
    if memory_tokens.ndim == 2:
        memory_tokens = memory_tokens.unsqueeze(0).expand(batch, -1, -1)
    elif memory_tokens.shape[0] != batch:
        raise ValueError("batched memory_tokens must match inputs_embeds batch size")
    if memory_tokens.shape[-1] != hidden:
        raise ValueError("memory token hidden dimension must match inputs_embeds")
    count = int(memory_tokens.shape[1])
    if not enabled or count == 0:
        return MemoryConditionedPrompt(inputs_embeds, attention_mask, 0,
                                       answer_start, False)
    memory_tokens = memory_tokens.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    prefix = inputs_embeds[:, :answer_start]
    suffix = inputs_embeds[:, answer_start:]
    conditioned = torch.cat((prefix, memory_tokens, suffix), dim=1)
    memory_mask = torch.ones((batch, count), dtype=attention_mask.dtype,
                             device=attention_mask.device)
    conditioned_mask = torch.cat((attention_mask[:, :answer_start], memory_mask,
                                  attention_mask[:, answer_start:]), dim=1)
    return MemoryConditionedPrompt(conditioned, conditioned_mask, count,
                                   answer_start, True)
