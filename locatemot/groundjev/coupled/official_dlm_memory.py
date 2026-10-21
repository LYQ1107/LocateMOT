"""Causal identity-memory injection for the official GroundAnything DLM.

The official Hierarchy decoder receives ``input_ids`` and recomputes clean
multimodal embeddings internally.  A plain ``inputs_embeds`` insertion would
therefore be ignored by its masked draft stream.  This adapter reserves
``<|vision_start|>`` marker positions in the clean prompt, lets the official
multimodal path compute positions and image features, and replaces only those
marker embeddings with persistent identity-state tokens.  The marker is
excluded from the noisy text stream by GroundAnything's existing vision-token
contract, so the same memory remains visible to causal prefill and MDM draft
steps.

The context manager is inference/training agnostic: gradients through the
memory tokens are preserved, and the original model method is restored on
exit.  It does not create a second visual encoder or alter the G2 weights.
"""
from __future__ import annotations

from types import MethodType
from typing import Any

import torch


class OfficialDLMIdentityMemory:
    """Temporarily inject projected identity tokens into official DLM clean input."""

    def __init__(self, model: Any, *, marker_token_id: int, memory_tokens: torch.Tensor) -> None:
        if memory_tokens.ndim != 2 or memory_tokens.shape[0] < 1:
            raise ValueError("memory_tokens must have shape [memory, hidden] and be non-empty")
        self.model = model
        self.marker_token_id = int(marker_token_id)
        self.memory_tokens = memory_tokens
        self._original: Any = None

    def __enter__(self) -> "OfficialDLMIdentityMemory":
        if self._original is not None:
            raise RuntimeError("official DLM memory adapter cannot be entered twice")
        original = self.model._embed_clean
        count = int(self.memory_tokens.shape[0])
        marker_id = self.marker_token_id
        memory_tokens = self.memory_tokens

        def _embed_clean_with_memory(instance: Any, input_ids: torch.Tensor, *args: Any, **kwargs: Any):
            embeds, positions = original(input_ids, *args, **kwargs)
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise ValueError("official DLM identity-memory adapter currently expects batch size 1")
            if input_ids.shape[1] < count:
                raise ValueError("prompt is shorter than the requested memory-token suffix")
            suffix = input_ids[:, -count:]
            expected = torch.full_like(suffix, marker_id)
            if not torch.equal(suffix, expected):
                raise ValueError("memory-token marker suffix is missing or was reordered")
            replacement = memory_tokens.to(device=embeds.device, dtype=embeds.dtype).unsqueeze(0)
            if replacement.shape[-1] != embeds.shape[-1]:
                raise ValueError(
                    f"memory hidden size {replacement.shape[-1]} != DLM hidden size {embeds.shape[-1]}"
                )
            # Clone only the prompt embedding tensor; the frozen base remains
            # untouched and gradients still reach the projected memory.
            embeds = embeds.clone()
            embeds[:, -count:] = replacement
            return embeds, positions

        self._original = original
        self.model._embed_clean = MethodType(_embed_clean_with_memory, self.model)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._original is None:
            return False
        self.model._embed_clean = self._original
        self._original = None
        return False


def append_memory_markers(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor | None,
    *,
    marker_token_id: int,
    memory_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Append vision markers that reserve clean-stream memory slots."""

    if memory_count < 1:
        raise ValueError("memory_count must be positive")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be [batch, sequence]")
    if input_ids.shape[0] != 1:
        raise ValueError("official DLM identity memory currently expects batch size 1")
    device = input_ids.device
    markers = torch.full(
        (1, memory_count), int(marker_token_id), dtype=input_ids.dtype, device=device
    )
    ones = torch.ones((1, memory_count), dtype=attention_mask.dtype, device=device)
    output_ids = torch.cat((input_ids, markers), dim=1)
    output_mask = torch.cat((attention_mask, ones), dim=1)
    if mm_token_type_ids is None:
        output_types = None
    else:
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError("mm_token_type_ids must match input_ids")
        zeros = torch.zeros((1, memory_count), dtype=mm_token_type_ids.dtype, device=device)
        output_types = torch.cat((mm_token_type_ids, zeros), dim=1)
    return output_ids, output_mask, output_types

