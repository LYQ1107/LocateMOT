"""Causal prefill adapter for identity-conditioned GroundAnything prompts.

GroundAnything's hybrid training stream requires ``input_ids``, labels and
embeddings to stay aligned.  The first safe Phase-C boundary is therefore the
causal generation path, whose prompt is complete immediately before the
answer tokens are generated.  This adapter inserts causal identity embeddings
at that terminal prompt boundary and extends the position/cache contract; it
does not run a second visual encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .memory_tokens import inject_identity_memory


@dataclass(frozen=True)
class CausalMemoryInput:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    inserted_tokens: int
    insertion_index: int
    memory_enabled: bool

    def audit_dict(self) -> dict[str, int | bool | list[int]]:
        return {
            "inserted_tokens": int(self.inserted_tokens),
            "insertion_index": int(self.insertion_index),
            "memory_enabled": bool(self.memory_enabled),
            "inputs_embeds_shape": list(self.inputs_embeds.shape),
            "attention_mask_shape": list(self.attention_mask.shape),
            "position_ids_shape": list(self.position_ids.shape),
        }


def _append_position_ids(position_ids: torch.Tensor, count: int) -> torch.Tensor:
    """Append causal positions for a terminal prompt insertion.

    Plain Qwen3 uses ``[1, batch, sequence]`` positions in the DLM wrapper;
    accepting ``[batch, sequence]`` as well keeps the adapter useful for small
    causal fixtures.  Multimodal axes are extended independently from their
    last prompt value, which preserves the existing prefix exactly.
    """
    if count == 0:
        return position_ids
    if position_ids.ndim == 2:
        if position_ids.shape[1] < 1:
            raise ValueError("position_ids cannot be empty when memory is enabled")
        last = position_ids[:, -1:]
        delta = torch.arange(1, count + 1, device=position_ids.device, dtype=position_ids.dtype)
        suffix = last + delta.view(1, -1)
        return torch.cat((position_ids, suffix), dim=1)
    if position_ids.ndim == 3:
        if position_ids.shape[2] < 1:
            raise ValueError("position_ids cannot be empty when memory is enabled")
        last = position_ids[:, :, -1:]
        delta = torch.arange(1, count + 1, device=position_ids.device, dtype=position_ids.dtype)
        suffix = last + delta.view(1, 1, -1)
        return torch.cat((position_ids, suffix), dim=2)
    raise ValueError("position_ids must have shape [batch, sequence] or [axes, batch, sequence]")


def build_causal_memory_input(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    memory_tokens: torch.Tensor,
    *,
    enabled: bool = True,
) -> CausalMemoryInput:
    """Insert causal identity tokens immediately before generation.

    The insertion is intentionally terminal: the caller has already rendered
    the complete user/assistant-prefix prompt and generation starts at
    ``sequence``.  This avoids silently changing multimodal image positions
    inside the prompt.  A disabled or empty memory path returns all original
    tensors by identity, making baseline equivalence directly testable.
    """
    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2 or attention_mask.shape[:2] != inputs_embeds.shape[:2]:
        raise ValueError("attention_mask must match inputs_embeds batch and sequence")
    if position_ids.ndim == 2:
        if position_ids.shape != inputs_embeds.shape[:2]:
            raise ValueError("2-D position_ids must match [batch, sequence]")
    elif position_ids.ndim == 3:
        if position_ids.shape[1:] != inputs_embeds.shape[:2]:
            raise ValueError("3-D position_ids must match [axes, batch, sequence]")
    else:
        raise ValueError("position_ids must have shape [batch, sequence] or [axes, batch, sequence]")
    sequence = int(inputs_embeds.shape[1])
    result = inject_identity_memory(
        inputs_embeds,
        attention_mask,
        memory_tokens,
        answer_start=sequence,
        enabled=enabled,
    )
    if not result.memory_enabled:
        return CausalMemoryInput(
            result.inputs_embeds,
            result.attention_mask,
            position_ids,
            0,
            sequence,
            False,
        )
    positions = _append_position_ids(position_ids, result.inserted_tokens)
    return CausalMemoryInput(
        result.inputs_embeds,
        result.attention_mask,
        positions,
        result.inserted_tokens,
        sequence,
        True,
    )


@torch.inference_mode()
def causal_prefill_with_memory(
    language_model: Any,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    memory_tokens: torch.Tensor,
    *,
    native_position_ids: Callable[[torch.Tensor], torch.Tensor] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Run exactly one causal prefill and expose next-token logits.

    ``language_model`` is the already-loaded GroundAnything language tower.
    The function is deliberately a prefill boundary rather than a new decoder:
    callers can continue with the model's existing KV-cache generation loop.
    """
    prepared = build_causal_memory_input(
        inputs_embeds,
        attention_mask,
        position_ids,
        memory_tokens,
        enabled=enabled,
    )
    native_positions = (native_position_ids or (lambda value: value))(prepared.position_ids)
    cache_position = torch.arange(
        prepared.inputs_embeds.shape[1],
        device=prepared.inputs_embeds.device,
        dtype=torch.long,
    )
    outputs = language_model(
        input_ids=None,
        inputs_embeds=prepared.inputs_embeds,
        position_ids=native_positions,
        attention_mask=prepared.attention_mask,
        past_key_values=None,
        use_cache=True,
        cache_position=cache_position,
        return_dict=True,
    )
    if not hasattr(outputs, "last_hidden_state"):
        raise TypeError("language_model prefill must return last_hidden_state")
    logits = lm_head(outputs.last_hidden_state[:, -1])
    return {
        "logits": logits,
        "outputs": outputs,
        "prepared": prepared,
        "prefill_calls": 1,
        "single_backbone": True,
        "second_vlm_loaded": False,
    }
