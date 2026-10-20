"""Typed JEV readout over shared GroundAnything evidence."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .dynamic_identity_slots import IdentitySlot, stack_slot_features
from .typed_decisions import (ADMISSION_CHOICES, LIFECYCLE_CHOICES, MEMORY_CHOICES,
                               TypedDecision, decode_identity)


class G2JEVReadout(nn.Module):
    """Small typed readout; it does not instantiate another VLM.

    Candidate and context projections are intentionally the only learned
    association parameters.  Dynamic identity slots are supplied by the
    causal tracker, so numeric IDs never become fixed semantic classes.
    """

    def __init__(self, input_dim: int, *, hidden_dim: int = 512,
                 contextual_dim: Optional[int] = None) -> None:
        super().__init__()
        contextual_dim = int(contextual_dim or input_dim)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_proj = nn.Linear(contextual_dim, hidden_dim)
        self.candidate_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.slot_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.action_slots = nn.ParameterDict({
            "NEW": nn.Parameter(torch.empty(hidden_dim)),
            "NO_MATCH": nn.Parameter(torch.empty(hidden_dim)),
            "REJECT": nn.Parameter(torch.empty(hidden_dim)),
        })
        for value in self.action_slots.values():
            nn.init.normal_(value, std=0.02)
        self.identity_temperature = nn.Parameter(torch.tensor(1.0))
        self.admit_head = nn.Linear(hidden_dim, 1)
        self.lifecycle_head = nn.Linear(hidden_dim, len(LIFECYCLE_CHOICES))
        self.memory_head = nn.Linear(hidden_dim, len(MEMORY_CHOICES))
        self.yesno_head = nn.Linear(hidden_dim, 1)
        self.birth_head = nn.Linear(hidden_dim, 1)
        self.context_norm = nn.LayerNorm(hidden_dim)

    def _context(self, candidate: torch.Tensor,
                 contextual_tokens: Optional[torch.Tensor],
                 *, memory_enabled: bool) -> torch.Tensor:
        parameter = next(self.parameters())
        candidate = candidate.to(device=parameter.device, dtype=parameter.dtype)
        candidate_hidden = self.candidate_proj(candidate)
        if contextual_tokens is None:
            return self.context_norm(candidate_hidden)
        else:
            if contextual_tokens.ndim == 3:
                contextual_tokens = contextual_tokens.mean(dim=0)
            contextual_tokens = contextual_tokens.to(device=parameter.device, dtype=parameter.dtype)
            context = contextual_tokens.mean(dim=0, keepdim=True).expand(candidate.shape[0], -1)
            context = self.context_proj(context)
            # ``memory_enabled`` controls the optional identity-token path in
            # the coupled model; the shared visual evidence remains active in
            # both modes so the disabled arm is a valid G2 baseline.
            return self.context_norm(context + candidate_hidden)

    def _slot_embedding(self, slot: IdentitySlot,
                        memory_features: Mapping[int, torch.Tensor],
                        *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if slot.is_existing:
            value = memory_features.get(int(slot.track_id))
            if value is None:
                value = torch.zeros(self.input_dim, device=device, dtype=dtype)
            if value.ndim != 1 or value.shape[0] != self.input_dim:
                raise ValueError(f"memory feature for identity {slot.track_id} must have dim {self.input_dim}")
            value = value.to(device=device, dtype=dtype)
            return self.slot_proj(value.unsqueeze(0))[0]
        return self.action_slots[slot.token].to(device=device, dtype=dtype)

    def forward(self, region_tokens: torch.Tensor,
                slots_per_candidate: Sequence[Sequence[IdentitySlot]],
                *, contextual_tokens: Optional[torch.Tensor] = None,
                memory_features: Optional[Mapping[int, torch.Tensor]] = None,
                memory_enabled: bool = True,
                candidate_indices: Optional[Sequence[int]] = None) -> list[TypedDecision]:
        if region_tokens.ndim != 2:
            raise ValueError("region_tokens must have shape [N,D]")
        if len(slots_per_candidate) != region_tokens.shape[0]:
            raise ValueError("one dynamic slot list is required per candidate")
        memory_features = memory_features or {}
        parameter = next(self.parameters())
        region_tokens = region_tokens.to(device=parameter.device, dtype=parameter.dtype)
        candidate_indices = list(candidate_indices or range(region_tokens.shape[0]))
        if len(candidate_indices) != region_tokens.shape[0]:
            raise ValueError("candidate_indices length mismatch")
        context = self._context(region_tokens, contextual_tokens, memory_enabled=memory_enabled)
        result: list[TypedDecision] = []
        scale = self.identity_temperature.abs().clamp_min(0.05)
        for row_index, (candidate, slots) in enumerate(zip(context, slots_per_candidate)):
            if not slots:
                raise ValueError("dynamic identity choices may not be empty")
            slot_vectors = torch.stack([
                self._slot_embedding(slot, memory_features,
                                     device=region_tokens.device,
                                     dtype=region_tokens.dtype)
                for slot in slots
            ], dim=0)
            query = F.normalize(candidate, dim=-1)
            options = F.normalize(slot_vectors, dim=-1)
            identity_logits = (query.unsqueeze(0) @ options.transpose(0, 1)).squeeze(0) / scale
            probabilities = F.softmax(identity_logits, dim=-1)
            token, track_id, decision, chosen_probability = decode_identity(probabilities, slots)
            admission_logits = self.admit_head(candidate).reshape(1)
            lifecycle_logits = self.lifecycle_head(candidate)
            memory_logits = self.memory_head(candidate)
            yesno_logit = self.yesno_head(candidate).reshape(())
            result.append(TypedDecision(
                candidate_index=int(candidate_indices[row_index]),
                identity_logits=identity_logits,
                identity_probabilities=probabilities,
                slots=tuple(slots),
                admission_logits=admission_logits,
                lifecycle_logits=lifecycle_logits,
                memory_logits=memory_logits,
                yesno_logit=yesno_logit,
                chosen_token=token,
                chosen_probability=chosen_probability,
                decision=decision,
                track_id=track_id,
                admission_probability=float(torch.sigmoid(admission_logits[0]).item()),
                lifecycle_probabilities=dict(zip(LIFECYCLE_CHOICES,
                                                  F.softmax(lifecycle_logits, dim=-1).detach().float().tolist())),
                memory_update_probability=float(F.softmax(memory_logits, dim=-1)[1].item()),
                metadata={"memory_enabled": bool(memory_enabled),
                          "typed_readout": "G2JEVReadout",
                          "second_vlm_loaded": False},
            ))
        return result
