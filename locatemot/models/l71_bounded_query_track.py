"""L71 bounded, explicit query--track correspondence head.

The head has no set competition, NULL head, tracker, or learned score scale.
It maps a causal observation history and a masked word-token sequence to
L2-normalized vectors and emits fixed-temperature cosine correspondence.
Track IDs are consumed only by the L71 indexer, never by this module.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class L71BoundedQueryTrack(nn.Module):
    def __init__(
        self,
        obs_dim: int = 1432,
        text_dim: int = 768,
        hidden: int = 192,
        max_history: int = 8,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if temperature != 0.07:
            raise ValueError("L71 temperature is fixed at 0.07")
        self.obs_dim = int(obs_dim)
        self.text_dim = int(text_dim)
        self.hidden = int(hidden)
        self.max_history = int(max_history)
        self.temperature = float(temperature)
        self.obs_norm = nn.LayerNorm(self.obs_dim)
        self.obs_proj = nn.Linear(self.obs_dim, self.hidden)
        self.text_norm = nn.LayerNorm(self.text_dim)
        self.text_proj = nn.Linear(self.text_dim, self.hidden)

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor, dim: int) -> Tensor:
        weight = mask.to(value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        denom = weight.sum(dim=dim).clamp_min(1.0)
        return (value * weight).sum(dim=dim) / denom

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        current = batch["current"]
        history = batch["history"]
        history_mask = batch["history_mask"].bool()
        text = batch["text"]
        text_mask = batch["text_mask"].bool()
        if current.ndim != 2 or current.shape[-1] != self.obs_dim:
            raise ValueError(f"current must be [N,{self.obs_dim}], got {tuple(current.shape)}")
        if history.ndim != 3 or history.shape[0] != current.shape[0] or history.shape[-1] != self.obs_dim:
            raise ValueError(f"history/current mismatch: {tuple(history.shape)} / {tuple(current.shape)}")
        if history.shape[1] != self.max_history or history_mask.shape != history.shape[:2]:
            raise ValueError("history mask/length mismatch")
        if text.ndim != 2 or text.shape[-1] != self.text_dim or text_mask.shape != text.shape[:1]:
            raise ValueError(f"text/mask mismatch: {tuple(text.shape)} / {tuple(text_mask.shape)}")
        if current.shape[0] == 0 or not history_mask.any(dim=1).all():
            raise ValueError("all candidates must have at least one valid causal history row")
        if not bool(text_mask.any()):
            raise ValueError("query token mask has no valid token")

        current_projected = self.obs_proj(self.obs_norm(current))
        history_projected = self.obs_proj(self.obs_norm(history))
        history_mean = self._masked_mean(history_projected, history_mask, dim=1)
        track_vector = F.normalize(0.5 * current_projected + 0.5 * history_mean, dim=-1)
        text_projected = self.text_proj(self.text_norm(text))
        query_vector = F.normalize(self._masked_mean(text_projected, text_mask, dim=0), dim=-1)
        correspondence_logits = track_vector @ query_vector / self.temperature
        return {
            "correspondence_logits": correspondence_logits,
            "query_vector": query_vector,
            "track_vector": track_vector,
        }

    def parameter_summary(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "obs_dim": self.obs_dim,
            "text_dim": self.text_dim,
            "hidden": self.hidden,
            "max_history": self.max_history,
            "temperature": self.temperature,
        }
