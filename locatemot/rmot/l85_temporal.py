"""Causal temporal utilities for the L85 factorized RMOT model."""
from __future__ import annotations

import torch
from torch import nn


class CausalHistoryEncoder(nn.Module):
    """Encode at most H observations without reading future rows."""

    def __init__(self, obs_dim: int = 1432, hidden: int = 256, max_history: int = 8) -> None:
        super().__init__()
        self.max_history = int(max_history)
        self.proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, observations: torch.Tensor, valid_mask: torch.Tensor,
                frame_ids: torch.Tensor | None = None, cutoff_frame: int | None = None) -> torch.Tensor:
        if observations.ndim != 3 or valid_mask.ndim != 2 or observations.shape[:2] != valid_mask.shape:
            raise ValueError("history expects [N,H,D] and [N,H]")
        if observations.shape[1] > self.max_history:
            raise ValueError("history length exceeds registered bound")
        if frame_ids is not None:
            if frame_ids.shape != valid_mask.shape:
                raise ValueError("history frame shape mismatch")
            if cutoff_frame is not None and bool((frame_ids[valid_mask] > int(cutoff_frame)).any()):
                raise AssertionError("future observation entered causal history")
        x = self.proj(observations.float())
        lengths = valid_mask.long().sum(dim=1).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(x.float(), lengths.cpu(), batch_first=True, enforce_sorted=False)
        # The verified Torch/CUDA build has no BF16 fused GRU kernel.  Keep
        # this recurrent operation FP32 under BF16 adapter autocast.
        with torch.autocast(device_type=observations.device.type, enabled=False):
            _, hidden_state = self.gru(packed)
        result = hidden_state[-1]
        empty = valid_mask.long().sum(dim=1) == 0
        if bool(empty.any()):
            result = result.masked_fill(empty[:, None], 0.0)
        return result


__all__ = ["CausalHistoryEncoder"]
