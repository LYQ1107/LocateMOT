"""Causal history and target-bag temporal utilities for L86."""
from __future__ import annotations

import torch
from torch import nn


class CausalHistoryEncoder(nn.Module):
    """Encode a bounded observation history without reading future rows."""

    def __init__(self, obs_dim: int = 1432, hidden: int = 256, max_history: int = 8) -> None:
        super().__init__()
        self.max_history = int(max_history)
        self.proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(
        self,
        observations: torch.Tensor,
        valid_mask: torch.Tensor,
        frame_ids: torch.Tensor | None = None,
        cutoff_frame: int | None = None,
    ) -> torch.Tensor:
        if observations.ndim != 3 or valid_mask.ndim != 2 or observations.shape[:2] != valid_mask.shape:
            raise ValueError("L86 history expects [N,H,D] and [N,H]")
        if observations.shape[1] > self.max_history:
            raise ValueError("L86 history length exceeds bound")
        if frame_ids is not None:
            if frame_ids.shape != valid_mask.shape:
                raise ValueError("L86 history frame shape mismatch")
            valid_frames = frame_ids[valid_mask]
            if cutoff_frame is not None and valid_frames.numel() and bool((valid_frames > int(cutoff_frame)).any()):
                raise AssertionError("future observation entered L86 causal history")
        x = self.proj(observations.float())
        lengths = valid_mask.long().sum(dim=1).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x.float(), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # Keep the recurrent kernel in FP32; this is the verified CUDA-safe path.
        with torch.autocast(device_type=observations.device.type, enabled=False):
            _, state = self.gru(packed)
        result = state[-1]
        empty = valid_mask.long().sum(dim=1) == 0
        if bool(empty.any()):
            result = result.masked_fill(empty[:, None], 0.0)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("nonfinite L86 history state")
        return result


__all__ = ["CausalHistoryEncoder"]
