"""Interface to the frozen U1 language backbone; no second random text model."""
from __future__ import annotations

import torch
from torch import nn


class SpecificationBridge(nn.Module):
    """Accept precomputed U1 BERT features and expose class text tokens/global state."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, spec_tokens: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if spec_tokens.ndim != 2 or spec_tokens.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError("spec_tokens must be [L,hidden] from the shared U1 language backbone")
        tokens = self.norm(spec_tokens)
        if mask is None:
            mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        if mask.shape != (tokens.shape[0],):
            raise ValueError("specification mask must be [L]")
        if not mask.any():
            raise ValueError("specification mask cannot be empty")
        return {"spec_tokens": tokens, "spec_global": tokens[mask].mean(dim=0)}
