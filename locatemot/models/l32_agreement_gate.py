"""Bounded, source-blind residual gate for LocateMOT Stage L32."""
from __future__ import annotations

import torch
from torch import nn


class L32AgreementGate(nn.Module):
    """Adjust frozen current-membership logits by a small learned residual.

    The six inputs are current membership, frozen visual association,
    recency, motion consistency, lifecycle consistency, and a query-independent
    visual confidence.  No pool/source/group/state identifier is accepted.
    """

    def __init__(self, input_dim: int = 6, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.residual_head = nn.Linear(hidden, 1)
        self.reject_head = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor, membership: torch.Tensor):
        hidden = self.body(torch.nan_to_num(features.float()))
        residual = 0.25 * torch.tanh(self.residual_head(hidden).squeeze(-1))
        reject_logit = self.reject_head(hidden).squeeze(-1)
        final = membership.float() + residual
        return {"residual": residual, "reject_logit": reject_logit, "final": final}
