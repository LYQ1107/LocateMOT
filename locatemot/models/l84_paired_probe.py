"""The canonical paired L84 frozen-representation probe."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class L84PairedProbeConfig:
    input_dim: int = 256
    hidden: int = 256
    dropout: float = 0.05


class L84PairedProbe(nn.Module):
    """Parameter-matched scalar interaction head used for every stage."""

    def __init__(self, config: L84PairedProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or L84PairedProbeConfig()
        c = self.config
        self.score = nn.Sequential(
            nn.LayerNorm(c.input_dim),
            nn.Linear(c.input_dim, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, 1),
        )

    def forward(self, representation: torch.Tensor) -> dict[str, torch.Tensor]:
        if representation.ndim != 3 or int(representation.shape[-1]) != self.config.input_dim:
            raise ValueError(f"expected [Q,N,{self.config.input_dim}], got {tuple(representation.shape)}")
        if int(representation.shape[0]) <= 0 or int(representation.shape[1]) <= 0:
            raise ValueError("empty query/candidate axes")
        if not bool(torch.isfinite(representation.float()).all()):
            raise FloatingPointError("nonfinite L84 representation")
        return {"interaction": self.score(representation).squeeze(-1)}

    def parameter_report(self) -> dict[str, object]:
        trainable = [(name, int(value.numel())) for name, value in self.named_parameters() if value.requires_grad]
        return {
            "config": asdict(self.config),
            "trainable_parameter_count": int(sum(count for _, count in trainable)),
            "total_parameter_count": int(sum(value.numel() for value in self.parameters())),
            "trainable_parameters": [name for name, _ in trainable],
            "primary_score": "interaction",
            "candidate_deletion": False,
            "candidate_truncation": False,
        }


__all__ = ["L84PairedProbe", "L84PairedProbeConfig"]
