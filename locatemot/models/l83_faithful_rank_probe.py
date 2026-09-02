"""The fixed-capacity interaction probe used for L83 faithful target bags."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class L83RankProbeConfig:
    input_dim: int = 256
    hidden: int = 256
    dropout: float = 0.05


class L83FaithfulRankProbe(nn.Module):
    """A parameter-matched scalar interaction probe.

    The representation is the only input.  Target-bag aggregation and all
    duplicate-aware supervision live in ``l83_target_bag_loss``; no IDs or
    candidate metadata enter this module.
    """

    def __init__(self, config: L83RankProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or L83RankProbeConfig()
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
            raise ValueError("empty query/candidate axis")
        if not bool(torch.isfinite(representation.float()).all()):
            raise FloatingPointError("nonfinite L83 representation")
        return {"interaction": self.score(representation).squeeze(-1)}

    def parameter_report(self) -> dict[str, object]:
        trainable = [(name, int(value.numel())) for name, value in self.named_parameters() if value.requires_grad]
        return {
            "config": asdict(self.config),
            "trainable_parameter_count": int(sum(count for _, count in trainable)),
            "total_parameter_count": int(sum(value.numel() for value in self.parameters())),
            "trainable_parameters": [name for name, _ in trainable],
            "primary_score": "interaction",
            "target_bag_score": "max per unique candidate_gt target",
            "candidate_deletion": False, "candidate_truncation": False,
        }


__all__ = ["L83FaithfulRankProbe", "L83RankProbeConfig"]
