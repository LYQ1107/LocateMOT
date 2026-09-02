"""The identical small probe used for all three L82 frozen representations."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class L82RankProbeConfig:
    input_dim: int = 256
    hidden: int = 256
    dropout: float = 0.05


class L82FactorizedRankProbe(nn.Module):
    """Primary candidate×query interaction plus nuisance main effects.

    ``interaction`` is the only trained/evaluated primary score.  The
    candidate-only and query-only outputs reuse the exact same probe weights
    after averaging one axis and are diagnostics, not semantic outputs.
    """

    def __init__(self, config: L82RankProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or L82RankProbeConfig()
        c = self.config
        self.score = nn.Sequential(
            nn.LayerNorm(c.input_dim),
            nn.Linear(c.input_dim, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, 1),
        )

    def _score(self, value: torch.Tensor) -> torch.Tensor:
        return self.score(value).squeeze(-1)

    def forward(self, representation: torch.Tensor) -> dict[str, torch.Tensor]:
        if representation.ndim != 3 or representation.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected [Q,N,{self.config.input_dim}], got {tuple(representation.shape)}")
        if representation.shape[0] <= 0 or representation.shape[1] <= 0:
            raise ValueError("L82 rank probe requires nonempty query and candidate axes")
        if not bool(torch.isfinite(representation.float()).all()):
            raise FloatingPointError("nonfinite L82 representation")
        interaction = self._score(representation)
        candidate_main = self._score(representation.mean(dim=0))
        query_main = self._score(representation.mean(dim=1))
        return {
            "interaction": interaction,
            "candidate_main": candidate_main,
            "query_main": query_main,
        }

    def parameter_report(self) -> dict[str, object]:
        trainable = [(name, int(value.numel())) for name, value in self.named_parameters() if value.requires_grad]
        return {
            "config": asdict(self.config),
            "trainable_parameter_count": int(sum(count for _, count in trainable)),
            "total_parameter_count": int(sum(value.numel() for value in self.parameters())),
            "trainable_parameters": [name for name, _ in trainable],
            "primary_score": "interaction",
            "nuisance_controls": ["candidate_main", "query_main"],
            "candidate_deletion": False,
            "candidate_truncation": False,
        }


__all__ = ["L82FactorizedRankProbe", "L82RankProbeConfig"]
