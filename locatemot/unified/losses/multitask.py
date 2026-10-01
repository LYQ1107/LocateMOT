"""Explicit multi-task loss composition without hidden task-specific models."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MultiTaskLoss:
    weights: dict[str, float]

    def __call__(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for name, weight in self.weights.items():
            if name not in values:
                continue
            term = values[name] * float(weight)
            total = term if total is None else total + term
        if total is None:
            raise ValueError("no registered loss terms present")
        return total
