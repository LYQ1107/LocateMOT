"""Deterministic task sampling contract for LocateMOT-U stages."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

TASKS = ("mot", "ovmot", "rmot", "point", "box", "mask", "grounding")


@dataclass(frozen=True)
class TaskSamplingConfig:
    probabilities: dict[str, float]
    seed: int = 20261001

    def __post_init__(self) -> None:
        unknown = set(self.probabilities) - set(TASKS)
        if unknown:
            raise ValueError(f"unknown tasks: {sorted(unknown)}")
        if not self.probabilities or abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("task probabilities must sum to one")
        if any(value <= 0 for value in self.probabilities.values()):
            raise ValueError("task probabilities must be positive")


@dataclass
class TaskSampler:
    config: TaskSamplingConfig

    def __post_init__(self) -> None:
        self._rng = Random(self.config.seed)

    def next_task(self) -> str:
        names = tuple(self.config.probabilities)
        weights = tuple(self.config.probabilities[name] for name in names)
        return self._rng.choices(names, weights=weights, k=1)[0]
