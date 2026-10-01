"""Evaluation boundary placeholder with legal-scope metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    task: str
    dataset: str
    metrics: dict[str, float]
    official_test_labels_read: bool = False
    metadata: dict[str, Any] | None = None
