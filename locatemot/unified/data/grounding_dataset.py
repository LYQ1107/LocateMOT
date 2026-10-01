"""Image-text grounding sample contract for U1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundingSample:
    image: Any
    text: str
    boxes: Any
    labels: Any

    task_type: str = "grounding"
