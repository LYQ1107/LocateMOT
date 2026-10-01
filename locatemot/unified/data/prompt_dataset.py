"""Point/box/mask prompt sample contracts for U4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptSample:
    frames: Any
    prompt: Any
    target: Any
    prompt_type: str

    def __post_init__(self) -> None:
        if self.prompt_type not in {"point", "box", "mask"}:
            raise ValueError(self.prompt_type)

    @property
    def task_type(self) -> str:
        return self.prompt_type
