"""Open-vocabulary MOT sample contract; C-TAO adapter is added in U2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .legal_scope import assert_legal_video


@dataclass(frozen=True)
class OVMOTSample:
    frames: Any
    annotations: Any
    class_specification: str
    dataset: str
    video: str

    task_type: str = "ovmot"

    def __post_init__(self) -> None:
        assert_legal_video(self.video)
