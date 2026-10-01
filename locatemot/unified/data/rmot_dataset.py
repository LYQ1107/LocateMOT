"""Legal Refer-KITTI sample contract for LocateMOT-U."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .legal_scope import assert_legal_video


@dataclass(frozen=True)
class RMOTSample:
    frames: Any
    expression: str
    targets: Any
    dataset: str
    video: str

    task_type: str = "rmot"

    def __post_init__(self) -> None:
        assert_legal_video(self.video, dataset=self.dataset)
