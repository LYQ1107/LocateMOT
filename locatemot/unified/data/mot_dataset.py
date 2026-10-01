"""Task-MOT sample contract; storage adapters are added in U2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .legal_scope import assert_legal_video


@dataclass(frozen=True)
class MOTSample:
    frames: Any
    annotations: Any
    dataset: str
    video: str

    task_type: str = "mot"

    def __post_init__(self) -> None:
        # The guard is intentionally shared with RMOT so a future storage
        # adapter cannot open a reserved Refer-KITTI video by accident.
        assert_legal_video(self.video)
