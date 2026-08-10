"""Per-track state for full-video association."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ACTIVE = "ACTIVE"
LOST = "LOST"
TERMINATED = "TERMINATED"
TENTATIVE = "TENTATIVE"


@dataclass
class Obs:
    frame: int
    box: np.ndarray  # xyxy pixels
    features: Dict[str, np.ndarray]
    gen_score: float = 0.0


@dataclass
class TrackState:
    track_id: int
    last_box: np.ndarray
    last_frame: int
    status: str = TENTATIVE
    hits: int = 1
    age: int = 1
    lost_age: int = 0
    prev_box: Optional[np.ndarray] = None
    last_features: Optional[Dict[str, np.ndarray]] = None
    anchor_features: Optional[Dict[str, np.ndarray]] = None
    ema_features: Optional[Dict[str, np.ndarray]] = None
    history: List[Obs] = field(default_factory=list)
    kalman: Optional[object] = None
    velocity: Optional[np.ndarray] = None
    observations: Dict[int, np.ndarray] = field(default_factory=dict)
    confidence: float = 0.0
    birth_frame: int = 0
    slot: int = -1

    @property
    def is_active(self):
        return self.status in (ACTIVE, TENTATIVE, LOST)

    def box_at(self, frame: int) -> Optional[np.ndarray]:
        if self.last_frame == frame:
            return self.last_box
        return None
