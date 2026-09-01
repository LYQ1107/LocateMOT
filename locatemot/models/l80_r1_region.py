"""L80-R1 region-interface-only variant.

R1 changes only the number of fixed spatial samples supplied by the frozen
CLIP pyramid.  The observation/history, text, query-to-region, set, heads and
loss remain the L80-R0 contract.  No old score, ID or candidate selection is
introduced.
"""
from __future__ import annotations

from dataclasses import dataclass

from locatemot.models.l80_raw_region_correspondence import L80Config, L80RawRegionCorrespondence


@dataclass(frozen=True)
class L80R1Config(L80Config):
    tokens_per_scale: int = 81  # 8x8 ROI + 4x4 context + 1 scene token


class L80R1RegionCorrespondence(L80RawRegionCorrespondence):
    """Same L80 head with a higher-resolution frozen-region interface."""

    def __init__(self, config: L80R1Config | None = None) -> None:
        super().__init__(config or L80R1Config())


__all__ = ["L80R1Config", "L80R1RegionCorrespondence"]
