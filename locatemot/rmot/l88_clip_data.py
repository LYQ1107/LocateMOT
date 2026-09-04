"""L88 data bridge over the already audited L69/L85/L86 contracts.

This module deliberately keeps the existing L86/L87 sidecar labels and causal
observation construction.  The only representation replaced by L88 is the
GroundingDINO Z1 tensor; callers must ignore the historical ``FrameExample.z1``
and provide the L88 adapted Z1 from :mod:`l88_grounding_runtime`.

The base worktree does not track the earlier L79/L80 compatibility modules, so
the asset-root package path is extended before importing the compact bridge.
No files are copied and no old module is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
import locatemot.rmot as _rmot_package  # noqa: E402

if str(ASSET_ROOT / "locatemot" / "rmot") not in [str(x) for x in _rmot_package.__path__]:
    _rmot_package.__path__.append(str(ASSET_ROOT / "locatemot" / "rmot"))

from locatemot.rmot.l80_data import L80BankStore  # noqa: E402
from locatemot.rmot.l86_clip_data import (  # noqa: E402
    FrameExample,
    L86ClipStore,
    HISTORY,
    OBS_DIM,
)


class L88ClipStore:
    """Expose L86 sidecar context while making the adapted-Z1 boundary explicit."""

    def __init__(self, cache_root: Path, *, load_cache_into_ram: bool = True) -> None:
        self._base = L86ClipStore(cache_root, load_cache_into_ram=load_cache_into_ram)
        self.groups = self._base.groups
        self.train_keys = self._base.train_keys
        self.dev_keys = self._base.dev_keys
        self.cache_root = Path(cache_root).resolve()

    def build_frame(self, group_key: str, query_ids: Any = None, *, temporal_enabled: bool) -> FrameExample:
        frame = self._base.build_frame(group_key, query_ids, temporal_enabled=temporal_enabled)
        self._assert_frame(frame)
        return frame

    def build_clip(self, anchor_key: str, *, temporal_enabled: bool, clip_length: int = 4) -> list[FrameExample]:
        frames = self._base.build_clip(anchor_key, temporal_enabled=temporal_enabled, clip_length=clip_length)
        for frame in frames:
            self._assert_frame(frame)
        return frames

    @staticmethod
    def _assert_frame(frame: FrameExample) -> None:
        candidate_count = len(frame.row_offsets)
        if candidate_count != len(frame.row_keys):
            raise AssertionError(f"L88 candidate row drift: {frame.group_key}")
        if frame.current_observation.shape != (candidate_count, OBS_DIM):
            raise AssertionError(f"L88 observation shape drift: {frame.group_key}")
        if frame.history_observations.shape != (candidate_count, HISTORY, OBS_DIM):
            raise AssertionError(f"L88 history shape drift: {frame.group_key}")
        if frame.history_mask.shape != (candidate_count, HISTORY):
            raise AssertionError(f"L88 history mask shape drift: {frame.group_key}")
        if frame.history_frame_ids.numel() and bool((frame.history_frame_ids[frame.history_mask] > int(frame.frame_id)).any()):
            raise AssertionError(f"L88 future history: {frame.group_key}")
        if frame.row_offsets != sorted(frame.row_offsets):
            raise AssertionError(f"L88 native row order drift: {frame.group_key}")
        for name, value in (
            ("current_observation", frame.current_observation),
            ("history_observations", frame.history_observations),
            ("z1_ignored", frame.z1),
            ("text_global", frame.text_global),
            ("frame_global", frame.frame_global),
        ):
            if not bool(torch.isfinite(value.float()).all()):
                raise FloatingPointError(f"nonfinite L88 {name}: {frame.group_key}")

    @property
    def bank_store(self) -> Any:
        return self._base.bank_store

    @property
    def cache_paths(self) -> dict[str, Path]:
        return self._base.cache_paths

    def release_loaded_cache_items(self) -> int:
        """Release lazily loaded L85 frame items while retaining the index.

        L88's query-independent encoder cache is read by a separate streaming
        reader.  The inherited sidecar cache may still hold the compact L85
        frame objects used for causal observations; clearing those objects at
        a group boundary keeps resident memory bounded without changing the
        immutable cache or its row contract.
        """
        count = len(self._base.cache_items)
        self._base.cache_items.clear()
        return count

    def close(self) -> None:
        self._base.close()


__all__ = ["ASSET_ROOT", "FrameExample", "HISTORY", "L80BankStore", "L88ClipStore", "OBS_DIM"]
