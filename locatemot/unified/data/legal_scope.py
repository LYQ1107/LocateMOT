"""Single source of truth for LocateMOT-U development label scope.

Every unified dataset adapter must call :func:`assert_legal_video` before
opening a video or label path.  Official Refer-KITTI evaluation videos are
represented as reserved metadata only and are never opened during development.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

# The forbidden set is the only video boundary used by unified adapters.  The
# partition below is the v2 split frozen before formal U1 evaluation.
FORBIDDEN_VIDEOS = frozenset({"0005", "0011", "0013", "0019"})
LEGAL_SPLIT_VERSION = "v2"
LEGAL_SPLITS = {
    "refer_kitti_v1": {
        "fit": ("0002", "0003", "0004", "0006", "0010", "0012", "0014", "0015", "0016"),
        "calibration": ("0001", "0007", "0020"),
        "validation": ("0008", "0009", "0018"),
        "official_eval_reserved": ("0005", "0011", "0013"),
    },
    "refer_kitti_v2": {
        "fit": ("0000", "0002", "0003", "0006", "0008", "0010", "0012", "0014", "0017"),
        "calibration": ("0001", "0009", "0020"),
        "validation": ("0007", "0015", "0016"),
        "official_eval_reserved": ("0005", "0011", "0013", "0019"),
    },
}


def assert_legal_video(video: str | int, *, dataset: str | None = None) -> str:
    value = str(video).zfill(4)
    if value in FORBIDDEN_VIDEOS:
        raise RuntimeError(f"official evaluation video is forbidden during development: {value}")
    if dataset is not None:
        if dataset not in LEGAL_SPLITS:
            raise KeyError(dataset)
        allowed = set(sum((list(LEGAL_SPLITS[dataset][part]) for part in ("fit", "calibration", "validation")), []))
        if value not in allowed:
            raise RuntimeError(f"video {value} is outside the legal {dataset} development scope")
    return value


def assert_legal_videos(videos: Iterable[str | int], *, dataset: str | None = None) -> tuple[str, ...]:
    values = tuple(assert_legal_video(video, dataset=dataset) for video in videos)
    if len(set(values)) != len(values):
        raise ValueError("duplicate video in legal scope")
    return values


def assert_legal_path(path: str | Path) -> Path:
    candidate = Path(path)
    parts = set(candidate.parts)
    forbidden = parts.intersection(FORBIDDEN_VIDEOS)
    if forbidden:
        raise RuntimeError(f"path enters forbidden official-evaluation video scope: {sorted(forbidden)}")
    return candidate


__all__ = [
    "FORBIDDEN_VIDEOS",
    "LEGAL_SPLIT_VERSION",
    "LEGAL_SPLITS",
    "assert_legal_path",
    "assert_legal_video",
    "assert_legal_videos",
]
