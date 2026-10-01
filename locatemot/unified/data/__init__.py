"""Unified task data contracts and legal-scope guards."""

from .legal_scope import (
    FORBIDDEN_VIDEOS,
    LEGAL_SPLITS,
    assert_legal_video,
    assert_legal_videos,
)

__all__ = ["FORBIDDEN_VIDEOS", "LEGAL_SPLITS", "assert_legal_video", "assert_legal_videos"]
