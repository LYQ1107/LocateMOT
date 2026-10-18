"""Causal semantic probability cache with explicit invalidation boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any


def normalize_expression(expression: str | None) -> str:
    return re.sub(r"\s+", " ", str(expression or "").strip().lower())


@dataclass(frozen=True)
class SemanticCacheKey:
    track_id: int
    task: str
    expression: str
    verifier_checkpoint_hash: str
    protocol_version: str = "semantic-cache-v1"

    def digest(self) -> str:
        payload = "|".join(
            (str(self.track_id), self.task, self.expression, self.verifier_checkpoint_hash, self.protocol_version)
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SemanticCacheEntry:
    p_yes: float
    frame: int
    evidence_frame_ids: tuple[int, ...]
    checkpoint_hash: str
    query_version: str
    track_id: int
    task: str
    expression: str


class SemanticTemporalCache:
    """Stores semantic probabilities only; association state remains external."""

    def __init__(self, *, ttl_frames: int = 30, enabled: bool = False) -> None:
        self.ttl_frames = max(0, int(ttl_frames))
        self.enabled = bool(enabled)
        self._entries: dict[str, SemanticCacheEntry] = {}

    def make_key(
        self,
        *,
        track_id: int,
        task: str,
        expression: str | None,
        verifier_checkpoint_hash: str,
        protocol_version: str = "semantic-cache-v1",
    ) -> SemanticCacheKey:
        return SemanticCacheKey(
            track_id=int(track_id),
            task=str(task),
            expression=normalize_expression(expression),
            verifier_checkpoint_hash=str(verifier_checkpoint_hash),
            protocol_version=protocol_version,
        )

    def get(self, key: SemanticCacheKey, *, current_frame: int) -> SemanticCacheEntry | None:
        if not self.enabled:
            return None
        entry = self._entries.get(key.digest())
        if entry is None:
            return None
        if int(current_frame) < entry.frame or int(current_frame) - entry.frame > self.ttl_frames:
            self._entries.pop(key.digest(), None)
            return None
        if any(frame >= int(current_frame) for frame in entry.evidence_frame_ids):
            self._entries.pop(key.digest(), None)
            return None
        return entry

    def put(self, key: SemanticCacheKey, *, p_yes: float, frame: int, evidence_frame_ids: tuple[int, ...], query_version: str) -> None:
        if not self.enabled:
            return
        if not 0.0 <= float(p_yes) <= 1.0:
            raise ValueError("semantic probability must be in [0, 1]")
        if any(int(evidence) > int(frame) for evidence in evidence_frame_ids):
            raise ValueError("semantic cache accepts causal evidence only")
        self._entries[key.digest()] = SemanticCacheEntry(
            p_yes=float(p_yes),
            frame=int(frame),
            evidence_frame_ids=tuple(int(value) for value in evidence_frame_ids),
            checkpoint_hash=key.verifier_checkpoint_hash,
            query_version=str(query_version),
            track_id=key.track_id,
            task=key.task,
            expression=key.expression,
        )

    def invalidate(
        self,
        *,
        track_id: int | None = None,
        task: str | None = None,
        expression: str | None = None,
        reason: str,
    ) -> int:
        """Invalidate on query change, reappearance, contradiction, or reset."""

        if reason not in {"query_change", "reappearance", "conflict", "track_terminated", "manual"}:
            raise ValueError(f"unknown cache invalidation reason: {reason}")
        normalized = normalize_expression(expression) if expression is not None else None
        removed = 0
        for digest in list(self._entries):
            entry = self._entries[digest]
            matches = (
                (track_id is None or entry.track_id == int(track_id))
                and (task is None or entry.task == str(task))
                and (normalized is None or entry.expression == normalized)
            )
            if not matches:
                continue
            self._entries.pop(digest, None)
            removed += 1
        return removed

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "SemanticCacheEntry",
    "SemanticCacheKey",
    "SemanticTemporalCache",
    "normalize_expression",
]
