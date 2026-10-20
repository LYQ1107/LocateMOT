"""Causal dynamic identity options for a GroundJEV frame."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..trackjev.decision_schema import NEW_TOKEN, NO_MATCH_TOKEN, REJECT_TOKEN


@dataclass(frozen=True)
class IdentitySlot:
    token: str
    track_id: int | None
    lost_age: int = 0
    kind: str = "existing"


@dataclass
class DynamicIdentitySlots:
    slots: list[IdentitySlot]

    @property
    def tokens(self) -> list[str]:
        return [slot.token for slot in self.slots]

    @property
    def existing(self) -> list[IdentitySlot]:
        return [slot for slot in self.slots if slot.track_id is not None]

    def index(self, token: str) -> int:
        return self.tokens.index(str(token))


def build_dynamic_slots(
    tracks: Iterable[object],
    *,
    max_existing: int = 31,
    allow_new: bool = True,
    allow_no_match: bool = True,
    allow_reject: bool = True,
) -> DynamicIdentitySlots:
    """Build options from current causal tracker state, never from GT."""
    def track_id(track: object) -> int:
        value = getattr(track, "track_id", None)
        if value is None:
            value = getattr(track, "persistent_track_id")
        return int(value)

    def lost_age(track: object) -> int:
        value = getattr(track, "lost_age", None)
        if value is None:
            value = getattr(track, "lost_duration", 0)
        return int(value or 0)

    def key(track: object):
        age = lost_age(track)
        return (age > 0, age, track_id(track))
    slots: list[IdentitySlot] = []
    seen: set[int] = set()
    for track in sorted(list(tracks), key=key)[: int(max_existing)]:
        current_id = track_id(track)
        if current_id in seen:
            continue
        seen.add(current_id)
        age = lost_age(track)
        slots.append(IdentitySlot(str(current_id), current_id, age,
                                  "reactivate" if age > 0 else "existing"))
    if allow_new:
        slots.append(IdentitySlot(NEW_TOKEN, None, 0, "new"))
    if allow_no_match:
        slots.append(IdentitySlot(NO_MATCH_TOKEN, None, 0, "no_match"))
    if allow_reject:
        slots.append(IdentitySlot(REJECT_TOKEN, None, 0, "reject"))
    return DynamicIdentitySlots(slots)
