"""Dynamic, auditable identity choices for one causal frame."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch

from ..trackjev.decision_schema import NO_MATCH_TOKEN, NEW_TOKEN, REJECT_TOKEN


@dataclass(frozen=True)
class IdentitySlot:
    token: str
    kind: str
    track_id: Optional[int] = None
    lost_age: int = 0
    state: str = "ACTIVE"
    text: str = ""
    retrieval_rank: Optional[int] = None

    @property
    def is_existing(self) -> bool:
        return self.kind == "existing"


def build_dynamic_slots(
    tracks: Iterable[Any],
    *,
    max_history: int = 31,
    allow_new: bool = True,
    allow_no_match: bool = True,
    allow_reject: bool = True,
) -> list[IdentitySlot]:
    """Build a per-frame shortlist without injecting GT identity labels."""
    rows = []
    for track in tracks:
        if isinstance(track, dict):
            # Native identity-memory exports use ``id`` while tracker
            # candidates commonly use ``track_id``.  Both are causal
            # identifiers and must map to the same dynamic slot contract.
            tid_value = track.get("track_id")
            if tid_value is None:
                tid_value = track.get("persistent_track_id")
            if tid_value is None:
                tid_value = track.get("id")
            if tid_value is None:
                raise ValueError("track dictionary has no causal identity id")
            tid = int(tid_value)
            lost_age = int(track.get("lost_age", track.get("lost_duration", 0)) or 0)
            state = str(track.get("state", "ACTIVE"))
            prior = float(track.get("retrieval_prior", track.get("prior", 0.0)) or 0.0)
        else:
            tid_value = getattr(track, "track_id", None)
            if tid_value is None:
                tid_value = getattr(track, "persistent_track_id")
            tid = int(tid_value)
            lost_age = int(getattr(track, "lost_age", 0) or 0)
            state = str(getattr(track, "state", "ACTIVE"))
            prior = float(getattr(track, "retrieval_prior", 0.0) or 0.0)
        rows.append((prior, lost_age, tid, state))
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    rows = rows[: int(max_history)]
    slots = [IdentitySlot(token=str(tid), kind="existing", track_id=tid,
                          lost_age=lost_age, state=state, retrieval_rank=rank,
                          text=(f"REACTIVATE identity {tid}" if lost_age else
                                f"MATCH identity {tid}"))
             for rank, (_, lost_age, tid, state) in enumerate(rows)]
    if allow_new:
        slots.append(IdentitySlot(token=NEW_TOKEN, kind="new",
                                  text="NEW physical entity"))
    if allow_no_match:
        slots.append(IdentitySlot(token=NO_MATCH_TOKEN, kind="no_match",
                                  text="valid entity, no supported shortlist identity"))
    if allow_reject:
        slots.append(IdentitySlot(token=REJECT_TOKEN, kind="reject",
                                  text="invalid or duplicate proposal"))
    return slots


def slot_tokens(slots: Sequence[IdentitySlot]) -> list[str]:
    tokens = [slot.token for slot in slots]
    if len(tokens) != len(set(tokens)):
        raise ValueError("dynamic identity options contain duplicate tokens")
    return tokens


def slot_index(slots: Sequence[IdentitySlot]) -> dict[str, int]:
    return {token: index for index, token in enumerate(slot_tokens(slots))}


def stack_slot_features(
    slots: Sequence[IdentitySlot],
    *,
    existing_features: Optional[dict[int, torch.Tensor]] = None,
    special_features: Optional[dict[str, torch.Tensor]] = None,
    hidden_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Materialize option embeddings in the exact advertised slot order."""
    existing_features = existing_features or {}
    special_features = special_features or {}
    values = []
    for slot in slots:
        value = (existing_features.get(int(slot.track_id)) if slot.is_existing
                 else special_features.get(slot.token))
        if value is None:
            if hidden_size is None:
                raise ValueError(f"missing feature for slot {slot.token}")
            value = torch.zeros(hidden_size, device=device, dtype=dtype or torch.float32)
        if value.ndim != 1:
            raise ValueError(f"slot feature for {slot.token} must be one-dimensional")
        values.append(value)
    if not values:
        size = int(hidden_size or 0)
        return torch.empty((0, size), device=device, dtype=dtype or torch.float32)
    return torch.stack(values, dim=0)
