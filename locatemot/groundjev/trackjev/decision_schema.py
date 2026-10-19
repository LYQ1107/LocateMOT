"""TrackJEV structured decision schema.

TrackJEV asks the pretrained OneJev model a small set of structured questions per
candidate group.  This module owns that vocabulary so that training labels,
runtime inference and evaluation all speak the same language.

Questions (one group per frame, one answer set per candidate):

* ``admit_i``   Noul  — is candidate i a valid physical entity at all?
* ``identity_i`` Choice — which identity owns it: MATCH(id) / REACTIVATE(id) /
                        NEW / REJECT (NO_MATCH is represented as HOLD + no
                        identity, never as NEW)
* ``memory_i``  Noul  — after the *actual* assignment, should this identity's
                        long-term memory be updated? (asked only for candidates
                        that receive an identity)
* ``risk_i``    Score (optional) — predicted risk that this decision damages the
                        identity state within the registered horizon

The option list is capped: OneJev's Choice question has an official option limit
and long-history reasoning is not helped by dumping 200 ids into one question.
``MAX_IDENTITY_OPTIONS`` reserves two slots for NEW and REJECT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence

from ..data_v2.schema import IdentityDecision

MAX_IDENTITY_OPTIONS = 34
RESERVED_OPTION_SLOTS = 3   # NEW, REJECT and NO_MATCH
MAX_SHORTLIST = MAX_IDENTITY_OPTIONS - RESERVED_OPTION_SLOTS

NEW_TOKEN = "NEW"
REJECT_TOKEN = "REJECT"
NO_MATCH_TOKEN = "NO_MATCH"

OPTION_TEXT_NEW = ("A genuinely new physical entity that has no identity in this video yet "
                   "and is not a duplicate of another current candidate")
OPTION_TEXT_REJECT = ("Invalid or duplicate proposal: background, a second box of the same "
                      "physical entity already accepted in this frame, or unusable evidence")
OPTION_TEXT_EXISTING = "Existing persistent identity {track_id}; compare history and visual evidence"
OPTION_TEXT_NO_MATCH = ("A valid physical object that was observed before in this video, but none of "
                        "the listed identities supports its match. Hold the association instead of "
                        "creating a new identity.")
OPTION_TEXT_REACTIVATE = ("Existing identity {track_id} that is currently not visible; it may "
                          "reappear after occlusion")

ADMIT_TEXT = ("Is current candidate {index} a valid single physical object in this frame? "
              "A second box of an object that is already admitted is not a valid entity.")
MEMORY_TEXT = ("Candidate {index} was assigned to identity {track_id}. Is this observation clean "
               "and reliable enough to update that identity's long-term memory? Reject occlusion, "
               "ambiguous identity and drifting boxes.")
RISK_TEXT = ("How likely is it that assigning candidate {index} to identity {track_id} will cause "
             "a wrong identity, an extra identity or memory contamination within the next "
             "{horizon} frames?")


@dataclass
class IdentityOption:
    token: str                      # str(track_id) | "NEW" | "REJECT"
    track_id: Optional[int] = None
    text: str = ""
    prior: Optional[float] = None    # retrieval score, diagnostics only
    lost_age: Optional[int] = None

    @property
    def kind(self) -> Literal["existing", "new", "reject", "no_match"]:
        if self.token == NEW_TOKEN:
            return "new"
        if self.token == REJECT_TOKEN:
            return "reject"
        if self.token == NO_MATCH_TOKEN:
            return "no_match"
        return "existing"


def build_identity_options(shortlist: Sequence[dict[str, Any]],
                           *, allow_new: bool = True,
                           allow_reject: bool = True,
                           allow_no_match: bool = True) -> list[IdentityOption]:
    """Build the Choice options for one candidate group.

    ``shortlist`` entries need ``track_id`` and may carry ``lost_age``/``prior``.
    The shortlist is truncated (not silently reordered) to ``MAX_SHORTLIST``;
    retrieval coverage of the true id is measured separately, so truncation is
    always visible in the diagnostics.
    """
    options: list[IdentityOption] = []
    for entry in list(shortlist)[:MAX_SHORTLIST]:
        track_id = int(entry["track_id"])
        lost_age = entry.get("lost_age")
        text = OPTION_TEXT_REACTIVATE.format(track_id=track_id) if (lost_age or 0) > 0 \
            else OPTION_TEXT_EXISTING.format(track_id=track_id)
        options.append(IdentityOption(token=str(track_id), track_id=track_id, text=text,
                                      prior=entry.get("prior"), lost_age=lost_age))
    if allow_new:
        options.append(IdentityOption(token=NEW_TOKEN, text=OPTION_TEXT_NEW))
    if allow_no_match:
        options.append(IdentityOption(token=NO_MATCH_TOKEN, text=OPTION_TEXT_NO_MATCH))
    if allow_reject:
        options.append(IdentityOption(token=REJECT_TOKEN, text=OPTION_TEXT_REJECT))
    return options


@dataclass
class ParsedDecision:
    """One candidate's decoded decision."""

    candidate_index: int
    decision: str                       # IdentityDecision value
    track_id: Optional[int] = None
    probability: float = 0.0
    margin: float = 0.0                 # best minus second best
    option_count: int = 0
    admitted: Optional[float] = None
    memory_update: Optional[float] = None
    risk: Optional[float] = None
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def parse_identity_answer(index: int, probabilities: dict[str, float],
                          options: Sequence[IdentityOption]) -> ParsedDecision:
    """Decode Choice probabilities into a TrackJEV decision."""
    valid = {option.token: float(probabilities.get(option.token, 0.0)) for option in options}
    ranked = sorted(valid.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return ParsedDecision(index, IdentityDecision.UNKNOWN.value, option_count=0)
    best_token, best_prob = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_prob - second
    lookup = {option.token: option for option in options}
    chosen = lookup.get(best_token)
    if chosen is None:
        return ParsedDecision(index, IdentityDecision.UNKNOWN.value, probability=best_prob,
                              margin=margin, option_count=len(options), probabilities=valid)
    if chosen.kind == "new":
        decision = IdentityDecision.NEW.value
        track_id = None
    elif chosen.kind == "no_match":
        decision = IdentityDecision.NO_MATCH.value
        track_id = None
    elif chosen.kind == "reject":
        decision = IdentityDecision.REJECT.value
        track_id = None
    else:
        decision = (IdentityDecision.REACTIVATE.value if (chosen.lost_age or 0) > 0
                    else IdentityDecision.MATCH.value)
        track_id = chosen.track_id
    return ParsedDecision(index, decision, track_id=track_id, probability=best_prob, margin=margin,
                          option_count=len(options), probabilities=valid)


def supervised_decision(*, physical_identity: Optional[int], identity_valid: bool,
                        is_duplicate: bool, is_valid_entity: bool,
                        identity_in_shortlist: Optional[bool],
                        is_reactivation: bool = False) -> tuple[str, Optional[int]]:
    """Ground-truth decision for a candidate, or UNKNOWN when evidence is missing.

    Order matters:
    1. duplicate proposal -> REJECT (even if it overlaps a real GT box)
    2. valid entity with a known identity -> MATCH / REACTIVATE
    3. valid entity with no known identity but *identity supervision present*
       -> NEW
    4. everything else -> UNKNOWN (never silently NEW or REJECT)
    """
    if is_duplicate:
        return IdentityDecision.REJECT.value, None
    if is_valid_entity and identity_valid and physical_identity is not None:
        return (IdentityDecision.REACTIVATE.value if is_reactivation
                else IdentityDecision.MATCH.value), int(physical_identity)
    if is_valid_entity and identity_valid and physical_identity is None:
        return IdentityDecision.NEW.value, None
    if is_valid_entity and not identity_valid and identity_in_shortlist is False:
        # valid object, identity supervision exists but the true id was not
        # retrieved: this is NO_MATCH/HOLD territory, not NEW.
        return IdentityDecision.NO_MATCH.value, None
    return IdentityDecision.UNKNOWN.value, None


def slot_targets(decision: str, track_id: Optional[int]) -> dict[str, Any]:
    """Turn a decision into the exact Choice slot used by the training loss."""
    if decision in (IdentityDecision.MATCH.value, IdentityDecision.REACTIVATE.value):
        assert track_id is not None, "MATCH/REACTIVATE requires a track id"
        return {"identity_choice": str(int(track_id))}
    if decision == IdentityDecision.NEW.value:
        return {"identity_choice": NEW_TOKEN}
    if decision == IdentityDecision.REJECT.value:
        return {"identity_choice": REJECT_TOKEN}
    if decision == IdentityDecision.NO_MATCH.value:
        return {"identity_choice": NO_MATCH_TOKEN}
    return {}          # UNKNOWN produces no identity slot target


def shortlist_coverage(shortlist: Iterable[int], true_id: Optional[int]) -> Optional[bool]:
    """Was the true identity inside the retrieval shortlist? (None when unknown)."""
    if true_id is None:
        return None
    return int(true_id) in {int(i) for i in shortlist}
