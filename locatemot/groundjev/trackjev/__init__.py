"""TrackJEV: joint identity decisions for unified online MOT/RMOT/OVMOT.

Modules:

* :mod:`decision_schema` — the structured question vocabulary and label contract
* :mod:`candidate_competition` — sparse competition graph, entity clusters,
  admission supervision and prompt rows
* :mod:`global_assignment` — one-to-one global identity allocation
* :mod:`native_onejev_adapter` — the pretrained OneJev-0.8B adapter (runtime)
* :mod:`model` — TrackJEV module used by the tracker
* :mod:`losses` — structured slot losses for the registered supervision
* :mod:`rollout` — own-state video rollout used for DAgger-style training
"""
from .decision_schema import (
    MAX_IDENTITY_OPTIONS,
    MAX_SHORTLIST,
    NEW_TOKEN,
    NO_MATCH_TOKEN,
    REJECT_TOKEN,
    IdentityOption,
    ParsedDecision,
    build_identity_options,
    parse_identity_answer,
    shortlist_coverage,
    slot_targets,
    supervised_decision,
)

__all__ = [
    "MAX_IDENTITY_OPTIONS", "MAX_SHORTLIST", "NEW_TOKEN", "NO_MATCH_TOKEN", "REJECT_TOKEN",
    "IdentityOption", "ParsedDecision", "build_identity_options",
    "parse_identity_answer", "shortlist_coverage", "slot_targets", "supervised_decision",
]
