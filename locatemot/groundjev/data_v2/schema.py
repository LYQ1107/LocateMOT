"""FullDataV2 unified training record schema.

One physical row of supervision for TrackJEV.  The schema is deliberately
explicit about *what is unknown*: UNKNOWN is not REJECT, an unannotated area is
not background, an identity missing from a retrieval shortlist is not NEW, and a
missed detection does not retire a real identity.

Nothing in this module reads images or GT files; it only defines the contract
that every dataset adapter must fill.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

SCHEMA_VERSION = "full-data-v2.1"


class IdentityDecision(str, Enum):
    """The vocabulary OneJev is trained and evaluated on."""

    MATCH = "MATCH"            # owned by an existing live identity
    REACTIVATE = "REACTIVATE"  # owned by an existing lost identity
    NEW = "NEW"                # a genuinely new physical entity
    REJECT = "REJECT"          # invalid/duplicate proposal, must not create an id
    NO_MATCH = "NO_MATCH"      # valid object, no supported identity in the shortlist
    UNKNOWN = "UNKNOWN"        # supervision absent or unreliable


class AnnotationQuality(str, Enum):
    HUMAN_EXHAUSTIVE = "human_exhaustive"
    HUMAN_PARTIAL = "human_partial"
    AUTOMATIC_VERIFIED = "automatic_verified"
    MASK_DERIVED = "mask_derived"
    UNVERIFIED = "unverified"


@dataclass
class TrainingRecord:
    """A single supervised (or explicitly unsupervised) training row."""

    # identity of the source
    dataset_name: str
    source_video_id: str
    canonical_video_uid: str
    split: str
    frame_index: int
    timestamp: float

    # pixels
    image_path: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    # candidate-level evidence (filled by the perception stage, not by parsers)
    candidate_box: Optional[list[float]] = None
    candidate_score: Optional[float] = None
    candidate_source: Optional[str] = None
    native_region_feature: Optional[list[float]] = None
    temporal_region_feature: Optional[list[float]] = None

    # physical identity supervision
    physical_identity: Optional[int] = None
    physical_identity_valid: bool = False
    identity_decision: str = IdentityDecision.UNKNOWN.value
    shortlisted_ids: list[int] = field(default_factory=list)
    correct_identity_shortlisted: Optional[bool] = None

    # referring-expression supervision
    query_text: Optional[str] = None
    query_id: Optional[str] = None
    query_target_ids: list[int] = field(default_factory=list)
    query_supervision_valid: bool = False

    # open-vocabulary category supervision
    category: Optional[str] = None
    category_supervision_valid: bool = False
    category_is_base: Optional[bool] = None

    # mask / annotation provenance
    mask: Optional[dict] = None
    annotation_quality: str = AnnotationQuality.UNVERIFIED.value
    annotation_exhaustive: bool = False

    # per-task label validity flags (all must be checked before use in a loss)
    candidate_duplicate_label_valid: bool = False
    identity_match_label_valid: bool = False
    birth_label_valid: bool = False
    memory_write_label_valid: bool = False

    # provenance
    source_annotation_path: Optional[str] = None
    source_annotation_sha256: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


def validate_record(row: dict[str, Any]) -> list[str]:
    """Return a list of contract violations; empty means the row is usable."""
    problems: list[str] = []
    for key in ("dataset_name", "source_video_id", "canonical_video_uid", "split"):
        if not row.get(key):
            problems.append(f"missing {key}")
    if row.get("frame_index") is None:
        problems.append("missing frame_index")
    decision = row.get("identity_decision")
    if decision not in {d.value for d in IdentityDecision}:
        problems.append(f"invalid identity_decision {decision!r}")
    if row.get("physical_identity_valid") and row.get("physical_identity") is None:
        problems.append("identity flagged valid without a physical_identity")
    if decision == IdentityDecision.REJECT.value and row.get("physical_identity_valid"):
        problems.append("REJECT row must not carry a valid physical identity")
    if row.get("identity_match_label_valid") and not row.get("physical_identity_valid"):
        problems.append("match label valid without identity supervision")
    if row.get("birth_label_valid") and decision not in (
        IdentityDecision.NEW.value, IdentityDecision.UNKNOWN.value,
        IdentityDecision.MATCH.value, IdentityDecision.REACTIVATE.value,
    ):
        problems.append("birth label on a non-object row")
    if row.get("query_supervision_valid") and not row.get("query_text"):
        problems.append("query supervision flagged valid without query_text")
    if row.get("category_supervision_valid") and not row.get("category"):
        problems.append("category supervision valid without category")
    if row.get("annotation_quality") not in {q.value for q in AnnotationQuality}:
        problems.append(f"invalid annotation_quality {row.get('annotation_quality')!r}")
    return problems


def iter_valid(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        if not validate_record(row):
            yield row
