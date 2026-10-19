"""Supervision targets and optional structured losses for TrackJEV.

Two responsibilities, kept apart on purpose:

1. :func:`build_supervision` turns physical frame data (GT boxes/ids, query
   targets, category labels) into the exact per-candidate slots the OneJev SFT
   exporter writes.  Every slot carries a *validity* flag, so a loss can never
   be computed on a label the dataset does not own.
2. :func:`structured_losses` is an optional torch objective for a head that
   predicts the same slots directly (used for ablations and for the non-JEV
   control arms), including the one-to-one consistency term between two
   candidates competing for the same identity.

Discrete Hungarian assignment is never differentiated; it only produces targets
and evaluation-time decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .candidate_competition import AdmissionTarget, entity_clusters, sparse_edges, admission_targets
from .decision_schema import NEW_TOKEN, REJECT_TOKEN, slot_targets, supervised_decision


@dataclass
class CandidateSupervision:
    """Everything a training row needs to know about one candidate."""

    index: int
    admit_target: Optional[int] = None          # 1 valid entity / 0 duplicate-or-background / None unknown
    admit_valid: bool = False
    identity_target: Optional[str] = None       # "12" | NEW | REJECT | None (no slot)
    identity_valid: bool = False
    memory_target: Optional[int] = None         # 1 update / 0 hold
    memory_valid: bool = False
    risk_target: Optional[float] = None         # consequence label when available
    risk_valid: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def build_supervision(*, boxes: Sequence[Sequence[float]],
                      gt_boxes: Sequence[Sequence[float]],
                      gt_ids: Sequence[int],
                      annotation_exhaustive: bool,
                      shortlist: Sequence[int],
                      assigned_ids: Optional[Sequence[int]] = None,
                      match_iou: float = 0.5,
                      reactivated_ids: Optional[Sequence[int]] = None) -> list[CandidateSupervision]:
    """Build per-candidate supervision with explicit validity.

    ``shortlist`` is the retrieval shortlist *actually shown to the model*; the
    true identity's presence in it decides MATCH-with-evidence versus
    insufficient-retrieval, and is recorded rather than hidden.

    ``assigned_ids``/``reactivated_ids`` come from the tracker's own causal state
    (never from GT repairs): an id is a REACTIVATE target only if the tracker
    itself had it in LOST state before this frame.
    """
    edges = sparse_edges(boxes, None)
    clusters = entity_clusters(boxes, edges)
    targets = admission_targets(boxes, gt_boxes, gt_ids, match_iou=match_iou,
                                annotation_exhaustive=annotation_exhaustive,
                                clusters=clusters)
    assigned = set(int(i) for i in (assigned_ids or []))
    reactivated = set(int(i) for i in (reactivated_ids or []))
    shortlist_set = set(int(i) for i in shortlist)

    rows: list[CandidateSupervision] = []
    for target in targets:
        row = CandidateSupervision(index=target.candidate, evidence={
            "admission_label": target.label, "matched_gt": target.matched_gt,
            "matched_iou": target.matched_iou, "duplicate_of": target.duplicate_of,
            "reason": target.reason,
        })
        # admission supervision is only valid when the frame annotation is usable
        if target.label in ("valid_entity", "duplicate"):
            row.admit_target = 1 if target.label == "valid_entity" else 0
            row.admit_valid = True
        elif target.label == "background" and annotation_exhaustive:
            row.admit_target = 0
            row.admit_valid = True
        # identity supervision
        if target.label == "duplicate":
            row.identity_target, _ = supervised_decision(
                physical_identity=target.matched_gt, identity_valid=True, is_duplicate=True,
                is_valid_entity=False, identity_in_shortlist=True)
            row.identity_valid = True
        elif target.label == "valid_entity" and target.matched_gt is not None:
            decision, track_id = supervised_decision(
                physical_identity=target.matched_gt, identity_valid=True, is_duplicate=False,
                is_valid_entity=True,
                identity_in_shortlist=target.matched_gt in shortlist_set,
                is_reactivation=target.matched_gt in reactivated)
            if decision in ("MATCH", "REACTIVATE") and target.matched_gt not in shortlist_set:
                # true id missing from the shortlist: the honest target is NO_MATCH/HOLD
                row.identity_target = None
                row.identity_valid = False
                row.evidence["retrieval_miss"] = True
            else:
                slots = slot_targets(decision, track_id)
                row.identity_target = slots.get("identity_choice")
                row.identity_valid = bool(slots)
        # memory supervision: only for identities that are actually assigned now
        if row.identity_target not in (None, NEW_TOKEN, REJECT_TOKEN):
            row.memory_target = 1
            row.memory_valid = int(row.identity_target) in assigned or \
                int(row.identity_target) == target.matched_gt
        rows.append(row)
    return rows


def summarize_supervision(rows: Sequence[CandidateSupervision]) -> dict[str, int]:
    """Counts used by the coverage/registry report (no loss math here)."""
    counts = {
        "candidates": len(rows),
        "admit_valid": sum(1 for r in rows if r.admit_valid),
        "identity_valid": sum(1 for r in rows if r.identity_valid),
        "memory_valid": sum(1 for r in rows if r.memory_valid),
        "risk_valid": sum(1 for r in rows if r.risk_valid),
        "unknown_identity": sum(1 for r in rows if not r.identity_valid),
        "retrieval_miss": sum(1 for r in rows if r.evidence.get("retrieval_miss")),
        "new_targets": sum(1 for r in rows if r.identity_target == NEW_TOKEN),
        "reject_targets": sum(1 for r in rows if r.identity_target == REJECT_TOKEN),
        "match_targets": sum(1 for r in rows if r.identity_target not in (None, NEW_TOKEN, REJECT_TOKEN)),
    }
    return counts


def structured_losses(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Optional torch objective for a directly-trained slot head.

    Imported lazily so that the data layer never requires torch.  Shapes:
    ``identity_logits`` is (candidates, options); ``option_tokens`` lists the
    option strings in column order.
    """
    import torch
    import torch.nn.functional as F

    identity_logits = kwargs["identity_logits"]
    identity_targets = kwargs["identity_targets"]      # list[Optional[str]]
    identity_valid = kwargs["identity_valid"]          # list[bool]
    option_tokens = list(kwargs["option_tokens"])
    admit_logits = kwargs.get("admit_logits")
    admit_targets = kwargs.get("admit_targets")
    admit_valid = kwargs.get("admit_valid")
    memory_logits = kwargs.get("memory_logits")
    memory_targets = kwargs.get("memory_targets")
    memory_valid = kwargs.get("memory_valid")
    weights = kwargs.get("weights", {})

    losses: dict[str, Any] = {}
    token_to_index = {token: i for i, token in enumerate(option_tokens)}

    rows, targets = [], []
    for index, (target, valid) in enumerate(zip(identity_targets, identity_valid)):
        if not valid or target is None or target not in token_to_index:
            continue
        rows.append(index)
        targets.append(token_to_index[target])

    if rows:
        logits = identity_logits[rows]
        target_tensor = torch.tensor(targets, dtype=torch.long, device=logits.device)
        losses["identity_ce"] = F.cross_entropy(logits, target_tensor)
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(target_tensor, num_classes=logits.shape[-1]).float()
        losses["identity_brier"] = ((probs - one_hot) ** 2).sum(dim=-1).mean()

        # one-to-one consistency: two candidates must not both be supervised to
        # the same existing identity, and the model must not strongly prefer it.
        seen: dict[str, int] = {}
        clashes = 0
        for row, target_index in zip(rows, targets):
            token = option_tokens[target_index]
            if token in (NEW_TOKEN, REJECT_TOKEN):
                continue
            if token in seen:
                clashes += 1
            seen[token] = row
        losses["identity_duplicate_target_clashes"] = torch.tensor(float(clashes))

    if admit_logits is not None and admit_targets is not None and admit_valid is not None:
        mask = [i for i, ok in enumerate(admit_valid) if ok]
        if mask:
            logits = admit_logits[mask].view(-1)
            target_tensor = torch.tensor([admit_targets[i] for i in mask], dtype=torch.float32,
                                         device=logits.device)
            losses["admit_bce"] = F.binary_cross_entropy_with_logits(logits, target_tensor)

    if memory_logits is not None and memory_targets is not None and memory_valid is not None:
        mask = [i for i, ok in enumerate(memory_valid) if ok]
        if mask:
            logits = memory_logits[mask].view(-1)
            target_tensor = torch.tensor([memory_targets[i] for i in mask], dtype=torch.float32,
                                         device=logits.device)
            losses["memory_bce"] = F.binary_cross_entropy_with_logits(logits, target_tensor)

    total = None
    for name, value in losses.items():
        if not torch.is_tensor(value) or value.ndim != 0 or name.endswith("clashes"):
            continue
        weight = float(weights.get(name, 1.0))
        total = value * weight if total is None else total + value * weight
    losses["total"] = total
    return losses
