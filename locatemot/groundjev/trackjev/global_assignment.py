"""Globally consistent identity assignment for TrackJEV.

The model produces per-candidate option probabilities.  Assignment turns them
into the frame's actual identity allocation under three hard constraints:

1. every candidate is assigned at most once (it is one box, one decision);
2. every identity is used at most once per frame (no two candidates share an id);
3. duplicates that lose their competition are suppressed as REJECT, never
   promoted to a second id.

Hungarian assignment (scipy) is used when available; otherwise a deterministic
greedy pass on (probability desc, candidate index asc) is applied.  Both paths
return the same diagnostics so results stay comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .decision_schema import NEW_TOKEN, REJECT_TOKEN, IdentityOption


@dataclass
class AssignmentResult:
    assignment: dict[int, str]                  # candidate index -> option token
    unmatched: list[int] = field(default_factory=list)
    suppressed_duplicates: list[int] = field(default_factory=list)
    reused_identity_conflicts: int = 0
    solver: str = "greedy"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"assignment": {str(k): v for k, v in self.assignment.items()},
                "unmatched": self.unmatched,
                "suppressed_duplicates": self.suppressed_duplicates,
                "reused_identity_conflicts": self.reused_identity_conflicts,
                "solver": self.solver,
                "diagnostics": self.diagnostics}


def _greedy(rows: Sequence[Sequence[float]], tokens: Sequence[str]) -> list[Optional[int]]:
    """Greedy one-to-one assignment: best (prob, candidate, option) first."""
    triples = []
    for candidate, row in enumerate(rows):
        for option, prob in enumerate(row):
            if prob > 0.0:
                triples.append((prob, candidate, option))
    triples.sort(key=lambda t: (-t[0], t[1], t[2]))
    taken_candidates: set[int] = set()
    taken_identities: set[str] = set()
    picks: list[Optional[int]] = [None] * len(rows)
    for _prob, candidate, option in triples:
        if candidate in taken_candidates:
            continue
        token = tokens[option]
        if token not in (NEW_TOKEN, REJECT_TOKEN) and token in taken_identities:
            continue
        picks[candidate] = option
        taken_candidates.add(candidate)
        if token not in (NEW_TOKEN, REJECT_TOKEN):
            taken_identities.add(token)
    return picks


def _hungarian(rows: Sequence[Sequence[float]]) -> Optional[list[int]]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return None
    matrix = np.asarray(rows, dtype=float)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    picks: list[Optional[int]] = [None] * len(rows)
    for r, c in zip(row_ind, col_ind):
        picks[int(r)] = int(c)
    return picks


def assign(probabilities: Sequence[dict[str, float]], options: Sequence[IdentityOption],
           *, solver: str = "auto", duplicate_penalty: float = 0.0) -> AssignmentResult:
    """Assign candidates to option tokens.

    ``probabilities`` is one dict per candidate keyed by option token; missing
    entries are zero.  ``duplicate_penalty`` subtracts a fixed amount from
    existing-identity options so a caller can express "prefer REJECT when two
    boxes compete for one identity".
    """
    tokens = [option.token for option in options]
    rows: list[list[float]] = []
    for candidate_probs in probabilities:
        row = []
        for option in options:
            prob = float(candidate_probs.get(option.token, 0.0))
            if duplicate_penalty and option.kind == "existing":
                prob = max(0.0, prob - duplicate_penalty)
            row.append(prob)
        rows.append(row)

    used_solver = "greedy"
    picks: Optional[list[Optional[int]]] = None
    if solver in ("auto", "hungarian"):
        picks = _hungarian(rows)
        if picks is not None:
            used_solver = "hungarian"
    if picks is None:
        picks = _greedy(rows, tokens)

    assignment: dict[int, str] = {}
    unmatched: list[int] = []
    suppressed: list[int] = []
    conflicts = 0
    identity_owner: dict[str, int] = {}
    order = sorted(range(len(rows)),
                   key=lambda i: (-(rows[i][picks[i]] if picks[i] is not None else 0.0), i))
    for candidate in order:
        option_index = picks[candidate]
        if option_index is None:
            unmatched.append(candidate)
            continue
        token = tokens[option_index]
        if token in (NEW_TOKEN, REJECT_TOKEN):
            assignment[candidate] = token
            continue
        if token in identity_owner:
            conflicts += 1
            assignment[candidate] = REJECT_TOKEN
            suppressed.append(candidate)
            continue
        identity_owner[token] = candidate
        assignment[candidate] = token

    return AssignmentResult(
        assignment=assignment,
        unmatched=sorted(unmatched),
        suppressed_duplicates=sorted(suppressed),
        reused_identity_conflicts=conflicts,
        solver=used_solver,
        diagnostics={
            "candidates": len(rows),
            "options": len(tokens),
            "new": sum(1 for token in assignment.values() if token == NEW_TOKEN),
            "reject": sum(1 for token in assignment.values() if token == REJECT_TOKEN),
            "matched": sum(1 for token in assignment.values()
                           if token not in (NEW_TOKEN, REJECT_TOKEN)),
            "identity_columns_used": len(identity_owner),
            "one_identity_per_frame": len(set(identity_owner)) == len(identity_owner),
        },
    )


def assignment_to_memory_ops(result: AssignmentResult) -> list[dict[str, Any]]:
    """Translate an assignment into memory write intents.

    The write decision itself belongs to the model (``memory_i``); this records
    only *which* identity a write would touch, so a HOLD abstains instead of
    silently writing.
    """
    ops: list[dict[str, Any]] = []
    for candidate, token in sorted(result.assignment.items()):
        if token == NEW_TOKEN:
            ops.append({"candidate": candidate, "op": "CREATE", "track_id": None})
        elif token == REJECT_TOKEN:
            ops.append({"candidate": candidate, "op": "NONE", "track_id": None})
        else:
            ops.append({"candidate": candidate, "op": "ASSIGN", "track_id": int(token)})
    return ops
