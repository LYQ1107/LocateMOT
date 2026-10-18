"""Selective, opt-in fusion policy for Yes/No verification.

The policy is intentionally downstream of physical candidate generation and
global association.  A semantic ``No`` can suppress an unconfirmed NEW
candidate or lower an existing candidate's score, but it never creates a new
identity and never rewrites the physical assignment pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from .yes_verification import YesNoResult


@dataclass(frozen=True)
class VerificationPolicyConfig:
    enabled: bool = False
    top2_margin_threshold: float = 0.10
    entropy_threshold: float = 0.85
    verify_reactivation: bool = True
    verify_contradiction: bool = True
    verify_query_ambiguity: bool = True
    max_verifications_per_frame: int = 8
    yes_threshold: float = 0.5
    semantic_weight: float = 0.25
    timeout_ms: float = 1500.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "VerificationPolicyConfig":
        value = value or {}
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: int
    association_scores: Mapping[str, float]
    reactivation: bool = False
    contradiction: bool = False
    query_ambiguity: bool = False
    entropy: float | None = None
    track_state: str = "ACTIVE"


@dataclass(frozen=True)
class VerificationDecision:
    verify: bool
    reasons: tuple[str, ...] = ()
    budget_remaining: int = 0


@dataclass
class SelectiveVerificationPolicy:
    config: VerificationPolicyConfig = field(default_factory=VerificationPolicyConfig)
    _used_this_frame: int = 0

    def begin_frame(self) -> None:
        self._used_this_frame = 0

    def should_verify(self, evidence: CandidateEvidence) -> VerificationDecision:
        budget_remaining = max(0, self.config.max_verifications_per_frame - self._used_this_frame)
        if not self.config.enabled or budget_remaining == 0:
            return VerificationDecision(False, (), budget_remaining)
        scores = sorted((float(value) for value in evidence.association_scores.values()), reverse=True)
        reasons: list[str] = []
        if len(scores) >= 2 and scores[0] - scores[1] < self.config.top2_margin_threshold:
            reasons.append("low_top2_margin")
        if evidence.entropy is not None and evidence.entropy > self.config.entropy_threshold:
            reasons.append("high_assignment_entropy")
        if self.config.verify_reactivation and evidence.reactivation:
            reasons.append("reactivation")
        if self.config.verify_contradiction and evidence.contradiction:
            reasons.append("contradiction")
        if self.config.verify_query_ambiguity and evidence.query_ambiguity:
            reasons.append("query_ambiguity")
        should = bool(reasons)
        if should:
            self._used_this_frame += 1
        return VerificationDecision(should, tuple(reasons), budget_remaining - int(should))

    def fuse_association_score(self, base_score: float, result: YesNoResult | None) -> float:
        """Fuse only an available semantic probability; disabled is identity."""

        if not self.config.enabled or result is None or result.p_yes is None or not result.used:
            return float(base_score)
        weight = min(1.0, max(0.0, float(self.config.semantic_weight)))
        return (1.0 - weight) * float(base_score) + weight * float(result.p_yes)

    def lifecycle_action(
        self,
        *,
        base_state: str,
        result: YesNoResult | None,
        candidate_is_new: bool,
    ) -> str:
        """Map semantic evidence to an action without treating No as NEW."""

        if not self.config.enabled or result is None or result.p_yes is None or not result.used:
            return base_state
        if result.p_yes < self.config.yes_threshold:
            return "IGNORE" if candidate_is_new else base_state
        return "ACTIVE" if candidate_is_new else base_state


def global_assignment(scores: Sequence[Sequence[float]]) -> dict[int, int]:
    """Return a maximum-score one-to-one assignment for a CPU-sized matrix."""

    if not scores:
        return {}
    rows, cols = len(scores), len(scores[0]) if scores[0] else 0
    if any(len(row) != cols for row in scores):
        raise ValueError("assignment score matrix must be rectangular")
    if cols == 0:
        return {}
    # Small deterministic dynamic program avoids a SciPy dependency in the
    # CPU preparation/tests; larger matrices use scipy when available.
    if rows <= 12 and cols <= 12:
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def solve(row: int, used: int) -> tuple[float, tuple[tuple[int, int], ...]]:
            if row == rows:
                return 0.0, ()
            best_value, best_pairs = solve(row + 1, used)  # leave row unmatched
            for col in range(cols):
                if used & (1 << col):
                    continue
                value, pairs = solve(row + 1, used | (1 << col))
                value += float(scores[row][col])
                candidate_pairs = ((row, col),) + pairs
                if value > best_value or (math.isclose(value, best_value) and candidate_pairs < best_pairs):
                    best_value, best_pairs = value, candidate_pairs
            return best_value, best_pairs

        return dict(solve(0, 0)[1])
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrix = np.asarray(scores, dtype=float)
        row_ids, col_ids = linear_sum_assignment(-matrix)
        return {int(row): int(col) for row, col in zip(row_ids, col_ids)}
    except ImportError as error:
        raise ValueError("large global assignment requires scipy") from error


def baseline_equivalent(config: VerificationPolicyConfig | Mapping[str, Any] | None = None) -> bool:
    """Document the default contract used by the regression test."""

    resolved = config if isinstance(config, VerificationPolicyConfig) else VerificationPolicyConfig.from_mapping(config)
    return resolved.enabled is False


__all__ = [
    "CandidateEvidence",
    "SelectiveVerificationPolicy",
    "VerificationDecision",
    "VerificationPolicyConfig",
    "baseline_equivalent",
    "global_assignment",
]
