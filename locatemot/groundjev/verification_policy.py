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
    fusion_mode: str = "fixed"
    calibration_temperature: float = 1.0
    calibration_bias: float = 0.0
    timeout_ms: float = 1500.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "VerificationPolicyConfig":
        value = value or {}
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})

    def __post_init__(self) -> None:
        if self.fusion_mode not in {"fixed", "calibrated"}:
            raise ValueError("fusion_mode must be 'fixed' or 'calibrated'")
        if float(self.calibration_temperature) <= 0.0:
            raise ValueError("calibration_temperature must be positive")


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
        verifier_probability = float(result.p_yes)
        if self.config.fusion_mode == "calibrated":
            verifier_probability = calibrate_probability(
                verifier_probability,
                temperature=float(self.config.calibration_temperature),
                bias=float(self.config.calibration_bias),
            )
        weight = min(1.0, max(0.0, float(self.config.semantic_weight)))
        return (1.0 - weight) * float(base_score) + weight * verifier_probability

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


def _logit(probability: float) -> float:
    value = min(1.0 - 1e-7, max(1e-7, float(probability)))
    return math.log(value / (1.0 - value))


def calibrate_probability(probability: float, *, temperature: float = 1.0, bias: float = 0.0) -> float:
    """Apply a registered one-dimensional logistic calibration transform."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    z = (_logit(probability) + float(bias)) / float(temperature)
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def fit_logistic_calibrator(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    steps: int = 200,
    learning_rate: float = 0.05,
) -> tuple[float, float]:
    """Fit temperature/bias on a declared calibration split only.

    This small CPU fitter is intentionally independent of the tracker and is
    suitable for a pre-registered calibration set.  It returns
    ``(temperature, bias)``; callers must record the split and result before
    enabling ``fusion_mode='calibrated'``.
    """

    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("calibrator requires equally sized non-empty inputs")
    if any(int(label) not in (0, 1) for label in labels):
        raise ValueError("calibration labels must be binary")
    temperature, bias = 1.0, 0.0
    for _ in range(max(1, int(steps))):
        grad_temperature = 0.0
        grad_bias = 0.0
        for probability, label in zip(probabilities, labels):
            x = _logit(float(probability))
            p = calibrate_probability(float(probability), temperature=temperature, bias=bias)
            error = p - int(label)
            grad_temperature += error * (-x / (temperature * temperature))
            grad_bias += error / temperature
        scale = 1.0 / len(probabilities)
        temperature = max(0.05, min(20.0, temperature - float(learning_rate) * grad_temperature * scale))
        bias = max(-20.0, min(20.0, bias - float(learning_rate) * grad_bias * scale))
    return float(temperature), float(bias)


__all__ = [
    "CandidateEvidence",
    "SelectiveVerificationPolicy",
    "VerificationDecision",
    "VerificationPolicyConfig",
    "baseline_equivalent",
    "calibrate_probability",
    "fit_logistic_calibrator",
    "global_assignment",
]
