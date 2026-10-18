"""Optional Yes/No semantic verification for the GroundJEV tracker.

The verifier is an adapter boundary: the baseline tracker never imports or
calls a model backend unless a caller explicitly enables the policy and
injects a backend.  Requests contain current visual evidence and causal track
history only; labels, GT boxes, and evaluation annotations are deliberately
outside the request object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence


YES_NO_PROTOCOL_VERSION = "yes-verification-request-v1"
_FORBIDDEN_PROMPT_MARKERS = (
    "ground truth",
    "ground_truth",
    "gt_box",
    "gt_label",
    "evaluation label",
    "oracle box",
)


class YesNoBackend(Protocol):
    def forward(self, request: "VerificationRequest") -> Any:
        ...


@dataclass(frozen=True)
class VerificationRequest:
    task: str
    expression: str | None
    question: str
    media: tuple[Any, ...]
    current_frame: int
    track_id: int | None = None
    candidate_id: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    history_frame_ids: tuple[int, ...] = ()
    evidence_reason: str = "selective_trigger"
    checkpoint_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(int(frame) >= int(self.current_frame) for frame in self.history_frame_ids):
            raise ValueError("verification history must be causal: future frame leaked into request")
        assert_prompt_has_no_ground_truth(self.question)

    def model_payload(self) -> dict[str, Any]:
        """Return the request fields that may cross a model boundary."""

        return {
            "protocol": YES_NO_PROTOCOL_VERSION,
            "task": self.task,
            "expression": self.expression,
            "question": self.question,
            "media": list(self.media),
            "current_frame": int(self.current_frame),
            "track_id": self.track_id,
            "candidate_id": self.candidate_id,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "history_frame_ids": list(self.history_frame_ids),
        }


@dataclass(frozen=True)
class YesNoResult:
    status: str
    used: bool
    p_yes: float | None
    answer: str | None
    question_type: str
    evidence_frame_ids: tuple[int, ...]
    checkpoint_hash: str | None
    latency_ms: float | None
    error: str | None = None
    raw_kind: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "used": self.used,
            "p_yes": self.p_yes,
            "answer": self.answer,
            "question_type": self.question_type,
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "checkpoint_hash": self.checkpoint_hash,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "raw_kind": self.raw_kind,
        }


def assert_prompt_has_no_ground_truth(prompt: str) -> None:
    lowered = str(prompt).lower()
    for marker in _FORBIDDEN_PROMPT_MARKERS:
        if marker in lowered:
            raise ValueError(f"GT/evaluation annotation marker found in model prompt: {marker}")


def build_verification_prompt(
    *,
    task: str,
    expression: str | None,
    bbox: Sequence[float] | None,
    question_type: str = "membership",
) -> str:
    box = "unknown" if bbox is None else " ".join(f"{float(value):.4f}" for value in bbox)
    if question_type == "identity":
        question = "Does this candidate depict the same persistent object as the supplied track history?"
    elif question_type == "ovmot_membership":
        question = "Does this candidate match the requested category or expression?"
    else:
        question = "Does this candidate satisfy the supplied referring expression?"
    target = expression.strip() if isinstance(expression, str) and expression.strip() else "the tracking target"
    prompt = (
        f"Task: {task}. Candidate box in the current image: [{box}]. "
        f"{question} Target description: {target}. Answer exactly Yes or No."
    )
    assert_prompt_has_no_ground_truth(prompt)
    return prompt


def _stable_two_way_probability(yes_logit: float, no_logit: float) -> float:
    maximum = max(float(yes_logit), float(no_logit))
    yes_exp = math.exp(float(yes_logit) - maximum)
    no_exp = math.exp(float(no_logit) - maximum)
    return yes_exp / max(yes_exp + no_exp, 1e-12)


def parse_yes_no_output(output: Any, *, yes_id: int | None = None, no_id: int | None = None) -> tuple[float, str, str]:
    """Normalize logits/probabilities/text with an explicit Yes direction."""

    if isinstance(output, Mapping):
        if "p_yes" in output:
            value = float(output["p_yes"])
            if not 0.0 <= value <= 1.0:
                raise ValueError("p_yes must be in [0, 1]")
            return value, ("Yes" if value >= 0.5 else "No"), "p_yes"
        if "yes_probability" in output:
            value = float(output["yes_probability"])
            if not 0.0 <= value <= 1.0:
                raise ValueError("yes_probability must be in [0, 1]")
            return value, ("Yes" if value >= 0.5 else "No"), "yes_probability"
        if "yes_logit" in output and "no_logit" in output:
            value = _stable_two_way_probability(float(output["yes_logit"]), float(output["no_logit"]))
            return value, ("Yes" if value >= 0.5 else "No"), "yes_no_logits"
        logits = output.get("logits")
        if isinstance(logits, Mapping):
            if "Yes" in logits and "No" in logits:
                value = _stable_two_way_probability(float(logits["Yes"]), float(logits["No"]))
                return value, ("Yes" if value >= 0.5 else "No"), "named_logits"
        text = output.get("text", output.get("answer"))
        if text is not None:
            output = text
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        values = list(output)
        if yes_id is not None and no_id is not None and max(yes_id, no_id) < len(values):
            value = _stable_two_way_probability(float(values[yes_id]), float(values[no_id]))
            return value, ("Yes" if value >= 0.5 else "No"), "token_logits"
        if len(values) == 2:
            value = _stable_two_way_probability(float(values[0]), float(values[1]))
            return value, ("Yes" if value >= 0.5 else "No"), "two_logits"
    if isinstance(output, str):
        match = re.search(r"\b(yes|no)\b", output.strip(), flags=re.IGNORECASE)
        if match:
            answer = match.group(1).capitalize()
            return (1.0 if answer == "Yes" else 0.0), answer, "text"
    raise ValueError("backend output does not contain a Yes/No probability or answer")


class YesNoVerifier:
    """Opt-in backend adapter with deterministic disabled behavior."""

    def __init__(
        self,
        backend: YesNoBackend | Callable[[VerificationRequest], Any] | None = None,
        *,
        enabled: bool = False,
        yes_id: int | None = None,
        no_id: int | None = None,
        checkpoint_hash: str | None = None,
        timeout_ms: float | None = None,
    ) -> None:
        self.backend = backend
        self.enabled = bool(enabled)
        self.yes_id = yes_id
        self.no_id = no_id
        self.checkpoint_hash = checkpoint_hash
        self.timeout_ms = timeout_ms

    def verify(self, request: VerificationRequest) -> YesNoResult:
        if not self.enabled:
            return YesNoResult(
                status="DISABLED",
                used=False,
                p_yes=None,
                answer=None,
                question_type=request.metadata.get("question_type", "membership"),
                evidence_frame_ids=request.history_frame_ids + (request.current_frame,),
                checkpoint_hash=self.checkpoint_hash or request.checkpoint_hash,
                latency_ms=0.0,
            )
        if self.backend is None:
            return YesNoResult(
                status="FAILED_NO_BACKEND",
                used=False,
                p_yes=None,
                answer=None,
                question_type=request.metadata.get("question_type", "membership"),
                evidence_frame_ids=request.history_frame_ids + (request.current_frame,),
                checkpoint_hash=self.checkpoint_hash or request.checkpoint_hash,
                latency_ms=0.0,
                error="verification enabled without an injected backend",
            )
        started = time.perf_counter()
        try:
            raw = self.backend.forward(request) if hasattr(self.backend, "forward") else self.backend(request)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if self.timeout_ms is not None and elapsed_ms > float(self.timeout_ms):
                return YesNoResult(
                    status="FAILED_TIMEOUT",
                    used=False,
                    p_yes=None,
                    answer=None,
                    question_type=request.metadata.get("question_type", "membership"),
                    evidence_frame_ids=request.history_frame_ids + (request.current_frame,),
                    checkpoint_hash=self.checkpoint_hash or request.checkpoint_hash,
                    latency_ms=elapsed_ms,
                    error=f"verification exceeded timeout_ms={self.timeout_ms}",
                )
            p_yes, answer, raw_kind = parse_yes_no_output(raw, yes_id=self.yes_id, no_id=self.no_id)
            return YesNoResult(
                status="VERIFIED",
                used=True,
                p_yes=p_yes,
                answer=answer,
                question_type=request.metadata.get("question_type", "membership"),
                evidence_frame_ids=request.history_frame_ids + (request.current_frame,),
                checkpoint_hash=self.checkpoint_hash or request.checkpoint_hash,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                raw_kind=raw_kind,
            )
        except Exception as error:  # backend failures are a recorded result, not a baseline mutation
            return YesNoResult(
                status="FAILED",
                used=False,
                p_yes=None,
                answer=None,
                question_type=request.metadata.get("question_type", "membership"),
                evidence_frame_ids=request.history_frame_ids + (request.current_frame,),
                checkpoint_hash=self.checkpoint_hash or request.checkpoint_hash,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=repr(error),
            )

    def verify_identity(
        self,
        *,
        track_id: int,
        candidate_id: int | None,
        current_image: Any,
        current_frame: int,
        history_images: Sequence[Any] = (),
        history_frame_ids: Sequence[int] = (),
        bbox: Sequence[float] | None = None,
        evidence_reason: str = "selective_identity_trigger",
    ) -> YesNoResult:
        """Verify a candidate against causal online track imagery.

        The caller supplies crops already retained by the online tracker;
        this method never accepts or serializes GT IDs or evaluation labels.
        ``history_frame_ids`` must be strictly earlier than ``current_frame``.
        """

        prompt = build_verification_prompt(
            task="mot",
            expression=None,
            bbox=bbox,
            question_type="identity",
        )
        request = VerificationRequest(
            task="mot",
            expression=None,
            question=prompt,
            media=tuple(history_images) + (current_image,),
            current_frame=int(current_frame),
            track_id=int(track_id),
            candidate_id=None if candidate_id is None else int(candidate_id),
            bbox=None if bbox is None else tuple(float(value) for value in bbox),
            history_frame_ids=tuple(int(value) for value in history_frame_ids),
            evidence_reason=evidence_reason,
            checkpoint_hash=self.checkpoint_hash,
            metadata={"question_type": "identity"},
        )
        return self.verify(request)

    def verify_semantic(
        self,
        *,
        task: str,
        expression: str,
        current_image: Any,
        current_frame: int,
        track_id: int | None = None,
        candidate_id: int | None = None,
        history_images: Sequence[Any] = (),
        history_frame_ids: Sequence[int] = (),
        bbox: Sequence[float] | None = None,
        question_type: str = "membership",
        evidence_reason: str = "selective_semantic_trigger",
    ) -> YesNoResult:
        """Verify an RMOT expression or OVMOT category for a physical ID."""

        if question_type not in {"membership", "ovmot_membership"}:
            raise ValueError("semantic question_type must be membership or ovmot_membership")
        prompt = build_verification_prompt(
            task=task,
            expression=expression,
            bbox=bbox,
            question_type=question_type,
        )
        request = VerificationRequest(
            task=str(task),
            expression=str(expression),
            question=prompt,
            media=tuple(history_images) + (current_image,),
            current_frame=int(current_frame),
            track_id=None if track_id is None else int(track_id),
            candidate_id=None if candidate_id is None else int(candidate_id),
            bbox=None if bbox is None else tuple(float(value) for value in bbox),
            history_frame_ids=tuple(int(value) for value in history_frame_ids),
            evidence_reason=evidence_reason,
            checkpoint_hash=self.checkpoint_hash,
            metadata={"question_type": question_type},
        )
        return self.verify(request)


def causal_history_frame_ids(history: Sequence[Mapping[str, Any]], current_frame: int) -> tuple[int, ...]:
    """Return only strictly earlier evidence, preserving caller order."""

    return tuple(int(row["frame"]) for row in history if int(row["frame"]) < int(current_frame))


__all__ = [
    "YES_NO_PROTOCOL_VERSION",
    "VerificationRequest",
    "YesNoBackend",
    "YesNoResult",
    "YesNoVerifier",
    "assert_prompt_has_no_ground_truth",
    "build_verification_prompt",
    "causal_history_frame_ids",
    "parse_yes_no_output",
]
