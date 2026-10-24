"""Native pretrained-OneJev adapter for TrackJEV.

This is the only module that talks to the real OneJev-0.8B engine
(``external/OneJev/qev``).  It builds the TrackJEV request (admission, identity,
memory and risk questions) and parses the answers back into decisions.

Design rules enforced here:

* identity options come from the caller's shortlist, capped at
  ``MAX_SHORTLIST``; truncation is recorded so retrieval coverage stays visible;
* NEW and REJECT are always present, so "no supported identity" never collapses
  into a false NEW;
* the memory question is only asked for candidates that received an identity,
  and it is asked *after* assignment (assignment-conditioned memory);
* the adapter never reads GT and never uses future frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .decision_schema import (
    ADMIT_TEXT,
    MAX_SHORTLIST,
    MEMORY_TEXT,
    RISK_TEXT,
    IdentityOption,
    ParsedDecision,
    build_identity_options,
    parse_identity_answer,
)
from .candidate_competition import competition_rows, entity_clusters, sparse_edges

ROOT = Path(__file__).resolve().parents[3]
ONEJEV_ROOT = ROOT / "external" / "OneJev"


@dataclass
class AdapterOutput:
    decisions: list[ParsedDecision]
    metadata: dict[str, Any] = field(default_factory=dict)
    request_count: int = 0
    question_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"decisions": [d.to_dict() for d in self.decisions],
                "metadata": self.metadata,
                "request_count": self.request_count,
                "question_count": self.question_count}


class NativeOneJevAdapter:
    """Wraps ``qev.mm_engine.MMDecisionEngine`` behind the TrackJEV contract."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda:0",
                 dtype: str = "bfloat16", batch_candidates: int = 8,
                 shortlist: int = MAX_SHORTLIST, evidence_format: str = "trackjev_v1",
                 risk_horizon: int = 30, ask_risk: bool = False,
                 engine: Optional[Any] = None) -> None:
        self.checkpoint = str(Path(checkpoint).resolve())
        self.batch_candidates = int(batch_candidates)
        self.shortlist = min(int(shortlist), MAX_SHORTLIST)
        self.evidence_format = evidence_format
        self.risk_horizon = int(risk_horizon)
        self.ask_risk = bool(ask_risk)
        self.device = device
        if engine is not None:
            self.engine = engine
        else:
            import sys

            sys.path.insert(0, str(ONEJEV_ROOT))
            from qev.mm_engine import MMDecisionEngine

            self.engine = MMDecisionEngine(self.checkpoint, device=device, dtype=dtype,
                                           cuda_graphs=False)

    # ------------------------------------------------------------------ state
    def build_state(self, *, image: Any, group: Sequence[dict[str, Any]],
                    competition: Optional[dict[str, Any]], tracks: Sequence[dict[str, Any]],
                    timestamp: float, specification: dict[str, Any],
                    task_query: Optional[str] = None) -> dict[str, Any]:
        state: dict[str, Any] = {
            "current_frame": "<image:1>",
            "timestamp": timestamp,
            "task": specification,
            "candidates": [{"index": i, "box": list(c["box"]),
                            **({"label": c["label"]} if c.get("label") else {})}
                           for i, c in enumerate(group)],
            "historical_tracks": [
                {"id": t["track_id"], "box": list(t.get("box") or []),
                 "predicted_box": list(t.get("predicted_box") or []),
                 "age": t.get("age"), "lost_age": t.get("lost_age"),
                 "state": t.get("state"), "last_reliable_timestamp": t.get("last_reliable_timestamp")}
                for t in tracks],
            "identity_option_limit": self.shortlist,
            "decision_vocabulary": {
                "existing": "an id in the option list (MATCH or REACTIVATE if the id is lost)",
                "NEW": "a genuinely new physical entity",
                "REJECT": "invalid or duplicate proposal that must not create an id",
            },
        }
        if task_query:
            state["referring_expression"] = task_query
        if competition:
            state.update(competition)
        return state

    # ---------------------------------------------------------------- request
    def build_request(self, *, image: Any, group: Sequence[dict[str, Any]],
                      tracks: Sequence[dict[str, Any]], timestamp: float,
                      specification: dict[str, Any], task_query: Optional[str] = None,
                      competition: Optional[dict[str, Any]] = None) -> tuple[Any, list[dict[str, Any]]]:
        from qev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

        shortlist_entries = [{"track_id": t["track_id"], "lost_age": t.get("lost_age"),
                              "prior": t.get("prior")} for t in tracks[:self.shortlist]]
        options = build_identity_options(shortlist_entries)
        state = self.build_state(image=image, group=group, competition=competition,
                                 tracks=tracks[:self.shortlist], timestamp=timestamp,
                                 specification=specification, task_query=task_query)
        media = [{"type": "image", "image": image}]

        questions: dict[str, Any] = {}
        for index in range(len(group)):
            questions[f"admit_{index}"] = NoulQuestion(
                type="noul", instructions=ADMIT_TEXT.format(index=index))
            criteria = {}
            for option in options:
                if option.kind == "existing":
                    criteria[option.token] = option.text
                else:
                    criteria[option.token] = option.text
            questions[f"identity_{index}"] = ChoiceQuestion(
                type="choice",
                instructions=(f"Which identity owns current candidate {index}? Judge only from the "
                              f"visual evidence, the competition rows and the identity history. "
                              f"A class label alone is never identity evidence."),
                criteria=criteria)
        for track in tracks[:self.shortlist]:
            questions[f"life_{track['track_id']}"] = ChoiceQuestion(
                type="choice",
                instructions=(f"What is the current lifecycle of identity {track['track_id']} "
                              f"given only current and past evidence?"),
                criteria={"ACTIVE": "currently visible and associated",
                          "LOST": "temporarily unobserved, memory retained",
                          "REACTIVATE": "returned after a gap",
                          "TERMINATE": "permanently leaves this tracked video"})
        request = SystemOneRequest(state=state, model="jev-latest", questions=questions)
        self._pending_options = options
        return request, media

    def build_memory_questions(self, assigned: dict[int, str]) -> dict[str, Any]:
        """Assignment-conditioned memory questions (asked after identity allocation)."""
        from qev.schema import NoulQuestion, ScoreQuestion

        questions: dict[str, Any] = {}
        for candidate, token in assigned.items():
            if token in ("NEW", "REJECT", "NO_MATCH"):
                continue
            questions[f"memory_{candidate}"] = NoulQuestion(
                type="noul",
                instructions=MEMORY_TEXT.format(index=candidate, track_id=token))
            if self.ask_risk:
                questions[f"risk_{candidate}"] = ScoreQuestion(
                    type="score",
                    instructions=RISK_TEXT.format(index=candidate, track_id=token,
                                                  horizon=self.risk_horizon),
                    criteria=["negligible", "low", "moderate", "high"])
        return questions

    # --------------------------------------------------------------- decode
    def decide(self, *, image: Any, group: Sequence[dict[str, Any]],
               tracks: Sequence[dict[str, Any]], timestamp: float,
               specification: dict[str, Any], task_query: Optional[str] = None,
               competition: Optional[dict[str, Any]] = None) -> AdapterOutput:
        """One inference over one candidate group (may split on context limits)."""
        if not group:
            return AdapterOutput(decisions=[], metadata={"backend": "trackjev_native_onejev",
                                                         "empty_group": True})

        collected: list[dict[str, float]] = [{} for _ in group]
        admission: list[Optional[float]] = [None] * len(group)
        lifecycle: dict[str, Any] = {}
        batches: list[dict[str, Any]] = []
        options = None

        pending = [list(range(len(group)))]
        split_events: list[dict[str, Any]] = []
        while pending:
            indices = pending.pop(0)
            subgroup = [group[i] for i in indices]
            # The full-frame competition matrix is quadratic in the number of
            # detector proposals. Rebuild it for a split subgroup; retaining
            # the original NxN matrix here would make even a one-candidate
            # retry exceed OneJev's context limit on dense frames.
            local_competition = competition
            if competition is not None and len(indices) != len(group):
                boxes = [c["box"] for c in subgroup]
                features = [c.get("native_region_feature") for c in subgroup]
                edges = sparse_edges(boxes, features if any(f is not None for f in features) else None,
                                     box_format="xyxy")
                clusters = entity_clusters(boxes, edges)
                local_competition = competition_rows(boxes, clusters, edges, box_format="xyxy")
            request, media = self.build_request(
                image=image, group=subgroup, tracks=tracks, timestamp=timestamp,
                specification=specification, task_query=task_query, competition=local_competition)
            options = getattr(self, "_pending_options", options)
            try:
                response, meta = self.engine.decide(request, media=media)
            except ValueError as error:
                if not str(error).startswith("state plus longest question is ") or len(subgroup) < 2:
                    raise
                middle = len(subgroup) // 2
                split_events.append({"group_size": len(subgroup), "error": str(error),
                                     "new_group_sizes": [middle, len(subgroup) - middle]})
                pending[:0] = [indices[:middle], indices[middle:]]
                continue
            batches.append({"size": len(subgroup), "metadata": meta,
                            "questions": len(request.questions)})
            for local_index, global_index in enumerate(indices):
                answer = response.answers.get(f"identity_{local_index}")
                probabilities = dict(getattr(answer, "probabilities", {}) or {})
                collected[global_index] = probabilities
                admit = response.answers.get(f"admit_{local_index}")
                if admit is not None:
                    admission[global_index] = float(getattr(admit, "noul", 0.0))
            for track in tracks[:self.shortlist]:
                key = f"life_{track['track_id']}"
                if key in response.answers and key not in lifecycle:
                    lifecycle[key] = response.answers[key].model_dump()

        options = options or []
        decisions = [parse_identity_answer(i, probs, options) for i, probs in enumerate(collected)]
        for index, decision in enumerate(decisions):
            decision.admitted = admission[index]
        shortlist_ids = [int(t["track_id"]) for t in tracks[:self.shortlist]]
        truncated = max(0, len(tracks) - self.shortlist)
        metadata = {
            "backend": "trackjev_native_onejev",
            "checkpoint": self.checkpoint,
            "evidence_format": self.evidence_format,
            "native_probabilities": True,
            "gt_read": False,
            "shortlist_limit": self.shortlist,
            "shortlist_ids": shortlist_ids,
            "shortlist_truncated_ids": truncated,
            "option_count": len(options),
            "batches": batches,
            "context_splits": split_events,
            "lifecycle": lifecycle,
            "memory_questions_deferred_until_after_assignment": True,
        }
        return AdapterOutput(decisions=decisions, metadata=metadata,
                             request_count=len(batches),
                             question_count=sum(b["questions"] for b in batches))

    # ------------------------------------------------------- memory decisions
    def decide_memory(self, *, image: Any, group: Sequence[dict[str, Any]],
                      assigned: dict[int, str], tracks: Sequence[dict[str, Any]],
                      timestamp: float, specification: dict[str, Any],
                      competition: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Ask UPDATE/HOLD for the identities that were actually assigned."""
        questions = self.build_memory_questions(assigned)
        if not questions:
            return {"asked": 0, "updates": {}, "policy": "no identity assigned in this frame"}
        from qev.schema import SystemOneRequest

        # Memory is queried only for candidates that received an existing
        # identity. Do not carry the full detector proposal bank (and its
        # quadratic competition rows) into this second request: dense frames
        # otherwise exceed the OneJev context limit even when one identity is
        # being updated. Keep the original candidate ids in the state for
        # auditability while using local question indices for the compact group.
        selected = sorted(int(i) for i in assigned)
        local_group = [group[i] for i in selected]
        local_assigned = {local: assigned[original]
                          for local, original in enumerate(selected)}
        local_questions = self.build_memory_questions(local_assigned)
        local_competition = None
        if competition is not None:
            boxes = [c["box"] for c in local_group]
            features = [c.get("native_region_feature") for c in local_group]
            edges = sparse_edges(boxes, features if any(f is not None for f in features) else None,
                                 box_format="xyxy")
            clusters = entity_clusters(boxes, edges)
            local_competition = competition_rows(boxes, clusters, edges, box_format="xyxy")
        state = self.build_state(image=image, group=local_group, competition=local_competition,
                                 tracks=tracks[:self.shortlist], timestamp=timestamp,
                                 specification=specification)
        state["assigned_identity_per_candidate"] = {
            str(original): assigned[original] for original in selected
        }
        state["assignment_condition"] = ("these ids are the actual final allocation of this frame; "
                                         "decide whether each one deserves a long-term memory update")
        request = SystemOneRequest(state=state, model="jev-latest", questions=local_questions)
        response, meta = self.engine.decide(request, media=[{"type": "image", "image": image}])
        updates: dict[int, dict[str, Any]] = {}
        for local, candidate in enumerate(selected):
            key = f"memory_{local}"
            if key in response.answers:
                updates[candidate] = {"noul": float(response.answers[key].noul)}
            risk_key = f"risk_{local}"
            if risk_key in response.answers:
                updates.setdefault(candidate, {})["risk"] = response.answers[risk_key].score
        return {"asked": len(questions), "updates": updates, "metadata": meta,
                "policy": "memory write conditioned on the final assigned identity"}
