"""TrackJEV runtime module: OneJev decisions -> global assignment -> memory ops.

This is the piece the tracker calls once per frame.  It contains no tracker
internals: it takes candidate boxes, the current identity memory summary and a
task specification, and returns decisions plus explicit operations the tracker
applies.  Keeping the boundary this narrow is what makes the ablations
(B0..B6) comparable: the only things that change are the switches below.

``TrackJEVConfig`` mirrors the registered experiment matrix:

| arm | use_competition | assignment_memory | own_state | consequence |
|-----|-----------------|-------------------|-----------|-------------|
| B2  | no              | no                | no        | no          |
| B3  | yes             | no                | no        | no          |
| B4  | yes             | yes               | no        | no          |
| B5  | yes             | yes               | yes       | no          |
| B6  | yes             | yes               | yes       | yes         |
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .candidate_competition import competition_rows, entity_clusters, sparse_edges
from .decision_schema import MAX_SHORTLIST, NEW_TOKEN, REJECT_TOKEN, ParsedDecision
from .global_assignment import assign, assignment_to_memory_ops


@dataclass
class TrackJEVConfig:
    use_competition: bool = True
    assignment_conditioned_memory: bool = True
    own_state_training: bool = False
    use_consequence_questions: bool = False
    shortlist: int = MAX_SHORTLIST
    duplicate_penalty: float = 0.0
    memory_update_threshold: float = 0.5
    apply_admission_gate: bool = False
    admission_threshold: float = 0.3
    solver: str = "auto"

    def arm_id(self) -> str:
        return ("B6" if self.use_consequence_questions else
                "B5" if self.own_state_training else
                "B4" if self.assignment_conditioned_memory else
                "B3" if self.use_competition else "B2")

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["arm_id"] = self.arm_id()
        return payload


@dataclass
class TrackJEVStep:
    decisions: list[ParsedDecision]
    assignment: dict[int, str]
    memory_updates: dict[int, dict[str, Any]]
    tracker_ops: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"decisions": [d.to_dict() for d in self.decisions],
                "assignment": {str(k): v for k, v in self.assignment.items()},
                "memory_updates": {str(k): v for k, v in self.memory_updates.items()},
                "tracker_ops": self.tracker_ops,
                "diagnostics": self.diagnostics}


class TrackJEVModel:
    """Adapter + competition + assignment, with ablation switches."""

    def __init__(self, adapter: Any, config: Optional[TrackJEVConfig] = None) -> None:
        self.adapter = adapter
        self.config = config or TrackJEVConfig()

    # ------------------------------------------------------------- shortlist
    def build_shortlist(self, candidates: Sequence[dict[str, Any]],
                        tracks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Geometry-first retrieval shortlist; never GT-informed."""
        from ..decision import overlap  # reuse the project's IoU helper

        ranked = sorted(
            tracks,
            key=lambda t: (-max((overlap(c["box"], t["predicted_box"]) for c in candidates), default=0.0),
                           t.get("lost_age", 0), t["track_id"]),
        )[: self.config.shortlist]
        return ranked

    # ------------------------------------------------------------ competition
    def build_competition(self, candidates: Sequence[dict[str, Any]],
                          features: Optional[Sequence[Optional[Sequence[float]]]] = None) -> Optional[dict[str, Any]]:
        if not self.config.use_competition:
            return None
        boxes = [c["box"] for c in candidates]
        # OnlineTracker/RunVideo candidates are pixel xyxy; keep the legacy
        # xywh default for historical SFT rows and controls.
        edges = sparse_edges(boxes, features, box_format="xyxy")
        clusters = entity_clusters(boxes, edges)
        return competition_rows(boxes, clusters, edges, box_format="xyxy")

    # ------------------------------------------------------------------ step
    def step(self, *, image: Any, candidates: Sequence[dict[str, Any]],
             tracks: Sequence[dict[str, Any]], timestamp: float,
             specification: dict[str, Any], task_query: Optional[str] = None,
             features: Optional[Sequence[Optional[Sequence[float]]]] = None) -> TrackJEVStep:
        shortlist = self.build_shortlist(candidates, tracks)
        competition = self.build_competition(candidates, features)

        output = self.adapter.decide(image=image, group=candidates, tracks=shortlist,
                                     timestamp=timestamp, specification=specification,
                                     task_query=task_query, competition=competition)
        decisions: list[ParsedDecision] = list(output.decisions)

        if self.config.apply_admission_gate:
            for decision in decisions:
                if decision.admitted is not None and decision.admitted < self.config.admission_threshold:
                    decision.decision = "REJECT"
                    decision.track_id = None

        probabilities = [d.probabilities for d in decisions]
        options = self._options_from(decisions)
        result = assign(probabilities, options, solver=self.config.solver,
                        duplicate_penalty=self.config.duplicate_penalty)
        assignment = dict(result.assignment)
        for index, decision in enumerate(decisions):
            if index in assignment:
                continue
            if decision.decision in (NEW_TOKEN, REJECT_TOKEN, "NO_MATCH"):
                assignment[index] = decision.decision
            else:
                assignment[index] = REJECT_TOKEN if decision.track_id is None else str(decision.track_id)

        tracker_ops: list[dict[str, Any]] = []
        memory_updates: dict[int, dict[str, Any]] = {}
        if self.config.assignment_conditioned_memory:
            memory = self.adapter.decide_memory(image=image, group=candidates,
                                                assigned=assignment, tracks=shortlist,
                                                timestamp=timestamp, specification=specification,
                                                competition=competition)
            raw_updates = memory.get("updates", {}) if isinstance(memory, dict) else {}
            for candidate, payload in raw_updates.items():
                value = payload.get("noul")
                memory_updates[int(candidate)] = {
                    "noul": value,
                    "update": bool(value is not None and value >= self.config.memory_update_threshold),
                    "risk": payload.get("risk"),
                }
        else:
            for candidate, token in assignment.items():
                if token not in (NEW_TOKEN, REJECT_TOKEN):
                    memory_updates[candidate] = {"noul": None, "update": True, "risk": None,
                                                 "policy": "unconditional_memory_control"}

        for op in assignment_to_memory_ops(result):
            candidate = op["candidate"]
            decision = memory_updates.get(candidate)
            if op["op"] == "ASSIGN" and decision is not None and not decision["update"]:
                tracker_ops.append({"candidate": candidate, "op": "ASSIGN_HOLD_MEMORY",
                                    "track_id": op["track_id"]})
            else:
                tracker_ops.append(op)

        diagnostics = {
            "config": self.config.to_dict(),
            "shortlist_size": len(shortlist),
            "shortlist_ids": [int(t["track_id"]) for t in shortlist],
            "competition_enabled": self.config.use_competition,
            "competition_groups": (competition or {}).get("competition_groups"),
            "assignment_solver": result.solver,
            "assignment_diagnostics": result.diagnostics,
            "adapter_metadata": output.metadata,
            "memory_updates_applied": sum(1 for v in memory_updates.values() if v.get("update")),
            "memory_holds": sum(1 for v in memory_updates.values() if not v.get("update")),
        }
        return TrackJEVStep(decisions=decisions, assignment=assignment,
                            memory_updates=memory_updates, tracker_ops=tracker_ops,
                            diagnostics=diagnostics)

    # ---------------------------------------------------------------- helper
    @staticmethod
    def _options_from(decisions: Sequence[ParsedDecision]) -> list[Any]:
        """Rebuild the option list from the union of decoded probability keys."""
        from .decision_schema import IdentityOption

        tokens: list[str] = []
        for decision in decisions:
            for token in decision.probabilities:
                if token not in tokens:
                    tokens.append(token)
        options: list[IdentityOption] = []
        for token in tokens:
            if token == NEW_TOKEN:
                options.append(IdentityOption(token=NEW_TOKEN, text="new entity"))
            elif token == REJECT_TOKEN:
                options.append(IdentityOption(token=REJECT_TOKEN, text="invalid or duplicate"))
            elif token == "NO_MATCH":
                options.append(IdentityOption(token="NO_MATCH", text="valid object, no supported id"))
            else:
                options.append(IdentityOption(token=token, track_id=int(token), text="existing id"))
        return options
