"""TrackJEV decision adapter for the project's ``OnlineTracker`` contract.

``OnlineTracker`` expects a decision object with

* ``decide(image, candidates, tracks, timestamp, specification)`` returning
  ``(decisions, lifecycle, metadata)`` where each decision carries a
  ``probabilities`` dict keyed by ``str(track_id)`` / ``NEW`` / ``IGNORE``
  (and optionally ``NO_MATCH``) plus ``update_probability``;
* optionally ``decide_assigned_updates(selected_existing)`` which is called
  **after** the global assignment, with the ids the tracker actually selected.

That second hook is exactly the assignment-conditioned memory interface, so
TrackJEV plugs into the existing tracker without duplicating assignment logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .candidate_competition import competition_rows, entity_clusters, sparse_edges
from .decision_schema import MAX_SHORTLIST, NO_MATCH_TOKEN, NEW_TOKEN, REJECT_TOKEN, IdentityOption
from .native_onejev_adapter import NativeOneJevAdapter

TRACKER_REJECT_TOKEN = "IGNORE"      # the tracker's historical token for "do not create an id"


@dataclass
class TrackerDecisionConfig:
    shortlist: int = MAX_SHORTLIST
    use_competition: bool = True
    assignment_conditioned_memory: bool = True
    ask_risk: bool = False
    memory_threshold: float = 0.5
    duplicate_penalty: float = 0.0


class TrackJEVTrackerDecision:
    """Adapter: TrackJEV native OneJev decisions -> OnlineTracker protocol."""

    def __init__(self, adapter: NativeOneJevAdapter, config: Optional[TrackerDecisionConfig] = None) -> None:
        self.adapter = adapter
        self.config = config or TrackerDecisionConfig()
        self._last: dict[str, Any] = {}

    # --------------------------------------------------------------- helpers
    def _shortlist(self, candidates: Sequence[dict[str, Any]],
                   tracks: Sequence[Any]) -> list[dict[str, Any]]:
        from locatemot.groundjev.decision import overlap

        ranked = sorted(
            tracks,
            key=lambda t: (-max((overlap(c["box"], t.predicted_box(self._timestamp)) for c in candidates),
                                default=0.0),
                           t.lost_age, t.track_id))[: self.config.shortlist]
        return [{"track_id": t.track_id, "lost_age": t.lost_age,
                 "box": list(t.box), "predicted_box": t.predicted_box(self._timestamp),
                 "age": t.age, "state": t.state,
                 "last_reliable_timestamp": t.last_reliable_timestamp} for t in ranked]

    def _competition(self, candidates: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not self.config.use_competition:
            return None
        boxes = [c["box"] for c in candidates]
        features = [c.get("native_region_feature") for c in candidates]
        edges = sparse_edges(boxes, features if any(f is not None for f in features) else None,
                             box_format="xyxy")
        clusters = entity_clusters(boxes, edges)
        return competition_rows(boxes, clusters, edges, box_format="xyxy")

    # ------------------------------------------------------------ main entry
    def decide(self, image: Any, candidates: Sequence[dict[str, Any]],
               tracks: Sequence[Any], timestamp: float,
               specification: dict[str, Any]):
        self._timestamp = timestamp
        shortlist = self._shortlist(candidates, tracks)
        competition = self._competition(candidates)
        output = self.adapter.decide(image=image, group=candidates, tracks=shortlist,
                                     timestamp=timestamp, specification=specification,
                                     competition=competition)
        decisions = []
        for decision in output.decisions:
            probabilities = dict(decision.probabilities)
            if REJECT_TOKEN in probabilities:
                probabilities[TRACKER_REJECT_TOKEN] = probabilities.pop(REJECT_TOKEN)
            probabilities.setdefault(TRACKER_REJECT_TOKEN, 0.0)
            probabilities.setdefault(NO_MATCH_TOKEN, 0.0)
            decisions.append({
                "probabilities": probabilities,
                "update_probability": 0.5 if self.config.assignment_conditioned_memory else 1.0,
                "reliability": decision.risk,
                "shortlisted_ids": [int(t["track_id"]) for t in shortlist],
            })
        metadata = dict(output.metadata)
        metadata.update(native_probabilities=True,
                        explicit_no_match=True,
                        trackjev_assignment_conditioned_memory=self.config.assignment_conditioned_memory,
                        trackjev_arm=("B6" if self.config.ask_risk else
                                      "B4" if self.config.assignment_conditioned_memory else
                                      "B3" if self.config.use_competition else "B2"))
        self._last = {"image": image, "candidates": list(candidates), "shortlist": shortlist,
                      "competition": competition, "timestamp": timestamp,
                      "specification": specification, "decisions": decisions}
        return decisions, {}, metadata

    # ------------------------------------------- assignment-conditioned memory
    def decide_assigned_updates(self, selected_existing: dict[int, int]):
        """Called by the tracker with the FINAL assignment of this frame."""
        if not self.config.assignment_conditioned_memory or not selected_existing:
            return {}, {"protocol": "disabled"}
        last = self._last
        assigned = {int(candidate): str(int(track_id))
                    for candidate, track_id in selected_existing.items()}
        result = self.adapter.decide_memory(image=last["image"], group=last["candidates"],
                                            assigned=assigned, tracks=last["shortlist"],
                                            timestamp=last["timestamp"],
                                            specification=last["specification"],
                                            competition=last["competition"])
        updates = {}
        for candidate, payload in (result.get("updates") or {}).items():
            value = payload.get("noul")
            if value is not None:
                updates[int(candidate)] = float(value)
        report = {
            "protocol": "assignment_conditioned_native_onejev",
            "asked": result.get("asked"),
            "assigned_ids": {str(k): v for k, v in assigned.items()},
            "answered": len(updates),
            "threshold": self.config.memory_threshold,
            "writes": sum(1 for v in updates.values() if v >= self.config.memory_threshold),
            "holds": sum(1 for v in updates.values() if v < self.config.memory_threshold),
        }
        return updates, report
