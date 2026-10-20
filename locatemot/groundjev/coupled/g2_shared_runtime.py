"""Causal single-backbone GroundJEV video runtime."""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .dynamic_identity_slots import DynamicIdentitySlots, build_dynamic_slots
from .g2_feature_bridge import G2FeatureBridge
from .groundjev_model import GroundJEVModel
from .identity_memory import IdentityMemoryState
from .lifecycle_controller import LifecycleController
from .memory_writer import MemoryWriter
from .typed_decisions import decode_typed_decisions


def _global_one_to_one(scores: torch.Tensor) -> list[tuple[int, int]]:
    """Solve a rectangular candidate/track assignment globally."""
    if scores.numel() == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment((-scores.detach().cpu()).numpy())
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    except Exception:
        # This fallback is only for minimal environments; it remains one-to-one
        # and is recorded by the caller when SciPy is unavailable.
        pairs = sorted(((float(scores[i, j]), i, j)
                        for i in range(scores.shape[0]) for j in range(scores.shape[1])), reverse=True)
        used_i: set[int] = set(); used_j: set[int] = set(); result = []
        for _, i, j in pairs:
            if i not in used_i and j not in used_j:
                used_i.add(i); used_j.add(j); result.append((i, j))
        return result


class CoupledGroundJEVRuntime:
    """One G2 feature forward, typed decision, assignment, then memory write."""

    def __init__(self, bridge: G2FeatureBridge, model: GroundJEVModel,
                 *, memory: IdentityMemoryState | None = None,
                 lifecycle: LifecycleController | None = None,
                 memory_writer: MemoryWriter | None = None,
                 admit_threshold: float = 0.5):
        self.bridge = bridge
        self.model = model
        self.memory = memory or IdentityMemoryState()
        self.lifecycle = lifecycle or LifecycleController()
        self.memory_writer = memory_writer or MemoryWriter()
        self.admit_threshold = float(admit_threshold)

    @torch.no_grad()
    def step(self, image: Any, boxes: Sequence[Sequence[float]], labels: Sequence[str], *,
             frame_id: int, image_key: str, task_query: str | None = None,
             scores: Sequence[float] | None = None) -> dict[str, Any]:
        self.memory.assert_causal(frame_id)
        state = self.bridge.encode(image, boxes, labels, frame_id=frame_id,
                                   image_key=image_key, task_query=task_query, scores=scores)
        slots = build_dynamic_slots(self.memory.ordered())
        outputs = self.model(state, self.memory, slots)
        decisions = decode_typed_decisions(outputs, slots.tokens)
        existing_slots = [slot for slot in slots.slots if slot.track_id is not None]
        assignments: dict[int, int] = {}
        assignment_solver = "none"
        if existing_slots:
            index = [slots.tokens.index(slot.token) for slot in existing_slots]
            probabilities = torch.softmax(outputs["identity_logits"], dim=-1)
            scores_matrix = probabilities[:, index]
            for candidate_index, track_index in _global_one_to_one(scores_matrix):
                decision = decisions[candidate_index]
                track_id = existing_slots[track_index].track_id
                if (track_id is not None and decision["identity"] == str(track_id) and
                        decision["admit_probability"] >= self.admit_threshold):
                    assignments[candidate_index] = int(track_id)
            assignment_solver = "hungarian" if _has_scipy() else "one_to_one_fallback"
        # Explicit NEW is the only route to a birth.  NO_MATCH never becomes NEW.
        for decision in decisions:
            i = int(decision["candidate_index"])
            if i in assignments or decision["admit_probability"] < self.admit_threshold:
                continue
            if decision["identity"] == "NEW":
                track = self.memory.create(state.candidates[i].region_token, state.candidates[i].box,
                                           timestamp=frame_id, confidence=decision["identity_probability"],
                                           semantic_evidence=labels[i])
                assignments[i] = track.persistent_track_id
        matched_ids = set(assignments.values())
        for track in self.memory.ordered():
            if track.persistent_track_id in matched_ids:
                self.lifecycle.apply(track, "ACTIVE", timestamp=frame_id, matched=True)
            elif track.state != "TERMINATED":
                self.lifecycle.apply(track, "LOST", timestamp=frame_id, matched=False)
        updates = {int(d["candidate_index"]): float(d["memory_update_probability"]) for d in decisions}
        report = self.memory_writer.write(self.memory, state.candidates, assignments, updates,
                                          timestamp=frame_id, labels=labels)
        return {
            "state": state, "slots": slots, "decisions": decisions,
            "assignments": assignments, "memory_report": report,
            "assignment_solver": assignment_solver,
            "bridge_forward_count": self.bridge.forward_count,
            "single_backbone": True,
        }


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False
