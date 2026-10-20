"""Online one-forward GroundJEV runtime and assignment replay."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

from .dynamic_identity_slots import IdentitySlot, build_dynamic_slots
from .g2_feature_bridge import G2FeatureBridge
from .groundjev_model import GroundJEVCoupledModel
from .identity_memory import IdentityMemoryStore
from .lifecycle_controller import LifecycleController
from .memory_writer import MemoryWriter
from .shared_ground_state import SharedGroundState
from .typed_decisions import TypedDecisionBatch


@dataclass
class RuntimeConfig:
    memory_enabled: bool = True
    admission_threshold: float = 0.5
    max_history: int = 31
    memory_update_threshold: float = 0.5


def global_one_to_one_assignment(decisions: TypedDecisionBatch, *,
                                 config: RuntimeConfig) -> dict[int, str]:
    """Solve a global assignment using the raw dynamic-slot probabilities."""
    from scipy.optimize import linear_sum_assignment

    rows = decisions.decisions
    if not rows:
        return {}
    existing = sorted({slot.token for row in rows for slot in row.slots if slot.is_existing},
                      key=lambda token: int(token))
    # NEW/NO_MATCH/REJECT are candidate-local actions.  Keeping one column per
    # candidate allows two legitimate births in the same frame while existing
    # identities remain globally one-to-one.
    action_columns = [(action, row) for row in range(len(rows))
                      for action in ("NEW", "NO_MATCH", "REJECT")]
    columns = existing + action_columns
    costs = torch.full((len(rows), len(columns)), 1e6, dtype=torch.float64)
    for i, row in enumerate(rows):
        probs = row.probability_dict()
        if row.admission_probability < config.admission_threshold:
            probs["REJECT"] = max(probs.get("REJECT", 0.0), 1.0 - row.admission_probability)
        for j, column in enumerate(columns):
            if isinstance(column, tuple):
                token, owner = column
                if owner != i:
                    continue
            else:
                token = column
            if token in probs:
                costs[i, j] = -math.log(max(float(probs[token]), 1e-12))
    row_ids, col_ids = linear_sum_assignment(costs.numpy())
    assignment = {}
    for row, column in zip(row_ids, col_ids):
        selected = columns[int(column)]
        assignment[int(row)] = selected[0] if isinstance(selected, tuple) else selected
    for index, row in enumerate(rows):
        if index not in assignment:
            assignment[index] = "REJECT"
    # A model's NO_MATCH is never silently converted to NEW.  Conversely, an
    # empty history keeps NEW feasible for every admitted candidate.
    return assignment


class G2SharedRuntime:
    """Encode -> typed decisions -> global assignment -> causal memory."""

    def __init__(self, bridge: G2FeatureBridge, model: GroundJEVCoupledModel,
                 *, store: Optional[IdentityMemoryStore] = None,
                 config: Optional[RuntimeConfig] = None) -> None:
        self.bridge = bridge
        self.model = model
        self.store = store or IdentityMemoryStore()
        self.config = config or RuntimeConfig()
        self.lifecycle = LifecycleController(self.store)
        self.writer = MemoryWriter(self.store)

    def _memory_views(self) -> dict[int, Any]:
        return {int(track_id): memory for track_id, memory in self.store.tracks.items()
                if memory.state != "TERMINATE"}

    def step(self, *, image: Any, boxes: Sequence[Sequence[float]],
             timestamp: float, frame_id: int | str, image_key: str,
             task_query: str = "", contextual_tokens: Optional[torch.Tensor] = None,
             metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if self.store.last_timestamp is not None and float(timestamp) <= self.store.last_timestamp:
            raise ValueError("runtime timestamps must strictly increase")
        state = self.bridge.encode(
            image, boxes, frame_id=frame_id, image_key=image_key,
            task_query=task_query, contextual_tokens=contextual_tokens,
            metadata=metadata,
        )
        return self.step_from_state(state, timestamp=timestamp)

    def step_from_state(self, state: SharedGroundState, *, timestamp: float) -> dict[str, Any]:
        """Run decisions and causal memory updates from one encoded frame.

        This is the explicit shared-feature path used by ablation replays: a
        frame can be compared across readouts without silently running the G2
        visual encoder once per arm.
        """
        if self.store.last_timestamp is not None and float(timestamp) <= self.store.last_timestamp:
            raise ValueError("runtime timestamps must strictly increase")
        boxes = state.boxes
        memories = self._memory_views()
        slots = build_dynamic_slots(list(memories.values()), max_history=self.config.max_history)
        slots_per_candidate = [list(slots) for _ in boxes]
        batch = self.model(state, slots_per_candidate, memories=memories,
                           memory_enabled=self.config.memory_enabled)
        assignment = global_one_to_one_assignment(batch, config=self.config)
        self.lifecycle.validate_assignment(assignment, num_candidates=len(boxes))
        update_probabilities = {row.candidate_index: row.memory_update_probability
                                for row in batch.decisions}
        confidence = {row.candidate_index: row.chosen_probability for row in batch.decisions}
        memory_report = self.writer.apply(
            assignments=assignment,
            candidate_tokens=state.region_tokens,
            boxes=[list(map(float, box)) for box in boxes],
            timestamp=float(timestamp),
            update_probabilities=update_probabilities,
            confidences=confidence,
        )
        seen_ids = [int(token) for token in assignment.values() if token.isdigit()]
        seen_ids.extend(int(value) for value in memory_report.get("new_assignments", {}).values())
        lifecycle_report = self.lifecycle.apply_unseen(
            seen_ids=seen_ids, timestamp=float(timestamp), lifecycle_outputs=None)
        batch.assignment = assignment
        batch.metadata.update({
            "memory_report": memory_report,
            "lifecycle_report": lifecycle_report,
            "visual_forward_calls_for_frame": state.metadata.get("visual_forward_calls_for_frame", 1),
            "bridge_forward_calls_total": self.bridge.forward_calls,
            "single_backbone": True,
            "second_vlm_loaded": False,
        })
        return {"state": state, "decisions": batch, "assignment": assignment,
                "memory": memory_report, "lifecycle": lifecycle_report,
                "persistent_state": self.store.state_dict()}
