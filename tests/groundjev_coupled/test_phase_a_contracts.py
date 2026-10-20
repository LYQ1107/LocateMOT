from __future__ import annotations

import json

import unittest
import torch

from locatemot.groundjev.coupled.dynamic_identity_slots import build_dynamic_slots
from locatemot.groundjev.coupled.g2_jev_readout import G2JEVReadout
from locatemot.groundjev.coupled.g2_shared_runtime import RuntimeConfig, global_one_to_one_assignment
from locatemot.groundjev.coupled.groundjev_model import GroundJEVCoupledModel
from locatemot.groundjev.coupled.identity_memory import IdentityMemoryStore
from locatemot.groundjev.coupled.lifecycle_controller import LifecycleController
from locatemot.groundjev.coupled.memory_writer import MemoryWriter
from locatemot.groundjev.coupled.region_tokens import pool_region_tokens
from locatemot.groundjev.coupled.shared_ground_state import SharedGroundState


def _state(n=2, dim=8):
    spatial = torch.arange(dim * 4 * 4, dtype=torch.float32).reshape(1, dim, 4, 4) / 100
    pooled, _ = pool_region_tokens(spatial, [[0, 0, 8, 8], [8, 8, 16, 16]][:n],
                                    image_size=(16, 16))
    return SharedGroundState(
        frame_id=1, image_key="frame-1", model_sha256="model-sha",
        boxes=tuple((float(v) for v in box) for box in [[0, 0, 8, 8], [8, 8, 16, 16]][:n]),
        vision_tokens=spatial, region_tokens=pooled, image_grid_thw=(1, 8, 8),
        spatial_layout=(4, 4), image_size=(16, 16), metadata={"visual_forward_calls_for_frame": 1},
    )


class PhaseAContracts(unittest.TestCase):
  def test_shared_state_and_region_geometry(self):
    state = _state()
    assert state.region_tokens.shape == (2, 8)
    assert state.audit_dict()["vision_tokens_shape"] == [1, 8, 4, 4]
    with self.assertRaises(ValueError):
        SharedGroundState(frame_id=1, image_key="", model_sha256="x",
                          boxes=(), vision_tokens=torch.zeros(1, 8, 4, 4),
                          region_tokens=torch.empty(0, 8), image_grid_thw=(1, 8, 8),
                          spatial_layout=(4, 4), image_size=(16, 16))

  def test_dynamic_identity_and_multi_birth_assignment(self):
    slots = build_dynamic_slots([], max_history=31)
    assert [slot.token for slot in slots] == ["NEW", "NO_MATCH", "REJECT"]
    exported = build_dynamic_slots([{"id": 7, "lost_duration": 2, "prior": 0.5}])
    assert exported[0].token == "7" and exported[0].lost_age == 2
    model = G2JEVReadout(8, hidden_dim=16)
    decisions = model(torch.randn(2, 8), [slots, slots])
    from locatemot.groundjev.coupled.typed_decisions import TypedDecisionBatch
    batch = TypedDecisionBatch(decisions=decisions, frame_id=1, image_key="f",
                               model_sha256="m", visual_forward_calls=1)
    # Force both candidate-local NEW columns to be feasible regardless of the
    # random readout output; the assignment solver must permit two births.
    for row in batch.decisions:
        row.identity_probabilities = torch.tensor([0.9, 0.05, 0.05])
    assignment = global_one_to_one_assignment(batch, config=RuntimeConfig())
    assert assignment == {0: "NEW", 1: "NEW"}


  def test_memory_write_is_assignment_conditioned_and_causal(self):
    store = IdentityMemoryStore()
    writer = MemoryWriter(store)
    report = writer.apply(assignments={0: "NEW", 1: "REJECT"},
                          candidate_tokens=torch.randn(2, 8),
                          boxes=[[0, 0, 4, 4], [1, 1, 5, 5]], timestamp=1,
                          update_probabilities={0: 1.0, 1: 1.0})
    assert report["assigned"] == 1 and len(store.tracks) == 1
    with self.assertRaises(ValueError):
        writer.apply(assignments={0: "1"}, candidate_tokens=torch.randn(1, 8),
                     boxes=[[0, 0, 4, 4]], timestamp=0,
                     update_probabilities={0: 1.0})


  def test_lifecycle_rejects_unsafe_terminate_and_global_duplicate(self):
    store = IdentityMemoryStore()
    track = store.allocate()
    controller = LifecycleController(store)
    assert controller.suggest(track, "TERMINATE", model_probability=0.1) == "LOST"
    with self.assertRaises(ValueError):
        controller.validate_assignment({0: "1", 1: "1"}, num_candidates=2)


  def test_typed_readout_backward_and_disabled_baseline(self):
    state = _state()
    slots = build_dynamic_slots([], max_history=31)
    model = GroundJEVCoupledModel(8, hidden_dim=16)
    model.train()
    out = model(state, [slots, slots], memory_enabled=False)
    loss = sum(row.identity_logits.square().mean() for row in out.decisions)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Replaying the same state with memory disabled is deterministic and does
    # not mutate model/store state.
    out2 = model(state, [slots, slots], memory_enabled=False)
    assert torch.allclose(out.decisions[0].identity_logits,
                          out2.decisions[0].identity_logits)


if __name__ == "__main__":
    unittest.main()
