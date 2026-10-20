import torch

from locatemot.groundjev.coupled.dynamic_identity_slots import build_dynamic_slots
from locatemot.groundjev.coupled.g2_jev_readout import G2JEVReadout
from locatemot.groundjev.coupled.groundjev_model import GroundJEVModel
from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.g2_shared_runtime import CoupledGroundJEVRuntime
from locatemot.groundjev.coupled.identity_memory import IdentityMemoryState
from locatemot.groundjev.coupled.lifecycle_controller import LifecycleConfig, LifecycleController
from locatemot.groundjev.coupled.losses import structured_groundjev_losses


class FakeG2Provider:
    def __init__(self):
        self.calls = 0

    def __call__(self, image, boxes):
        self.calls += 1
        n = len(boxes)
        tokens = torch.arange(max(n, 1) * 8, dtype=torch.float32).reshape(max(n, 1), 8)[:n]
        return tokens, tokens[:, None, :], {"frame_id": self.calls}


def test_typed_readout_has_real_backward_path():
    model = G2JEVReadout(8, hidden_dim=16)
    tokens = torch.randn(3, 8)
    slots = torch.randn(4, 16)
    outputs = model(tokens, slots)
    losses = structured_groundjev_losses(
        outputs,
        identity_targets=[0, 1, 2], identity_valid=[True, True, True],
        admit_targets=[1, 1, 0], admit_valid=[True, True, True],
        yesno_targets=[1, 0, 1], yesno_valid=[True, True, True],
        lifecycle_targets=[0, 1, 2], lifecycle_valid=[True, True, True],
        memory_targets=[1, 0, 1], memory_valid=[True, True, True],
    )
    assert losses["total"] is not None
    losses["total"].backward()
    assert model.region_projector.projection.weight.grad is not None


def test_dynamic_slots_and_safe_termination():
    class Track:
        def __init__(self, track_id, lost_age):
            self.track_id = track_id
            self.lost_age = lost_age

    slots = build_dynamic_slots([Track(3, 0), Track(2, 4)], max_existing=2)
    assert slots.tokens[:2] == ["3", "2"]
    assert slots.tokens[-3:] == ["NEW", "NO_MATCH", "REJECT"]
    memory = IdentityMemoryState()
    track = memory.create(torch.ones(8), (0, 0, 1, 1), timestamp=0)
    controller = LifecycleController()
    controller.apply(track, "TERMINATE", timestamp=1, confidence=1.0, matched=False)
    assert track.state == "LOST"


def test_runtime_uses_one_feature_forward_and_causal_memory():
    provider = FakeG2Provider()
    bridge = G2FeatureBridge(provider, model_sha256="fixture", device="cpu")
    model = GroundJEVModel(8, hidden_dim=16)
    runtime = CoupledGroundJEVRuntime(bridge, model)
    result = runtime.step(object(), [(0, 0, 2, 2), (3, 3, 5, 5)], ["person", "person"],
                          frame_id=0, image_key="fixture/0.jpg")
    assert result["single_backbone"] is True
    assert result["bridge_forward_count"] == 1
    assert provider.calls == 1
    assert runtime.memory.timestamp == 0
    assert len(set(result["assignments"].values())) == len(result["assignments"])


def test_runtime_executes_lifecycle_and_timeout_without_memory_reactivation():
    class ControlledModel:
        def __call__(self, state, memory, slots):
            count = int(state.vision_tokens.shape[0])
            option_count = len(slots.tokens)
            identity = torch.full((count, option_count), -8.0)
            lifecycle = torch.full((count, 4), -8.0)
            if count:
                # With no history the first slot is NEW; after birth the first
                # slot is the existing persistent id.
                identity[:, 0] = 8.0
                lifecycle[:, 1 if memory.tracks else 0] = 8.0  # LOST / ACTIVE
            return {
                "candidate_hidden": state.vision_tokens,
                "slot_hidden": state.vision_tokens.new_empty((option_count, 16)),
                "identity_logits": identity,
                "admit_logits": torch.full((count,), 8.0),
                "yesno_logits": torch.zeros(count),
                "lifecycle_logits": lifecycle,
                "memory_logits": torch.full((count,), 8.0),
            }

    provider = FakeG2Provider()
    bridge = G2FeatureBridge(provider, model_sha256="fixture", device="cpu")
    runtime = CoupledGroundJEVRuntime(
        bridge, ControlledModel(),
        lifecycle=LifecycleController(LifecycleConfig(
            terminate_after_lost=2, terminate_probability=0.95
        )),
    )
    first = runtime.step(object(), [(0, 0, 2, 2)], ["person"],
                         frame_id=0, image_key="fixture/0.jpg")
    track_id = first["assignments"][0]
    second = runtime.step(object(), [(0, 0, 2, 2)], ["person"],
                          frame_id=1, image_key="fixture/1.jpg")
    track = runtime.memory.tracks[track_id]
    assert second["lifecycle_actions"][track_id] == "LOST"
    assert track.state == "LOST"
    assert track.memory_update_status == "HOLD"
    assert second["memory_report"]["skipped"] == 1

    third = runtime.step(object(), [], [], frame_id=2, image_key="fixture/2.jpg")
    assert track_id in third["terminated_track_ids"]
    assert runtime.memory.tracks[track_id].state == "TERMINATED"
