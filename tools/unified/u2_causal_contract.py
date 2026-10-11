#!/usr/bin/env python3
"""Deterministic U2 causal, isolation, and save/reload contract checks."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
from locatemot.unified.runtime.u2_online_tracker import U2OnlineTracker


def candidate(feature: torch.Tensor, *, x: float = 0.0) -> dict[str, torch.Tensor]:
    n, h = feature.shape
    boxes = torch.tensor([[x, 0.0, x + 1.0, 1.0]] * n, dtype=feature.dtype)
    return {
        "boxes": boxes,
        "scores": torch.full((n,), 0.9, dtype=feature.dtype),
        "id_visual": feature,
        "semantic_visual": feature * 0.7,
        "decoder_query": feature * 0.4,
    }


def deterministic_core(hidden_dim: int = 32) -> SharedTrackCore:
    torch.manual_seed(20261002)
    core = SharedTrackCore(hidden_dim=hidden_dim, max_age=30).eval()
    # Make contract transitions independent of an untrained random head while
    # retaining the real Hungarian, state, GRU, and semantic code paths.
    torch.nn.init.constant_(core.association.pair[-1].bias, 3.0)
    torch.nn.init.constant_(core.association.no_match[-1].bias, -3.0)
    torch.nn.init.constant_(core.association.new[-1].bias, 3.0)
    return core


def state_equal(left, right) -> bool:
    names = (
        "track_ids", "id_memory", "identity_anchor", "appearance_ema", "semantic_memory",
        "boxes", "velocity", "age", "hits", "lost_age", "last_frame", "confidence", "alive",
    )
    return left.next_track_id == right.next_track_id and left.frame_index == right.frame_index and all(
        torch.equal(getattr(left, name), getattr(right, name)) for name in names
    )


def main() -> int:
    torch.manual_seed(20261002)
    feature = torch.randn(1, 32)
    frame0 = candidate(feature)
    frame1 = candidate(feature + 0.01)
    core = deterministic_core()

    first = core(frame0, task_type="mot")
    birth_id = first["track_ids"].tolist()
    assert birth_id == [0], f"birth failed: {birth_id}"
    second = core(frame1, first["state"], task_type="mot")
    assert second["track_ids"].tolist() == birth_id, "matched identity was not persistent"

    occluded = core({"boxes": frame0["boxes"].new_empty((0, 4)), "scores": frame0["scores"].new_empty((0,)), "id_visual": frame0["id_visual"].new_empty((0, 32))}, second["state"], task_type="mot")
    assert occluded["state"].num_tracks == 1 and occluded["state"].lost_age.tolist() == [1]
    recovered = core(frame1, occluded["state"], task_type="mot")
    assert recovered["track_ids"].tolist() == birth_id, "occlusion recycled the identity"

    prefix_a = core(frame0, task_type="mot")
    prefix_b = core(frame0, task_type="mot")
    _ = core(candidate(feature * -3.0, x=4.0), prefix_a["state"], task_type="mot")
    _ = core(candidate(feature * 5.0, x=9.0), prefix_b["state"], task_type="mot")
    assert state_equal(prefix_a["state"], prefix_b["state"]), "frame t output depends on future input"

    text_a = torch.randn(3, 32)
    text_b = torch.randn(3, 32)
    mot_a = core(frame1, second["state"].clone(), task_type="mot", class_text=text_a)
    mot_b = core(frame1, second["state"].clone(), task_type="mot", class_text=text_b)
    assert torch.equal(mot_a["identity_pair_logits"], mot_b["identity_pair_logits"])
    assert torch.equal(mot_a["no_match_logits"], mot_b["no_match_logits"])
    assert torch.equal(mot_a["new_logits"], mot_b["new_logits"])
    assert state_equal(mot_a["state"], mot_b["state"]), "MOT specification changed identity state"

    ov_a = core(frame1, second["state"].clone(), task_type="ovmot", class_text=text_a)
    ov_b = core(frame1, second["state"].clone(), task_type="ovmot", class_text=text_b)
    assert torch.equal(ov_a["identity_pair_logits"], ov_b["identity_pair_logits"])
    assert torch.equal(ov_a["state"].id_memory, ov_b["state"].id_memory)
    assert torch.all((ov_a["semantic_gate"] >= 0) & (ov_a["semantic_gate"] <= 1))
    assert torch.allclose(core(frame1, second["state"], task_type="mot", class_text=text_a)["semantic_gate"], torch.zeros_like(ov_a["semantic_gate"]))

    original_task_embeddings = core.ovmot.task_embeddings.weight.detach().clone()
    with torch.no_grad():
        core.ovmot.task_embeddings.weight.add_(3.)
    changed_mot = core(frame1, second["state"].clone(), task_type="mot", class_text=text_a)
    changed_ov = core(frame1, second["state"].clone(), task_type="ovmot", class_text=text_a)
    assert state_equal(changed_mot["state"], mot_a["state"]), "Task embeddings leaked into ordinary MOT"
    assert torch.equal(changed_ov["identity_pair_logits"], ov_a["identity_pair_logits"])
    assert torch.equal(changed_ov["state"].id_memory, ov_a["state"].id_memory)
    assert not torch.equal(changed_ov["semantic_gate"], ov_a["semantic_gate"])
    with torch.no_grad():
        core.ovmot.task_embeddings.weight.copy_(original_task_embeddings)

    sequence = [frame0, frame1, candidate(feature + 0.02, x=0.1)]
    continuous = U2OnlineTracker(copy.deepcopy(core))
    continuous_outputs = [continuous.step(item, task_type="mot") for item in sequence]
    split = U2OnlineTracker(copy.deepcopy(core))
    split.step(sequence[0], task_type="mot")
    with tempfile.TemporaryDirectory(prefix="locatemot_u2_state_") as directory:
        state_path = Path(directory) / "state.pt"
        split.save_state(state_path)
        resumed = U2OnlineTracker(copy.deepcopy(core))
        resumed.load_state(state_path)
        resumed_outputs = [resumed.step(item, task_type="mot") for item in sequence[1:]]
    assert all(torch.equal(a["track_ids"], b["track_ids"]) for a, b in zip(continuous_outputs[1:], resumed_outputs))
    assert state_equal(continuous_outputs[-1]["state"], resumed_outputs[-1]["state"])

    payload = {
        "format": "locatemot-u2-causal-contract-v1",
        "status": "pass",
        "birth_persistence": "pass",
        "occlusion_retention": "pass",
        "no_future_dependency": "pass",
        "mot_specification_invariance": "pass",
        "ovmot_semantic_isolation": "pass",
        "task_embedding_identity_isolation": "pass",
        "online_state_save_reload": "pass",
        "max_age": core.max_age,
        "official_test_labels_read": False,
        "refer_kitti_official_eval_labels_read": False,
    }
    output = Path("outputs/unified/u2/eval/causal_contract.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
