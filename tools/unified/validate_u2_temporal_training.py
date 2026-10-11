#!/usr/bin/env python3
"""Targeted causal/temporal-supervision proof on synthetic identity clips."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from locatemot.unified.losses.u2_association import FrameSupervision, candidate_gt_assignment
from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
from locatemot.unified.runtime.u2_temporal_training import (
    U2MOTTemporalTraining, clip_length, teacher_forcing_probability, update_identity_bookkeeping,
)


def synthetic_clip(hidden: int = 32, device: str = "cpu") -> list[dict]:
    visual = torch.randn(4, hidden, device=device)
    result = []
    for index in range(4):
        # Two approaching objects cross in the final frame. Third is a real
        # unmatched proposal; fourth is inside an ignored region.
        boxes = torch.tensor([[4. * index, 0., 20. + 4. * index, 40.],
                              [24. - 4. * index, 0., 44. - 4. * index, 40.],
                              [80., 0., 100., 40.], [120., 0., 140., 40.]], device=device)
        gt_boxes = boxes[:2].clone()
        keep = torch.tensor([0, 2, 3], device=device) if index == 2 else torch.arange(4, device=device)
        candidates = {"boxes": boxes[keep], "scores": boxes.new_full((len(keep),), .9),
                      "id_visual": visual[keep], "semantic_visual": visual[keep] * .7,
                      "decoder_query": visual[keep] * .4}
        gt = FrameSupervision(gt_boxes, torch.tensor([10, 20], device=device), boxes[3:], {10, 20})
        result.append({"frame_index": index * 4, "candidates": candidates, "supervision": gt})
    return result


def main() -> int:
    torch.manual_seed(20261012)
    torch.set_num_threads(2)
    core = SharedTrackCore(hidden_dim=32)
    torch.nn.init.constant_(core.association.pair[-1].bias, 3.)
    torch.nn.init.constant_(core.association.no_match[-1].bias, -3.)
    torch.nn.init.constant_(core.association.new[-1].bias, 3.)
    trainer = U2MOTTemporalTraining(core).train()
    clip = synthetic_clip()
    captured = []

    def capture(module, args, output):
        if output["state"].id_memory.requires_grad:
            output["state"].id_memory.retain_grad()
        if output["new_logits"].requires_grad:
            output["new_logits"].retain_grad()
        captured.append(output)

    handle = core.register_forward_hook(capture)
    result = trainer([clip], 3000, forcing_decisions=[[True, True, False, True]], augment=False)
    result["losses"]["total"].backward()
    handle.remove()
    assert all(torch.isfinite(value) for value in result["losses"].values())
    assert captured[0]["state"].id_memory.grad.abs().sum() > 0, "Temporal graph detached after birth"
    branches = {"candidate_id_projection": core.candidate_encoder.id_projection[0].weight,
                "identity_gru": core.identity_gru.weight_hh, "pair_head": core.association.pair[0].weight,
                "set_interaction": core.set_interaction.encoder.layers[0].linear1.weight,
                "motion": core.motion.head[0].weight, "alive": core.lifecycle.head[0].weight,
                "birth": core.association.new[0].weight}
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in branches.values())
    assert captured[0]["birth_candidates"] == [0, 1], "False/ignored proposals taught NEW"
    assert captured[0]["new_logits"].grad[2] > 0, "False-birth loss does not suppress the logit"
    assert captured[0]["new_logits"].grad[3] == 0, "Ignored proposal used as a negative"
    assert [d["teacher_forced"] for d in result["diagnostics"]] == [True, True, False, True]
    assert result["diagnostics"][2]["model_birth_candidates"], "Model rollout was replaced by teacher state"
    assert all(set(output["state"].metadata) == {"max_age"} for output in captured), "GT leaked into runtime state"

    # State augmentation must change actual track rows, using past false
    # proposals, while leaving the incoming state and permanent anchors intact.
    state = captured[0]["state"]
    pool = {"candidate_id_tokens": captured[0]["candidate_id_tokens"][2:3],
            "candidate_sem_tokens": captured[0]["candidate_sem_tokens"][2:3],
            "boxes": clip[0]["candidates"]["boxes"][2:3], "scores": clip[0]["candidates"]["scores"][2:3]}
    core.query_interaction.drop_probability = 0.
    core.query_interaction.false_positive_probability = 1.
    augmented, info = core.query_interaction.augment_state(state, pool)
    assert augmented.num_tracks == state.num_tracks + 1 and info["injected_track_ids"] == [2]
    assert torch.equal(augmented.identity_anchor[-1], pool["candidate_id_tokens"][0])
    core.query_interaction.drop_probability = 1.
    dropped, info_drop = core.query_interaction.augment_state(state, None)
    assert dropped.num_tracks == 0 and info_drop["dropped_track_ids"] == [0, 1]
    assert state.num_tracks == 2

    # A switch cannot be hidden by relabeling an already-known runtime track.
    swapped = {**captured[0], "candidate_indices": torch.tensor([1, 0]), "track_ids": torch.tensor([0, 1])}
    assert update_identity_bookkeeping({0: 10, 1: 20}, swapped, torch.tensor([10, 20, -1, -1])) == {0: 10, 1: 20}
    ids, eligible = candidate_gt_assignment(clip[0]["candidates"]["boxes"], clip[0]["supervision"])
    assert ids.tolist() == [10, 20, -1, -1] and eligible.tolist() == [True, True, True, False]

    core.eval()
    with torch.no_grad():
        first = core(clip[0]["candidates"], frame_index=0)
        # Single identity avoids ambiguous untrained Hungarian choices.
        one = {name: value[:1] for name, value in clip[0]["candidates"].items()}
        initial = core(one, frame_index=0)
        moving = {name: value.clone() for name, value in one.items()}
        moving["boxes"][:, [0, 2]] += 4
        matched = core(moving, initial["state"], frame_index=4)
        assert torch.allclose(matched["state"].velocity[:, 0], torch.tensor([.05]))
        empty = {name: value[:0] for name, value in one.items()}
        missing = core(empty, matched["state"], frame_index=8)
        assert missing["state"].lost_age.tolist() == [4] and missing["state"].age.tolist() == [9]
        assert torch.allclose(missing["state"].boxes[:, 0], torch.tensor([8.]))
        assert torch.allclose(missing["state"].id_memory, matched["state"].id_memory * .995 ** 4)
        expired = core(one, missing["state"], frame_index=35)
        assert expired["track_ids"].tolist() == [1], "Expired identity revived beyond the 30-source-frame cap"
        try:
            core(one, first["state"], forced_assignment=([], {0}), training=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Teacher forcing enabled at inference")
    assert [clip_length(step) for step in (0, 999, 1000, 2499, 2500, 3999, 4000, 5999)] == [2, 2, 4, 4, 6, 6, 8, 8]
    assert [teacher_forcing_probability(step) for step in (0, 500, 1500, 2500, 5999)] == [1., 1., .7, .4, .4]
    report = {"format": "locatemot-u2-temporal-supervision-contract-v1", "status": "PASS",
              "input_source": "synthetic crossing/occlusion clip; not benchmark metrics",
              "losses": {key: float(value.detach()) for key, value in result["losses"].items()},
              "nonzero_gradient_branches": sorted(branches), "connected_temporal_gradient": True,
              "actual_state_drop_and_past_negative_injection": True, "mixed_teacher_model_rollout": True,
              "false_birth_suppression": True, "official_ignore_region_excluded_from_negatives": True,
              "canonical_identity_not_relabelled_after_switch": True, "physical_frame_gap": 4,
              "expired_identity_not_revived_after_max_age": True,
              "gt_in_runtime_state": False, "formal_training_launched": False, "official_test_labels_read": False}
    path = ROOT / "outputs/unified/u2/eval/temporal_supervision_contract.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
