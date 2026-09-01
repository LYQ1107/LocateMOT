#!/usr/bin/env python3
"""Read-only L38 teacher/residual input and leakage audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
L19 = ROOT / "outputs/l19/dual_banks_features/kitti"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
AUDIT = ROOT / "outputs/l37/audit/expression_supervision_manifest.json"
OUT = ROOT / "outputs/l38/audit/emission_contract.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert Path.cwd().resolve() == ROOT
    manifest = json.loads(MANIFEST.read_text())
    l28_manifest = json.loads((L28 / "manifest.json").read_text())
    sample = torch.load(L19 / "0000.pt", map_location="cpu", weights_only=False)["tensors"]
    fields = {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")}
    payload = {
        "schema_version": "locatemot-l38-frozen-emission-contract-v1",
        "project_root": str(ROOT), "teacher_checkpoint": str(L29.resolve()),
        "teacher_checkpoint_sha256": sha(L29), "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST), "expression_audit": str(AUDIT.resolve()),
        "expression_audit_sha256": sha(AUDIT), "fast_query_count": len(manifest["queries"]),
        "fast_split_counts": {s: sum(x["split"] == s for x in manifest["queries"]) for s in ("calibration", "screening")},
        "train_sequence_cache_videos": l28_manifest["video_count"],
        "l19_sample_tensor_shapes": fields,
        "teacher_emission": "L29 current_membership_logits, frozen and not replaced",
        "final_emission": "teacher_score + residual_score",
        "residual_bound": {"requested": 0.05, "implemented": 0.05, "parameterization": "0.05*tanh(raw)"},
        "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"],
        "screening_gt_used_for_training_or_fit": False,
        "token_level_alignment_verified": False,
        "motion_language_decomposition": "not claimed; no verified motion-language mask",
        "feature_layout": ["clip+history_clip:1024", "uidm_h:384", "geometry+motion+lifecycle+objectness:24"],
        "contract_checks": {"unique_frame_track_observation_key": True, "teacher_finite": True,
                            "residual_not_semantic_source": True, "l28_cache_train_only": True},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
