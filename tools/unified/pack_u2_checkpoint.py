#!/usr/bin/env python3
"""Package a recomputed eligible joint selection and its actual foundation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def selected_inputs(args):
    from tools.unified.train_u2_joint import preflight
    from tools.unified.eval_u2_joint import select_joint
    from tools.unified.build_u2_candidate_cache import digest, write_json
    ordinary, simple, reference, ovmot, labels, text = preflight(args)
    provided = json.loads(args.joint_selection.read_text())
    provenance = {"foundation_sha256": ordinary["spec"]["foundation_sha256"],
                  "ordinary_cache_sha256": digest(args.cache_manifest), "ovmot_cache_sha256": digest(args.ovmot_cache),
                  "class_text_sha256": digest(args.class_text), "mot_only_checkpoint_sha256": reference["checkpoint_sha256"],
                  "tao_gt_view_sha256": digest(args.tao_gt_view)}
    selected = select_joint(args.joint_selection.parent, reference, provenance)
    if provided != selected or selected["status"] != "U2_B_JOINT_SELECTED" or selected["selected"] is None:
        raise ValueError("Final packing requires all full joint measurements and unchanged passing gates")
    return ordinary, simple, reference, ovmot, labels, text, selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-manifest", "simple-baseline", "mot-only-selection", "ovmot-cache", "ovmot-supervision", "class-text", "tao-gt-view"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--joint-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from tools.unified.build_u2_candidate_cache import digest, write_json
    ordinary, simple, reference, ovmot, labels, text, selected = selected_inputs(args)
    import torch
    from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
    from locatemot.unified.runtime.grounding_inference import DEFAULT_BERT, load_grounding_runtime
    from locatemot.unified.runtime.u2_checkpoint import make_payload, save_payload
    chosen = selected["selected"]
    checkpoint = Path(chosen["checkpoint"])
    if digest(checkpoint) != chosen["checkpoint_sha256"]:
        raise ValueError("Selected joint tensor file differs from its full measurement")
    trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (trained["format"] != "locatemot-u2-joint-checkpoint-v1" or trained["next_step"] != chosen["step"]
            or trained["metadata"]["foundation_weights_modified"]):
        raise ValueError("This packer requires the exact selected frozen-foundation U2-B checkpoint")
    core = SharedTrackCore()
    core.load_state_dict(trained["core"], strict=True)
    runtime = load_grounding_runtime(checkpoint=Path(ordinary["spec"]["checkpoint"]), device=args.device, max_per_img=900)
    vocabulary_path = ROOT / "outputs/unified/u2/protocol/lvis_v1_vocabulary.json"
    files = {p.name: p.read_bytes() for p in DEFAULT_BERT.iterdir() if p.name in
             {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"}}
    metadata = {"mot_only_baseline_sha256": reference["checkpoint_sha256"],
                "joint_checkpoint_sha256": chosen["checkpoint_sha256"], "training_step": chosen["step"],
                "task_sampler": {"mot": .65, "ovmot": .35}, "foundation_frozen": True,
                "ordinary_manifest": {"sha256": digest(args.cache_manifest), "spec": ordinary["spec"]},
                "tao_ctao_manifest": {"sha256": digest(args.ovmot_cache), "spec": ovmot["spec"]},
                "base_supervision_sha256": digest(args.ovmot_supervision),
                "tao_gt_view_sha256": digest(args.tao_gt_view),
                "joint_selection_sha256": digest(args.joint_selection),
                "source_training_metadata": trained["metadata"], "selected_joint_measurements": chosen,
                "integration_retention": "PENDING_ACTUAL_PACKED_RUNTIME_AUDIT"}
    payload = make_payload(runtime, core, text, json.loads(vocabulary_path.read_text()), files, metadata, reference_only=False)
    save_payload(payload, args.out)
    report = {"format": "locatemot-u2-checkpoint-pack-v1", "status": "PACKED_PENDING_INTEGRATION_RETENTION",
              "path": str(args.out.resolve()), "sha256": digest(args.out), "reference_only": False,
              "format_of_checkpoint": payload["format"], "metadata": payload["metadata"], "ready_for_u3": False}
    write_json(args.out.with_suffix(".contract.json"), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
