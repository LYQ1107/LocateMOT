#!/usr/bin/env python3
"""Actual F0 one-file reconstruction and small U1 integration retention proof."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import BertModel
    from mmengine.config import Config
    from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
    from locatemot.unified.models.u2.track_state import UnifiedTrackState
    from locatemot.unified.runtime.grounding_inference import DEFAULT_BERT, load_grounding_runtime, run_grounding, prediction_arrays
    from locatemot.unified.runtime.u2_grounding_candidates import run_u2_candidates
    from locatemot.unified.runtime.u2_class_text import encode_class_text
    from locatemot.unified.runtime.u2_checkpoint import make_payload, save_payload, load_u2_checkpoint, parameter_sharing
    from tools.unified.u2_mot_portability import CHECKPOINT, FOUNDATION_SHA
    from tools.unified.u2_causal_contract import state_equal
    from tools.unified.u1_formal_expression_gate import query_records, read_gt, iou, average_precision
    from tools.unified.u1_formal_generic_gate import read_gt as read_generic_gt
    from tools.unified.build_u2_candidate_cache import digest, write_json

    torch.set_num_threads(2)
    torch.manual_seed(20261012)
    runtime = load_grounding_runtime(checkpoint=CHECKPOINT, device=args.device, max_per_img=900)
    assert runtime.checkpoint_sha256 == FOUNDATION_SHA
    core = SharedTrackCore().to(args.device).eval()
    torch.nn.init.constant_(core.association.pair[-1].bias, 3.)
    torch.nn.init.constant_(core.association.new[-1].bias, 3.)
    torch.nn.init.constant_(core.association.no_match[-1].bias, -3.)
    # Save nonzero task conditioning too, so omission cannot pass this proof.
    with torch.no_grad():
        core.ovmot.task_embeddings.weight[1].fill_(.1)
    text = torch.load(ROOT / "outputs/unified/u2/ovmot_semantics/f0_class_text.pt", map_location="cpu", weights_only=False)
    vocabulary = json.loads((ROOT / "outputs/unified/u2/protocol/lvis_v1_vocabulary.json").read_text())
    assets = {p.name: p.read_bytes() for p in DEFAULT_BERT.iterdir() if p.name in
              {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"}}
    metadata = {"training_step": 0, "foundation_frozen": True,
                "core_source": "explicitly random contract core; deterministic birth/match biases and task-embedding probe",
                "integration_retention": "REFERENCE_ONLY", "mot_only_baseline_sha256": None}
    payload = make_payload(runtime, core, text, vocabulary, assets, metadata, reference_only=True)
    checkpoint = ROOT / "outputs/unified/u2/checkpoint_contract/reference_native_pack.pth"
    save_payload(payload, checkpoint)
    del payload
    print(json.dumps({"stage": "SAVED_ACTUAL_FOUNDATION_REFERENCE_PACK", "bytes": checkpoint.stat().st_size}), flush=True)
    real_load = torch.load
    opened = []

    def only_pack(path, *args, **kwargs):
        assert Path(path).resolve() == checkpoint.resolve(), "Loader read an external tensor file"
        opened.append(str(path))
        return real_load(path, *args, **kwargs)

    with patch.object(BertModel, "from_pretrained", side_effect=AssertionError("External BERT weights requested")), \
            patch.object(Config, "fromfile", side_effect=AssertionError("External model config requested")), \
            patch.object(torch, "load", side_effect=only_pack):
        packed = load_u2_checkpoint(checkpoint, device=args.device)
    assert len(opened) == 1
    assert all(torch.equal(value, packed.foundation.state_dict()[name]) for name, value in runtime.model.state_dict().items())
    assert all(torch.equal(value, packed.core.state_dict()[name]) for name, value in core.state_dict().items())
    embeddings, _ = encode_class_text(packed.runtime, vocabulary["categories"])
    assert torch.equal(embeddings, text["embeddings"]), "Packed shared BERT class features changed"
    assert parameter_sharing(packed.foundation, packed.core) == packed.metadata["parameter_sharing"]
    print(json.dumps({"stage": "STRICT_NATIVE_RECONSTRUCTION_PASS", "class_text": "BIT_EXACT"}), flush=True)

    protocol = ROOT / "outputs/unified/u2/protocol"
    mot = json.loads((protocol / "mot17_half.json").read_text())["sequences"][0]
    mot_image = Path(mot["image_dir"]) / mot["validation_frames"][0]
    tao = json.loads((protocol / "tao_portability.json").read_text())
    annotation = json.loads(Path(tao["annotation_path"]).read_text())
    tao_image = Path(tao["image_root"]) / annotation["images"][0]["file_name"]
    del annotation
    tasks = []
    with torch.inference_mode():
        for task, image, names, cap in [("mot", mot_image, "person .", 150),
                                      ("ovmot", tao_image, [c["name"] for c in vocabulary["categories"]], 300)]:
            candidates = run_u2_candidates(runtime, image, names, max_candidates=cap)
            result = packed(image, task_type=task, frame_index=0, vocabulary="person .")
            for field in ("boxes", "scores", "id_visual", "semantic_visual", "decoder_query", "predicted_class", "query_index", "chunk_index"):
                assert torch.equal(candidates[field], result["candidates"][field]), (task, field)
            native = core(candidates, task_type=task, frame_index=0, class_text=text["embeddings"].to(args.device) if task == "ovmot" else None)
            assert state_equal(native["state"], result["online_state"])
            state_file = checkpoint.parent / (task + "_nonempty_online_state.pt")
            torch.save(result["online_state"].as_dict(), state_file)
            resumed_state = UnifiedTrackState(**torch.load(state_file, map_location=args.device, weights_only=False))
            assert resumed_state.num_tracks > 0
            native_next = core(candidates, native["state"], task_type=task, frame_index=3,
                               class_text=text["embeddings"].to(args.device) if task == "ovmot" else None)
            packed_next = packed.core(result["candidates"], resumed_state, task_type=task, frame_index=3,
                                      class_text=packed.class_text if task == "ovmot" else None)
            assert state_equal(native_next["state"], packed_next["state"])
            for field in ("track_ids", "identity_pair_logits", "pair_logits", "new_logits", "no_match_logits", "semantic_gate"):
                assert torch.equal(native_next[field], packed_next[field]), (task, field)
            tasks.append({"task": task, "image": str(image), "candidate_count": len(candidates["boxes"]),
                          "candidate_features": "BIT_EXACT", "nonempty_state_resume": "BIT_EXACT"})
            print(json.dumps({"stage": "REAL_IMAGE_TASK_PASS", "task": task}), flush=True)

    from tools.unified.audit_u2_checkpoint_retention import small_u1_retention
    retention = small_u1_retention(runtime, packed.runtime)
    report = {"format": "locatemot-u2-native-checkpoint-contract-v1", "status": "PASS",
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
              "foundation_sha256": FOUNDATION_SHA, "reference_only": True,
              "core_is_random_contract_fixture": True, "actual_foundation_all_tensors": "BIT_EXACT",
              "actual_core_all_tensors_including_task_embeddings": "BIT_EXACT",
              "external_checkpoint_loads": 0, "hf_pretrained_weight_loads": 0, "external_config_loads": 0,
              "full_1203_shared_bert_class_features": "BIT_EXACT", "supported_tasks": tasks,
              "parameter_sharing": packed.metadata["parameter_sharing"],
              "small_u1_integration_scope": retention,
              "original_full_u1_references_modified": False,
              "selected_final_checkpoint_exists": False, "formal_track_training_launched": False,
              "ready_for_u3": False, "official_test_labels_read": False}
    write_json(ROOT / "outputs/unified/u2/eval/native_checkpoint_contract.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
