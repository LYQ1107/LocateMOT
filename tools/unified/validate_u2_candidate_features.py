#!/usr/bin/env python3
"""Validate real candidate/feature alignment without starting a feature cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from locatemot.unified.runtime.grounding_inference import (
        load_grounding_runtime, run_grounding, run_grounding_chunked,
    )
    from locatemot.unified.runtime.u2_grounding_candidates import run_u2_candidates
    from locatemot.unified.models.u2.perception_bridge import PerceptionBridge
    from tools.unified.u2_mot_portability import CHECKPOINT, FOUNDATION_SHA

    torch.set_num_threads(2)
    runtime = load_grounding_runtime(checkpoint=CHECKPOINT, device=args.device, max_per_img=900)
    assert runtime.checkpoint_sha256 == FOUNDATION_SHA
    assert not runtime.missing_keys and not runtime.unexpected_keys
    protocol_dir = ROOT / "outputs/unified/u2/protocol"
    mot = json.loads((protocol_dir / "mot17_half.json").read_text())
    sequence = mot["sequences"][0]
    image = Path(sequence["image_dir"]) / sequence["validation_frames"][0]
    maps, queries = [], []

    def save_maps(module, inputs, output):
        maps.append(tuple(level.detach().clone() for level in output))

    def save_queries(module, inputs, output):
        queries.append(inputs[0][-1, 0].detach().clone())

    def check_parity(reference, candidates, top_k, chunked):
        predictions = reference.pred_instances
        if not chunked:
            predictions = predictions[torch.argsort(predictions.scores, descending=True)[:top_k]]
        assert torch.equal(candidates["boxes"], predictions.bboxes), "native candidate boxes changed"
        assert torch.equal(candidates["scores"], predictions.scores), "native candidate scores changed"
        assert torch.equal(candidates["predicted_class"], predictions.labels), "native classes changed"
        assert candidates["id_visual"].shape == (len(predictions), 256)
        assert candidates["source_metadata"]["backbone_calls"] == 1
        assert not candidates["source_metadata"]["annotations_read"]
        expected = torch.stack([queries[chunk][query]
                                for chunk, query in zip(candidates["chunk_index"].tolist(),
                                                        candidates["query_index"].tolist())])
        assert torch.equal(candidates["decoder_query"], expected), "decoder/query provenance changed"
        assert torch.equal(candidates["semantic_visual"], expected)
        assert not {"gt_ids", "track_ids", "gt_classes", "gt_boxes"}.intersection(candidates)

    reference = run_grounding(runtime, image, mot["vocabulary"])
    neck_hook = runtime.model.neck.register_forward_hook(save_maps)
    head_hook = runtime.model.bbox_head.register_forward_hook(save_queries)
    try:
        ordinary = run_u2_candidates(runtime, image, mot["vocabulary"], max_candidates=150)
        check_parity(reference, ordinary, 150, False)
        person_maps = maps[0]
        maps.clear()
        queries.clear()
        alternate = run_u2_candidates(runtime, image, "car . bus . truck .", max_candidates=150)
        assert all(torch.equal(left, right) for left, right in zip(person_maps, maps[0])), \
            "pre-text visual maps depend on vocabulary"
        meta = alternate["source_metadata"]
        pad_h, pad_w = meta["batch_input_shape"]
        fixed_boxes = ordinary["boxes"] * ordinary["boxes"].new_tensor(meta["scale_factor"]).repeat(2)
        fixed_boxes = fixed_boxes / fixed_boxes.new_tensor([pad_w, pad_h, pad_w, pad_h])
        fixed_identity = PerceptionBridge.sample_id_visual(maps[0], fixed_boxes)
        assert torch.equal(ordinary["id_visual"], fixed_identity), "fixed-box identity depends on vocabulary"
        print(json.dumps({"ordinary_candidate_parity": "PASS", "fixed_box_text_invariance": "PASS"}), flush=True)

        # The full 1,203-category path exercises provenance after all 38 chunks
        # and the actual cross-chunk NMS, rather than a reduced-vocabulary proxy.
        tao = json.loads((protocol_dir / "tao_portability.json").read_text())
        annotation = json.loads(Path(tao["annotation_path"]).read_text())
        tao_image = Path(tao["image_root"]) / annotation["images"][0]["file_name"]
        del annotation
        vocabulary = json.loads((protocol_dir / "lvis_v1_vocabulary.json").read_text())
        entities = [category["name"] for category in vocabulary["categories"]]
        reference = run_grounding_chunked(runtime, tao_image, entities, chunked_size=32,
                                          merge_iou=.8, top_k=300)
        maps.clear()
        queries.clear()
        chunked = run_u2_candidates(runtime, tao_image, entities, max_candidates=300)
        check_parity(reference, chunked, 300, True)
        assert len(maps) == 1 and len(queries) == 38
        assert chunked["source_metadata"]["chunk_count"] == 38
    finally:
        neck_hook.remove()
        head_hook.remove()
    assert "_predict_by_feat_single" not in runtime.model.bbox_head.__dict__
    report = {"format": "locatemot-u2-real-candidate-feature-contract-v1", "status": "PASS",
              "foundation_sha256": runtime.checkpoint_sha256, "device": args.device,
              "ordinary": {"image": str(image), "candidates": len(ordinary["boxes"]),
                           "native_box_score_class_parity": "BIT_EXACT", "decoder_alignment": "BIT_EXACT"},
              "ovmot": {"image": str(tao_image), "vocabulary_size": len(entities), "chunk_count": 38,
                        "candidates": len(chunked["boxes"]), "backbone_calls": 1,
                        "native_box_score_class_nms_parity": "BIT_EXACT", "decoder_alignment": "BIT_EXACT"},
              "fixed_box_identity_text_invariance": "BIT_EXACT",
              "identity_source": ordinary["source_metadata"]["identity_source"],
              "coordinate_contract": ordinary["source_metadata"]["identity_coordinate_system"],
              "feature_payload_has_gt_labels": False, "full_cache_started": False,
              "track_training_launched": False, "official_test_labels_read": False}
    destination = ROOT / "outputs/unified/u2/eval/real_candidate_feature_contract.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
