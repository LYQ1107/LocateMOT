#!/usr/bin/env python3
"""Cache full textual LVIS vocabulary from the shared real foundation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.unified.build_u2_candidate_cache import digest, write_json
from tools.unified.select_u2_perception_adaptation import verify_selected_foundation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-foundation-sha256", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.reference_only:
        if args.selection is None:
            parser.error("Formal class text cache requires complete foundation selection")
        verify_selected_foundation(args.selection, args.expected_foundation_sha256)
    if digest(args.checkpoint) != args.expected_foundation_sha256:
        raise ValueError("Class text checkpoint differs from its registered foundation")
    import torch
    from locatemot.unified.runtime.grounding_inference import load_grounding_runtime
    from locatemot.unified.runtime.u2_class_text import encode_class_text
    runtime = load_grounding_runtime(checkpoint=args.checkpoint, device=args.device, max_per_img=900)
    if runtime.missing_keys or runtime.unexpected_keys:
        raise ValueError("Native foundation class-text weights do not match the checkpoint")
    vocabulary_path = ROOT / "outputs/unified/u2/protocol/lvis_v1_vocabulary.json"
    categories = json.loads(vocabulary_path.read_text())["categories"]
    if [category["id"] for category in categories] != list(range(1, 1204)):
        raise ValueError("Class embedding order must retain the full frozen 1203-way textual vocabulary")
    backbone_calls = 0

    def count_backbone(module, args, output):
        nonlocal backbone_calls
        backbone_calls += 1

    hook = runtime.model.backbone.register_forward_hook(count_backbone)
    embeddings, metadata = encode_class_text(runtime, categories)
    hook.remove()
    if backbone_calls:
        raise ValueError("Text feature extraction unexpectedly used image features")
    payload = {"format": "locatemot-u2-shared-class-text-cache-v1", "embeddings": embeddings,
               "category_ids": [category["id"] for category in categories],
               "names": [category["name"] for category in categories],
               "frequencies": [category["frequency"] for category in categories],
               "foundation_sha256": args.expected_foundation_sha256,
               "vocabulary_sha256": digest(vocabulary_path), "reference_only": args.reference_only,
               "metadata": metadata}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(args.out)
    report = {"format": "locatemot-u2-shared-class-text-contract-v1", "status": "PASS",
              "foundation_sha256": args.expected_foundation_sha256,
              "class_text_cache": str(args.out.resolve()), "cache_sha256": digest(args.out),
              "vocabulary_sha256": payload["vocabulary_sha256"], "classes": 1203,
              "base_classes": sum(category["frequency"] in {"f", "c"} for category in categories),
              "novel_classes": sum(category["frequency"] == "r" for category in categories),
              "embedding_shape": list(embeddings.shape), "max_chunk_tokens": max(metadata["chunk_token_counts"]),
              "chunk_count": metadata["chunk_count"], "image_backbone_calls": backbone_calls,
              "reference_only": args.reference_only, "second_text_encoder": False,
              "novel_training_labels_used": False, "formal_joint_optimization_launched": False}
    write_json(args.out.with_suffix(".contract.json"), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
