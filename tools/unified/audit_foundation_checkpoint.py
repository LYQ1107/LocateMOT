#!/usr/bin/env python3
"""Audit exact MM-GroundingDINO-B checkpoint compatibility."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.runtime.grounding_inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_grounding_runtime,
    runtime_metadata,
)

# The released MMDetection checkpoint has no denoising label embedding because
# it is an inference checkpoint.  The three BERT pooler/position tensors are
# metadata/unused modules in the local no-pooler BERT contract.  No backbone,
# encoder, decoder, text-fusion or bbox-regression parameter is permitted here.
EXPECTED_MISSING = frozenset({"dn_query_generator.label_embedding.weight"})
EXPECTED_UNEXPECTED = frozenset({
    "language_model.language_backbone.body.model.embeddings.position_ids",
    "language_model.language_backbone.body.model.pooler.dense.bias",
    "language_model.language_backbone.body.model.pooler.dense.weight",
})


def main() -> int:
    runtime = load_grounding_runtime(
        config_path=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        training_contract=True,
    )
    missing = frozenset(runtime.missing_keys)
    unexpected = frozenset(runtime.unexpected_keys)
    if missing != EXPECTED_MISSING:
        raise RuntimeError(f"checkpoint missing whitelist mismatch: {sorted(missing)}")
    if unexpected != EXPECTED_UNEXPECTED:
        raise RuntimeError(f"checkpoint unexpected whitelist mismatch: {sorted(unexpected)}")
    forbidden_prefixes = ("backbone.", "encoder.", "decoder.", "bbox_head.reg_branches.", "text_feat_map.")
    forbidden = sorted(key for key in missing if key.startswith(forbidden_prefixes))
    if forbidden:
        raise RuntimeError(f"core model keys were not initialized: {forbidden}")
    payload = {
        "format": "locatemot-u-foundation-checkpoint-audit-v2",
        "status": "PASS_CONDITIONAL_TRAINING_ONLY_MISSING",
        **runtime_metadata(runtime),
        "expected_missing": sorted(EXPECTED_MISSING),
        "expected_unexpected": sorted(EXPECTED_UNEXPECTED),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "core_missing_keys": forbidden,
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
    }
    out = PROJECT_ROOT / "outputs/unified/protocol/foundation_checkpoint_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
