#!/usr/bin/env python3
"""Reference-only actual-F0 integration of the selected-core audit entry."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch
    from locatemot.unified.runtime.u2_checkpoint import load_u2_checkpoint
    from locatemot.unified.runtime.u2_grounding_candidates import run_u2_candidates
    from tools.unified.audit_u2_selected_core import core_contract
    from tools.unified.build_u2_candidate_cache import write_json
    torch.set_num_threads(2)
    checkpoint = ROOT / "outputs/unified/u2/checkpoint_contract/reference_native_pack.pth"
    packed = load_u2_checkpoint(checkpoint, device="cuda:0")
    assert packed.reference_only
    original = {k: v.detach().clone() for k,v in packed.core.state_dict().items()}
    mot = json.loads((ROOT / "outputs/unified/u2/protocol/mot17_half.json").read_text())
    group = mot["sequences"][0]
    clip = [{"candidates": run_u2_candidates(packed.runtime, Path(group["image_dir"]) / frame, mot["vocabulary"], max_candidates=150),
             "frame_index": int(Path(frame).stem), "frame": frame} for frame in group["validation_frames"][:4]]
    with tempfile.TemporaryDirectory(prefix="locatemot-u2-selected-audit-contract-") as directory:
        results = core_contract(packed.core, clip, packed.class_text, Path(directory))
    assert all(torch.equal(v,packed.core.state_dict()[k]) for k,v in original.items())
    report = {"format": "locatemot-u2-selected-core-audit-entry-contract-v1", "status": "PASS", "reference_only": True,
              "core_source": "existing explicitly random reference pack; no weights altered during this audit",
              "foundation_sha256": packed.metadata["foundation_sha256"], "results": results,
              "frames": [item["frame"] for item in clip], "actual_selected_checkpoint_audited": False,
              "ready_for_u3": False, "official_test_labels_read": False}
    write_json(ROOT / "outputs/unified/u2/eval/selected_core_audit_entry_contract.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
