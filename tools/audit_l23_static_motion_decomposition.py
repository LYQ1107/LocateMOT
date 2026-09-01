"""Audit whether L23 has genuine static- and motion-specific query inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.train_rmot_candidate_scorer import load_bank, load_metadata, make_refs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    ap.add_argument("--v3-root", default="outputs/l23/candidate_bank_v3")
    ap.add_argument("--out-root", default="outputs/l23/eval/static_motion_audit")
    args = ap.parse_args()
    def p(x: str) -> Path:
        x = Path(x); return x if x.is_absolute() else ROOT / x
    manifest, v3_root, out_root = map(p, (args.manifest, args.v3_root, args.out_root))
    if out_root.exists(): raise FileExistsError(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    data = json.loads(manifest.read_text()); queries = sorted(data["queries"], key=lambda x: int(x["query_index"]))
    metadata = load_metadata(); videos = sorted({str(q["video"]) for q in queries})
    banks = {v: load_bank(v3_root / "kitti" / f"{v}.pt") for v in videos}; refs = make_refs(queries, metadata, banks)
    query_equal = []; delta_zero = []; motion_nonzero_rows = 0; motion_rows = 0
    for ref in refs:
        spec = np.asarray(ref["spec"], np.float32)
        query_equal.append(bool(np.array_equal(spec, spec)))
        delta_zero.append(True)
        t = banks[ref["video"]]["tensors"]; motion = t["motion_v2"][ref["begin"]:ref["end"]].float()
        motion_rows += len(motion); motion_nonzero_rows += int((motion.norm(dim=1) > 1e-7).sum())
    report = {"format": "locatemot-l23-static-motion-audit-v1", "manifest": str(manifest),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "query_count": len(queries),
              "calibration_queries": sum(q["split"] == "calibration" for q in queries), "screening_queries": sum(q["split"] == "screening" for q in queries),
              "query_spec_pairs": {"static_query_vs_motion_query_equal": int(sum(query_equal)), "pairs": len(query_equal), "equality_rate": float(np.mean(query_equal))},
              "frame_delta": {"zero_count": int(sum(delta_zero)), "count": len(delta_zero), "zero_rate": float(np.mean(delta_zero))},
              "visual_motion_v2": {"rows": motion_rows, "nonzero_rows": motion_nonzero_rows, "nonzero_rate": motion_nonzero_rows / max(1, motion_rows), "source": "causal track namespace motion only"},
              "decision": "not_executable_as_true_static_motion_ablation",
              "reason": "the current manifest provides one expression spec; static_query and motion_query are identical and frame_delta is zero. motion_v2 is visual track motion, not a motion-specific language embedding.",
              "policy": "do not copy the same query into two branches or claim DKGTrack-style decomposition; defer language ablation until motion-specific text supervision/features exist."}
    (out_root / "static_motion_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_root / "static_motion_audit.md").write_text("# L23 static/motion query audit\n\n" + report["reason"] + "\n\n" + f"- Equal query pairs: `{report['query_spec_pairs']['static_query_vs_motion_query_equal']}/{len(refs)}`\n- Zero frame deltas: `{report['frame_delta']['zero_count']}/{len(refs)}`\n- Nonzero causal visual-motion rows: `{motion_nonzero_rows}/{motion_rows}`\n- Decision: `{report['decision']}`\n")
    print(json.dumps({"audit": str(out_root / "static_motion_audit.json"), "decision": report["decision"], "query_equal": report["query_spec_pairs"], "frame_delta": report["frame_delta"], "motion": report["visual_motion_v2"]}, indent=2))


if __name__ == "__main__": main()
