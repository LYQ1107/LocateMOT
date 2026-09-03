#!/usr/bin/env python3
"""Assemble the immutable L86 evidence into one compact final status.

This helper only reads completed L86 JSON outputs and never reads screening or
official-test labels.  It is intentionally separate from training and
evaluation so the final status cannot change a checkpoint or an emission rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
BASELINE = {
    "refer_kitti_v1": {"HOTA": 25.0548, "DetA": 14.9853, "AssA": 42.4542,
                        "LocA": 89.8828, "DetRe": 47.4128, "DetPr": 17.8400,
                        "IDF1": 21.3342, "IDSW": 2330},
    "refer_kitti_v2": {"HOTA": 17.2924, "DetA": 9.7879, "AssA": 30.8389,
                        "LocA": 88.2524, "DetRe": 45.9492, "DetPr": 10.9991,
                        "IDF1": 12.9430, "IDSW": 18374},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def trackeval_metrics(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    result: dict[str, dict[str, float]] = {}
    for item in payload["datasets"]:
        dataset = str(item["dataset"])
        percent = item["metrics_percent"]
        counts = item["metrics_counts"]
        names = {"HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC",
                 "LocA": "LocA___AUC", "DetRe": "DetRe___AUC", "DetPr": "DetPr___AUC"}
        result[dataset] = {**{key: float(percent[value]) for key, value in names.items()},
                           "IDF1": float(percent["IDF1"]), "IDSW": float(counts["IDSW"]),
                           "CLR_FP": float(counts["CLR_FP"]), "CLR_FN": float(counts["CLR_FN"]),
                           "sequence_count": int(item["sequence_count"])}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", type=Path, default=ROOT / "outputs/l86/eval/fixed_semantic_attempt2/semantic.json")
    parser.add_argument("--trackeval", type=Path, default=ROOT / "outputs/l86/trackeval/fullvideo_eval_attempt1/trackeval_summary.json")
    parser.add_argument("--train", type=Path, default=ROOT / "outputs/l86/train/joint40/provenance.json")
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/l86/eval/dev_selection_attempt1/checkpoint_selection.json")
    parser.add_argument("--oracle", type=Path, default=ROOT / "outputs/l86/trackeval/semantic_oracle_attempt2/summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l86")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("fixed manifest SHA drift")
    semantic = json.loads(args.semantic.resolve().read_text())
    train = json.loads(args.train.resolve().read_text())
    selection = json.loads(args.selection.resolve().read_text())["selected"]
    trackeval = trackeval_metrics(args.trackeval.resolve())
    material = all(trackeval[dataset]["HOTA"] >= BASELINE[dataset]["HOTA"] + 3.0 for dataset in BASELINE)
    semantic_pass = semantic.get("decision") == "semantic_gate_pass_pending_supervisor"
    status = "full_rmot_hota_complete_semantic_gate_pass" if semantic_pass else (
        "full_rmot_hota_complete_improved" if material else "full_rmot_hota_complete_no_material_gain")
    payload = {
        "format": "locatemot-l86-final-status-v1", "status": status,
        "evidence_type": "completed L86 internal full-RMOT validation; semantic gate and TrackEval are separate",
        "cwd": str(ROOT), "luna_thread": THREAD, "manifest_sha256": MANIFEST_SHA,
        "training": {"epochs": 40, "world_size": int(train["world_size"]),
                      "effective_clip_batch": int(train["effective_clip_batch"]),
                      "final_checkpoint": train["final_checkpoint"],
                      "selected_checkpoint": selection["checkpoint_info"],
                      "selected_rule": selection["rule_fit"], "seed": int(train["seed"])},
        "fixed_semantic": {
            "source": str(args.semantic.resolve()), "decision": semantic["decision"],
            "validation": semantic["validation"]["final_frozen_rule"],
            "gate": semantic["gate"]["checks"],
        },
        "trackeval": {"source": str(args.trackeval.resolve()), "scope": "internal_full_video_validation",
                       "metrics": trackeval, "baseline_l85": BASELINE,
                       "delta_vs_l85": {dataset: {key: trackeval[dataset][key] - value
                                                   for key, value in BASELINE[dataset].items()}
                                       for dataset in BASELINE},
                       "material_fullvideo_improvement": bool(material)},
        "semantic_oracle": {"source": str(args.oracle.resolve()), "oracle_only": True},
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
        "no_hota_or_trackeval": False, "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED", "z1_representation_changed": False,
        "groundingdino_lora_used": False, "next_action": "stop and await supervisor review",
    }
    write_json(args.out.resolve() / "final_status.json", payload)
    write_json(args.out.resolve() / "final_summary.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
