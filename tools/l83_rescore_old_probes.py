#!/usr/bin/env python3
"""Read-only corrected target-bag rescore of the L81/L59/L82 probe packages."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
SPLIT = ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
CHECKPOINTS = {
    "l81_candidate_evidence": ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l81_candidate_evidence_checkpoint_epoch10.pt",
    "l59_fused_roi": ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l59_fused_roi_checkpoint_epoch10.pt",
    "l82_candidate_reference": ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l82_candidate_reference_checkpoint_epoch10.pt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def meta(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def load_probe(path: Path, device: torch.device) -> torch.nn.Module:
    from locatemot.models.l82_rank_probe import L82FactorizedRankProbe, L82RankProbeConfig
    package = torch.load(path, map_location="cpu", weights_only=False)
    config = L82RankProbeConfig(**package.get("model_config", {}))
    model = L82FactorizedRankProbe(config).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict load failed for {path}: {result}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L83 baseline output: {out}")
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != EXPECTED_MANIFEST:
        raise AssertionError("fixed manifest SHA drift")
    if not torch.cuda.is_available():
        raise RuntimeError("L83 baseline rescore requires CUDA for the audited GroundingDINO runtime")
    torch.cuda.set_device(0)
    from tools.l82_train_frozen_rank_probe import build_local_groups, load_groups
    from locatemot.evaluation.l83_target_bag_metrics import aggregate_group_metrics, breakdowns, group_metrics
    groups, train_keys, dev_keys = load_groups()
    started = time.perf_counter()
    data, build_info = build_local_groups(dev_keys, groups, torch.device("cuda:0"))
    if {item.group_key for item in data} != set(dev_keys):
        raise AssertionError("dev group key drift")
    by_rep: dict[str, list[dict[str, Any]]] = {key: [] for key in CHECKPOINTS}
    aux_by_rep: dict[str, list[dict[str, Any]]] = {key: [] for key in CHECKPOINTS}
    for representation, checkpoint in CHECKPOINTS.items():
        model = load_probe(checkpoint, torch.device("cuda:0"))
        for item in data:
            values = item.features[representation].to(device="cuda:0", dtype=torch.float32).clone()
            with torch.inference_mode():
                output = model(values)
            record, auxiliary = group_metrics(item, output["interaction"].cpu())
            by_rep[representation].append(record)
            aux_by_rep[representation].append(auxiliary)
            del values, output
        del model
        gc.collect()
        torch.cuda.empty_cache()
    metrics = {}
    compact_records = []
    for representation, records in by_rep.items():
        metrics[representation] = {
            "aggregate": aggregate_group_metrics(records, aux_by_rep[representation]),
            "breakdowns": breakdowns(records, aux_by_rep[representation]),
            "group_records": len(records), "all_dev_groups_present": len(records) == len(dev_keys),
            "checkpoint": meta(CHECKPOINTS[representation]),
        }
        compact_records.extend({key: value for key, value in record.items() if not key.startswith("_")} | {"representation": representation} for record in records)
    payload = {
        "format": "locatemot-l83-corrected-old-probe-metrics-v1", "status": "complete",
        "stage": "phase_5_corrected_old_probe_rescore", "command": " ".join([sys.executable] + sys.argv),
        "cwd": str(ROOT), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "train_group_count": len(train_keys), "dev_group_count": len(dev_keys),
        "metrics": metrics, "feature_build": build_info,
        "metric_contract": {"target_bag_score": "max per unique candidate_gt target", "background": "singleton rows", "query_swap_auc": "independent rank-based ROC-AUC", "row_metrics": "ROW_DIAGNOSTIC"},
        "inputs": {"manifest": meta(MANIFEST), "split": meta(SPLIT), "checkpoints": {key: meta(value) for key, value in CHECKPOINTS.items()}},
        "elapsed_seconds": time.perf_counter() - started,
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "training_run": False, "hota_trackeval_run": False, "candidate_deletion": False, "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
    }
    write_json(out / "corrected_old_probe_metrics.json", payload)
    (out / "dev_group_metrics.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True, default=str) for row in compact_records) + "\n")
    write_json(out / "provenance.json", {"format": "locatemot-l83-corrected-baseline-provenance-v1", "status": "complete", "command": " ".join([sys.executable] + sys.argv), "inputs": payload["inputs"], "labels": "fit-only labels attached after feature construction", "selection": "none; no new model or metric choice", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False})
    write_json(out / "status.json", {"format": "locatemot-l83-corrected-baseline-status-v1", "status": "complete", "failure_root_cause": None, "next_action": "run the preregistered faithful target-bag probe", "command": " ".join([sys.executable] + sys.argv)})
    print(json.dumps({"status": "complete", "dev_groups": len(data), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
