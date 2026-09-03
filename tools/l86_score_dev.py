#!/usr/bin/env python3
"""Cheap target-bag scores for every even L86 checkpoint on internal dev.

This is an internal fit/dev selection artifact.  It does not inspect the
fixed validation slice and does not run TrackEval.  Every current L69 row is
scored in native order; target labels are attached only after the complete
frame/features have been built.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DEFAULT_CACHE = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l86_full_rmot import L86Config, L86FullRMOT  # noqa: E402
from locatemot.rmot.l86_clip_data import L86ClipStore  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def load_model(path: Path, device: torch.device) -> tuple[L86FullRMOT, dict[str, Any]]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    model = L86FullRMOT(L86Config(**package["model_config"]))
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict checkpoint load failed: {loaded}")
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"path": str(path.resolve()), "sha256": sha256_file(path),
                   "epoch": int(package.get("epoch", 0)), "step": int(package.get("step", 0)),
                   "model_config": package["model_config"], "strict_reload": True}


def score_checkpoint(model: L86FullRMOT, store: L86ClipStore, keys: list[str], device: torch.device,
                     checkpoint_info: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with torch.inference_mode():
        for key in keys:
            frame = store.build_frame(key, temporal_enabled=True)
            output = model(
                frame.z1.to(device), frame.text_global.to(device), frame.frame_global.to(device),
                frame.current_observation.to(device), frame.history_observations.to(device),
                frame.history_mask.to(device), frame.history_frame_ids.to(device), frame.frame_id,
                temporal_enabled=True)
            scores = output["candidate_energy"].float().cpu().numpy()
            r_total = output["r_total"].float().cpu().numpy()
            prior = output["candidate_prior"].float().cpu().numpy()
            presence = output["presence_logit"].float().cpu().numpy()
            null = output["null_logit"].float().cpu().numpy()
            if not all(np.isfinite(value).all() for value in (scores, r_total, prior, presence, null)):
                raise FloatingPointError(f"nonfinite dev output {key}")
            candidate_count = len(frame.row_offsets)
            for q, labels in enumerate(frame.labels):
                values = scores[q]
                if values.shape != (candidate_count,) or len(labels["labels"]) != candidate_count:
                    raise AssertionError(f"L86 dev row count drift {labels['unit_key']}")
                result.append({
                    "format": "locatemot-l86-dev-score-v1", "checkpoint": checkpoint_info,
                    "unit_key": str(labels["unit_key"]), "group_key": str(frame.group_key),
                    "dataset": str(frame.dataset), "video": str(frame.video),
                    "query_id": int(labels["query_id"]), "frame_id": int(frame.frame_id),
                    "candidate_count": int(candidate_count), "row_offsets": [int(x) for x in frame.row_offsets],
                    "row_keys": [list(x) for x in frame.row_keys],
                    "candidate_indices": [int(x) for x in frame.candidate_indices],
                    "track_ids": [int(x) for x in frame.track_ids],
                    "score": values.astype(np.float64).tolist(),
                    "r_total": r_total[q].astype(np.float64).tolist(),
                    "candidate_prior": prior.astype(np.float64).tolist(),
                    "presence_logit": float(presence[q]), "null_logit": float(null[q]),
                    "labels": [bool(x) for x in labels["labels"].tolist()],
                    "target_ids": [str(x) for x in labels["target_ids"]],
                    "candidate_gt": [None if x is None else str(x) for x in labels["candidate_gt"]],
                    "positive_count": int(labels["positive_count"]),
                    "category": str(labels["category"]), "coverage_mask": bool(labels["coverage_mask"]),
                    "candidate_rows_retained": True, "candidate_deletion": False,
                    "candidate_truncation": False, "future_history_count": int((frame.history_frame_ids > frame.frame_id).sum()),
                    "labels_attached_after_feature_construction": True, "finite_scores": True,
                })
            del output, frame
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L86 dev output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        checkpoint_paths = sorted(args.checkpoint_dir.resolve().glob("checkpoint_l86_epoch*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"no even L86 checkpoints in {args.checkpoint_dir}")
        if any(int(path.stem.split("epoch")[-1]) % 2 for path in checkpoint_paths):
            raise AssertionError("cheap dev checkpoint list contains an odd epoch")
        store = L86ClipStore(args.cache.resolve(), load_cache_into_ram=True)
        keys = [str(value) for value in store.dev_keys]
        records_path = out / "score_records.jsonl"
        checkpoint_summary: list[dict[str, Any]] = []
        with records_path.open("w") as handle:
            for path in checkpoint_paths:
                model, info = load_model(path, device)
                values = score_checkpoint(model, store, keys, device, info)
                for row in values:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                checkpoint_summary.append({"checkpoint": info, "record_count": len(values),
                                           "group_count": len(keys), "all_candidate_rows": True,
                                           "candidate_deletion": False, "candidate_truncation": False})
                del model, values
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        payload = {
            "format": "locatemot-l86-cheap-dev-scores-v1", "status": "complete", "evidence_type": "internal fit/dev",
            "command": command, "cwd": str(ROOT), "luna_thread": THREAD, "seed": 20260829,
            "cache": str(args.cache.resolve()), "cache_summary_sha256": sha256_file(args.cache / "summary.json"),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()), "checkpoint_count": len(checkpoint_summary),
            "checkpoints": checkpoint_summary, "dev_group_count": len(keys),
            "record_count": sum(item["record_count"] for item in checkpoint_summary),
            "score_records": str(records_path.resolve()), "thresholds_not_selected_here": True,
            "labels_scope": "fit/dev only; no fixed calibration/validation labels",
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "z1_representation_changed": False,
            "groundingdino_lora_used": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started, "failure_root_cause": None,
            "next_action": "apply the frozen cheap-dev checkpoint/rule selection tuple",
        }
        write_json(out / "cheap_dev_scores.json", payload); write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L86 cheap dev — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l86-cheap-dev-scores-v1", "status": "incomplete",
                                         "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md",
                                         "screening_gt_used": False, "official_test_labels_read": False,
                                         "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
