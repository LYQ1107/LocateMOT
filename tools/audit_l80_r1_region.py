#!/usr/bin/env python3
"""Label-free contract audit for the L80-R1 high-resolution region interface."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))
from locatemot.models.l80_r1_region import L80R1Config, L80R1RegionCorrespondence  # noqa: E402
from locatemot.rmot.l80_data import EXPECTED_MANIFEST_SHA, MANIFEST, L80BankStore, key_only, load_fit_units, sha256_file  # noqa: E402
from locatemot.rmot.l80_r1_runtime import CLIP_SHA256, CLIP_WEIGHT, FrameFeatureCache, load_clip, raw_inputs_for_unit_r1  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 audit output {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA or sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
            raise AssertionError("immutable manifest/CLIP SHA mismatch")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        # The source rows are immediately reduced to key/text/frame metadata;
        # no sidecar or target field is used by this label-free audit.
        source = load_fit_units()
        selected = []
        seen = set()
        per_dataset = {"refer_kitti_v1": 0, "refer_kitti_v2": 0}
        for row in source:
            pair = (str(row["dataset"]), str(row["video"]))
            if pair in seen or per_dataset.get(pair[0], 0) >= 4:
                continue
            selected.append(key_only(row))
            seen.add(pair)
            per_dataset[pair[0]] = per_dataset.get(pair[0], 0) + 1
            if len(selected) == 8 and all(value == 4 for value in per_dataset.values()):
                break
        if len(selected) != 8 or any(value != 4 for value in per_dataset.values()):
            raise AssertionError(f"R1 audit sampling lost V1/V2 coverage: {per_dataset}")
        model = L80R1RegionCorrespondence(L80R1Config()).to(device=device, dtype=torch.float32).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        clip_model = load_clip(device)
        cache = FrameFeatureCache(max_items=16)
        store = L80BankStore(max_history=8)
        records = []
        for meta in selected:
            batch = store.build_unit(meta)
            raw = raw_inputs_for_unit_r1(clip_model, batch, device, cache)
            with torch.inference_mode():
                result = model(
                    raw["visual_tokens"], raw["text_tokens"], raw["text_mask"],
                    batch.history_observations.to(device).clone(), batch.history_mask.to(device).clone(),
                    batch.history_frame_ids.to(device).clone(), int(batch.frame_id),
                )
            scores = result["candidate_logits"].float().cpu()
            records.append({
                "unit_key": batch.unit_key, "dataset": batch.dataset, "video": batch.video,
                "frame_id": int(batch.frame_id), "candidate_count": int(batch.candidate_count),
                "row_keys": [list(x) for x in batch.row_keys],
                "candidate_indices": [int(x) for x in batch.candidate_indices],
                "pool_ids": [int(x) for x in batch.pool_ids],
                "region_shape": list(raw["visual_tokens"].shape),
                "pyramid_shape": raw["pyramid_shape"], "roi_audit": raw["roi_audit"],
                "score_shape": list(scores.shape), "score_finite": bool(torch.isfinite(scores).all()),
                "score_std": float(scores.std()), "future_history_rows": int((batch.history_frame_ids > batch.frame_id).sum()),
                "candidate_deletion": False, "candidate_truncation": False,
                "sidecar_labels_loaded": False, "raw_cache_persistent": False,
            })
            del result, raw, batch
        before = copy.deepcopy(model.state_dict())
        reload_model = L80R1RegionCorrespondence(L80R1Config()).to(device=device, dtype=torch.float32)
        load = reload_model.load_state_dict(before, strict=True)
        if load.missing_keys or load.unexpected_keys:
            raise AssertionError(f"R1 strict reload mismatch {load}")
        reload_model.eval()
        strict_same = all(torch.equal(before[name], reload_model.state_dict()[name].cpu() if before[name].device.type == "cpu" else reload_model.state_dict()[name]) for name in before)
        payload = {
            "format": "locatemot-l80-r1-region-contract-v1", "status": "complete",
            "stage": "R1-region-interface-only", "candidate_rows": int(sum(x["candidate_count"] for x in records)),
            "records": records, "region_contract": {"roi_grid": 8, "context_grid": 4, "tokens_per_scale": 81, "region_tokens": 243},
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "finite": all(x["score_finite"] for x in records), "history_future_rows": sum(x["future_history_rows"] for x in records),
            "strict_reload": bool(strict_same), "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "frozen_detector": True, "no_grad_detector": True, "sidecar_labels_loaded": False,
            "manifest_sha256": sha256_file(MANIFEST), "clip_sha256": sha256_file(CLIP_WEIGHT),
            "elapsed_sec": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "next_action": "run R1 bounded fit probe if contract passes",
        }
        write_json(out / "contract.json", payload)
        write_json(out / "provenance.json", {"format": "locatemot-l80-r1-region-provenance-v1", "status": "complete", "command": " ".join([sys.executable] + sys.argv), "project_root": str(ROOT), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "inputs": {"manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST), "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT), "l69_features": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"), "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl")}, "region_only_change": "ROI 8x8/context 4x4, same frozen CLIP taps and same L80 head/loss/sampler", "labels_used": False, "raw_dense_cache_written": False, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        write_json(out / "status.json", {"format": "locatemot-l80-r1-status-v1", "status": "complete", "command": " ".join(sys.argv), "failure_root_cause": None, "next_action": "run R1 bounded fit probe", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        print(json.dumps({"status": "complete", "output": str(out), "records": len(records)}, indent=2))
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text("# L80-R1 contract — INCOMPLETE\n\n" + __import__("traceback").format_exc() + "\n")
        raise
    finally:
        gc.collect()
        if "device" in locals() and device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
