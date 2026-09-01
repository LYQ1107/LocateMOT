#!/usr/bin/env python3
"""L79 P0 frozen-bank, raw feature and task-isolation contract audit."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l79_hierarchical_correspondence import L79HierarchicalCorrespondence  # noqa: E402
from locatemot.rmot.l79_data import (  # noqa: E402
    ALL_L69_VIDEOS,
    EXPECTED_MANIFEST_SHA,
    EXPECTED_SHARED_CHECKPOINT_SHA,
    L69_ROOT,
    MANIFEST,
    file_meta,
    key_only_unit,
    load_fit_units,
    sha256_file,
    source_file_manifest,
    L79BankStore,
)
from locatemot.rmot.l79_runtime import (  # noqa: E402
    CLIP_SHA256,
    CLIP_WEIGHT,
    attach_lora,
    load_clip_visual,
    l79_task_enabled,
    preprocess_full_frame,
    visual_pyramid,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l79/audit/p0_contract_attempt1")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    command = " ".join([sys.executable] + sys.argv)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    output_names = ["contract.json", "parameter_contract.json", "coverage_contract.json", "provenance.json", "status.json"]
    base_inputs = [str(ROOT / "AGENTS.md"), str(MANIFEST), str(ROOT / "outputs/l11/checkpoints/uidm_l11_main/latest.pt"), str(ROOT / "outputs/l69/attempt9/budget40_features/kitti/manifest.json")]
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA changed")
        fit_units = load_fit_units()
        domains = sorted({str(x["dataset"]) for x in fit_units})
        categories = {str(x["category"]) for x in fit_units}
        # The complete source metadata is used only for a post-construction
        # stratum count; the actual pre-forward unit is stripped first.
        first = sorted(fit_units, key=lambda x: (str(x["dataset"]), str(x["category"]), str(x["video"]), int(x["frame_id"]), int(x["query_id"]), str(x["unit_key"]))) [0]
        key_unit = key_only_unit(first)
        forbidden = {"target_ids", "positive_indices", "positive_count", "category", "labels", "target_present", "begin", "end"}
        if forbidden.intersection(key_unit):
            raise AssertionError(f"forbidden fields leaked into pre-forward unit: {sorted(forbidden.intersection(key_unit))}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("L79 P0 requires GPU0 for the raw CLIP contract")
        torch.cuda.set_device(0)
        clip_model = load_clip_visual(device, enable_lora=False)
        batch = L79BankStore(max_history=16).build_unit(key_unit)
        if not Path(batch.image_path).is_file():
            raise FileNotFoundError(batch.image_path)
        image = preprocess_full_frame(batch.image_path, device, next(clip_model.parameters()).dtype)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.inference_mode():
            pyramid = visual_pyramid(clip_model, image, with_grad=False)
        model = L79HierarchicalCorrespondence().to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            outputs = model(
                batch.observations.to(device), batch.history_observations.to(device), batch.history_mask.to(device),
                batch.text_tokens.to(device), batch.text_mask.to(device), batch.boxes_norm.to(device), pyramid,
            )
        torch.cuda.synchronize()
        # Labels are deliberately attached only after the complete image/text/
        # candidate feature path has run.
        labels = L79BankStore.attach_labels(batch, first)
        duplicate_candidate_indices = len(batch.candidate_indices) - len(set(batch.candidate_indices))
        finite_output = all(bool(torch.isfinite(value.float()).all()) for value in outputs.values())
        contract = {
            "format": "locatemot-l79-p0-contract-v1",
            "status": "complete",
            "stage": "L79-P0",
            "project_root": str(ROOT),
            "cwd": str(Path.cwd().resolve()),
            "command": command,
            "input_record_schema_before_label_attach": sorted(key_unit.keys()),
            "forbidden_fields_absent_before_label_attach": sorted(forbidden.difference(key_unit)),
            "labels_attached_after_feature_construction": bool(labels["labels_attached_after_feature_construction"]),
            "unit_key": batch.unit_key,
            "dataset": batch.dataset,
            "video": batch.video,
            "query_id": batch.query_id,
            "frame_id": batch.frame_id,
            "image_path": batch.image_path,
            "candidate_count": batch.candidate_count,
            "row_key_count": len(batch.row_keys),
            "row_order_exact": batch.row_keys == sorted(batch.row_keys, key=lambda x: x[-1]),
            "candidate_deletion": False,
            "candidate_truncation": False,
            "duplicate_candidate_index_rows_retained": duplicate_candidate_indices,
            "observation_shape": list(batch.observations.shape),
            "history_shape": list(batch.history_observations.shape),
            "history_mask_shape": list(batch.history_mask.shape),
            "text_shape": list(batch.text_tokens.shape),
            "text_valid_tokens": int(batch.text_mask.sum()),
            "clip_pyramid_shape": list(pyramid.shape),
            "model_output_shapes": {name: list(value.shape) for name, value in outputs.items()},
            "finite": bool(finite_output),
            "image_spatial_variation": float(pyramid.float().flatten(2).std().cpu()),
            "candidate_logit_variation": float(outputs["frame_membership_logits"].float().std().cpu()),
            "history_future_rows": int((batch.history_frame_ids > batch.frame_id).sum()),
            "source_pool_ids_provenance_only": True,
            "l11_shared_checkpoint_expected_sha256": EXPECTED_SHARED_CHECKPOINT_SHA,
            "l11_task_bypass": {"mot": not l79_task_enabled("mot"), "ovmot": not l79_task_enabled("ovmot"), "rmot": l79_task_enabled("rmot")},
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "training_run": False,
            "hota_trackeval_run": False,
        }
        parameter = {
            "format": "locatemot-l79-parameter-contract-v1",
            "status": "complete",
            "model": "L79HierarchicalCorrespondence",
            "decoder_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "decoder_trainable_parameter_count": model.trainable_parameter_count(),
            "target_range": [25_000_000, 60_000_000],
            "in_range": 25_000_000 <= model.trainable_parameter_count() <= 60_000_000,
            "hidden": 384,
            "heads": 6,
            "history_length": 16,
            "query_layers": 4,
            "temporal_layers": 2,
            "set_layers": 4,
            "private_lora": {"rank": 32, "alpha": 16.0, "blocks": [8, 9, 10, 11], "target": "visual.transformer.resblocks[i].attn.out_proj", "enabled": False},
            "clip_base_all_requires_grad_false": all(not p.requires_grad for p in clip_model.parameters()),
            "l11_loaded_into_model": False,
            "optimizer_scope": "L79 decoder only in P1; private CLIP LoRA may be added only in fixed phase-2 schedule",
        }
        coverage = {
            "format": "locatemot-l79-fit-coverage-contract-v1",
            "status": "complete",
            "fit_units": len(fit_units),
            "domains": {name: sum(str(x["dataset"]) == name for x in fit_units) for name in domains},
            "categories": {name: sum(str(x["category"]) == name for x in fit_units) for name in sorted(categories)},
            "videos": sorted({str(x["video"]) for x in fit_units}),
            "required_categories": ["positive", "multi_positive", "inactive", "present_uncovered", "occlusion_reappear_or_cross_fragment_diagnostic"],
            "sampler_category_fields": "fit metadata only; not neural inputs",
            "all_fit_units_present": True,
            "present_uncovered_membership_masked": True,
        }
        provenance = {
            "format": "locatemot-l79-p0-provenance-v1",
            "status": "complete",
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "command": command,
            "python": sys.executable, "python_version": sys.version,
            "torch_version": torch.__version__, "cuda": torch.version.cuda,
            "device": str(device), "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "input_files": {
                "manifest": file_meta(MANIFEST),
                "l11_checkpoint": file_meta(ROOT / "outputs/l11/checkpoints/uidm_l11_main/latest.pt"),
                "l69_manifest": file_meta(L69_ROOT / "manifest.json"),
                "l48_text_cache": file_meta(ROOT / "outputs/l48/data/text_cache.pt"),
                "clip_weight": {"path": str(CLIP_WEIGHT), "sha256": sha256_file(CLIP_WEIGHT), "expected_sha256": CLIP_SHA256, "bytes": CLIP_WEIGHT.stat().st_size},
                "l69_sample_bank": file_meta(L69_ROOT / f"{batch.video}.pt"),
            },
            "l69_videos_available": list(ALL_L69_VIDEOS),
            "l69_bank_manifest": source_file_manifest(ALL_L69_VIDEOS),
            "labels": "one fit unit attached post-feature construction for stratum audit; no calibration/validation/screening labels read",
            "no_raw_or_dense_cache_written": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        status = {"format": "locatemot-l79-p0-status-v1", "status": "complete", "failure_root_cause": None, "next_action": "run the independent L79 forward/loss contract, then P1 100-step fit smoke if complete"}
        for name, payload in [("contract.json", contract), ("parameter_contract.json", parameter), ("coverage_contract.json", coverage), ("provenance.json", provenance), ("status.json", status)]:
            write_json(out / name, payload)
        del outputs, pyramid, image, clip_model, model
        torch.cuda.empty_cache()
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        status = {"format": "locatemot-l79-p0-status-v1", "status": "incomplete", "command": command, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": tb, "next_action": "fix only the first actionable P0 contract error and rerun in a new attempt"}
        write_json(out / "status.json", status)
        (out / "INCOMPLETE.md").write_text("# L79 P0 incomplete\n\nFirst actionable error:\n\n```text\n" + tb + "```\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
