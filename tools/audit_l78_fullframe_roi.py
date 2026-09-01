#!/usr/bin/env python3
"""Label-free contract audit for the L78 full-frame spatial ROI path."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l78_fullframe_roi_set import L78FullFrameROISet
from tools.l78_common import (
    CLIP_WEIGHTS, EXPECTED_CLIP_SHA, EXPECTED_MANIFEST_SHA, MANIFEST,
    FORBIDDEN_LABEL_FIELDS,
    StreamingOpenAIClipFullFrame, L78Bank, boxes_to_normalized, dist,
    fixed_key_order, image_path, make_fit_schedule, select_label_free_audit_units,
    sha256_file, simple_svg_boxes, unit_key, write_json,
)


def _ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out = out.resolve()
    _ensure_empty(out)
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA mismatch")
    if sha256_file(CLIP_WEIGHTS) != EXPECTED_CLIP_SHA:
        raise AssertionError("CLIP SHA mismatch")
    seed = 20260829
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    units = select_label_free_audit_units()
    if len(units) != 8:
        raise AssertionError(f"expected 8 declared audit strata, got {len(units)}")
    device = torch.device(args.device)
    encoder = StreamingOpenAIClipFullFrame(str(device))
    frozen = encoder.frozen_contract()
    model = L78FullFrameROISet(hidden=128, heads=4, roi_grid=4).to(device).eval()
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise AssertionError("CLIP parameter unexpectedly trainable")
    rows = []
    unit_summaries = []
    image_map_stds, roi_stds, score_stds = [], [], []
    expr_deltas, perm_deltas = [], []
    all_finite = True
    total_candidates = 0
    start_time = time.time()
    first_reload_diff = None
    for unit in units:
        bank = L78Bank(str(unit["video"]))
        record = bank.label_free_record(unit)
        # The sidecar is deliberately not touched here.  Complete native rows
        # and raw features are constructed before the explicit label phase.
        path = image_path(record["video"], record["frame_id"])
        spatial, global_token, image_geometry = encoder.image_map(path)
        normalized, box_details = boxes_to_normalized(record["boxes"], image_geometry, padding=0.10)
        text, text_mask, token_ids = encoder.text_tokens(record["sentence"])
        with torch.inference_mode():
            base_output = model({
                "spatial_map": spatial.to(device), "global_token": global_token.to(device),
                "text": text.to(device), "text_mask": text_mask.to(device),
                "boxes": normalized.to(device),
            })
            unrelated_text, unrelated_mask, unrelated_ids = encoder.text_tokens(
                "an unrelated object in a distant location"
            )
            unrelated_output = model({
                "spatial_map": spatial.to(device), "global_token": global_token.to(device),
                "text": unrelated_text.to(device), "text_mask": unrelated_mask.to(device),
                "boxes": normalized.to(device),
            })
            permutation = torch.arange(normalized.shape[0] - 1, -1, -1, device=device)
            perm_output = model({
                "spatial_map": spatial.to(device), "global_token": global_token.to(device),
                "text": text.to(device), "text_mask": text_mask.to(device),
                "boxes": normalized.to(device)[permutation],
            })
        score = base_output["match_logits"].float().cpu()
        unrelated_score = unrelated_output["match_logits"].float().cpu()
        roi = base_output["roi_tokens"].float().cpu()
        perm_roi = perm_output["roi_tokens"].float().cpu()
        if score.shape != (record["candidate_count"],) or roi.shape != (record["candidate_count"], 16, 512):
            raise AssertionError(f"L78 output shape failed {record['unit_key']}")
        tensors = (spatial, global_token, text, score, unrelated_score, roi, perm_roi)
        if not all(bool(torch.isfinite(value.float()).all()) for value in tensors):
            raise FloatingPointError(f"nonfinite label-free output {record['unit_key']}")
        map_std = float(spatial.float().std())
        roi_row_mean = roi.mean(dim=(1, 2))
        roi_std = float(roi_row_mean.std()) if len(roi_row_mean) > 1 else 0.0
        score_std = float(score.std()) if len(score) > 1 else 0.0
        expression_delta = float(torch.linalg.vector_norm(score - unrelated_score) / torch.linalg.vector_norm(score).clamp_min(1e-6))
        permutation_delta = float(torch.linalg.vector_norm(roi - perm_roi) / torch.linalg.vector_norm(roi).clamp_min(1e-6))
        image_map_stds.append(map_std); roi_stds.append(roi_std); score_stds.append(score_std)
        expr_deltas.append(expression_delta); perm_deltas.append(permutation_delta)
        total_candidates += int(record["candidate_count"])
        unit_summaries.append({
            "unit_key": record["unit_key"], "dataset": record["dataset"], "video": record["video"],
            "query_id": record["query_id"], "frame_id": record["frame_id"],
            "candidate_count": record["candidate_count"], "row_key_count": len(record["row_keys"]),
            "row_order_exact": record["row_offsets"] == list(range(record["row_offsets"][0], record["row_offsets"][-1] + 1)),
            "duplicate_candidate_index_count": int(record["candidate_count"] - len(set(record["candidate_index_provenance"]))),
            "image_path": str(path), "image_geometry": image_geometry,
            "pixel_shape": image_geometry["pixel_shape"], "spatial_map_shape": list(spatial.shape),
            "global_token_shape": list(global_token.shape), "text_shape": list(text.shape),
            "text_valid_tokens": int(text_mask.sum()), "token_id_count": int(token_ids.numel()),
            "unrelated_valid_tokens": int(unrelated_mask.sum()), "unrelated_token_id_count": int(unrelated_ids.numel()),
            "map_spatial_std": map_std, "roi_row_mean_std": roi_std, "score_std": score_std,
            "expression_score_relative_l2": expression_delta,
            "box_permutation_roi_relative_l2": permutation_delta,
            "all_finite": True, "sidecar_labels_loaded": False,
            "candidate_deletion": False, "candidate_truncation": False,
        })
        for index, (key, detail, row_score) in enumerate(zip(record["row_keys"], box_details, score.tolist())):
            rows.append({
                "format": "locatemot-l78-label-free-row-v1", "unit_key": record["unit_key"],
                "row_key": key, "row_offset": int(record["row_offsets"][index]),
                "candidate_index_provenance": int(record["candidate_index_provenance"][index]),
                "pool_id_provenance": int(record["pool_id_provenance"][index]),
                "normalized_box": detail["normalized_box"], "match_logit_diagnostic": float(row_score),
                "finite": True, "sidecar_labels_loaded": False,
            })
        if first_reload_diff is None:
            audit_ckpt = out / "adapter_contract_state.pt"
            torch.save({"model": model.state_dict(), "format": "locatemot-l78-audit-adapter-v1"}, audit_ckpt)
            reloaded = L78FullFrameROISet(hidden=128, heads=4, roi_grid=4).to(device).eval()
            package = torch.load(audit_ckpt, map_location=device, weights_only=False)
            reloaded.load_state_dict(package["model"], strict=True)
            with torch.inference_mode():
                reload_output = reloaded({
                    "spatial_map": spatial.to(device), "global_token": global_token.to(device),
                    "text": text.to(device), "text_mask": text_mask.to(device), "boxes": normalized.to(device),
                })
            first_reload_diff = float(torch.max(torch.abs(reload_output["match_logits"].float().cpu() - score)))
            del reloaded, package, reload_output
        simple_svg_boxes(out / f"boxes_{len(unit_summaries):02d}.svg", image_geometry, normalized, score.tolist())
        del bank, record, spatial, global_token, text, text_mask, token_ids, normalized
        del base_output, unrelated_output, perm_output, unrelated_text, unrelated_mask, unrelated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del encoder, model
    finite = bool(all_finite and all(x["all_finite"] for x in unit_summaries))
    label_free_pass = bool(
        finite and len(rows) == total_candidates and total_candidates > 0
        and min(image_map_stds) > 0.0 and max(roi_stds) > 0.0 and max(score_stds) > 0.0
        and max(expr_deltas) > 1e-4 and max(perm_deltas) > 1e-4 and first_reload_diff is not None
        and first_reload_diff <= 1e-6
        and all(not x["sidecar_labels_loaded"] for x in unit_summaries)
    )
    status = "complete" if label_free_pass else "INCOMPLETE"
    diagnostics = {
        "format": "locatemot-l78-label-free-fullframe-roi-audit-v1", "status": status,
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "decision": "label_free_contract_pass" if label_free_pass else "label_free_contract_fail",
        "audit_units": len(units), "candidate_rows": total_candidates,
        "unit_summaries": unit_summaries,
        "map_spatial_std": dist(image_map_stds), "roi_row_mean_std": dist(roi_stds),
        "score_std": dist(score_stds), "expression_relative_l2": dist(expr_deltas),
        "box_permutation_relative_l2": dist(perm_deltas),
        "strict_reload_max_abs_diff": first_reload_diff,
        "primary_contract": {
            "full_frame_visual_map": "CLIP ViT-B/16 final 14x14 patch grid after ln_post+visual.proj",
            "roi_pool": "differentiable align_corners=False grid_sample, fixed 4x4 cell-center lattice",
            "letterbox": "fixed aspect-preserving 224 square canvas with CLIP mean background",
            "candidate_padding": 0.10, "text_mask": "OpenAI CLIP token_id != 0",
            "token_region_alignment": "UNALIGNED",
        },
        "pre_forward_schema": [
            "format", "status", "unit_key", "dataset", "video", "query_id", "frame_id",
            "sentence", "expression", "frame_index", "bank_path", "row_offsets", "row_keys",
            "candidate_index_provenance", "track_id_provenance", "pool_id_provenance",
            "raw_rank_provenance", "candidate_count", "boxes", "image_size_bank",
            "candidate_deletion", "candidate_truncation", "old_l49_ranges_used",
        ],
        "forbidden_label_fields_absent": all(all(field not in item for field in FORBIDDEN_LABEL_FIELDS) for item in unit_summaries),
        "sidecar_labels_loaded": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
    }
    write_json(out / "label_free_diagnostics.json", diagnostics)
    with (out / "unit_records.jsonl").open("w") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_json(out / "config.json", {
        "format": "locatemot-l78-label-free-config-v1", "seed": seed, "device": str(device),
        "audit_units_selected_by_declared_fit_strata": len(units),
        "declared_strata": ["refer_kitti_v1/positive", "refer_kitti_v1/multi_positive", "refer_kitti_v1/inactive", "refer_kitti_v1/present_uncovered", "refer_kitti_v2/positive", "refer_kitti_v2/multi_positive", "refer_kitti_v2/inactive", "refer_kitti_v2/present_uncovered"],
        "labels_used_for_representation": False, "candidate_rows_all_retained": True,
        "candidate_deletion": False, "candidate_truncation": False,
        "roi_grid": 4, "padding": 0.10, "align_corners": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
    })
    write_json(out / "provenance.json", {
        "format": "locatemot-l78-label-free-provenance-v1", "status": status,
        "command": " ".join([str(Path.cwd() / "tools/audit_l78_fullframe_roi.py")] + list(__import__("sys").argv[1:])),
        "inputs": {
            "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
            "l62_fixed_records": str(ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"),
            "l49_fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
            "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
            "clip_weights": str(CLIP_WEIGHTS), "clip_weights_sha256": sha256_file(CLIP_WEIGHTS),
        },
        "outputs": {"label_free_diagnostics": str(out / "label_free_diagnostics.json"), "unit_records": str(out / "unit_records.jsonl")},
        "feature_construction_precedes_label_attach": True, "sidecar_labels_loaded": False,
        "model_parameter_summary": L78FullFrameROISet(hidden=128, heads=4, roi_grid=4).parameter_summary(),
        "runtime": {"elapsed_sec": time.time() - start_time, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None, "environment": sys.version if False else "recorded by interpreter"},
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
    })
    write_json(out / "status.json", {"format": "locatemot-l78-status-v1", "status": status, "decision": diagnostics["decision"], "failure_root_cause": None if label_free_pass else "raw_fullframe_roi_contract", "next_action": "attach expression labels and run fit smoke" if label_free_pass else "preserve audit and stop L78", "command": " ".join(__import__("sys").argv), "inputs": [str(ROOT / "outputs/l69/attempt9/budget40_features/kitti")], "outputs": [str(out)], "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True})
    if not label_free_pass:
        (out / "INCOMPLETE.md").write_text("L78 label-free contract failed; see label_free_diagnostics.json for the first contract-level diagnosis. No labels or training were used.\n")
        raise RuntimeError("L78 label-free contract failed")
    print(json.dumps({"status": status, "decision": diagnostics["decision"], "candidate_rows": total_candidates, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
