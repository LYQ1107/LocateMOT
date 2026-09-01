#!/usr/bin/env python3
"""L42 train-side current-frame expression grounding contract audit."""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.audit_l28_identity_bank import BANK_ROOT, load_labels
from tools.l40_raw_data import RAW_ROOT, crop_box, sha256
from tools.train_l26_crossmodal_adapter import EXP, REC1, REC2, load_expressions

L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
TRAIN_VIDEOS = ("0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0020")


def main():
    assert Path.cwd().resolve() == ROOT
    out = ROOT / "outputs/l42/audit"; out.mkdir(parents=True, exist_ok=True); started = time.time()
    expressions = [x for x in load_expressions() if str(x["video"]) in TRAIN_VIDEOS]
    expression_count = len(expressions); frame_units = candidate_rows = positive_rows = hard_rows = multi_units = inactive_units = 0
    missing_images = []; invalid_boxes = []; finite = True; token_lengths = []; video_stats = {}
    for video in TRAIN_VIDEOS:
        path = BANK_ROOT / f"{video}.pt"; bank = torch.load(path, map_location="cpu", weights_only=False); t = bank["tensors"]; count = int(t["track_id"].numel()); labels, label_path = load_labels(path, count, tensors=t)
        meta = bank.get("metadata", {}); width, height = [int(x) for x in meta.get("image_size", [1242, 375])]
        for frame in t["frame_ids"].tolist():
            p = RAW_ROOT / video / f"{int(frame):06d}.png"
            if not p.exists(): missing_images.append(str(p))
        for row in range(count):
            box = t["box"][row].float(); finite = finite and bool(torch.isfinite(box).all()) and bool(torch.isfinite(t["geometry"][row].float()).all()) and bool(torch.isfinite(t["motion"][row].float()).all()) and bool(torch.isfinite(t["lifecycle"][row].float()).all())
            c = crop_box(box.tolist(), width, height)
            if c[2] <= c[0] or c[3] <= c[1]: invalid_boxes.append({"video": video, "row": row, "box": box.tolist()})
        by_frame = {int(f): (int(b), int(e)) for f, b, e in zip(t["frame_ids"].tolist(), t["frame_ptr"][:-1].tolist(), t["frame_ptr"][1:].tolist())}
        vframes = 0; vpos = 0
        for e in expressions:
            if str(e["video"]) != video: continue
            text = str(e.get("sentence", e.get("expression", ""))); token_lengths.append(len(text.split()))
            targets = {int(k): {str(x) for x in v} for k, v in e.get("label", {}).items()}
            for frame, (begin, end) in by_frame.items():
                frame_units += 1; vframes += 1; n = end - begin; candidate_rows += n
                target_ids = targets.get(frame, set()); y = np.asarray([labels[r] is not None and str(labels[r]) in target_ids for r in range(begin, end)], bool)
                positive_rows += int(y.sum()); hard_rows += int((~y).sum()) if target_ids else 0; vpos += int(y.sum())
                multi_units += int(y.sum() > 1); inactive_units += int(not target_ids)
        video_stats[video] = {"bank_rows": count, "bank_frames": len(by_frame), "expressions": sum(str(e["video"]) == video for e in expressions), "image_paths_missing": sum(str(RAW_ROOT / video / f"{int(f):06d}.png") in missing_images for f in t["frame_ids"].tolist()), "label_sidecar": str(label_path), "positive_rows": vpos, "frame_units": vframes}
    fast = json.loads(FAST.read_text()); payload = {"schema_version": "locatemot-l42-current-frame-grounding-contract-v1", "stage": "L42", "project_root": str(ROOT), "started_at": started, "completed_at": time.time(), "train_videos": list(TRAIN_VIDEOS), "train_video_count": len(TRAIN_VIDEOS), "expression_count": expression_count, "frame_unit_count": frame_units, "candidate_row_count": candidate_rows, "positive_candidate_rows": positive_rows, "same_frame_hard_negative_rows_on_target_frames": hard_rows, "multi_positive_frame_units": multi_units, "inactive_or_null_frame_units": inactive_units, "expression_tokenization": {"method": "whitespace audit of raw sentence; model retains existing word-level token sequence", "token_count_mean": float(np.mean(token_lengths)) if token_lengths else None, "token_count_max": int(max(token_lengths)) if token_lengths else None, "raw_expression_sources": [str(x) for x in EXP]}, "video_stats": video_stats, "raw_image_mapping": {"root": str(RAW_ROOT), "coordinate_system": "KITTI image pixels xyxy", "crop_rule": "candidate observation box with 10 percent padding, clipped to image, frozen CLIP preprocess", "missing_frame_paths": len(missing_images), "invalid_boxes": len(invalid_boxes), "gt_driven_sampling": False, "persistent_dense_or_embedding_cache": False}, "bank_cache_fields": ["frame", "track_id", "box", "geometry", "motion", "lifecycle", "objectness", "pool_id"], "finite_observation_fields": finite, "l29_teacher": {"checkpoint": str(L29.resolve()), "checkpoint_sha256": sha256(L29), "logit_field": "current_membership_logits", "frozen_primary_emission": True}, "supervision_labels": {"expression_to_current_frame_membership": "EXPRESSION_LEVEL_VERIFIED plus GT-derived candidate membership", "same_frame_hard_negative": "GT_PRIVILEGED_ORACLE", "multi_positive": "GT_PRIVILEGED_ORACLE; all positive candidates retained", "inactive_null": "GT_PRIVILEGED_ORACLE", "token_span_to_region": "UNALIGNED", "static_motion_language_mask": "UNALIGNED/not claimed"}, "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key"], "source_usage": "diagnostic stratification only", "fixed_fast_manifest": {"path": str(FAST), "sha256": sha256(FAST), "query_count": len(fast["queries"]), "calibration": 64, "screening": 96, "used_for_training": False, "used_for_structure_selection": False, "screening_gt_used": False}, "leakage_audit": {"training_videos": list(TRAIN_VIDEOS), "screening_gt_used_for_training_or_selection": False, "training_supervision_source": "train-side expression/frame/GT labels only"}, "decision": {"current_frame_expression_supervision_available": bool(expression_count > 0 and candidate_rows > 0 and finite and not missing_images and not invalid_boxes and L29.exists()), "enter_l42_smoke": bool(expression_count > 0 and candidate_rows > 0 and finite and not missing_images and not invalid_boxes and L29.exists())}, "elapsed_sec": time.time() - started}
    (out / "current_frame_grounding_contract.json").write_text(json.dumps(payload, indent=2) + "\n"); (out / "README.md").write_text("# L42 current-frame grounding contract\n\nExpression-level supervision is train-side only. Candidate crops are streamed; no dense embedding cache is written.\n")
    if not payload["decision"]["enter_l42_smoke"]: (out / "NO_CURRENT_FRAME_SUPERVISION.md").write_text("# NO_CURRENT_FRAME_SUPERVISION\n\nExpression, candidate, frame, or raw-image contract failed.\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
