#!/usr/bin/env python3
"""Train-only L46 query/region/track input-contract audit.

The audit deliberately does not train, run TrackEval, or consume screening
labels.  It validates the existing frozen candidate bank, L28 sequence cache,
word-token cache, raw-image reversibility, expression-level labels, and a small
L29 teacher mapping replay before the L46 RMOT-only model is allowed to smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.audit_l28_identity_bank import load_labels
from tools.audit_l44_integrated_contract import (
    FAST,
    L19,
    L28,
    L29,
    SPLIT,
    TRAIN_VIDEOS,
    load_queries,
    sha256,
    valid_teacher_indices,
)
from tools.l40_raw_data import RAW_ROOT, WEIGHTS, image_path
from tools.train_l26_crossmodal_adapter import V5
from tools.train_l28_track_set_decoder import state_at

OUT = ROOT / "outputs/l46/audit"
REPORT = ROOT / "reports/l46_end_to_end_contract_audit.md"


def finite(value) -> bool:
    x = torch.as_tensor(value)
    return bool(torch.isfinite(x.float()).all())


def load_bank(video: str):
    path = L19 / f"{video}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tensors = blob["tensors"]
    labels, label_path = load_labels(path, int(tensors["track_id"].numel()), tensors=tensors)
    return {
        "path": path,
        "metadata": blob["metadata"],
        "box": tensors["box"].float(),
        "frame": tensors["frame"].long(),
        "candidate_index": tensors["candidate_index"].long(),
        "track": tensors["track_id"].long(),
        "pool_id": tensors["pool_id"].long(),
        "objectness": tensors["objectness"].float(),
        "clip": tensors["clip"].float(),
        "history_clip": tensors["history_clip"].float(),
        "geometry": tensors["geometry"].float(),
        "motion": tensors["motion"].float(),
        "context": tensors["context"].float(),
        "lifecycle": tensors["lifecycle"].float(),
        "uidm_h": tensors["uidm_h"].float(),
        "frame_ids": tensors["frame_ids"].long(),
        "ptr": tensors["frame_ptr"].long(),
        "labels": labels,
        "label_path": str(label_path),
    }


def query_rows():
    queries = load_queries()
    if len(queries) != 7757:
        raise AssertionError(f"expected 7757 train expressions, found {len(queries)}")
    return queries


def feature_shapes(bank):
    return {key: list(value.shape) for key, value in bank.items()
            if isinstance(value, torch.Tensor)}


def inspect_anchors():
    anchor_path = ROOT / "outputs/l44/audit/frozen_anchors.json"
    old = json.loads(anchor_path.read_text()) if anchor_path.exists() else {}
    paths = {
        "l11_uidm": ROOT / "outputs/l11/checkpoints/uidm_l11_main/step11000.pt",
        "l8_adapter": ROOT / "outputs/l8/checkpoints/uidm_l8_final/latest.pt",
        "online_tracker": ROOT / "locatemot/tracking/online_tracker.py",
        "ordinary_mot_entry": ROOT / "tools/eval_l8_ordinary.py",
        "ovmot_entry": ROOT / "tools/eval_l8_ovmot.py",
        "l8_unified": ROOT / "locatemot/models/l8_unified.py",
    }
    current = {}
    for name, path in paths.items():
        current[name] = {"path": str(path), "exists": path.exists(),
                         "sha256": sha256(path) if path.exists() else None}
    return {"previous_anchor_file": str(anchor_path), "previous": old,
            "current": current}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-sample", type=int, default=8)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    queries = query_rows()
    banks = {video: load_bank(video) for video in TRAIN_VIDEOS}
    caches = {video: torch.load(L28 / f"{video}.pt", map_location="cpu", weights_only=False)
              for video in TRAIN_VIDEOS}
    required = ("box", "frame", "candidate_index", "track", "objectness", "clip",
                "geometry", "motion", "context", "lifecycle", "uidm_h",
                "frame_ids", "ptr", "labels")
    missing_fields = {}
    nonfinite = Counter()
    invalid_boxes = []
    boundary_clipped_boxes = []
    missing_images = []
    duplicate_rows = 0
    frame_sizes = []
    per_video = {}
    cache_alignment_errors = []

    for video in TRAIN_VIDEOS:
        bank = banks[video]
        missing = [key for key in required if key not in bank]
        if missing:
            missing_fields[video] = missing
        n = len(bank["track"])
        for key in ("box", "objectness", "clip", "geometry", "motion", "context",
                    "lifecycle", "uidm_h"):
            if not finite(bank[key]):
                nonfinite[key] += 1
        # candidate_index is local to each source pool (main/reserve).  The
        # stable row identity therefore includes pool, track and candidate;
        # checking candidate_index alone would report every normal cross-pool
        # pair as a false duplicate.
        for fi, frame in enumerate(bank["frame_ids"].tolist()):
            begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi + 1])
            frame_sizes.append(end - begin)
            row_identity = list(zip(
                bank["pool_id"][begin:end].tolist(),
                bank["track"][begin:end].tolist(),
                bank["candidate_index"][begin:end].tolist(),
            ))
            duplicate_rows += len(row_identity) - len(set(row_identity))
            path = image_path(video, int(frame))
            if not path.exists():
                missing_images.append(str(path))
            else:
                # Check image dimensions once per frame so boxes are reversible.
                from PIL import Image
                with Image.open(path) as image:
                    width, height = image.size
                boxes = bank["box"][begin:end].numpy()
                for local, box in enumerate(boxes):
                    eps = 1e-3
                    if (not np.isfinite(box).all() or box[2] <= box[0] or
                            box[3] <= box[1] or box[0] < -eps or box[1] < -eps or
                            box[2] > width + eps or box[3] > height + eps):
                        invalid_boxes.append({"video": video, "frame": int(frame),
                                              "row": begin + local, "box": box.tolist(),
                                              "image_size": [width, height]})
                    elif (box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height):
                        boundary_clipped_boxes.append({
                            "video": video, "frame": int(frame), "row": begin + local,
                            "box": box.tolist(), "image_size": [width, height],
                        })

        cache = caches[video]
        ptr = cache["track_ptr"].numpy()
        frames = cache["obs_frame"].numpy()
        rows_by_track = defaultdict(list)
        for row, track in enumerate(bank["track"].tolist()):
            rows_by_track[int(track)].append(row)
        for values in rows_by_track.values():
            values.sort(key=lambda row: (int(bank["frame"][row]), row))
        for ti, track in enumerate(cache["track_ids"].tolist()):
            cb, ce = int(ptr[ti]), int(ptr[ti + 1])
            cache_frames = [int(x) for x in frames[cb:ce].tolist()]
            bank_frames = [int(bank["frame"][row]) for row in rows_by_track.get(int(track), [])]
            if cache_frames != bank_frames:
                cache_alignment_errors.append({"video": video, "track_id": int(track),
                                               "cache_frames": cache_frames,
                                               "bank_frames": bank_frames})
        per_video[video] = {
            "split": bank["metadata"].get("split"),
            "bank_sha256": sha256(bank["path"]),
            "bank_metadata_format": bank["metadata"].get("format"),
            "frame_count": int(len(bank["frame_ids"])),
            "candidate_rows": int(n),
            "track_count": int(len(cache["track_ids"])),
            "feature_shapes": feature_shapes(bank),
            "sequence_feature_shape": list(cache["obs_features"].shape),
            "sequence_feature_dim": int(cache["obs_features"].shape[1]),
            "label_sidecar": bank["label_path"],
        }

    population = Counter()
    per_video_queries = Counter()
    source_counts = Counter()
    for bank in banks.values():
        source_counts["main_rows"] += int((bank["pool_id"] == 0).sum())
        source_counts["reserve_rows"] += int((bank["pool_id"] == 1).sum())
    hard_definition = "all same-frame non-positive candidates; class identity is not used because no verified class field is present"
    continuation_positive_rows = 0
    query_attribute_counts = Counter()
    representative_units = []

    for query in queries:
        video = query["video"]
        bank = banks[video]
        cache = caches[video]
        per_video_queries[video] += 1
        target = query["target"]
        ptr = bank["ptr"]
        for fi, frame in enumerate(bank["frame_ids"].tolist()):
            begin, end = int(ptr[fi]), int(ptr[fi + 1])
            ids = {str(x) for x in target.get(int(frame), set())}
            labels = np.asarray([
                bank["labels"][row] is not None and str(bank["labels"][row]) in ids
                for row in range(begin, end)
            ], dtype=bool)
            pos = int(labels.sum())
            neg = int(len(labels) - pos)
            population["frame_units"] += 1
            population["candidate_rows"] += end - begin
            population["positive_rows"] += pos
            population["negative_rows"] += neg
            population["positive_negative_pairs"] += pos * neg
            population["target_frame_units"] += int(bool(ids))
            population["positive_frame_units"] += int(bool(pos))
            population["multi_positive_frame_units"] += int(pos > 1)
            population["multi_positive_rows"] += pos if pos > 1 else 0
            population["inactive_or_null_frame_units"] += int(not ids)
            population["target_without_candidate_frame_units"] += int(bool(ids) and not pos)
            population["same_frame_hard_negative_candidates"] += neg if pos else 0
            # Verify expression-level history labels without using screening.
            track_to_index = {int(track): i for i, track in enumerate(cache["track_ids"].tolist())}
            cptr = cache["track_ptr"].tolist()
            cframes = cache["obs_frame"].tolist()
            cgt = cache["obs_gt_ids"]
            for offset, row in enumerate(range(begin, end)):
                if not labels[offset]:
                    continue
                ti = track_to_index.get(int(bank["track"][row]))
                if ti is None:
                    continue
                cb, ce = int(cptr[ti]), int(cptr[ti + 1])
                earlier = [j for j in range(cb, ce) if int(cframes[j]) < int(frame)]
                if any(cgt[j] is not None and str(cgt[j]) in ids for j in earlier):
                    continuation_positive_rows += 1
            if len(representative_units) < 16 and (pos > 1 or not ids or pos == 1):
                representative_units.append({"video": video, "expression": query["expression"],
                                             "query_index": int(query["text_index"]),
                                             "frame": int(frame), "candidate_rows": end - begin,
                                             "positive_rows": pos, "inactive": not ids})
        text = query["sentence"].lower()
        for token in ("black", "red", "white", "silver", "left", "right", "front",
                      "behind", "moving", "standing", "turning", "opposite"):
            query_attribute_counts[token] += int(token in text)

    # Text cache is train-only here.  The fixed manifest is recorded and hashed,
    # but its screening labels never enter this audit's counts or decisions.
    text_blob = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden = text_blob["token_hidden"].float()
    text_mask = text_blob["attention_mask"].bool()
    query_indices = torch.as_tensor([int(q["text_index"]) for q in queries])
    text_checks = {
        "hidden_shape": list(text_hidden.shape),
        "mask_shape": list(text_mask.shape),
        "finite": finite(text_hidden),
        "all_train_queries_present": bool(query_indices.max() < text_hidden.shape[0]),
        "nonempty_train_masks": int(text_mask[query_indices].any(1).sum()),
        "query_count_checked": len(queries),
        "word_level_sequence_retained": True,
        "token_span_region_verified": False,
        "static_motion_language_mask_verified": False,
    }

    # Small train-only L29 mapping regression: use track IDs from the L28 cache,
    # never row position, and do not compute an L29 replay for the full population.
    teacher = L29FrameMembershipSetDecoder()
    teacher_blob = torch.load(L29, map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_blob["model"], strict=True)
    teacher.eval()
    samples = []
    for q in queries:
        b = banks[q["video"]]
        for fi, frame in enumerate(b["frame_ids"].tolist()):
            begin, end = int(b["ptr"][fi]), int(b["ptr"][fi + 1])
            ids = {str(x) for x in q["target"].get(int(frame), set())}
            y = np.asarray([b["labels"][row] is not None and str(b["labels"][row]) in ids
                            for row in range(begin, end)], bool)
            category = "multi_positive" if y.sum() > 1 else "positive" if y.any() else "inactive"
            if not any(x[0] == category for x in samples):
                samples.append((category, q, fi, y))
            if len(samples) >= min(args.teacher_sample, 3):
                break
        if len(samples) >= min(args.teacher_sample, 3):
            break
    teacher_examples = []
    teacher_missing = 0
    teacher_nonfinite = 0
    with torch.inference_mode():
        for category, q, fi, y in samples:
            b = banks[q["video"]]; cache = caches[q["video"]]
            frame = int(b["frame_ids"][fi]); begin, end = int(b["ptr"][fi]), int(b["ptr"][fi + 1])
            obs, om, ot, _, _ = state_at(cache, frame, history=8)
            encoded = teacher.encode_observations(obs, om, ot)
            result = teacher.forward_encoded(encoded, encoded[1],
                                             text_hidden[q["text_index"]],
                                             text_mask[q["text_index"]])
            logits = result["current_membership_logits"].float()
            valid = valid_teacher_indices(cache, frame)
            valid_ids = cache["track_ids"][torch.as_tensor(valid)].tolist()
            by_track = {int(track): float(score) for track, score in zip(valid_ids, logits.tolist())}
            values = [by_track.get(int(b["track"][row]), math.nan) for row in range(begin, end)]
            teacher_missing += sum(not math.isfinite(value) for value in values)
            teacher_nonfinite += int(not finite(logits))
            teacher_examples.append({"category": category, "video": q["video"],
                                     "expression": q["expression"], "frame": frame,
                                     "candidate_rows": end - begin, "positive_rows": int(y.sum()),
                                     "mapped_track_count": len(by_track),
                                     "teacher_min": float(np.min(values)),
                                     "teacher_max": float(np.max(values))})

    fast = json.loads(FAST.read_text())
    anchors = inspect_anchors()
    population["continuation_positive_rows"] = continuation_positive_rows
    decision_checks = {
        "missing_fields": not missing_fields,
        "missing_images": not missing_images,
        "invalid_boxes": not invalid_boxes,
        "duplicate_row_keys": duplicate_rows == 0,
        "cache_alignment": not cache_alignment_errors,
        "finite_bank_fields": not nonfinite,
        "text_finite": text_checks["finite"],
        "text_masks_nonempty": text_checks["nonempty_train_masks"] == len(queries),
        "teacher_mapping": teacher_missing == 0 and teacher_nonfinite == 0,
        "train_split_only": all(banks[v]["metadata"].get("split") == "train" for v in TRAIN_VIDEOS),
    }
    payload = {
        "schema_version": "locatemot-l46-end-to-end-query-region-track-contract-v1",
        "stage": "L46-A-train-only-contract-audit",
        "project_root": str(ROOT),
        "started_at": started,
        "completed_at": time.time(),
        "train_videos": list(TRAIN_VIDEOS),
        "train_video_count": len(TRAIN_VIDEOS),
        "expression_count": len(queries),
        "population_counts": dict(population),
        "per_video": per_video,
        "per_video_expression_counts": dict(per_video_queries),
        "candidate_count_quantiles": {
            "q0": int(np.min(frame_sizes)), "q50": float(np.median(frame_sizes)),
            "q90": float(np.quantile(frame_sizes, .90)), "q99": float(np.quantile(frame_sizes, .99)),
            "q100": int(np.max(frame_sizes)),
        },
        "source_diagnostics": {
            "counts": dict(source_counts),
            "semantic_input": False,
            "used_only_for_stratification": True,
        },
        "hard_negative_definition": hard_definition,
        "bank_contract": {
            "path": str(L19.resolve()), "row_key": "(video,frame,candidate_index,track_id,observation_row)",
            "label_sidecars_are_gt_derived": True, "missing_fields": missing_fields,
            "duplicate_candidate_indices": duplicate_rows, "raw_image_root": str(RAW_ROOT),
            "duplicate_identity": "(frame,pool_id,track_id,candidate_index)",
            "boundary_clipped_boxes": boundary_clipped_boxes,
            "crop_rule": "existing frozen crop helper may be used by B0; A only checks reversible image/box mapping",
            "persistent_dense_cache_written": False,
        },
        "sequence_cache_contract": {
            "path": str(L28.resolve()), "manifest_sha256": sha256(L28 / "manifest.json"),
            "feature_dim": int(caches[TRAIN_VIDEOS[0]]["obs_features"].shape[1]),
            "history_length": int(caches[TRAIN_VIDEOS[0]]["history_length"]),
            "track_frame_alignment_errors": len(cache_alignment_errors),
            "row_key": "(video,track_id,observation_index) with frame carried by observation",
        },
        "text_contract": {
            "path": str((V5 / "text_tokens.pt").resolve()),
            "sha256": sha256(V5 / "text_tokens.pt"),
            **text_checks,
            "query_source": "train expressions only",
            "attribute_word_counts": dict(query_attribute_counts),
        },
        "teacher_contract": {
            "checkpoint": str(L29.resolve()), "sha256": sha256(L29),
            "role": "auxiliary distillation/control only; not the sole L46 emission",
            "mapping": "L29 current_membership_logits mapped to candidate track_id",
            "sample_count": len(teacher_examples), "missing_candidate_rows": teacher_missing,
            "nonfinite_replays": teacher_nonfinite, "examples": teacher_examples,
        },
        "fixed_fast_manifest": {
            "path": str(FAST.resolve()), "sha256": sha256(FAST),
            "expected_sha256": "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa",
            "query_count": len(fast["queries"]), "calibration_queries": 64,
            "screening_queries": 96, "used_for_training": False,
            "used_for_structure_or_threshold_selection": False,
            "screening_gt_used": False,
        },
        "frozen_runtime_anchors": anchors,
        "labels": {
            "current_membership": "EXPRESSION_LEVEL_VERIFIED / GT-derived frame-to-track membership",
            "multi_positive": "GT_PRIVILEGED_ORACLE for training labels; every positive retained",
            "same_frame_negative": "GT_PRIVILEGED_ORACLE non-positive candidate; no verified class field",
            "inactive_null": "GT-derived no-target/inactive frame",
            "fragment_continuation": "GT_PRIVILEGED_ORACLE from train-only L28 observation history",
            "token_span_region": "UNALIGNED",
            "static_motion_language_mask": "UNALIGNED / not claimed",
        },
        "audit_checks": {
            **decision_checks,
            "missing_images_count": len(missing_images),
            "invalid_boxes_count": len(invalid_boxes),
            "boundary_clipped_boxes_count": len(boundary_clipped_boxes),
            "cache_alignment_errors_count": len(cache_alignment_errors),
            "nonfinite_fields": dict(nonfinite),
            "representative_units": representative_units,
        },
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key"],
        "decision": "enter_b0_smoke" if all(decision_checks.values()) and fast["queries"] else "incomplete",
        "elapsed_sec": time.time() - started,
    }
    output = OUT / "end_to_end_contract.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (OUT / "README.md").write_text(
        "# L46 end-to-end query-region-track contract\n\n"
        "Train-only audit. It checks frozen L19/L28 features, expression-level labels, "
        "word-token masks, raw-image reversibility, L29 track-id mapping and screening isolation. "
        "No training or TrackEval is performed.\n"
    )
    if payload["decision"] != "enter_b0_smoke":
        (OUT / "INCOMPLETE.md").write_text(
            "# INCOMPLETE\n\nL46 contract checks failed; see end_to_end_contract.json. "
            "No training was started.\n"
        )
    report = [
        "# Stage L46 — End-to-end query/region/track contract audit", "",
        f"- Decision: **{payload['decision']}**",
        f"- Train videos: {len(queries) and len(TRAIN_VIDEOS)}; expressions: {len(queries)}",
        f"- Frame units: {population['frame_units']:,}; candidate rows: {population['candidate_rows']:,}; positive rows: {population['positive_rows']:,}",
        f"- Multi-positive units: {population['multi_positive_frame_units']:,}; inactive/no-target units: {population['inactive_or_null_frame_units']:,}",
        f"- Same-frame positive-negative pairs: {population['positive_negative_pairs']:,}; continuation-positive rows: {continuation_positive_rows:,}",
        f"- Candidate count q0/q50/q90/q99/q100: {payload['candidate_count_quantiles']['q0']}/{payload['candidate_count_quantiles']['q50']}/{payload['candidate_count_quantiles']['q90']}/{payload['candidate_count_quantiles']['q99']}/{payload['candidate_count_quantiles']['q100']}",
        "", "## Contract checks", "",
        f"- Missing images: {len(missing_images)}; invalid boxes: {len(invalid_boxes)}; boundary-clipped boxes: {len(boundary_clipped_boxes)}; duplicate stable row keys: {duplicate_rows}",
        f"- L19↔L28 track/frame alignment errors: {len(cache_alignment_errors)}; nonfinite fields: {dict(nonfinite)}",
        f"- L29 sample missing candidate rows: {teacher_missing}; nonfinite teacher replays: {teacher_nonfinite}",
        f"- Text hidden/mask shapes: {text_checks['hidden_shape']} / {text_checks['mask_shape']}; nonempty train masks: {text_checks['nonempty_train_masks']}/{len(queries)}",
        "- Source/pool/group/state IDs are excluded from semantic inputs; source is diagnostic only.",
        "- Fixed 64/96 manifest is recorded but not used for training, structure or threshold selection; screening GT is not read by the audit.",
        "", "## Provenance", "",
        f"- L19 bank: `{L19.resolve()}`", f"- L28 sequence cache: `{L28.resolve()}`",
        f"- L29 teacher: `{L29.resolve()}` (SHA256 `{sha256(L29)}`)",
        f"- Text cache: `{(V5 / 'text_tokens.pt').resolve()}`",
        f"- Raw image root: `{RAW_ROOT}`; no pixels or dense cache were persisted.",
        f"- Machine-readable output: `{output.resolve()}`", "",
        "Expression-level frame-to-track labels are valid RMOT supervision. Token/span→region and static/motion language masks remain UNALIGNED; this audit makes no verified word-level claim.",
    ]
    REPORT.write_text("\n".join(report) + "\n")
    print(json.dumps({"decision": payload["decision"], "output": str(output),
                      "report": str(REPORT), "audit_checks": payload["audit_checks"],
                      "population_counts": dict(population)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
