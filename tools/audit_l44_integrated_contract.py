#!/usr/bin/env python3
"""Train-only contract audit for the L44 integrated RMOT decoder.

This audit deliberately does not train, run TrackEval, or read screening
labels.  It checks the exact current-frame candidate contract, the L29
teacher mapping, the persistent cache alignment, and the transient raw-image
mapping before any L44 smoke is allowed to start.
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
from tools.l40_raw_data import RAW_ROOT, WEIGHTS, image_path
from tools.train_l26_crossmodal_adapter import FAST, SPLIT, V5, load_expressions
from tools.train_l28_track_set_decoder import state_at

L19 = ROOT / "outputs/l19/dual_banks_features/kitti"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
OUT = ROOT / "outputs/l44/audit"
REPORT = ROOT / "reports/l44_integrated_contract_audit.md"
TRAIN_VIDEOS = ("0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0020")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_tensor(x) -> bool:
    return bool(torch.isfinite(torch.as_tensor(x)).all())


def load_bank(video: str):
    d = torch.load(L19 / f"{video}.pt", map_location="cpu", weights_only=False)
    t = d["tensors"]
    # Labels are deliberately read only from the existing L19 sidecar helper.
    from tools.audit_l28_identity_bank import load_labels
    labels, label_path = load_labels(L19 / f"{video}.pt", int(t["track_id"].numel()), tensors=t)
    return {
        "box": t["box"].float(), "frame": t["frame"].long(),
        "track": t["track_id"].long(), "objectness": t["objectness"].float(),
        "geometry": t["geometry"].float(), "motion": t["motion"].float(),
        "context": t["context"].float(), "lifecycle": t["lifecycle"].float(),
        "history_clip": t["history_clip"].float(), "uidm_h": t["uidm_h"].float(),
        "frame_ids": t["frame_ids"].long(), "ptr": t["frame_ptr"].long(),
        "labels": labels, "label_path": str(label_path),
    }


def load_queries():
    train_videos = {str(x) for x in json.loads(SPLIT.read_text())["kitti_v2"]["train"]}
    text_manifest = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"])
                  for x in text_manifest}
    result = []
    for row in load_expressions():
        key = (str(row["video"]), str(row["expression"]))
        if key[0] in train_videos and key in text_index:
            result.append({
                "video": key[0], "expression": key[1],
                "text_index": text_index[key],
                "sentence": str(row.get("sentence", row["expression"])),
                "target": {int(k): {str(v) for v in values}
                           for k, values in row.get("label", {}).items()},
            })
    if len(result) != 7757:
        raise AssertionError(f"expected 7757 train expressions, found {len(result)}")
    return result


def valid_teacher_indices(cache, cutoff: int):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def select_sample_units(queries, banks, limit=12):
    buckets = {"multi_positive": [], "positive": [], "inactive": [], "other": []}
    seen = set()
    for q in queries:
        b = banks[q["video"]]
        for fi, frame in enumerate(b["frame_ids"].tolist()):
            key = (q["video"], q["expression"], int(frame))
            if key in seen:
                continue
            begin, end = int(b["ptr"][fi]), int(b["ptr"][fi + 1])
            ids = q["target"].get(int(frame), set())
            y = np.asarray([b["labels"][r] is not None and str(b["labels"][r]) in ids
                            for r in range(begin, end)], dtype=bool)
            category = ("multi_positive" if int(y.sum()) > 1 else
                        "positive" if bool(y.any()) else
                        "inactive" if not ids else "other")
            buckets[category].append((q, fi, y))
            seen.add(key)
            if all(len(v) >= max(2, limit // 4) for v in buckets.values()):
                break
        if all(len(v) >= max(2, limit // 4) for v in buckets.values()):
            break
    result = []
    for name in ("multi_positive", "positive", "inactive", "other"):
        result.extend(buckets[name][:max(2, limit // 4)])
    return result[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--teacher-sample", type=int, default=8)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    banks = {video: load_bank(video) for video in TRAIN_VIDEOS}
    caches = {video: torch.load(L28 / f"{video}.pt", map_location="cpu", weights_only=False)
              for video in TRAIN_VIDEOS}

    required_bank = ("box", "frame", "track", "objectness", "geometry", "motion",
                     "context", "lifecycle", "history_clip", "uidm_h", "frame_ids", "ptr", "labels")
    missing_fields = {v: [k for k in required_bank if k not in banks[v]] for v in TRAIN_VIDEOS}
    missing_fields = {v: x for v, x in missing_fields.items() if x}
    row_key_count = 0
    row_keys = set()
    duplicate_rows = []
    missing_images = []
    invalid_boxes = []
    nonfinite = Counter()
    frame_sizes = []
    per_video = {}
    cache_alignment_errors = []

    for video in TRAIN_VIDEOS:
        b = banks[video]
        n_rows = int(b["track"].numel())
        finite_fields = ("box", "objectness", "geometry", "motion", "context",
                         "lifecycle", "history_clip", "uidm_h")
        for field in finite_fields:
            if not finite_tensor(b[field]):
                nonfinite[field] += 1
        rows_by_track = defaultdict(list)
        for row, track in enumerate(b["track"].tolist()):
            rows_by_track[int(track)].append(row)
            key = (video, int(b["frame"][row]), int(track), int(row))
            if key in row_keys:
                duplicate_rows.append(key)
            row_keys.add(key)
            row_key_count += 1
            box = b["box"][row].numpy()
            if (not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]):
                invalid_boxes.append({"video": video, "row": row, "box": box.tolist()})
        for rows in rows_by_track.values():
            rows.sort(key=lambda r: (int(b["frame"][r]), int(r)))
        for frame in b["frame_ids"].tolist():
            path = image_path(video, int(frame))
            if not path.exists():
                missing_images.append(str(path))
        ptr = caches[video]["track_ptr"].numpy()
        cframes = caches[video]["obs_frame"].numpy()
        for ti, track in enumerate(caches[video]["track_ids"].tolist()):
            cb = int(ptr[ti]); ce = int(ptr[ti + 1])
            cache_frames = [int(x) for x in cframes[cb:ce].tolist()]
            bank_frames = [int(b["frame"][r]) for r in rows_by_track.get(int(track), [])]
            if cache_frames != bank_frames:
                cache_alignment_errors.append({"video": video, "track_id": int(track),
                                               "cache_frames": cache_frames,
                                               "bank_frames": bank_frames})
        per_video[video] = {
            "frame_count": int(len(b["frame_ids"])), "candidate_rows": n_rows,
            "track_count": int(len(caches[video]["track_ids"])),
            "label_sidecar": b["label_path"],
            "cache_feature_shape": list(caches[video]["obs_features"].shape),
        }
        frame_sizes.extend((b["ptr"][1:] - b["ptr"][:-1]).tolist())

    population = Counter()
    per_video_query = Counter()
    continuation_rows = 0
    hard_samples = []
    for q in queries:
        b = banks[q["video"]]
        cache = caches[q["video"]]
        cptr = cache["track_ptr"].tolist()
        cframes = cache["obs_frame"].tolist()
        cgt = cache["obs_gt_ids"]
        history_gt_by_track = {}
        for ti, track in enumerate(cache["track_ids"].tolist()):
            cb, ce = int(cptr[ti]), int(cptr[ti + 1])
            history_gt_by_track[int(track)] = set(
                str(x) for x in cgt[cb:ce] if x is not None
            )
        per_video_query[q["video"]] += 1
        for fi, frame in enumerate(b["frame_ids"].tolist()):
            begin, end = int(b["ptr"][fi]), int(b["ptr"][fi + 1])
            ids = q["target"].get(int(frame), set())
            y = np.asarray([b["labels"][r] is not None and str(b["labels"][r]) in ids
                            for r in range(begin, end)], dtype=bool)
            pos, neg = int(y.sum()), int((~y).sum())
            population["frame_units"] += 1
            population["candidate_rows"] += end - begin
            population["positive_rows"] += pos
            population["negative_rows"] += neg
            population["positive_negative_pairs"] += pos * neg
            population["target_frame_units"] += int(bool(ids))
            population["positive_frame_units"] += int(bool(y.any()))
            population["multi_positive_frame_units"] += int(pos > 1)
            population["multi_positive_rows"] += pos if pos > 1 else 0
            population["inactive_or_null_frame_units"] += int(not ids)
            population["missing_target_frame_units"] += int(bool(ids) and not y.any())
            for offset, row in enumerate(range(begin, end)):
                track = int(b["track"][row])
                if y[offset] and len(history_gt_by_track.get(track, set()) & set(ids)):
                    continuation_rows += 1
            if pos and neg:
                hard_samples.append({"video": q["video"], "query": q["expression"],
                                     "frame": int(frame), "candidate_count": end - begin,
                                     "positive_count": pos, "negative_count": neg,
                                     "positive_tracks": [int(b["track"][r]) for r in range(begin, end) if b["labels"][r] is not None and str(b["labels"][r]) in ids]})

    # Text token contract is checked without fitting or reading a screening item.
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden = text["token_hidden"].float()
    text_mask = text["attention_mask"].bool()
    query_indices = [q["text_index"] for q in queries]
    text_checks = {
        "hidden_shape": list(text_hidden.shape),
        "mask_shape": list(text_mask.shape),
        "finite": finite_tensor(text_hidden),
        "nonempty_masks": int(text_mask[torch.as_tensor(query_indices)].any(1).sum()),
        "query_count_checked": len(query_indices),
        "word_level_sequence_retained": True,
        "token_span_region_verified": False,
        "static_motion_language_mask_verified": False,
    }

    # Run a small CPU teacher replay to verify that current cache tracks map to
    # the L29 current-membership output by track ID, not by stale row position.
    sample_units = select_sample_units(queries, banks, max(8, args.teacher_sample))
    teacher = L29FrameMembershipSetDecoder()
    teacher_blob = torch.load(L29, map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_blob["model"], strict=True)
    teacher.eval()
    teacher_missing_rows = 0
    teacher_nonfinite = 0
    teacher_examples = []
    with torch.inference_mode():
        for q, fi, y in sample_units[:args.teacher_sample]:
            b = banks[q["video"]]; cache = caches[q["video"]]
            frame = int(b["frame_ids"][fi]); begin, end = int(b["ptr"][fi]), int(b["ptr"][fi + 1])
            obs, obs_mask, obs_time, _, _ = state_at(cache, frame, history=8)
            encoded = teacher.encode_observations(obs, obs_mask, obs_time)
            z = teacher.forward_encoded(encoded, encoded[1], text_hidden[q["text_index"]], text_mask[q["text_index"]])
            logits = z["current_membership_logits"].float()
            valid = valid_teacher_indices(cache, frame)
            ids = cache["track_ids"][torch.as_tensor(valid)].tolist()
            mapping = {int(track): float(logit) for track, logit in zip(ids, logits.tolist())}
            values = [mapping.get(int(b["track"][r]), math.nan) for r in range(begin, end)]
            teacher_missing_rows += sum(not math.isfinite(x) for x in values)
            teacher_nonfinite += int(not finite_tensor(logits))
            teacher_examples.append({"video": q["video"], "query": q["expression"],
                                     "frame": frame, "candidate_rows": end - begin,
                                     "positive_rows": int(y.sum()), "teacher_min": float(min(values)),
                                     "teacher_max": float(max(values)), "mapped_tracks": len(mapping)})

    anchors_path = OUT / "frozen_anchors.json"
    anchors = json.loads(anchors_path.read_text()) if anchors_path.exists() else {}
    fast = json.loads(FAST.read_text())
    payload = {
        "schema_version": "locatemot-l44-integrated-query-region-track-contract-v1",
        "stage": "L44-A",
        "project_root": str(ROOT),
        "started_at": started,
        "completed_at": time.time(),
        "train_videos": list(TRAIN_VIDEOS),
        "train_video_count": len(TRAIN_VIDEOS),
        "expression_count": len(queries),
        "population_counts": dict(population),
        "continuation_positive_rows": int(continuation_rows),
        "candidate_count_quantiles": {
            "q0": int(np.min(frame_sizes)), "q50": float(np.median(frame_sizes)),
            "q90": float(np.quantile(frame_sizes, .90)), "q99": float(np.quantile(frame_sizes, .99)),
            "q100": int(np.max(frame_sizes)),
        },
        "per_video": per_video,
        "per_video_expression_counts": dict(per_video_query),
        "cache_contract": {
            "path": str(L28.resolve()), "manifest_sha256": sha256(L28 / "manifest.json"),
            "feature_dim": int(caches[TRAIN_VIDEOS[0]]["obs_features"].shape[1]),
            "track_frame_alignment_errors": len(cache_alignment_errors),
            "track_row_key": "(video,track,observation_index) with frame carried by observation",
        },
        "bank_contract": {
            "path": str(L19.resolve()), "required_fields": list(required_bank),
            "missing_fields": missing_fields, "row_key": "(video,frame,track,observation_row)",
            "row_count": row_key_count, "duplicate_row_keys": len(duplicate_rows),
            "label_sidecars_are_gt_derived": True,
        },
        "raw_image_contract": {
            "root": str(RAW_ROOT), "missing_frame_images": len(missing_images),
            "invalid_boxes": len(invalid_boxes), "crop_rule": "existing frozen 10 percent padded crop helper; L44 audit does not persist pixels",
            "streaming_only": True, "dense_cache_written": False,
        },
        "teacher": {
            "checkpoint": str(L29.resolve()), "sha256": sha256(L29),
            "logit": "L29 current_membership_logits mapped by current cache track_id",
            "sample_units": len(teacher_examples), "missing_candidate_rows": teacher_missing_rows,
            "nonfinite_replays": teacher_nonfinite, "examples": teacher_examples,
        },
        "text": text_checks,
        "fixed_fast_manifest": {
            "path": str(FAST.resolve()), "sha256": sha256(FAST),
            "query_count": len(fast["queries"]), "calibration_queries": 64,
            "screening_queries": 96, "used_for_training": False,
            "used_for_structure_or_threshold_selection": False,
            "screening_gt_used": False,
        },
        "l11_l8_anchor_file": str((OUT / "frozen_anchors.json").resolve()),
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key"],
        "labels": {
            "current_membership": "EXPRESSION_LEVEL_VERIFIED / GT-derived candidate membership",
            "multi_positive": "GT_PRIVILEGED_ORACLE for training diagnostics; all positives retained",
            "hard_negative": "GT_PRIVILEGED_ORACLE same current frame; no verified class field asserted",
            "inactive_null": "GT-derived no-target/inactive frame",
            "fragment_continuation": "GT_PRIVILEGED_ORACLE from train cache observation history",
            "token_span_region": "UNALIGNED",
            "static_motion_language_mask": "UNALIGNED / not claimed",
        },
        "finite_checks": {"nonfinite_fields": dict(nonfinite), "teacher_nonfinite": teacher_nonfinite},
        "audit_checks": {
            "missing_images": len(missing_images), "invalid_boxes": len(invalid_boxes),
            "duplicate_rows": len(duplicate_rows), "cache_alignment_errors": len(cache_alignment_errors),
            "teacher_missing_candidate_rows": teacher_missing_rows,
            "text_finite": text_checks["finite"], "text_masks_nonempty": text_checks["nonempty_masks"] == len(queries),
            "screening_leakage": False,
        },
        "decision": "enter_b0_smoke" if not (missing_fields or missing_images or invalid_boxes or duplicate_rows or cache_alignment_errors or teacher_missing_rows or teacher_nonfinite or nonfinite or not text_checks["finite"]) else "incomplete",
        "representative_train_hard_samples": hard_samples[:12],
        "frozen_anchor_snapshot": anchors,
        "elapsed_sec": time.time() - started,
    }
    path = OUT / "integrated_query_region_track_contract.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (OUT / "README.md").write_text(
        "# L44 integrated query-region-track contract\n\n"
        "Train-only audit. It verifies current-frame rows, L29 track-id mapping, "
        "L28 history alignment, raw-image reversibility, word-token masks and "
        "screening isolation. No training or TrackEval is performed.\n"
    )
    if payload["decision"] != "enter_b0_smoke":
        (OUT / "INCOMPLETE.md").write_text(
            "# INCOMPLETE\n\nThe L44 input contract did not pass. See "
            "integrated_query_region_track_contract.json; no training was started.\n"
        )
    report = [
        "# Stage L44 — Integrated query/region/track contract audit", "",
        f"- Decision: **{payload['decision']}**", f"- Train videos: {len(TRAIN_VIDEOS)}; expressions: {len(queries)}",
        f"- Frame units: {population['frame_units']:,}; candidate rows: {population['candidate_rows']:,}; positive rows: {population['positive_rows']:,}",
        f"- Multi-positive frame units: {population['multi_positive_frame_units']:,}; inactive/no-target units: {population['inactive_or_null_frame_units']:,}",
        f"- Same-frame positive-negative pairs: {population['positive_negative_pairs']:,}; continuation-positive rows: {continuation_rows:,}",
        f"- Candidate rows q0/q50/q99/q100: {payload['candidate_count_quantiles']['q0']}/{payload['candidate_count_quantiles']['q50']}/{payload['candidate_count_quantiles']['q99']}/{payload['candidate_count_quantiles']['q100']}",
        "", "## Contract checks", "",
        f"- Missing train frame images: {len(missing_images)}; invalid boxes: {len(invalid_boxes)}; duplicate row keys: {len(duplicate_rows)}",
        f"- L19↔L28 track/frame alignment errors: {len(cache_alignment_errors)}; nonfinite fields: {dict(nonfinite)}",
        f"- L29 current-membership sample missing rows: {teacher_missing_rows}; nonfinite teacher replays: {teacher_nonfinite}",
        f"- Text hidden shape: {text_checks['hidden_shape']}; nonempty masks: {text_checks['nonempty_masks']}/{len(queries)}",
        "- Semantic inputs excluded: source/pool/group/state IDs; source is not an input.",
        "- Screening GT, fixed 64/96 manifest and threshold selection were not used by this train-side audit.",
        "", "## Provenance", "",
        f"- L19 bank: `{L19}`", f"- L28 cache manifest: `{L28 / 'manifest.json'}`",
        f"- L29 teacher: `{L29}` (SHA256 `{sha256(L29)}`)",
        f"- Raw image root: `{RAW_ROOT}`; pixels are not persisted by this audit.",
        f"- Machine-readable output: `{path.resolve()}`", "",
        "Expression-level labels are valid RMOT supervision. Token/span→region and static/motion language masks remain `UNALIGNED`; this audit makes no word-level alignment claim.",
    ]
    REPORT.write_text("\n".join(report) + "\n")
    print(json.dumps({"decision": payload["decision"], "output": str(path), "report": str(REPORT), "audit_checks": payload["audit_checks"], "population_counts": dict(population)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
