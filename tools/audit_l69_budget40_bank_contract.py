#!/usr/bin/env python3
"""Audit the L69 budget-40 dual bank and its L19 identity view.

The audit is deliberately post-materialization and read-only with respect to
all L16/L18/L19 inputs.  GT sidecars are used only for coverage and identity
oracle descriptions after row construction has been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MAIN_ROOT = ROOT / "outputs/l16/track_banks_dedup/kitti"
RECORD_ROOT = ROOT / "outputs/l16/data/kitti_missing/records"
L11_RECORD_ROOT = ROOT / "outputs/l11/data/rmot_kitti"
DINO_PATH = ROOT / "outputs/l18/cache/dino_kitti_trainval.pkl"
CLIP_PATH = Path("/home/lwr/.cache/clip/ViT-B-32.pt")
UNITS_PATH = ROOT / "outputs/l49/data/validation_units.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_DINO = "ce0cc5b342ecf0cd7195fd4f67fcc6ec1c915b170d501e3969b3b2e1c25e1c9d"
EXPECTED_CLIP = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
VIDEOS = ("0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018", "0020")
V2_VALIDATION = ("0016", "0017", "0020")
MAIN_FIELDS = ("frame", "candidate_index", "track_id", "box", "objectness", "clip", "history_clip", "pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd", "geometry", "motion", "context", "lifecycle", "frame_ptr", "frame_ids")
FEATURE_FIELDS = ("frame", "candidate_index", "track_id", "box", "objectness", "clip", "history_clip", "pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd", "geometry", "motion", "context", "lifecycle", "pool_id", "source_score", "frame_ptr", "frame_ids", "raw_rank")
# L19 intentionally replaces these reserve-side memory fields with its
# nonzero identity views.  Equality is therefore checked for the invariant
# observation fields and for the complete pool-0 subsequence; reserve memory
# is audited separately for schema/nonzero values.
FEATURE_INVARIANT_FIELDS = ("frame", "candidate_index", "track_id", "box", "objectness", "clip", "geometry", "context", "pool_id", "source_score", "frame_ptr", "frame_ids", "raw_rank")
PICKLE_ALIAS_USED = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pickle_compat(path: Path):
    global PICKLE_ALIAS_USED
    try:
        return pickle.load(path.open("rb"))
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        import numpy as np_local
        sys.modules["numpy._core"] = np_local.core
        sys.modules["numpy._core.numeric"] = np_local.core.numeric
        PICKLE_ALIAS_USED = True
        return pickle.load(path.open("rb"))


def iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-8, aa + bb - inter)


def stats(values):
    x = [float(v) for v in values]
    if not x:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"count": len(x), "mean": float(statistics.fmean(x)),
            "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "max": float(max(x))}


def load_units():
    rows = []
    for line in UNITS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") == "refer_kitti_v2" and row.get("split") == "validation":
            if str(row["video"]) not in V2_VALIDATION:
                raise AssertionError(f"unexpected V2 validation video {row['video']}")
            rows.append(row)
    if len(rows) != 768:
        raise AssertionError(f"expected 768 V2 validation rows, got {len(rows)}")
    return rows


def frame_rows(tensors, frame_index):
    ptr = tensors["frame_ptr"].long()
    begin, end = int(ptr[frame_index]), int(ptr[frame_index + 1])
    return begin, end


def source_record(video):
    path = RECORD_ROOT / f"{video}.pkl"
    if not path.exists():
        path = L11_RECORD_ROOT / f"{video}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path, load_pickle_compat(path)


def raw_unique(video_dino, frame, budget=40):
    entry = video_dino[frame]
    boxes = np.asarray(entry.get("boxes", []), np.float32).reshape(-1, 4)
    scores = np.asarray(entry.get("scores", []), np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")[:budget]
    seen, unique = set(), []
    for raw_index in order.tolist():
        key = np.asarray(boxes[raw_index], np.float32).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append((int(raw_index), boxes[raw_index], float(scores[raw_index])))
    return unique


def check_finite(tensors, names):
    result = {}
    for name in names:
        if name not in tensors:
            continue
        value = tensors[name]
        if not torch.is_floating_point(value):
            continue
        finite = bool(torch.isfinite(value.float()).all())
        result[name] = finite
        if not finite:
            raise AssertionError(f"nonfinite tensor {name}")
    return result


def audit_video(video, dual_path, feature_path, dino_video, units):
    main_path = MAIN_ROOT / f"{video}.pt"
    main_labels_path = main_path.with_suffix(".labels.json")
    main_blob = torch.load(main_path, map_location="cpu")
    dual_blob = torch.load(dual_path, map_location="cpu")
    feature_blob = torch.load(feature_path, map_location="cpu")
    main, dual, feature = main_blob["tensors"], dual_blob["tensors"], feature_blob["tensors"]
    main_labels = json.loads(main_labels_path.read_text())["candidate_gt"]
    dual_labels = json.loads(dual_path.with_suffix(".labels.json").read_text())["candidate_gt"]
    feature_labels = json.loads(feature_path.with_suffix(".labels.json").read_text())["candidate_gt"]
    if dual_labels != feature_labels:
        raise AssertionError(f"dual/feature label sidecar mismatch {video}")
    if len(main_labels) != int(main["track_id"].numel()):
        raise AssertionError(f"L16 label length mismatch {video}")
    if len(dual_labels) != int(dual["track_id"].numel()) or len(feature_labels) != int(feature["track_id"].numel()):
        raise AssertionError(f"new label length mismatch {video}")
    if int(dual_blob["metadata"].get("reserve_budget", -1)) != 40 or int(feature_blob["metadata"].get("reserve_budget", -1)) != 40:
        raise AssertionError(f"budget metadata mismatch {video}")
    if not (dual_blob["metadata"].get("causal") and dual_blob["metadata"].get("query_independent") and dual_blob["metadata"].get("rmot_only_reserve_namespace")):
        raise AssertionError(f"dual metadata flags mismatch {video}")
    if not (feature_blob["metadata"].get("causal") and feature_blob["metadata"].get("query_independent") and feature_blob["metadata"].get("rmot_only_reserve_namespace")):
        raise AssertionError(f"feature metadata flags mismatch {video}")
    if not feature_blob["metadata"].get("preserve_source_ids"):
        raise AssertionError(f"source IDs were not preserved {video}")
    check_finite(main, MAIN_FIELDS)
    check_finite(dual, FEATURE_FIELDS)
    check_finite(feature, FEATURE_FIELDS)
    if set(dual.keys()) - set(feature.keys()) - {"observation_group_id", "cross_pool_duplicate"}:
        raise AssertionError(f"feature fields missing dual fields {video}")
    if dual["frame_ids"].tolist() != feature["frame_ids"].tolist() or dual["frame_ptr"].tolist() != feature["frame_ptr"].tolist():
        raise AssertionError(f"feature temporal order mismatch {video}")
    pool = dual["pool_id"].long()
    main_rows = torch.nonzero(pool == 0, as_tuple=False).flatten()
    reserve_rows = torch.nonzero(pool == 1, as_tuple=False).flatten()
    if len(main_rows) != int(main["track_id"].numel()):
        raise AssertionError(f"main row count changed {video}")
    if len(reserve_rows) != int(dual_blob["metadata"].get("reserve_observations", -1)):
        raise AssertionError(f"reserve metadata row count mismatch {video}")
    # Every frozen L16 field must be byte-identical in the pool-0 subsequence.
    for name in MAIN_FIELDS:
        if name not in main or name not in dual:
            raise AssertionError(f"missing main field {name} {video}")
        left, right = main[name], dual[name][main_rows] if name not in ("frame_ptr", "frame_ids") else dual[name]
        if name in ("frame_ptr", "frame_ids"):
            # The dual pointer includes reserve rows, so compare frame IDs and
            # reconstruct main counts instead of comparing its pointer.
            if name == "frame_ids" and left.tolist() != right.tolist():
                raise AssertionError(f"main frame_ids changed {video}")
            continue
        if left.dtype != right.dtype or left.shape != right.shape or left.contiguous().numpy().tobytes() != right.contiguous().numpy().tobytes():
            raise AssertionError(f"main byte equality failed {video}/{name}")
    main_label_subsequence = [dual_labels[int(i)] for i in main_rows.tolist()]
    if main_label_subsequence != main_labels:
        raise AssertionError(f"main labels changed {video}")
    # The identity view must preserve every common dual-bank value and sidecar.
    for name in FEATURE_INVARIANT_FIELDS:
        if name not in dual or name not in feature:
            raise AssertionError(f"feature field missing {video}/{name}")
        if dual[name].dtype != feature[name].dtype or dual[name].shape != feature[name].shape or dual[name].contiguous().numpy().tobytes() != feature[name].contiguous().numpy().tobytes():
            raise AssertionError(f"dual/feature field order mismatch {video}/{name}")
    for name in ("frame", "candidate_index", "track_id", "box", "objectness", "clip", "geometry", "context", "pool_id", "source_score", "raw_rank"):
        if dual[name][main_rows].contiguous().numpy().tobytes() != feature[name][main_rows].contiguous().numpy().tobytes():
            raise AssertionError(f"feature main subsequence changed {video}/{name}")
    for name in ("pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd", "history_clip", "motion", "lifecycle"):
        value = feature[name][reserve_rows].float()
        if value.ndim == 2:
            nonzero = int((value.abs().sum(dim=1) > 1e-6).sum())
        else:
            nonzero = int((value.abs().reshape(len(value), -1).sum(dim=1) > 1e-6).sum())
        if name in ("pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd") and nonzero != len(reserve_rows):
            raise AssertionError(f"reserve identity field has zero rows {video}/{name}")
    frame_ids = [int(x) for x in dual["frame_ids"].tolist()]
    frame_map = {frame: i for i, frame in enumerate(frame_ids)}
    frame_audit = []
    raw_duplicate_total = 0
    cross_pool_iou50_rows = 0
    cross_pool_iou70_rows = 0
    reserve_raw_ranks = []
    for fi, frame in enumerate(frame_ids):
        start, end = frame_rows(dual, fi)
        if dual["frame"][start:end].long().tolist() != [frame] * (end - start):
            raise AssertionError(f"frame row mismatch {video}/{frame}")
        raws = raw_unique(dino_video, frame, 40)
        reserve_indices = [i for i in range(start, end) if int(pool[i]) == 1]
        if len(reserve_indices) != len(raws):
            raise AssertionError(f"raw top40/dedup count mismatch {video}/{frame}")
        ranks = dual["raw_rank"][reserve_indices].long().tolist()
        expected_ranks = [raw_index + 1 for raw_index, _box, _score in raws]
        if ranks != expected_ranks:
            raise AssertionError(f"raw rank/order mismatch {video}/{frame}")
        raw_duplicate_count = min(40, len(dino_video[frame]["boxes"])) - len(raws)
        raw_duplicate_total += raw_duplicate_count
        reserve_raw_ranks.extend(ranks)
        frame_main_boxes = dual["box"][start:end][pool[start:end] == 0].float().tolist()
        frame_reserve_boxes = dual["box"][start:end][pool[start:end] == 1].float().tolist()
        frame_iou50 = sum(any(iou(box, other) >= .50 for other in frame_main_boxes)
                          for box in frame_reserve_boxes)
        frame_iou70 = sum(any(iou(box, other) >= .70 for other in frame_main_boxes)
                          for box in frame_reserve_boxes)
        cross_pool_iou50_rows += frame_iou50
        cross_pool_iou70_rows += frame_iou70
        frame_audit.append({"frame_id": frame, "main_rows": sum(int(pool[i]) == 0 for i in range(start, end)),
                            "reserve_rows": len(reserve_indices), "raw_top40_unique": len(raws),
                            "raw_exact_duplicates_removed": raw_duplicate_count,
                            "cross_pool_iou50_rows": frame_iou50,
                            "cross_pool_iou70_rows": frame_iou70,
                            "reserve_raw_rank_min": min(ranks) if ranks else None,
                            "reserve_raw_rank_max": max(ranks) if ranks else None,
                            "candidate_count": end - start})
    # Descriptive identity oracle, explicitly not a model metric.
    labels = [None if x is None else str(x) for x in feature_labels]
    tracks = feature["track_id"].long().tolist()
    sources = feature["pool_id"].long().tolist()
    gt_frames = defaultdict(lambda: defaultdict(set))
    gt_tracks = defaultdict(set)
    for idx, gid in enumerate(labels):
        if gid is not None:
            frame = int(feature["frame"][idx])
            gt_frames[gid][int(sources[idx])].add(frame)
            gt_tracks[gid].add(int(tracks[idx]))
    gt_both = [g for g, p in gt_frames.items() if 0 in p and 1 in p]
    gt_later = [g for g in gt_both if max(gt_frames[g][1]) > min(gt_frames[g][0])]
    switch_total, switch_changes = 0, 0
    for gid in set(labels) - {None}:
        seq = defaultdict(list)
        for idx, value in enumerate(labels):
            if value == gid:
                seq[int(feature["frame"][idx])].append(int(tracks[idx]))
        ordered = sorted((f, min(ids)) for f, ids in seq.items())
        for (_fa, ta), (_fb, tb) in zip(ordered, ordered[1:]):
            switch_total += 1
            switch_changes += ta != tb
    same_frame_iou50 = 0
    same_frame_iou70 = 0
    duplicate_candidate = 0
    duplicate_candidate_by_pool = {"main": 0, "reserve": 0}
    for fi, _frame in enumerate(frame_ids):
        start, end = frame_rows(feature, fi)
        ci = feature["candidate_index"][start:end].long().tolist()
        for pool_value, pool_name in ((0, "main"), (1, "reserve")):
            pool_ci = feature["candidate_index"][start:end][feature["pool_id"][start:end] == pool_value].long().tolist()
            duplicate_candidate_by_pool[pool_name] += len(pool_ci) - len(set(pool_ci))
        duplicate_candidate += len(ci) - len(set(ci))
        boxes = feature["box"][start:end].float().tolist()
        for i in range(end - start):
            if sources[start + i] != 1:
                continue
            same_frame_iou50 += any(iou(boxes[i], boxes[j]) >= .50 for j in range(end - start) if sources[start + j] == 0)
            same_frame_iou70 += any(iou(boxes[i], boxes[j]) >= .70 for j in range(end - start) if sources[start + j] == 0)
    positive_reserve_ranks = [int(feature["raw_rank"][i]) for i, gid in enumerate(feature_labels)
                              if int(sources[i]) == 1 and gid is not None and int(feature["raw_rank"][i]) > 0]
    rank_bins = {"1-20": 0, "21-40": 0, "41-80": 0, "81-300": 0, "gt_positive": len(positive_reserve_ranks)}
    for rank in positive_reserve_ranks:
        if rank <= 20:
            rank_bins["1-20"] += 1
        elif rank <= 40:
            rank_bins["21-40"] += 1
        elif rank <= 80:
            rank_bins["41-80"] += 1
        elif rank <= 300:
            rank_bins["81-300"] += 1
    return {
        "format": "locatemot-l69-budget40-per-video-v1", "status": "complete", "video": video,
        "dual_bank": str(dual_path), "feature_bank": str(feature_path),
        "dual_sha256": sha256(dual_path), "feature_sha256": sha256(feature_path),
        "main_rows": int(len(main_rows)), "reserve_rows": int(len(reserve_rows)), "rows": int(len(dual_rows := dual["track_id"])),
        "frames": len(frame_ids), "tracks": len(set(tracks)),
        "reserve_budget": 40, "frame_audit": frame_audit,
        "reserve_rows_per_frame": stats([x["reserve_rows"] for x in frame_audit]),
        "union_rows_per_frame": stats([x["candidate_count"] for x in frame_audit]),
        "raw_exact_duplicates_removed": int(raw_duplicate_total),
        "cross_pool_iou50_rows": int(cross_pool_iou50_rows),
        "cross_pool_iou70_rows": int(cross_pool_iou70_rows),
        "reserve_raw_rank_distribution": {str(x): reserve_raw_ranks.count(x) for x in sorted(set(reserve_raw_ranks))},
        "gt_positive_reserve_raw_rank_bins": rank_bins,
        "raw_rank_contract": True, "main_byte_equality": True, "feature_common_fields_byte_equal": True,
        "reserve_identity_nonzero": {name: int((feature[name][reserve_rows].float().abs().reshape(len(reserve_rows), -1).sum(dim=1) > 1e-6).sum()) for name in ("pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd")},
        "identity_oracle": {"gt_with_main_and_reserve": len(gt_both), "gt_with_later_reserve": len(gt_later),
                            "gt_fragment_count_distribution": stats([len(x) for x in gt_tracks.values()]),
                            "track_switch_proxy": switch_changes / max(1, switch_total), "track_switch_transitions": switch_total,
                            "same_frame_cross_pool_iou50_rows": same_frame_iou50,
                            "same_frame_cross_pool_iou70_rows": same_frame_iou70,
                            "duplicate_candidate_index_rows": duplicate_candidate,
                            "duplicate_candidate_index_rows_by_pool": duplicate_candidate_by_pool},
        "units_seen": len(units),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual-root", required=True, type=Path)
    ap.add_argument("--feature-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--scope", choices=("targeted", "full"), required=True)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = " ".join([sys.executable] + sys.argv)
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        if (sha256(MANIFEST) != EXPECTED_MANIFEST or
                sha256(DINO_PATH) != EXPECTED_DINO or
                not CLIP_PATH.is_file() or sha256(CLIP_PATH) != EXPECTED_CLIP):
            raise AssertionError("frozen hash mismatch")
        videos = list(dict.fromkeys(args.videos))
        expected = list(V2_VALIDATION if args.scope == "targeted" else VIDEOS)
        if videos != expected:
            raise AssertionError(f"video scope/order mismatch: {videos} != {expected}")
        if any(v not in VIDEOS for v in videos):
            raise AssertionError("official-eval or out-of-scope video requested")
        units = load_units()
        dino = load_pickle_compat(DINO_PATH)
        per_video = []
        for video in videos:
            dual = (args.dual_root if args.dual_root.is_absolute() else ROOT / args.dual_root).resolve() / f"{video}.pt"
            feature = (args.feature_root if args.feature_root.is_absolute() else ROOT / args.feature_root).resolve() / f"{video}.pt"
            if not dual.is_file() or not feature.is_file():
                raise FileNotFoundError(f"missing L69 bank {video}: {dual} / {feature}")
            per_video.append(audit_video(video, dual, feature, dino[video], [x for x in units if str(x["video"]) == video]))
            print(f"[l69-audit] checked {video}", flush=True)
        # Coverage is only defined for the V2 validation rows and is post-hoc.
        coverage_rows = []
        for video in V2_VALIDATION:
            dual_path = (args.dual_root if args.dual_root.is_absolute() else ROOT / args.dual_root).resolve() / f"{video}.pt"
            if not dual_path.is_file():
                if args.scope == "targeted":
                    raise FileNotFoundError(dual_path)
                continue
            blob = torch.load(dual_path, map_location="cpu")
            t, labels = blob["tensors"], json.loads(dual_path.with_suffix(".labels.json").read_text())["candidate_gt"]
            frame_map = {int(x): i for i, x in enumerate(t["frame_ids"].tolist())}
            for unit in [x for x in units if str(x["video"]) == video]:
                fi = frame_map[int(unit["frame_id"])]
                start, end = frame_rows(t, fi)
                target = {str(x) for x in unit.get("target_ids", [])}
                present = sorted({str(g) for g in labels[start:end] if g is not None and str(g) in target})
                coverage_rows.append({"unit_key": unit["unit_key"], "video": video, "category": unit.get("category"),
                                     "target_ids": sorted(target), "covered_target_ids": present,
                                     "target_present": bool(target), "candidate_covered": bool(present),
                                     "candidate_count": end - start})
            del blob, t, labels
        present = [x for x in coverage_rows if x["target_present"]]
        covered = [x for x in present if x["candidate_covered"]]
        target_total = sum(len(x["target_ids"]) for x in present)
        target_hit = sum(len(x["covered_target_ids"]) for x in present)
        coverage = {"units": len(coverage_rows), "target_present_units": len(present), "covered_units": len(covered),
                    "unit_coverage": len(covered) / max(1, len(present)), "target_ids": target_total,
                    "covered_target_ids": target_hit, "target_micro_coverage": target_hit / max(1, target_total),
                    "present_uncovered_units": sum(x["target_present"] and not x["candidate_covered"] for x in coverage_rows),
                    "inactive_units": sum(not x["target_present"] for x in coverage_rows),
                    "label_use": "post-hoc coverage/oracle only", "scope": "V2 validation 0016/0017/0020"}
        output_files = ["contract.json", "per_video.jsonl", "identity_oracle.json", "provenance.json", "status.json", "v2_validation_coverage.json"]
        contract = {"format": "locatemot-l69-budget40-contract-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                    "scope": args.scope, "videos": videos, "reserve_budget": 40,
                    "inputs": {"l16_main": str(MAIN_ROOT), "dino_cache": str(DINO_PATH), "dino_sha256": EXPECTED_DINO,
                               "clip_weight": str(CLIP_PATH), "clip_sha256": EXPECTED_CLIP,
                               "record_compatibility_interpreter": "/home/lwr/anaconda3/envs/masaenv_debug/bin/python",
                               "train_pool_records": str(RECORD_ROOT), "validation_units": str(UNITS_PATH), "manifest": str(MANIFEST), "manifest_sha256": EXPECTED_MANIFEST},
                    "outputs": [str(out / x) for x in output_files], "failure_root_cause": None, "next_action": "separate RMOT-only persistent identity/semantic fast probe",
                    "construction": {"stable_descending_score": True, "top_k": 40, "reserve_exact_duplicate": "float32 box.tobytes", "cross_pool_overlap_retained": True,
                                     "causal_linker": "L18 IoU/CLIP linker, max_gap=2", "reserve_id_offset": 1000000,
                                     "feature_view": "L19 reserve_identity_features max_gap=2 preserve_source_ids"},
                    "main_equality": all(x["main_byte_equality"] for x in per_video), "feature_equality": all(x["feature_common_fields_byte_equal"] for x in per_video),
                    "per_video": per_video, "v2_validation_coverage": coverage,
                    "flags": {"screening_gt_used": False, "official_test_labels_read": False, "training_run": False, "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False, "detector_forward": False, "clip_forward": True, "clip_crop_forward_during_materialization": True, "dense_or_raw_cache_written": False}}
        (out / "per_video.jsonl").write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in per_video))
        identity = {"format": "locatemot-l69-identity-oracle-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                    "label_type": "GT_PRIVILEGED_ORACLE", "not_model_performance": True, "per_video": [{"video": x["video"], **x["identity_oracle"]} for x in per_video],
                    "inputs": contract["inputs"], "outputs": [str(out / "identity_oracle.json")], "failure_root_cause": None,
                    "next_action": contract["next_action"], "screening_gt_used": False, "official_test_labels_read": False,
                    "training_run": False, "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False}
        provenance = {"format": "locatemot-l69-provenance-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                      "inputs": contract["inputs"], "outputs": contract["outputs"], "scope": args.scope, "videos": videos,
                      "clip_weight": str(CLIP_PATH), "clip_sha256": EXPECTED_CLIP,
                      "l16_main_bank_hashes": {v: sha256(MAIN_ROOT / f"{v}.pt") for v in videos},
                      "old_l19_unchanged_hashes": {v: sha256(ROOT / "outputs/l19/dual_banks_features/kitti" / f"{v}.pt") for v in videos if (ROOT / "outputs/l19/dual_banks_features/kitti" / f"{v}.pt").exists()},
                      "source_record_policy": "L16 missing-records first; existing L11 train-pool records only for non-missing train videos; no official records",
                      "numpy_pickle_alias_used": PICKLE_ALIAS_USED,
                      "numpy_pickle_compatibility": "DINO cache/read-only; L11 records were read in the verified masaenv_debug subprocess during materialization",
                      "main_source_frozen": True, "no_source_write_attempt": True,
                      "screening_gt_used": False, "official_test_labels_read": False, "training_run": False, "hota_trackeval_run": False,
                      "ordinary_mot_ovmot_touched": False, "dense_or_raw_cache_written": False, "failure_root_cause": None,
                      "next_action": contract["next_action"]}
        (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (out / "v2_validation_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        (out / "identity_oracle.json").write_text(json.dumps(identity, indent=2) + "\n")
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        status = {"format": "locatemot-l69-status-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                  "inputs": contract["inputs"], "outputs": contract["outputs"], "failure_root_cause": None, "next_action": contract["next_action"],
                  "scope": args.scope, "videos": videos, "elapsed_sec": time.time() - started, **contract["flags"]}
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    except Exception:
        (out / "status.json").write_text(json.dumps({"format": "locatemot-l69-status-v1", "status": "INCOMPLETE", "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                                                       "outputs": [str(out / "INCOMPLETE.md")], "failure_root_cause": traceback.format_exc(), "next_action": "fix only first actionable contract error in a new attempt",
                                                       "training_run": False, "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False}, indent=2) + "\n")
        (out / "INCOMPLETE.md").write_text("# L69 INCOMPLETE\n\n```text\n" + traceback.format_exc() + "\n```\n")
        raise


if __name__ == "__main__":
    main()
