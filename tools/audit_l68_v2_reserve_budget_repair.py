#!/usr/bin/env python3
"""Read-only simulation of the L18 reserve-budget sweep for L68.

This deliberately does not run a detector or materialize a bank.  It repeats
the proposal ordering and exact reserve duplicate rule from
``build_l18_dual_track_bank.py`` and joins the simulated rows to the frozen
L16 main rows only for post-hoc coverage auditing.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA = ROOT / "outputs/l49/data"
MAIN_ROOT = ROOT / "outputs/l16/track_banks_dedup/kitti"
DINO_PATH = ROOT / "outputs/l18/cache/dino_kitti_trainval.pkl"
RECORD_ROOT = ROOT / "outputs/l16/data/kitti_missing/records"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L67_ROOT = ROOT / "outputs/l67/audit/v2_candidate_coverage_attempt7"
VIDEOS = ("0016", "0017", "0020")
BUDGETS = (5, 10, 20, 40, 80)
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DINO_SHA = "ce0cc5b342ecf0cd7195fd4f67fcc6ec1c915b170d501e3969b3b2e1c25e1c9d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_default(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-8, aa + bb - inter)


def match_gt(boxes, gt_by_frame, threshold):
    pairs = []
    for i, box in enumerate(boxes):
        for gid, target in gt_by_frame.items():
            value = iou(box, target)
            if value >= threshold:
                pairs.append((value, i, str(gid)))
    pairs.sort(reverse=True)
    used_i, used_g = set(), set()
    labels = [None] * len(boxes)
    for _value, i, gid in pairs:
        if i in used_i or gid in used_g:
            continue
        labels[i] = gid
        used_i.add(i)
        used_g.add(gid)
    return labels


def load_pickle_compat(path):
    # Records/cache were produced under NumPy 2 in some runs.  This alias only
    # lets the read-only audit deserialize them; the source pickle is untouched.
    try:
        return pickle.load(path.open("rb"))
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        import numpy as np_local
        sys.modules["numpy._core"] = np_local.core
        sys.modules["numpy._core.numeric"] = np_local.core.numeric
        return pickle.load(path.open("rb"))


def load_units():
    rows = []
    for line in (DATA / "validation_units.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") == "refer_kitti_v2" and row.get("split") == "validation":
            if str(row["video"]) not in VIDEOS:
                raise AssertionError(f"unexpected validation video {row['video']}")
            rows.append(row)
    if len(rows) != 768:
        raise AssertionError(f"expected 768 V2 validation units, got {len(rows)}")
    return rows


def unit_key(row):
    return str(row["unit_key"])


def l67_expected():
    p = L67_ROOT / "coverage.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    x = json.loads(p.read_text())
    c = x["coverage_ceiling"]
    return {
        "target_present_units": int(c["target_present_units"]),
        "covered_units": int(c["covered_target_present_units"]),
        "unit_coverage": float(c["target_present_unit_coverage"]),
        "target_ids": int(c["target_ids"]),
        "covered_target_ids": int(c["covered_target_ids"]),
        "target_micro": float(c["target_level_micro_coverage"]),
        "present_uncovered": int(c["present_uncovered_units"]),
        "inactive": int(c["inactive_units"]),
    }


def l67_records():
    p = L67_ROOT / "unit_records.jsonl"
    result = {}
    for line in p.read_text().splitlines():
        if line.strip():
            x = json.loads(line)
            result[x["unit_key"]] = x
    return result


def load_video(video, dino):
    main_path = MAIN_ROOT / f"{video}.pt"
    label_path = main_path.with_suffix(".labels.json")
    bank = torch.load(main_path, map_location="cpu")
    t = bank["tensors"]
    main_labels = json.loads(label_path.read_text())["candidate_gt"]
    if len(main_labels) != int(t["track_id"].numel()):
        raise AssertionError(f"main label length mismatch {video}")
    record = load_pickle_compat(RECORD_ROOT / f"{video}.pkl")
    gt_by_frame = {int(x["frame"]): {str(k): v for k, v in x.get("gt_boxes", {}).items()}
                   for x in record["frames"]}
    frame_ids = [int(x) for x in t["frame_ids"].tolist()]
    if frame_ids != list(range(len(frame_ids))):
        raise AssertionError(f"unexpected non-contiguous frame ids {video}")
    dino_video = dino.get(video)
    if not isinstance(dino_video, dict):
        raise AssertionError(f"missing DINO video {video}")
    frames = {}
    for frame in frame_ids:
        entry = dino_video.get(frame)
        if entry is None:
            raise AssertionError(f"missing DINO frame {video}/{frame}")
        boxes = np.asarray(entry.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(entry.get("scores", []), dtype=np.float32).reshape(-1)
        if len(boxes) != len(scores) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise AssertionError(f"invalid DINO arrays {video}/{frame}")
        frames[frame] = {"boxes": boxes, "scores": scores}
    return {"bank": bank, "tensor": t, "main_labels": main_labels,
            "gt": gt_by_frame, "dino": frames, "main_path": str(main_path)}


def reserve_for(entry, budget, gt):
    boxes, scores = entry["boxes"], entry["scores"]
    order = np.argsort(-scores, kind="stable")[:budget]
    selected = boxes[order]
    selected_scores = scores[order]
    keep, seen = [], set()
    for i, box in enumerate(selected):
        key = np.asarray(box, dtype=np.float32).tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    if keep:
        kept = selected[np.asarray(keep, dtype=np.int64)]
        kept_scores = selected_scores[np.asarray(keep, dtype=np.int64)]
    else:
        kept = np.zeros((0, 4), np.float32)
        kept_scores = np.zeros((0,), np.float32)
    labels = match_gt(kept, gt, 0.5)
    return kept, kept_scores, labels, order[:len(selected)][keep] if keep else np.zeros(0, np.int64)


def frame_rows(video_data, frame, budget):
    t = video_data["tensor"]
    fi = frame
    begin, end = int(t["frame_ptr"][fi]), int(t["frame_ptr"][fi + 1])
    main_boxes = t["box"][begin:end].float().numpy().astype(np.float32)
    main = [{"source": "main", "pool": 0, "box": box, "gt": video_data["main_labels"][begin + i],
             "candidate_index": int(t["candidate_index"][begin + i]),
             "track_id": int(t["track_id"][begin + i]), "raw_rank": None}
            for i, box in enumerate(main_boxes)]
    dino_entry = video_data["dino"][frame]
    reserve, scores, labels, raw_indices = reserve_for(dino_entry, budget, video_data["gt"].get(frame, {}))
    reserve_rows = [{"source": "reserve", "pool": 1, "box": box, "gt": labels[i],
                     "candidate_index": i, "track_id": 1000000 + i,
                     "raw_rank": int(raw_indices[i]) + 1, "score": float(scores[i])}
                    for i, box in enumerate(reserve)]
    return main + reserve_rows, {"raw_count": len(dino_entry["boxes"]),
                                "reserve_count": len(reserve),
                                "exact_duplicates_removed": int(min(budget, len(dino_entry["boxes"])) - len(reserve)),
                                "raw_boxes": dino_entry["boxes"], "raw_scores": dino_entry["scores"]}


def stats(values):
    vals = [float(x) for x in values]
    if not vals:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"count": len(vals), "mean": float(statistics.fmean(vals)),
            "p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95)),
            "max": float(max(vals))}


def audit_unit(unit, rows, video_data, budget, l67_by_key, frame_info):
    targets = {str(x) for x in unit.get("target_ids", [])}
    positive = [i for i, r in enumerate(rows) if r["gt"] is not None and str(r["gt"]) in targets]
    covered_ids = sorted({str(rows[i]["gt"]) for i in positive})
    source_sets = {s: sorted({str(r["gt"]) for r in rows if r["source"] == s and r["gt"] in targets})
                   for s in ("main", "reserve")}
    gt_boxes = video_data["gt"].get(int(unit["frame_id"]), {})
    threshold_hits = {}
    for threshold in (0.50, 0.75):
        threshold_hits[str(threshold)] = sorted({str(gid) for gid in targets
            if any(iou(r["box"], gt_boxes.get(str(gid), ())) >= threshold for r in rows
                   if str(gid) in gt_boxes)})
    boxes = [r["box"].tolist() for r in rows]
    pair_count = 0
    for i in positive:
        for j, box in enumerate(boxes):
            if j not in positive and iou(boxes[i], box) >= 0.30:
                pair_count += 1
    candidates = [r["candidate_index"] for r in rows]
    tracks = [r["track_id"] for r in rows]
    cross50 = 0
    cross70 = 0
    for r in rows:
        if r["source"] != "reserve":
            continue
        overlap = max((iou(r["box"], x["box"]) for x in rows if x["source"] == "main"), default=0.0)
        cross50 += overlap >= 0.50
        cross70 += overlap >= 0.70
    rec = {
        "format": "locatemot-l68-v2-budget-unit-v1", "status": "complete",
        "unit_key": unit_key(unit), "dataset": unit["dataset"], "video": str(unit["video"]),
        "query_id": int(unit["query_id"]), "frame_id": int(unit["frame_id"]),
        "category": str(unit.get("category", "unknown")), "budget": int(budget),
        "target_present": bool(targets), "target_ids": sorted(targets),
        "target_id_count": len(targets), "covered_target_ids": covered_ids,
        "target_ids_covered": len(covered_ids),
        "iou50_target_ids_covered": threshold_hits["0.5"],
        "iou75_target_ids_covered": threshold_hits["0.75"],
        "iou50_unit_covered": bool(threshold_hits["0.5"]),
        "iou75_unit_covered": bool(threshold_hits["0.75"]),
        "target_id_coverage": len(covered_ids) / max(1, len(targets)) if targets else None,
        "candidate_covered": bool(covered_ids), "candidate_count": len(rows),
        "positive_count": len(positive), "present_uncovered": bool(targets and not covered_ids),
        "inactive": not bool(targets), "empty_candidate": not bool(rows),
        "duplicate_candidate_index_count": len(rows) - len(set(candidates)),
        "duplicate_track_id_count": len(rows) - len(set(tracks)),
        "cross_pool_iou50_duplicate_rows": int(cross50),
        "cross_pool_iou70_duplicate_rows": int(cross70),
        "main_candidate_count": sum(r["source"] == "main" for r in rows),
        "reserve_candidate_count": sum(r["source"] == "reserve" for r in rows),
        "main_covered_target_ids": source_sets["main"], "reserve_covered_target_ids": source_sets["reserve"],
        "same_frame_positive_negative_iou_ge_030": int(pair_count),
        "candidate_rows": [[unit["dataset"], str(unit["video"]), int(unit["query_id"]), int(unit["frame_id"]),
                             int(r["candidate_index"]), int(r["track_id"]), r["source"], r["raw_rank"],
                             int(i)] for i, r in enumerate(rows)],
        "reserve_exact_duplicates_removed": int(frame_info["exact_duplicates_removed"]),
        "labels_posthoc_only": True, "candidate_truncation": False,
        "key_contract": "dataset,video,query_id,frame_id,candidate_index,track_id,source,raw_rank",
        "l67_budget20_reference": l67_by_key.get(unit_key(unit)) if budget == 20 else None,
    }
    return rec


def rank_records(units, video_data_by_video, budget):
    result = []
    for unit in units:
        video_data = video_data_by_video[str(unit["video"])]
        entry = video_data["dino"][int(unit["frame_id"])]
        rows = []
        for idx, (box, score) in enumerate(zip(entry["boxes"], entry["scores"])):
            labels = match_gt(np.asarray([box], np.float32), video_data["gt"].get(int(unit["frame_id"]), {}), .5)
            if labels[0] is not None and str(labels[0]) in {str(x) for x in unit.get("target_ids", [])}:
                rank = idx + 1
                bucket = "1-20" if rank <= 20 else "21-40" if rank <= 40 else "41-80" if rank <= 80 else "81-300"
                result.append({"unit_key": unit_key(unit), "video": str(unit["video"]),
                               "frame_id": int(unit["frame_id"]), "target_id": str(labels[0]),
                               "first_hit_rank": rank, "rank_bin": bucket, "raw_score": float(score),
                               "budget": int(budget)})
                break
        else:
            result.append({"unit_key": unit_key(unit), "video": str(unit["video"]),
                           "frame_id": int(unit["frame_id"]), "first_hit_rank": None,
                           "rank_bin": "no_hit_in_300", "raw_score": None, "budget": int(budget)})
    return result


def summarize(records):
    present = [r for r in records if r["target_present"]]
    covered = [r for r in present if r["candidate_covered"]]
    target_total = sum(r["target_id_count"] for r in present)
    target_hit = sum(r["target_ids_covered"] for r in present)
    def one(xs):
        p = [r for r in xs if r["target_present"]]
        c = [r for r in p if r["candidate_covered"]]
        tt = sum(r["target_id_count"] for r in p)
        th = sum(r["target_ids_covered"] for r in p)
        iou50 = [r for r in p if r["iou50_unit_covered"]]
        iou75 = [r for r in p if r["iou75_unit_covered"]]
        def source_one(source):
            covered_lists = [r["main_covered_target_ids"] if source == "main" else r["reserve_covered_target_ids"] for r in p]
            ids = sum(len(g) for g in covered_lists)
            return {"covered_units": sum(bool(r["main_covered_target_ids"] if source == "main" else r["reserve_covered_target_ids"]) for r in p),
                    "unit_coverage": sum(bool(r["main_covered_target_ids"] if source == "main" else r["reserve_covered_target_ids"]) for r in p) / max(1, len(p)),
                    "covered_target_ids": ids, "target_micro_coverage": ids / max(1, tt)}
        return {"units": len(xs), "target_present_units": len(p), "covered_units": len(c),
                "unit_coverage": len(c) / max(1, len(p)), "inactive_units": sum(not r["target_present"] for r in xs),
                "present_uncovered_units": sum(r["present_uncovered"] for r in xs),
                "target_ids": tt, "covered_target_ids": th,
                "target_micro_coverage": th / max(1, tt),
                "target_macro_coverage": float(np.mean([r["target_id_coverage"] for r in p])) if p else None,
                "iou50_unit_coverage": len(iou50) / max(1, len(p)),
                "iou75_unit_coverage": len(iou75) / max(1, len(p)),
                "iou50_target_micro_coverage": sum(len(r["iou50_target_ids_covered"]) for r in p) / max(1, tt),
                "iou75_target_micro_coverage": sum(len(r["iou75_target_ids_covered"]) for r in p) / max(1, tt),
                "main_only": source_one("main"), "reserve_only": source_one("reserve"),
                "candidate_count": stats([r["candidate_count"] for r in xs]),
                "reserve_rows_per_frame": stats([r["reserve_candidate_count"] for r in xs]),
                "union_rows_per_frame": stats([r["candidate_count"] for r in xs]),
                "positive_count": stats([r["positive_count"] for r in xs]),
                "duplicate_candidate_index_rows": sum(r["duplicate_candidate_index_count"] for r in xs),
                "reserve_exact_duplicates_removed": sum(r["reserve_exact_duplicates_removed"] for r in xs),
                "cross_pool_iou50_duplicate_rows": sum(r["cross_pool_iou50_duplicate_rows"] for r in xs),
                "cross_pool_iou70_duplicate_rows": sum(r["cross_pool_iou70_duplicate_rows"] for r in xs)}
    return {"all": one(records), "by_video": {v: one([r for r in records if r["video"] == v]) for v in VIDEOS},
            "by_category": {c: one([r for r in records if r["category"] == c]) for c in ("positive", "multi_positive", "present_uncovered", "inactive")},
            "coverage": {"target_present_units": len(present), "covered_units": len(covered),
                         "unit_coverage": len(covered) / max(1, len(present)), "target_ids": target_total,
                         "covered_target_ids": target_hit, "target_micro": target_hit / max(1, target_total),
                         "target_macro": float(np.mean([r["target_id_coverage"] for r in present])) if present else None,
                         "present_uncovered_units": sum(r["present_uncovered"] for r in records),
                         "inactive_units": sum(not r["target_present"] for r in records)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.time()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("manifest SHA mismatch")
        if sha256(DINO_PATH) != DINO_SHA:
            raise AssertionError("DINO cache SHA mismatch")
        units = load_units()
        expected = l67_expected()
        l67_all = l67_records()
        v2_keys = {unit_key(x) for x in units}
        if not v2_keys.issubset(set(l67_all)):
            missing = sorted(v2_keys - set(l67_all))[:4]
            raise AssertionError(f"L67 V2 record keys missing: {missing}")
        # L67 intentionally appended fixed-slice V1 context rows.  They are
        # not part of this primary V2 reproduction and are never aggregated.
        l67_by_key = {key: l67_all[key] for key in v2_keys}
        dino = load_pickle_compat(DINO_PATH)
        if "__meta__" not in dino:
            raise AssertionError("DINO provenance metadata missing")
        video_data = {}
        for video in VIDEOS:
            video_data[video] = load_video(video, dino)
            print(f"[l68] loaded {video}", flush=True)
        all_summaries, all_records = {}, {}
        for budget in BUDGETS:
            records = []
            for unit in units:
                vd = video_data[str(unit["video"])]
                rows, _info = frame_rows(vd, int(unit["frame_id"]), budget)
                records.append(audit_unit(unit, rows, vd, budget, l67_by_key, _info))
            all_records[str(budget)] = records
            all_summaries[str(budget)] = summarize(records)
        raw_records = rank_records(units, video_data, 300)
        base20 = all_summaries["20"]["coverage"]
        checks = {"target_present_units": base20["target_present_units"] == expected["target_present_units"],
                  "covered_units": base20["covered_units"] == expected["covered_units"],
                  "unit_coverage": abs(base20["unit_coverage"] - expected["unit_coverage"]) < 1e-12,
                  "target_ids": base20["target_ids"] == expected["target_ids"],
                  "covered_target_ids": base20["covered_target_ids"] == expected["covered_target_ids"],
                  "target_micro": abs(base20["target_micro"] - expected["target_micro"]) < 1e-12,
                  "present_uncovered": base20["present_uncovered_units"] == expected["present_uncovered"],
                  "inactive": base20["inactive_units"] == expected["inactive"]}
        if not all(checks.values()):
            raise AssertionError(f"budget20 reproduction failed: {checks}; got={base20}; expected={expected}")
        rows20 = all_records["20"]
        rows40 = all_records["40"]
        rescue = []
        for a, b in zip(rows20, rows40):
            if a["unit_key"] != b["unit_key"]:
                raise AssertionError("unit ordering drift")
            rescue.append({"unit_key": a["unit_key"], "video": a["video"], "category": a["category"],
                           "budget20_covered": a["candidate_covered"], "budget40_covered": b["candidate_covered"],
                           "rescued_unit": (not a["candidate_covered"] and b["candidate_covered"]),
                           "target_ids": a["target_ids"], "budget20_missing": sorted(set(a["target_ids"]) - set(a["covered_target_ids"])),
                           "budget40_missing": sorted(set(b["target_ids"]) - set(b["covered_target_ids"])),
                           "budget40_rescued_target_ids": sorted(set(b["covered_target_ids"]) - set(a["covered_target_ids"]))})
        added40 = []
        for unit in units:
            vd = video_data[str(unit["video"])]
            r20, _ = frame_rows(vd, int(unit["frame_id"]), 20)
            r40, _ = frame_rows(vd, int(unit["frame_id"]), 40)
            old = {tuple(np.asarray(r["box"], np.float32).tolist()) for r in r20 if r["source"] == "reserve"}
            for r in r40:
                if r["source"] == "reserve" and tuple(np.asarray(r["box"], np.float32).tolist()) not in old:
                    added40.append({"unit_key": unit_key(unit), "video": str(unit["video"]), "raw_rank": r["raw_rank"], "score": r["score"],
                                    "box": np.asarray(r["box"], np.float32).tolist(), "gt": r["gt"]})
        # Estimate from frozen L19 metadata, not a materialized budget-40 bank.
        # Read only compact L19 audit metadata for the frozen budget-20
        # denominator; do not load or create a new dual bank.
        l19_meta = {}
        for video in VIDEOS:
            meta_path = ROOT / "outputs/l19/dual_banks_features/kitti" / f"{video}.audit.json"
            l19_meta[video] = json.loads(meta_path.read_text())
        estimates = {}
        for budget in BUDGETS:
            vals = all_summaries[str(budget)]["all"]["reserve_rows_per_frame"]
            total_frames = sum(len(video_data[v]["dino"]) for v in VIDEOS)
            reserve_est = vals["mean"] * total_frames
            main_rows = sum(int(video_data[v]["tensor"]["track_id"].numel()) for v in VIDEOS)
            l19_rows = sum(int(l19_meta[v]["observations"]) for v in VIDEOS)
            estimates[str(budget)] = {"estimated_v2_reserve_observations": reserve_est,
                                      "frozen_l19_v2_rows": l19_rows, "frozen_l19_main_rows": main_rows,
                                      "estimated_union_rows": main_rows + reserve_est,
                                      "estimated_row_multiplier_vs_l19": (main_rows + reserve_est) / max(1, l19_rows),
                                      "estimated_disk_multiplier_vs_l19": (main_rows + reserve_est) / max(1, l19_rows),
                                      "estimated_clip_crop_forward_multiplier_vs_l19": reserve_est / max(1, main_rows),
                                      "note": "estimate only; no bank/crop feature was materialized"}
        decision = "budget40_candidate_repair_supported"
        if not checks["target_present_units"] or all_summaries["40"]["coverage"]["unit_coverage"] < .7233333 or all_summaries["40"]["coverage"]["target_micro"] < .80:
            decision = "budget40_not_supported"
        # L19 baseline row count is the frozen union; require <=2x.
        if estimates["40"]["estimated_row_multiplier_vs_l19"] > 2.0:
            decision = "budget40_not_supported"
        if len(rescue) != 768 or len({x["unit_key"] for x in rescue}) != 768:
            raise AssertionError("rescue matrix incomplete")
        raw_target_present = [r for r in raw_records
                              if l67_by_key[r["unit_key"]].get("target_ids")]
        raw_target_missing = sum(r["first_hit_rank"] is None for r in raw_target_present)
        failure_reasons = {}
        for budget in BUDGETS:
            rs = all_records[str(budget)]
            failure_reasons[str(budget)] = {
                "no_candidate_rows": {"count": sum(r["empty_candidate"] for r in rs), "denominator": len(rs), "examples": [r["unit_key"] for r in rs if r["empty_candidate"]][:8]},
                "candidate_rows_but_no_target_id": {"count": sum(r["present_uncovered"] for r in rs), "denominator": sum(r["target_present"] for r in rs), "examples": [r["unit_key"] for r in rs if r["present_uncovered"]][:8]},
                "sidecar_mismatch": {"count": 0, "denominator": len(rs), "examples": []},
                "target_metadata_missing": {"count": 0, "denominator": len(rs), "examples": []},
                "inactive_not_a_miss": True,
            }
        added_boxes = [np.asarray(x["box"], np.float32) for x in added40]
        sizes = [[max(0.0, float(b[2] - b[0])), max(0.0, float(b[3] - b[1]))] for b in added_boxes]
        image_sizes = {v: video_data[v]["bank"].get("metadata", {}).get("image_size", [0, 0]) for v in VIDEOS}
        boundary = []
        for x, b in zip(added40, added_boxes):
            width, height = image_sizes[x["video"]] if x["video"] in image_sizes else [0, 0]
            boundary.append(bool(b[0] <= 0 or b[1] <= 0 or b[2] >= width or b[3] >= height))
        coverage = {"format": "locatemot-l68-v2-reserve-budget-sweep-v1", "status": "complete",
                    "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                    "inputs": {"dino_cache": str(DINO_PATH), "dino_cache_sha256": DINO_SHA,
                               "l16_main_banks": [str(MAIN_ROOT / f"{v}.pt") for v in VIDEOS],
                               "l16_records": [str(RECORD_ROOT / f"{v}.pkl") for v in VIDEOS],
                               "validation_units": str(DATA / "validation_units.jsonl"),
                               "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA,
                               "l67_budget20": str(L67_ROOT)},
                    "outputs": [str(out / x) for x in ("coverage_budget_sweep.json", "unit_budget_records.jsonl", "rank_rescue_records.jsonl", "provenance.json", "status.json")],
                    "failure_root_cause": None, "next_action": None,
                    "scope": {"dataset": "refer_kitti_v2", "split": "validation", "videos": list(VIDEOS), "units": 768},
                    "protocol": {"budgets": list(BUDGETS), "stable_descending_scores": True,
                                 "exact_duplicate_rule": "float32 box.tobytes within reserve", "cross_pool_overlap_retained": True,
                                 "iou_thresholds": [.50, .75], "labels_posthoc_only": True,
                                 "raw_oracle": "top-300 first-hit ranks only; not formal budget"},
                    "budget_summaries": all_summaries, "materialization_estimates": estimates,
                    "rescue": {"budget40_vs20": rescue, "rescued_units": sum(x["rescued_unit"] for x in rescue),
                               "rescued_target_ids": len({g for x in rescue for g in x["budget40_rescued_target_ids"]})},
                    "raw_oracle_top300": {"records": len(raw_records), "target_present_records": len(raw_target_present),
                                          "target_present_no_hit_in_300": raw_target_missing,
                                          "rank_bin_counts": dict(Counter(x["rank_bin"] for x in raw_records)),
                                          "target_present_rank_bin_counts": dict(Counter(x["rank_bin"] for x in raw_target_present)),
                                          "first_hit_rank_records": raw_records},
                    "top40_added_candidates": {"count": len(added40), "score": stats([x["score"] for x in added40]),
                                               "gt_matched_posthoc": sum(x["gt"] is not None for x in added40),
                                               "width": stats([x[0] for x in sizes]), "height": stats([x[1] for x in sizes]),
                                               "boundary_touch_fraction": sum(boundary) / max(1, len(boundary)),
                                               "records": added40[:2000]},
                    "failure_reason_decomposition": failure_reasons,
                    "budget20_reproduction": {"checks": checks, "expected": expected, "observed": base20},
                    "decision": decision,
                    "labels": {"source": "L16 sidecar/main and record gt_boxes, only after candidate construction; ORACLE audit", "screening_gt_used": False, "official_test_labels_read": False},
                    "training_run": False, "detector_forward": False, "clip_forward": False, "bank_materialized": False,
                    "dense_or_raw_cache_written": False, "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False}
        out_records = []
        for budget in BUDGETS:
            out_records.extend(all_records[str(budget)])
        (out / "unit_budget_records.jsonl").write_text("".join(json.dumps(x, default=json_default, separators=(",", ":")) + "\n" for x in out_records))
        (out / "rank_rescue_records.jsonl").write_text("".join(json.dumps(x, default=json_default, separators=(",", ":")) + "\n" for x in raw_records))
        (out / "coverage_budget_sweep.json").write_text(json.dumps(coverage, indent=2, default=json_default) + "\n")
        prov = {"format": "locatemot-l68-provenance-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                "inputs": coverage["inputs"], "outputs": coverage["outputs"], "numpy_pickle_alias_used": True,
                "builder_source": str(ROOT / "tools/build_l18_dual_track_bank.py"), "builder_source_sha256": sha256(ROOT / "tools/build_l18_dual_track_bank.py"),
                "pool_mapping": {"0": "main", "1": "reserve", "basis": "L19 builder source and frozen metadata"},
                "budgets": list(BUDGETS), "labels_posthoc_oracle_only": True, "training_run": False, "detector_forward": False, "clip_forward": False,
                "bank_materialized": False, "dense_or_raw_cache_written": False, "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "decision": decision, "next_action": None}
        if decision == "budget40_candidate_repair_supported":
            prov["next_action"] = "separate authorized RMOT-only budget-40 bank materialization; not performed in L68"
        else:
            prov["next_action"] = "proposal-quality/rank analysis; do not select budget80 automatically"
        (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=json_default) + "\n")
        status = {"format": "locatemot-l68-status-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                  "inputs": coverage["inputs"], "outputs": coverage["outputs"], "failure_root_cause": None, "next_action": prov["next_action"],
                  "decision": decision, "elapsed_sec": time.time() - started, "training_run": False, "detector_forward": False,
                  "clip_forward": False, "bank_materialized": False, "screening_gt_used": False, "official_test_labels_read": False,
                  "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False}
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps({"status": "complete", "decision": decision, "budget20": base20, "budget40": all_summaries["40"]["coverage"], "output": str(out)}, indent=2), flush=True)
    except Exception as exc:
        failure = {"format": "locatemot-l68-status-v1", "status": "INCOMPLETE", "project_root": str(ROOT), "cwd": str(Path.cwd()), "command": command,
                   "inputs": {"dino_cache": str(DINO_PATH), "manifest": str(MANIFEST)}, "outputs": [str(out / "INCOMPLETE.md")],
                   "failure_root_cause": repr(exc), "next_action": "inspect first actionable root cause and run one new attempt", "training_run": False,
                   "detector_forward": False, "clip_forward": False, "screening_gt_used": False, "official_test_labels_read": False,
                   "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False}
        (out / "status.json").write_text(json.dumps(failure, indent=2) + "\n")
        (out / "INCOMPLETE.md").write_text("# L68 INCOMPLETE\n\n```text\n" + traceback.format_exc() + "\n```\n")
        raise


if __name__ == "__main__":
    main()
