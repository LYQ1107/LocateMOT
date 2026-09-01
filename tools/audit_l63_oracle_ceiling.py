#!/usr/bin/env python3
"""Read-only L63 ceiling audit over the immutable L49 40-unit slice.

The detector ROI branch is streamed and discarded after compact statistics are
computed.  Labels are loaded only after each unit's feature construction.  No
score, threshold, candidate selection, or feature tensor is persisted.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import random
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA = ROOT / "outputs/l49/data"
BANK_ROOT = ROOT / "outputs/l19/dual_banks_features/kitti"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
FIXED_SCORE_ROWS = ROOT / "outputs/l59/eval/semantic_16cal_24val/score_records.jsonl"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
OUT = ROOT / "outputs/l63/audit/oracle_ceiling"
sys_path_added = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stat(values):
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None,
                "max": None, "p05": None, "p50": None, "p95": None}
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(a.size), "mean": float(a.mean()),
            "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "p05": float(np.quantile(a, .05)),
            "p50": float(np.quantile(a, .50)),
            "p95": float(np.quantile(a, .95))}


def cosine(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    den = max(1e-8, float(np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.dot(a, b) / den)


def auc(pos, neg):
    if not pos or not neg:
        return None
    values = sorted([(float(v), 1) for v in pos] + [(float(v), 0) for v in neg])
    neg_seen = wins = 0.0
    for value, label in values:
        del value
        if label:
            wins += neg_seen
        else:
            neg_seen += 1
    return float(wins / max(1, len(pos) * len(neg)))


def pr_auc(pos, neg):
    if not pos or not neg:
        return None
    values = sorted([(float(v), 1) for v in pos] + [(float(v), 0) for v in neg], reverse=True)
    positives = len(pos)
    tp = fp = 0
    points = [(0.0, 1.0)]
    for _, label in values:
        if label:
            tp += 1
        else:
            fp += 1
        points.append((tp / positives, tp / max(1, tp + fp)))
    return float(sum((x2 - x1) * (y1 + y2) * .5
                     for (x1, y1), (x2, y2) in zip(points, points[1:])))


def bootstrap_mean(values, seed=20260829, rounds=300):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return {"count": int(values.size), "mean": float(values.mean()) if values.size else None,
                "ci95": None, "finite_sample_note": "fewer than two units"}
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(values, size=values.size, replace=True).mean()
                          for _ in range(rounds)])
    return {"count": int(values.size), "mean": float(values.mean()),
            "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            "finite_sample_note": f"nonparametric bootstrap over {values.size} fixed units; not population inference"}


def load_fixed_units():
    rows = [json.loads(line) for line in FIXED_SCORE_ROWS.read_text().splitlines() if line.strip()]
    if len(rows) != 40:
        raise AssertionError(f"expected immutable 40 score rows, got {len(rows)}")
    wanted = {(str(r["unit_key"]), i < 16) for i, r in enumerate(rows)}
    units = {}
    for name in ("calibration_units.jsonl", "validation_units.jsonl"):
        for line in (DATA / name).read_text().splitlines():
            if not line.strip():
                continue
            unit = json.loads(line)
            key = str(unit["unit_key"])
            if any(key == x[0] for x in wanted):
                units[key] = unit
    result = []
    for i, score in enumerate(rows):
        key = str(score["unit_key"])
        if key not in units:
            raise KeyError(f"fixed score unit missing from L49 units: {key}")
        unit = dict(units[key])
        unit["audit_split"] = "calibration" if i < 16 else "validation"
        unit["fixed_order"] = i
        unit["historical_scores"] = {
            "l29": score.get("l29"), "l59_fused_roi": score.get("l59"),
            "l54_continuous": score.get("m54"), "l53_m0": score.get("m0")}
        result.append(unit)
    if len(result) != 40 or len({u["unit_key"] for u in result}) != 40:
        raise AssertionError("fixed unit key collision")
    return result


def vector_specs(tensors):
    names = ("clip", "history_clip", "uidm_h", "pbd", "uidm_ref_pbd", "uidm_anchor_pbd")
    specs = {}
    for name in names:
        if name not in tensors:
            continue
        value = tensors[name].float()
        if value.dim() != 2:
            continue
        specs[name] = value
    return specs


def load_sidecar(path, count):
    labels_path = path.with_suffix(".labels.json")
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    labels = json.loads(labels_path.read_text()).get("candidate_gt", [])
    if len(labels) != count:
        raise AssertionError(f"label rows {len(labels)} != bank rows {count}")
    return [None if x is None else str(x) for x in labels], labels_path


def pair_metrics(same, different):
    if not same or not different:
        return {"same_count": len(same), "different_count": len(different),
                "roc_auc": None, "pr_auc": None, "pair_order_violation": None}
    # Exact rank comparison without materializing the Cartesian product.
    # This is a deterministic audit statistic, not a learned selection rule.
    ordered_different = np.sort(np.asarray(different, dtype=np.float64))
    violating = sum(int(len(ordered_different) - np.searchsorted(ordered_different, value, side="left"))
                    for value in same)
    total = len(same) * len(different)
    return {"same_count": len(same), "different_count": len(different),
            "roc_auc": auc(same, different), "pr_auc": pr_auc(same, different),
            "pair_order_violation": float(violating / max(1, total)),
            "same_similarity": stat(same), "different_similarity": stat(different)}


def score_summary(records, field):
    unit_values = []
    all_pos = []
    all_neg = []
    top1 = top5 = 0
    present = 0
    selected_pos = selected = 0
    fp_frame = []
    strict = []
    best = []
    average = []
    multi = []
    inactive_max = []
    present_max = []
    by_category = defaultdict(list)
    for r in records:
        s = np.asarray(r["scores"][field], np.float64)
        y = np.asarray(r["label"], bool)
        if len(s) != len(y):
            raise AssertionError("score/label length drift")
        if not np.isfinite(s).all():
            raise AssertionError(f"nonfinite score {field}")
        has_pos = bool(y.any())
        if has_pos:
            present += 1
            order = np.argsort(-s, kind="stable")
            top1 += int(bool(y[order[0]]))
            top5 += int(bool(y[order[:5]].any()))
            pos, neg = s[y], s[~y]
            all_pos.extend(pos.tolist()); all_neg.extend(neg.tolist())
            strict.append(float(pos.min() - neg.max()) if len(neg) else None)
            best.append(float(pos.max() - neg.max()) if len(neg) else None)
            average.append(float(pos.mean() - neg.max()) if len(neg) else None)
            if r["category"] == "multi_positive":
                k = int(y.sum())
                multi.append(float(y[np.argsort(-s, kind="stable")[:k]].mean()))
            present_max.append(float(s.max()))
        else:
            inactive_max.append(float(s.max()))
        # Threshold-free candidate accounting is deliberately omitted here;
        # a ceiling audit must not invent a threshold.  Set recall at k=|P|
        # is a ranking statistic, not candidate deletion.
        by_category[r["category"]].append({"top1": int(has_pos and bool(y[np.argmax(s)])),
                                             "has_pos": int(has_pos)})
        unit_values.append({"top1": int(has_pos and bool(y[np.argmax(s)])),
                            "top5": int(has_pos and bool(y[np.argsort(-s, kind="stable")[:5]].any())),
                            "category": r["category"]})
    strict = [x for x in strict if x is not None]
    best = [x for x in best if x is not None]
    average = [x for x in average if x is not None]
    return {"units": len(records), "present_units": present,
            "candidate_rows": int(sum(len(r["label"]) for r in records)),
            "positive_rows": int(sum(sum(r["label"]) for r in records)),
            "top1": top1 / max(1, present), "top5": top5 / max(1, present),
            "top1_ci95": bootstrap_mean([x["top1"] for x in unit_values if x["top1"] is not None]),
            "candidate_score_auc": auc(all_pos, all_neg), "candidate_score_pr_auc": pr_auc(all_pos, all_neg),
            "strict_min_positive_margin": stat(strict),
            "best_positive_margin": stat(best), "average_positive_margin": stat(average),
            "hard_violation": float(np.mean([x <= 0 for x in strict])) if strict else None,
            "multi_positive_set_recall_at_positive_count": stat(multi),
            "inactive_max_score": stat(inactive_max), "present_max_score": stat(present_max),
            "by_category_ranking_units": {k: {"units": len(v),
                "top1": float(np.mean([x["top1"] for x in v])) if v else None}
                for k, v in sorted(by_category.items())}}


def svg_case(case, path):
    width, height = 1242, 375
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             f'<text x="8" y="18" font-size="14">{html.escape(case["unit_key"])} hard negative</text>']
    for label, box, color, score in (("positive", case["positive_box"], "#2563eb", case["positive_score"]),
                                     ("hard_negative", case["negative_box"], "#dc2626", case["negative_score"])):
        x1, y1, x2, y2 = box
        lines.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{max(1,x2-x1):.2f}" '
                     f'height="{max(1,y2-y1):.2f}" fill="none" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{max(2,x1):.2f}" y="{max(32,y1-3):.2f}" font-size="11" fill="{color}">' 
                     f'{label} score={score:.4f} gt={html.escape(str(case["positive_gt"]))}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty audit dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "representative_hard_negatives").mkdir()
    start = time.time()
    units = load_fixed_units()
    manifest_sha = sha256(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST:
        raise AssertionError(f"manifest sha mismatch: {manifest_sha}")
    # Import the verified fused-memory utility only after the fixed unit and
    # provenance contracts are established.
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.l59_fused_common import build_detector, detector_provenance, sha256 as helper_sha, stream_fused_roi

    detector, cfg, load = build_detector()
    provenance = {
        "format": "locatemot-l63-oracle-ceiling-provenance-v1",
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "manifest": str(MANIFEST), "manifest_sha256": manifest_sha,
        "fixed_score_source": str(FIXED_SCORE_ROWS), "fixed_score_sha256": sha256(FIXED_SCORE_ROWS),
        "fixed_calibration_units": 16, "fixed_validation_units": 24,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "persistent_dense_or_raw_cache_written": False,
        "gt_used_for": "post-feature oracle labels/metrics only",
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_language_mask": "UNALIGNED",
        "detector": detector_provenance(load),
        "historical_score_controls": "L29/L53/L54/L62 read-only fields from immutable L59 records",
        "feature_construction": "stream_fused_roi; ROI tensors deleted after compact statistics; no feature tensor persisted",
        "literature_status": "see reports/l63_oracle_ceiling.md; URLs/remote HEADs are structural references only",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    records = []
    identity_pairs = defaultdict(lambda: {"same_gt_cross_frame": [], "different_gt_same_frame": [],
                                           "same_gt_cross_fragment": []})
    selected_rows = defaultdict(list)
    hard_cases = []
    category_counts = Counter()
    coverage_rows = []
    expression_control = []
    control_keys = {str(units[i]["unit_key"]) for i in (0, 1, 2, 3, 16, 17, 18, 19)}
    for index, unit in enumerate(units):
        bank_path = Path(unit["bank_path"]).resolve()
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        tensors = bank["tensors"]
        begin, end = int(unit["begin"]), int(unit["end"])
        if end - begin != int(unit["candidate_count"]):
            raise AssertionError(f"candidate range drift {unit['unit_key']}")
        # Construct all visual/geometry features before opening the labels.
        specs = vector_specs(tensors)
        boxes = tensors["box"][begin:end].float().numpy()
        row_offsets = list(range(begin, end))
        row_keys = [(str(unit["video"]), int(unit["frame_id"]), str(bank_path), int(row)) for row in row_offsets]
        roi, fused_text, text_valid, numeric, meta = stream_fused_roi(detector, unit, bank)
        roi_np = roi.detach().cpu().numpy()
        text_np = fused_text.detach().cpu().numpy()
        if text_np.ndim != 3 or text_np.shape[0] != 1:
            raise AssertionError(f"unexpected fused text shape {text_np.shape}")
        text_tokens = text_np[0]
        text_mask_np = np.asarray(text_valid.detach().cpu()).reshape(-1).astype(bool)
        if text_mask_np.size != text_tokens.shape[0]:
            raise AssertionError(f"text mask/token drift {text_mask_np.size} vs {text_tokens.shape[0]}")
        valid_text = text_tokens[text_mask_np]
        if valid_text.size == 0:
            valid_text = text_np
        text_mean = valid_text.mean(0)
        # Raw post-fusion ROI probes: fixed mean, per-level mean, and point max.
        roi_scores = {"roi_mean": [], "roi_point_max": [], "roi_level_mean_0": [],
                      "roi_level_mean_1": [], "roi_level_mean_2": [], "roi_level_mean_3": []}
        for row_tokens in roi_np:
            sims = np.asarray([cosine(token, word) for token in row_tokens for word in valid_text], np.float32)
            roi_scores["roi_mean"].append(cosine(row_tokens.mean(0), text_mean))
            roi_scores["roi_point_max"].append(float(sims.max()))
            for level in range(4):
                roi_scores[f"roi_level_mean_{level}"].append(cosine(row_tokens[level * 16:(level + 1) * 16].mean(0), text_mean))
        # Only after feature construction do we read row labels.
        labels, label_path = load_sidecar(bank_path, int(tensors["track_id"].numel()))
        y = [bool(i in set(unit.get("positive_indices", []))) for i in range(end - begin)]
        row_gt = [labels[row] for row in row_offsets]
        historical = unit["historical_scores"]
        scores = {k: np.asarray(v, np.float64) for k, v in roi_scores.items()}
        scores["l29"] = np.asarray(historical["l29"], np.float64)
        if len(scores["l29"]) != end - begin:
            raise AssertionError(f"historical score length drift {unit['unit_key']}")
        category_counts[unit["category"]] += 1
        coverage_rows.append({"unit_key": unit["unit_key"], "dataset": unit["dataset"],
                              "video": unit["video"], "category": unit["category"],
                              "candidate_count": end - begin, "positive_count": int(sum(y)),
                              "target_id_count": len(unit.get("target_ids", [])),
                              "candidate_covered": bool(sum(y) > 0),
                              "present_uncovered": bool(unit["category"] == "present_uncovered"),
                              "inactive_or_null": bool(unit["category"] == "inactive")})
        records.append({"unit_key": unit["unit_key"], "dataset": unit["dataset"], "video": unit["video"],
                        "frame_id": int(unit["frame_id"]), "category": unit["category"],
                        "audit_split": unit["audit_split"], "label": y, "scores": scores,
                        "row_keys": row_keys, "row_gt": row_gt, "boxes": boxes.tolist(),
                        "candidate_count": end - begin, "labels_source": str(label_path),
                        "roi_meta": {k: v for k, v in meta.items() if k not in ("roi_valid_counts_per_candidate_level",)},
                        "roi_valid_fraction_per_level": meta["roi_valid_fraction_per_level"],
                        "fused_text_shape": list(fused_text.shape),
                        "text_valid_count": int(text_valid.sum().item())})
        for offset, row in enumerate(row_offsets):
            selected_rows[(str(unit["dataset"]), str(unit["video"]), int(unit["frame_id"]), row)].append(unit["unit_key"])
        # Hard-negative visualization uses only audit labels after feature construction.
        if sum(y) and sum(not v for v in y):
            s = scores["roi_point_max"]
            pos = np.flatnonzero(y)
            neg = np.flatnonzero(~np.asarray(y))
            p = int(pos[np.argmin(s[pos])]); n = int(neg[np.argmax(s[neg])])
            hard_cases.append({"unit_key": unit["unit_key"], "dataset": unit["dataset"],
                               "video": unit["video"], "frame_id": int(unit["frame_id"]),
                               "positive_box": boxes[p].tolist(), "negative_box": boxes[n].tolist(),
                               "positive_gt": row_gt[p], "positive_score": float(s[p]),
                               "negative_score": float(s[n]), "score_gap": float(s[p] - s[n])})
        # The full fused ROI is retained only in RAM.  This compact perturbation
        # slice is an expression-conditioning diagnostic, never a selector.
        if unit["unit_key"] in control_keys:
            control_sentence = "an unrelated object in a remote scene"
            with torch.inference_mode():
                c_roi, c_text, c_mask, _, c_meta = stream_fused_roi(detector, unit, bank, sentence=control_sentence)
            delta = (c_roi - roi).float()
            exact_text_mean = fused_text.float()[0, text_valid[0].bool()].mean(0)
            control_text_valid = c_mask[0].bool() if c_mask.dim() > 1 else c_mask.bool()
            control_text_mean = c_text.float()[0, control_text_valid].mean(0)
            expression_control.append({"unit_key": unit["unit_key"],
                                       "control_sentence": control_sentence,
                                       "roi_relative_l2": float(delta.norm() / roi.float().norm().clamp_min(1e-8)),
                                       "text_mean_relative_l2": float((control_text_mean - exact_text_mean).norm() /
                                                                        exact_text_mean.norm().clamp_min(1e-8)),
                                       "per_level_relative_l2": [float(delta[:, i*16:(i+1)*16].norm() /
                                                                           roi[:, i*16:(i+1)*16].float().norm().clamp_min(1e-8)) for i in range(4)],
                                       "control_roi_finite": bool(torch.isfinite(c_roi).all()),
                                       "control_text_finite": bool(torch.isfinite(c_text).all()),
                                       "control_meta": {"candidate_count": c_meta["candidate_count"]}})
        del roi, fused_text, text_valid, numeric, roi_np, text_np, bank, tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[l63-ceiling] {index + 1}/{len(units)} {unit['unit_key']}", flush=True)

    # Identity pairs from selected immutable rows.  Features are loaded only
    # from the frozen bank; pair labels are used after row construction.
    selected = {}
    for r in records:
        path = Path(next(u["bank_path"] for u in units if u["unit_key"] == r["unit_key"])).resolve()
        bank = torch.load(path, map_location="cpu", weights_only=False)
        tensors = bank["tensors"]
        labels, _ = load_sidecar(path, int(tensors["track_id"].numel()))
        for offset, row in enumerate(range(int(next(u["begin"] for u in units if u["unit_key"] == r["unit_key"])),
                                      int(next(u["end"] for u in units if u["unit_key"] == r["unit_key"])) )):
            selected[(str(r["dataset"]), str(r["video"]), int(r["frame_id"]), row)] = {
                "frame": int(r["frame_id"]), "gt": labels[row], "pool": int(tensors["pool_id"][row]),
                "features": {name: tensors[name][row].float().numpy() for name in vector_specs(tensors)}}
        del bank, tensors
    for left_key, right_key in combinations(sorted(selected), 2):
        left, right = selected[left_key], selected[right_key]
        if left_key[:3] != right_key[:3]:
            continue
        if left["gt"] is None or right["gt"] is None:
            continue
        if left["gt"] == right["gt"]:
            relation = "same_gt_same_frame"
        else:
            relation = "different_gt_same_frame"
        for name in left["features"]:
            identity_pairs[name]["different_gt_same_frame" if relation == "different_gt_same_frame" else "same_gt_cross_frame"].append(
                cosine(left["features"][name], right["features"][name]))
    # Cross-frame pairs, including main/reserve transitions.
    selected_items = list(selected.items())
    for (lk, left), (rk, right) in combinations(selected_items, 2):
        if lk[:2] != rk[:2] or left["frame"] == right["frame"]:
            continue
        if left["gt"] is None or right["gt"] is None or left["gt"] != right["gt"]:
            continue
        for name in left["features"]:
            value = cosine(left["features"][name], right["features"][name])
            identity_pairs[name]["same_gt_cross_frame"].append(value)
            if left["pool"] != right["pool"]:
                identity_pairs[name]["same_gt_cross_fragment"].append(value)
    identity_summary = {}
    for name, groups in identity_pairs.items():
        same = groups["same_gt_cross_frame"]
        diff = groups["different_gt_same_frame"]
        identity_summary[name] = {"all_same_gt_vs_same_frame_different_gt": pair_metrics(same, diff),
                                  "cross_fragment_same_gt": stat(groups["same_gt_cross_fragment"]),
                                  "cross_frame_same_gt": stat(same)}

    method_summary = {"l29": score_summary(records, "l29")}
    for name in ("roi_mean", "roi_point_max", "roi_level_mean_0", "roi_level_mean_1", "roi_level_mean_2", "roi_level_mean_3"):
        method_summary[name] = score_summary(records, name)
    # L62 scalar is explicitly a historical learned-score control only; raw
    # post-fusion ROI methods above are the actual expression-region probes.
    l59_available = all("l59_fused_roi" in r["historical_scores"] and
                        r["historical_scores"]["l59_fused_roi"] is not None for r in units)
    if l59_available:
        for r in records:
            r["scores"]["l59_historical_scalar_control"] = np.asarray(
                next(u for u in units if u["unit_key"] == r["unit_key"])["historical_scores"]["l59_fused_roi"], float)
        method_summary["l59_historical_scalar_control"] = score_summary(records, "l59_historical_scalar_control")

    for case in sorted(hard_cases, key=lambda x: x["score_gap"])[:12]:
        svg_case(case, out / "representative_hard_negatives" / (case["unit_key"].replace("|", "_") + ".svg"))
    with (out / "representative_hard_negatives.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["unit_key", "dataset", "video", "frame_id", "positive_gt",
                                                     "positive_box", "negative_box", "positive_score",
                                                     "negative_score", "score_gap"])
        writer.writeheader()
        writer.writerows(sorted(hard_cases, key=lambda x: x["score_gap"])[:12])
    for r in records:
        r.pop("boxes", None)
        r.pop("row_gt", None)
        r["scores"] = {k: [float(x) for x in v] for k, v in r["scores"].items()}
    (out / "unit_records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    coverage = {"units": len(coverage_rows), "candidate_rows": sum(x["candidate_count"] for x in coverage_rows),
                "target_frame_units": sum(x["category"] != "inactive" for x in coverage_rows),
                "covered_target_units": sum(x["candidate_covered"] for x in coverage_rows if x["category"] != "inactive"),
                "visible_target_coverage": sum(x["candidate_covered"] for x in coverage_rows if x["category"] != "inactive") /
                max(1, sum(x["category"] != "inactive" for x in coverage_rows)),
                "by_dataset_category": {}}
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        coverage["by_dataset_category"][dataset] = {}
        for category in ("positive", "multi_positive", "inactive", "present_uncovered"):
            subset = [x for x in coverage_rows if x["dataset"] == dataset and x["category"] == category]
            coverage["by_dataset_category"][dataset][category] = {
                "units": len(subset), "candidate_rows": sum(x["candidate_count"] for x in subset),
                "positive_rows": sum(x["positive_count"] for x in subset),
                "target_ids": sum(x["target_id_count"] for x in subset),
                "covered": sum(x["candidate_covered"] for x in subset),
                "present_uncovered": sum(x["present_uncovered"] for x in subset),
                "inactive": sum(x["inactive_or_null"] for x in subset),
            }
    payload = {
        "format": "locatemot-l63-oracle-ceiling-v1", "status": "complete",
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "scope": "fixed immutable 16 calibration + 24 validation units; oracle/probe only",
        "provenance": provenance, "category_counts": dict(category_counts),
        "coverage_ceiling": coverage, "identity_ceiling": identity_summary,
        "expression_region_ceiling": {
            "raw_post_fusion_roi_available": True,
            "methods": method_summary,
            "token_span_alignment": "UNALIGNED",
            "static_motion_language_mask": "UNALIGNED",
            "interpretation": "frozen expression-conditioned ROI probes, not a trained model or HOTA",
            "conditioning_controls": expression_control,
        },
        "sequence_identity_ceiling": {
            "gt_conditioned_candidate_ceiling": coverage["visible_target_coverage"],
            "frozen_identity_pair_summary": identity_summary,
            "short_gap_reappearance": "only selected fixed units; no deployed DP/Viterbi; finite-sample audit",
            "multiple_positive_policy": "all labelled positive rows retained; ranking set recall at k=positive_count",
        },
        "decision_rules": {
            "candidate_coverage_blocked": "coverage or present-uncovered makes requested recall impossible",
            "identity_oracle_insufficient": "no repeatable same-GT versus same-frame different-GT separation or fragment recall",
            "representation_semantic_ceiling_insufficient": "coverage/identity adequate but expression-region probes lack stable hard/multi-positive separation",
            "representation_has_ceiling_but_decoder_missing": "coverage and frozen identity/region separation stable while L62 aggregation is missing",
            "null_calibration_blocked": "only inactive/no-match separation remains inadequate",
            "automatic_selection": "implemented in report from denominators and multiple fixed metrics; validation labels were not used to fit any parameter",
        },
        "elapsed_sec": time.time() - start,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "label_free_detector_forward_count": len(units) + len(expression_control),
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "persistent_dense_or_raw_cache_written": False,
        "all_candidate_rows_retained": True,
    }
    (out / "oracle_ceiling.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"status": "complete", "output": str(out / "oracle_ceiling.json"),
                      "units": len(units), "elapsed_sec": payload["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
