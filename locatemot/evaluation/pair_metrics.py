"""End-to-end and candidate-conditional metrics for two-frame association."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate_assignments(pred_assignments, records, cur_gt_boxes_by_key):
    """pred_assignments: list aligned with records; each is list of (track_i, tag)."""
    total = correct = 0
    cond_total = cond_correct = 0
    nm_tp = nm_fp = nm_fn = 0
    id_tp = id_fp = id_fn = 0
    loc05 = loc07 = loc_total = 0
    duplicates = 0
    unassigned_visible = 0
    per_key = defaultdict(lambda: {"total": 0, "correct": 0, "cond_total": 0, "cond_correct": 0,
                                    "nm_tp": 0, "nm_fp": 0, "nm_fn": 0})

    for rec, preds in zip(records, pred_assignments):
        targets = {t["track_id"]: t["candidate_index"] for t in rec["assignment_targets"]}
        no_match_ids = set(rec["no_match_targets"])
        missing_ids = set(rec["candidate_missing_targets"])
        ref_ids = [t["track_id"] for t in rec["reference_targets"]]
        pred_by_id = {}
        for t in rec["reference_targets"]:
            pred_by_id[t["track_id"]] = None
        for ti, tag in preds:
            if ti < len(ref_ids):
                pred_by_id[ref_ids[ti]] = tag
        used_cands = defaultdict(int)
        cur_key = rec["current_token_id"]
        cur_gt = cur_gt_boxes_by_key.get(cur_key, {})
        for tid in ref_ids:
            tag = pred_by_id.get(tid)
            if tag is not None and tag.startswith("candidate:"):
                used_cands[tag] += 1
            key = (rec["split"], rec["dataset"], rec["protocol"], rec["temporal_gap"],
                   len(ref_ids), rec.get("current_candidate_count", 0))
            per_key[key]["total"] += 1
            if tid in targets:
                expected = f"candidate:{targets[tid]}"
                is_correct = tag == expected
                total += 1
                correct += int(is_correct)
                # conditional metrics (candidate present)
                cond_total += 1
                per_key[key]["cond_total"] += 1
                if is_correct:
                    cond_correct += 1
                    per_key[key]["cond_correct"] += 1
                elif tag is not None and tag.startswith("candidate:"):
                    id_fp += 1
                if is_correct and tag.startswith("candidate:"):
                    j = int(tag.split(":")[1])
                    gt_box = cur_gt.get(str(tid))
                    if gt_box is not None and j < len(rec.get("candidate_boxes", [])):
                        iou = _iou(rec["candidate_boxes"][j], gt_box)
                        loc_total += 1
                        loc05 += iou >= 0.5
                        loc07 += iou >= 0.7
                id_tp += int(is_correct)
                id_fn += int(not is_correct)
            elif tid in no_match_ids:
                total += 1
                is_nm = tag == "NO_MATCH"
                correct += int(is_nm)
                if is_nm:
                    nm_tp += 1
                    per_key[key]["nm_tp"] += 1
                else:
                    nm_fn += 1
                    per_key[key]["nm_fn"] += 1
                    if tag is not None and tag.startswith("candidate:"):
                        nm_fp += 1
                        id_fp += 1
            elif tid in missing_ids:
                total += 1  # candidate missing -> e2e failure regardless
                if tag is None or tag == "NO_MATCH":
                    unassigned_visible += 1
                else:
                    unassigned_visible += 0
        # duplicates
        for tag, c in used_cands.items():
            if c > 1:
                duplicates += 1

    nm_prec = nm_tp / max(1, nm_tp + nm_fp)
    nm_rec = nm_tp / max(1, nm_tp + nm_fn)
    nm_f1 = 2 * nm_prec * nm_rec / max(1e-9, nm_prec + nm_rec)
    id_prec = id_tp / max(1, id_tp + id_fp)
    id_rec = id_tp / max(1, id_tp + id_fn)
    id_f1 = 2 * id_prec * id_rec / max(1e-9, id_prec + id_rec)
    return {
        "e2e_accuracy": correct / max(1, total),
        "conditional_accuracy": cond_correct / max(1, cond_total),
        "loc_success_0_5": loc05 / max(1, loc_total),
        "loc_success_0_7": loc07 / max(1, loc_total),
        "no_match_precision": nm_prec,
        "no_match_recall": nm_rec,
        "no_match_f1": nm_f1,
        "id_precision": id_prec,
        "id_recall": id_rec,
        "id_f1": id_f1,
        "duplicates": duplicates,
        "unassigned_visible": unassigned_visible,
        "total_refs": total,
        "conditional_refs": cond_total,
    }
