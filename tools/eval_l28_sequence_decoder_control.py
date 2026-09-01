#!/usr/bin/env python3
"""Stage L28 no-training causal sequence decoder control on frozen L27 caches."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.eval_l27_fast_rmot import frame_groups, load_gt, make_entries
from tools.summarize_l27_fast_rmot import load_caches

OUT = ROOT / "outputs/l28/eval/sequence_decoder_control_final"
SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
FORMAL = ROOT / "outputs/l27/fast_rmot_validation_formal/summary.json"


def annotate_gt_ids(data, entry):
    """Attach GT IDs only for fixed-strategy reporting, never selection."""
    gt = load_gt(str(entry["video"]))
    targets = {int(frame): {str(x) for x in ids}
               for frame, ids in entry.get("label", {}).items()}
    result = []
    for frame, box in zip(data["frame"].tolist(), data["box"]):
        best_id, best_iou = None, 0.0
        for gt_id in targets.get(int(frame), set()):
            target = gt.get(int(frame), {}).get(str(gt_id))
            if target is None:
                continue
            x1 = max(float(box[0]), float(target[0])); y1 = max(float(box[1]), float(target[1]))
            x2 = min(float(box[2]), float(target[2])); y2 = min(float(box[3]), float(target[3]))
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            area_b = max(0.0, float(target[2] - target[0])) * max(0.0, float(target[3] - target[1]))
            iou = inter / max(1e-8, area_a + area_b - inter)
            if iou >= 0.5 and iou > best_iou:
                best_id, best_iou = str(gt_id), iou
        result.append(best_id)
    data["gt_id"] = np.asarray(result, dtype=object)
    return data


def frame_metrics(records, strategy):
    selected = []; labels = []; fp_frame = []; empty = []; null_accept = []
    track_recall = []; query_frames = defaultdict(list)
    for record in records:
        y = record["label"].astype(bool); keep = record["selected"]
        selected.append(keep); labels.append(y)
        pred, pos = bool(keep.any()), bool(y.any())
        empty.append(not pred); null_accept.append((not pos) and pred)
        fp_frame.append(int((keep & ~y).sum()))
        positive_tracks = set(record["track_id"][y].tolist())
        selected_tracks = set(record["track_id"][keep].tolist())
        track_recall.append(float(bool(positive_tracks & selected_tracks)) if pos else 0.0)
        query_frames[record["query_index"]].append(record)
    s = np.concatenate(selected) if selected else np.zeros(0, bool)
    y = np.concatenate(labels) if labels else np.zeros(0, bool)
    tp = int((s & y).sum()); fp = int((s & ~y).sum()); fn = int((~s & y).sum())
    switches = 0; continuity_checks = 0
    for qrecords in query_frames.values():
        previous = {}
        for record in qrecords:
            current = {}
            for gt in set(record["gt_id"][record["label"].astype(bool)].tolist()):
                rows = np.flatnonzero(record["label"].astype(bool) &
                                      (record["gt_id"] == gt) & record["selected"])
                if len(rows):
                    best = rows[np.argmax(record["score"][rows])]
                    current[gt] = int(record["track_id"][best])
            for gt, track in current.items():
                if gt in previous:
                    continuity_checks += 1
                    switches += int(previous[gt] != track)
            previous.update(current)
    return {
        "frame_units": len(records), "selected": int(s.sum()),
        "positive_rows": int(y.sum()), "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, int(y.sum()))),
        "predictions_per_gt_positive": float(s.sum() / max(1, int(y.sum()))),
        "false_positive_candidates_per_frame": float(np.mean(fp_frame)) if fp_frame else None,
        "empty_output_rate": float(np.mean(empty)) if empty else None,
        "null_frame_false_acceptance": float(np.mean(null_accept)) if null_accept else None,
        "track_recall": float(np.mean(track_recall)) if track_recall else None,
        "identity_switches": int(switches),
        "identity_continuity_checks": int(continuity_checks),
    }


def online_sequence(query_data, threshold, null_threshold, use_null, cap):
    """Causal per-track persistence, intentionally independent per track."""
    by_frame = frame_groups(query_data)
    state = {}  # track -> last active frame index
    records = []
    for frame_number, idx in by_frame:
        score = query_data["score"][idx]
        tracks = query_data["track_id"][idx]
        per_track = {}
        for local, track in enumerate(tracks.tolist()):
            track = int(track)
            if track not in per_track or score[local] > per_track[track][1]:
                per_track[track] = (int(idx[local]), float(score[local]))
        active = []
        for track, (row, value) in per_track.items():
            last = state.get(track)
            if value >= threshold or (last is not None and frame_number - last <= 2 and value >= threshold - 0.75):
                active.append((track, row, value))
                state[track] = frame_number
        if use_null and not active:
            top = int(idx[np.argmax(score)]) if len(idx) else None
            if top is not None and float(query_data["score"][top]) >= float(null_threshold):
                active = [(int(query_data["track_id"][top]), top, float(query_data["score"][top]))]
                state[int(query_data["track_id"][top])] = frame_number
        if not use_null and not active and len(idx):
            top = int(idx[np.argmax(score)])
            active = [(int(query_data["track_id"][top]), top, float(query_data["score"][top]))]
        chosen_tracks = {track for track, _, _ in active}
        keep = np.asarray([int(track) in chosen_tracks for track in tracks], bool)
        records.append({"frame": int(frame_number), "selected": keep,
                        "label": query_data["label"][idx].astype(bool),
                        "track_id": query_data["track_id"][idx].astype(np.int64),
                        "gt_id": np.asarray(query_data["gt_id"][idx], dtype=object),
                        "score": score.astype(np.float32),
                        "query_index": int(query_data.get("query_index", -1))})
        if len(records) >= cap:
            break
    return records


def baseline_records(query_data, threshold, kind, cap):
    result = []
    for frame, idx in frame_groups(query_data)[:cap]:
        order = idx[np.argsort(-query_data["score"][idx], kind="stable")]
        keep = np.zeros(len(idx), bool)
        if kind == "threshold":
            keep = query_data["score"][idx] >= threshold
        elif kind == "top2":
            chosen = order[:2]
            keep[np.isin(idx, chosen[query_data["score"][chosen] >= threshold])] = True
        else:
            raise ValueError(kind)
        result.append({"frame": int(frame), "selected": keep,
                        "label": query_data["label"][idx].astype(bool),
                        "track_id": query_data["track_id"][idx].astype(np.int64),
                        "gt_id": np.asarray(query_data["gt_id"][idx], dtype=object),
                        "score": query_data["score"][idx].astype(np.float32),
                        "query_index": int(query_data.get("query_index", -1))})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    entries = make_entries()
    formal = json.loads(FORMAL.read_text())
    all_results = {}
    for model in ("A_C1_S2000", "B_F4_bounded_residual"):
        caches = load_caches(SCORE_ROOT, entries, (model,))[model]
        calibration = formal["candidate_metrics"][model]["calibration"]
        threshold = calibration["precision_first"]["threshold"]
        null_threshold = calibration["null_max_threshold"]
        if threshold is None:
            raise ValueError(f"{model}: precision threshold is None")
        if null_threshold is None:
            raise ValueError(f"{model}: NULL threshold is None; explicit no-null control required")
        items = [(entry, caches[(entry["video"], entry["expression"])])
                 for entry in entries if entry["split"] == "screening"]
        # Preserve query/video/frame identity and take the first 100 units in
        # manifest order; no screening label is used to choose a strategy.
        records_by_strategy = {name: [] for name in
                               ("l27_threshold", "l27_top2", "sequence_no_null", "sequence_null")}
        used = 0
        for entry, data in items:
            if used >= 100:
                break
            data = {key: np.asarray(value).copy() for key, value in data.items()}
            data["query_index"] = int(entry["query_index"])
            data = annotate_gt_ids(data, entry)
            remaining = 100 - used
            for name in ("l27_threshold", "l27_top2"):
                records_by_strategy[name].extend(
                    baseline_records(data, float(threshold),
                                     "threshold" if name.endswith("threshold") else "top2",
                                     remaining))
            records_by_strategy["sequence_no_null"].extend(
                online_sequence(data, float(threshold), float(null_threshold), False, remaining))
            records_by_strategy["sequence_null"].extend(
                online_sequence(data, float(threshold), float(null_threshold), True, remaining))
            used += min(remaining, len(frame_groups(data)))
        all_results[model] = {
            "threshold": float(threshold),
            "null_threshold": float(null_threshold),
            "calibration_null_samples": int(calibration["null_max_samples"]),
            "screening_gt_used_for_strategy_selection": False,
            "smoke_frame_units": used,
            "strategies": {name: frame_metrics(records, name) for name, records in records_by_strategy.items()},
        }
        del caches
    payload = {
        "format": "locatemot-l28-sequence-decoder-control-v1",
        "manifest": str(ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"),
        "score_root": str(SCORE_ROOT), "calibration_source": str(FORMAL),
        "scope": "100 screening frame units; no TrackEval; frozen L27 scores only",
        "models": all_results,
    }
    (OUT / "control.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# L28 sequence decoder control\n\n"
        "Causal per-track persistence control over immutable L27 score caches. "
        "This is candidate-level only; it is not a TrackEval result.\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
