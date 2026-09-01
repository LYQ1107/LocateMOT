#!/usr/bin/env python3
"""Read-only Stage L28 persistent identity-bank upper-bound audit."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
BANK_ROOT = ROOT / "outputs/l19/dual_banks_features/kitti"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPRESSIONS = (
    ROOT / "outputs/l11/data/rmot_kitti/expressions.json",
    ROOT / "outputs/l16/data/kitti_missing/records/expressions.json",
)
RECORD_ROOTS = (
    ROOT / "outputs/l11/data/rmot_kitti",
    ROOT / "outputs/l16/data/kitti_missing/records",
)
OUT = ROOT / "outputs/l28/audit/identity_bank_upper_bound_corrected"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    a = np.asarray(values, np.float64)
    return {"count": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "p05": float(np.quantile(a, .05)),
            "p95": float(np.quantile(a, .95))}


def cosine(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.dot(a, b) / max(1e-8, np.linalg.norm(a) * np.linalg.norm(b)))


def pair_similarity(left, right):
    return {name: cosine(left[name], right[name])
            for name in ("clip", "history_clip", "uidm_h")}


def record_path(video):
    for root in RECORD_ROOTS:
        path = root / f"{video}.pkl"
        if path.exists():
            return path
    return None


def load_expressions():
    result = {}
    for path in EXPRESSIONS:
        for video, rows in json.loads(path.read_text()).items():
            for row in rows:
                result[(str(video), str(row["expression"]))] = {
                    "video": str(video), **row}
    return result


def label_path(bank_path, count, tensors=None):
    candidates = [
        bank_path.with_suffix(".labels.json"),
        ROOT / "outputs/l19/dual_banks/kitti" / bank_path.name.replace(".pt", ".labels.json"),
        ROOT / "outputs/l18/dual_banks/kitti" / bank_path.name.replace(".pt", ".labels.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        labels = json.loads(path.read_text()).get("candidate_gt", [])
        if len(labels) == count:
            return path, labels
    raw = record_path(bank_path.stem)
    if raw is not None:
        record = pickle.loads(raw.read_bytes())
        gt_by_frame = {int(row["frame"]): row.get("gt_boxes", {})
                       for row in record.get("frames", [])}
        if tensors is None:
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)
            tensors = bank["tensors"]
        labels = [None] * count
        for fi, frame in enumerate(tensors["frame_ids"].tolist()):
            begin, end = int(tensors["frame_ptr"][fi]), int(tensors["frame_ptr"][fi + 1])
            gt_items = [(str(gt), np.asarray(box, np.float32))
                        for gt, box in gt_by_frame.get(int(frame), {}).items()]
            for row in range(begin, end):
                box = np.asarray(tensors["box"][row], np.float32)
                best_gt, best_iou = None, 0.0
                for gt, target in gt_items:
                    x1 = max(float(box[0]), float(target[0])); y1 = max(float(box[1]), float(target[1]))
                    x2 = min(float(box[2]), float(target[2])); y2 = min(float(box[3]), float(target[3]))
                    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
                    area_b = max(0.0, float(target[2] - target[0])) * max(0.0, float(target[3] - target[1]))
                    iou = inter / max(1e-8, area_a + area_b - inter)
                    if iou >= 0.5 and iou > best_iou:
                        best_gt, best_iou = gt, iou
                labels[row] = best_gt
        return raw, labels
    raise FileNotFoundError(
        f"no row-aligned labels sidecar for {bank_path} rows={count}; checked {candidates}")


def load_labels(bank_path, count, tensors=None):
    path, labels = label_path(bank_path, count, tensors=tensors)
    return [None if x is None else str(x) for x in labels], path


def item(row, tensors, labels):
    return {
        "row": int(row), "track_id": int(tensors["track_id"][row]),
        "pool_id": int(tensors["pool_id"][row]),
        "source": "reserve" if int(tensors["pool_id"][row]) else "main",
        "box": [float(x) for x in tensors["box"][row]],
        "gt_id": labels[row], "objectness": float(tensors["objectness"][row]),
        "clip": tensors["clip"][row].numpy(),
        "history_clip": tensors["history_clip"][row].numpy(),
        "uidm_h": tensors["uidm_h"][row].numpy(),
    }


def public_item(value):
    return {k: v for k, v in value.items()
            if k not in ("clip", "history_clip", "uidm_h")}


def pair_case(video, frame, left, right, relation, similarity):
    return {"video": video, "frame": int(frame), "relation": relation,
            "left": public_item(left), "right": public_item(right),
            "similarity": similarity}


def draw_case(case, path):
    width, height = 1242, 375
    colors = {"main": "#2563eb", "reserve": "#dc2626"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="8" y="18" font-size="14">{case["relation"]} {case["video"]}/{case["frame"]}</text>',
    ]
    for side in ("left", "right"):
        value = case[side]
        x1, y1, x2, y2 = value["box"]
        color = colors.get(value["source"], "#111827")
        label = f'{side}:{value["source"]} tr={value["track_id"]} gt={value.get("gt_id")}'
        lines.append(
            f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{max(1,x2-x1):.2f}" '
            f'height="{max(1,y2-y1):.2f}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(
            f'<text x="{max(2,x1):.2f}" y="{max(30,y1-3):.2f}" font-size="10" '
            f'fill="{color}">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def rank_auc(pos, neg):
    if not pos or not neg:
        return None
    values = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    neg_seen, wins = 0, 0.0
    for _, label in values:
        if label:
            wins += neg_seen
        else:
            neg_seen += 1
    return float(wins / max(1, len(pos) * len(neg)))


def audit_video(video):
    path = BANK_ROOT / f"{video}.pt"
    bank = torch.load(path, map_location="cpu", weights_only=False)
    tensors = bank["tensors"]
    count = int(tensors["track_id"].numel())
    labels, labels_path = load_labels(path, count, tensors=tensors)
    fields = ("frame", "candidate_index", "track_id", "box", "objectness",
              "clip", "history_clip", "pbd", "uidm_h", "uidm_ref_pbd",
              "uidm_anchor_pbd", "geometry", "motion", "context", "lifecycle",
              "pool_id", "source_score")
    finite = {name: bool(torch.isfinite(tensors[name].float()).all())
              for name in fields if name in tensors}
    frames, ptr = tensors["frame_ids"].tolist(), tensors["frame_ptr"].tolist()
    frame_data = []
    for fi, frame in enumerate(frames):
        begin, end = int(ptr[fi]), int(ptr[fi + 1])
        by_gt = defaultdict(list)
        for row in range(begin, end):
            if labels[row] is not None:
                by_gt[labels[row]].append(row)
        reps = {gt: max(rows, key=lambda r: float(tensors["objectness"][r]))
                for gt, rows in by_gt.items()}
        frame_data.append({"frame": int(frame), "begin": begin, "end": end,
                           "reps": reps})

    same, cross, different = ({n: [] for n in ("clip", "history_clip", "uidm_h")}
                               for _ in range(3))
    continuation = {key: [0, 0] for key in ("all", "cross_fragment",
                                              "main_to_reserve")}
    source_counts = Counter()
    fragments = defaultdict(set)
    previous = {}
    inactive = [0, 0]
    cases = []
    for current in frame_data:
        reps = current["reps"]
        for gt, row in reps.items():
            fragments[gt].add(int(tensors["track_id"][row]))
            source_counts["reserve" if int(tensors["pool_id"][row]) else "main"] += 1
        rep_items = list(reps.items())
        for (_, left_row), (_, right_row) in combinations(rep_items, 2):
            left, right = item(left_row, tensors, labels), item(right_row, tensors, labels)
            sim = pair_similarity(left, right)
            for name, value in sim.items():
                different[name].append(value)
            if len(cases) < 6:
                cases.append(pair_case(video, current["frame"], left, right,
                                       "same-frame-different-GT", sim))
        for gt, row in reps.items():
            if gt not in previous:
                continue
            prior = previous[gt]
            left, right = item(prior, tensors, labels), item(row, tensors, labels)
            sim = pair_similarity(left, right)
            for name, value in sim.items():
                same[name].append(value)
                if left["track_id"] != right["track_id"]:
                    cross[name].append(value)
            changed = left["track_id"] != right["track_id"]
            # The continuation metric is a real retrieval test: use the
            # previous representative's frozen appearance to rank every
            # candidate in the current frame, then inspect the retrieved
            # candidate's audit-only label.  Do not select the known same-GT
            # row as the prediction.
            current_rows = list(range(current["begin"], current["end"]))
            if current_rows:
                similarities = np.asarray([
                    cosine(left["clip"], tensors["clip"][candidate].numpy())
                    for candidate in current_rows], np.float32)
                nearest = current_rows[int(np.argmax(similarities))]
                nearest_item = item(nearest, tensors, labels)
                retrieved_correct = int(labels[nearest] == gt)
            else:
                nearest_item = right
                retrieved_correct = 0
            continuation["all"][1] += 1
            continuation["all"][0] += retrieved_correct
            if changed:
                continuation["cross_fragment"][1] += 1
                continuation["cross_fragment"][0] += retrieved_correct
                if left["source"] == "main" and right["source"] == "reserve":
                    continuation["main_to_reserve"][1] += 1
                    continuation["main_to_reserve"][0] += retrieved_correct
            if len(cases) < 12 and changed:
                retrieved_sim = pair_similarity(left, nearest_item)
                cases.append(pair_case(video, current["frame"], left, nearest_item,
                                       "cross-fragment-nearest-retrieval", retrieved_sim))
        current_tracks = {int(tensors["track_id"][r])
                          for r in range(current["begin"], current["end"])}
        for prior in previous.values():
            track = int(tensors["track_id"][prior])
            if track in current_tracks:
                inactive[1] += 1
                current_positive = any(
                    labels[r] is not None and int(tensors["track_id"][r]) == track
                    for r in range(current["begin"], current["end"]))
                inactive[0] += int(not current_positive)
        previous = reps

    report = {
        "video": video, "bank": str(path.resolve()), "bank_sha256": sha(path),
        "metadata": bank.get("metadata", {}), "observations": count,
        "labels_source": (str(labels_path.resolve()) if isinstance(labels_path, Path)
                           else f"derived_from_record:{labels_path}"),
        "labels_source_type": ("sidecar" if str(labels_path).endswith(".labels.json")
                                else "derived_from_raw_record_iou50"),
        "frames": len(frames), "finite_fields": finite, "label_count": len(labels),
        "gt_representative_observations": sum(len(x["reps"]) for x in frame_data),
        "unique_gt_ids": len(fragments),
        "fragment_count_by_gt": stats([len(x) for x in fragments.values()]),
        "source_representative_counts": dict(source_counts),
        "identity_similarity": {
            name: {"same_gt_cross_frame": stats(same[name]),
                   "same_gt_cross_fragment": stats(cross[name]),
                   "same_frame_different_gt": stats(different[name]),
                   "same_gt_vs_different_gt_auc": rank_auc(same[name], different[name])}
            for name in same
        },
        "continuation": {
            key: {"correct": int(value[0]), "attempts": int(value[1]),
                  "recall": value[0] / max(1, value[1])}
            for key, value in continuation.items()
        },
        "inactive_false_identity_continuity": {
            "false_persistent_tracks": int(inactive[0]),
            "persistent_track_checks": int(inactive[1]),
            "false_rate": inactive[0] / max(1, inactive[1]),
        },
        "raw_record": str(record_path(video).resolve()) if record_path(video) else None,
        "raw_record_available": record_path(video) is not None,
    }
    del bank, tensors
    return report, cases


def fast_oracle(expressions):
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    queries_by_video = defaultdict(list)
    for query in manifest["queries"]:
        queries_by_video[str(query["video"])].append(query)
    for video, video_queries in queries_by_video.items():
        path = BANK_ROOT / f"{video}.pt"
        bank = torch.load(path, map_location="cpu", weights_only=False)
        tensors = bank["tensors"]
        labels, _labels_path = load_labels(path, int(tensors["track_id"].numel()), tensors=tensors)
        frame_rows = [(int(tensors["frame_ptr"][fi]), int(tensors["frame_ptr"][fi + 1]), int(frame))
                      for fi, frame in enumerate(tensors["frame_ids"].tolist())]
        for query in video_queries:
            expression = expressions.get((video, str(query["expression"])))
            if expression is None:
                continue
            target = {int(frame): {str(x) for x in ids}
                      for frame, ids in expression.get("label", {}).items()}
            visible = covered = 0
            tracks = defaultdict(set)
            for begin, end, frame in frame_rows:
                for gt in target.get(frame, set()):
                    visible += 1
                    matched = [r for r in range(begin, end) if labels[r] == gt]
                    if matched:
                        covered += 1
                        tracks[gt].update(int(tensors["track_id"][r]) for r in matched)
            rows.append({
                "query_index": int(query["query_index"]), "video": video,
                "expression": query["expression"], "split": query["split"],
                "visible_target_frames": visible, "covered_target_frames": covered,
                "oracle_observation_recall": covered / max(1, visible),
                "oracle_association_accuracy": 1.0 if covered else None,
                "oracle_id_switches": 0,
                "covered_gt_fragment_count": int(sum(len(x) for x in tracks.values())),
            })
        del bank, tensors
    result = {"manifest": str(MANIFEST.resolve()), "manifest_sha256": sha(MANIFEST),
              "query_count": len(rows), "interpretation":
              "GT-conditioned semantic/identity upper bound; not a model result"}
    result["by_split"] = {}
    for split in ("calibration", "screening"):
        subset = [x for x in rows if x["split"] == split]
        result["by_split"][split] = {
            "queries": len(subset),
            "oracle_observation_recall":
                float(np.mean([x["oracle_observation_recall"] for x in subset])) if subset else None,
            "oracle_association_accuracy": 1.0 if subset else None,
            "oracle_id_switches": 0, "query_rows": subset}
    return result


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    videos = sorted(p.stem for p in BANK_ROOT.glob("*.pt"))
    expressions = load_expressions()
    reports, cases = {}, []
    for index, video in enumerate(videos, 1):
        reports[video], video_cases = audit_video(video)
        cases.extend(video_cases)
        print(f"[l28-audit] {video} {index}/{len(videos)} rows={reports[video]['observations']}",
              flush=True)
    payload = {
        "format": "locatemot-l28-identity-bank-audit-v1",
        "scope": "read-only L19 main+reserve identity upper bound",
        "bank_root": str(BANK_ROOT.resolve()), "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST), "gt_used_for": "oracle labels and audit only",
        "gt_used_for_training_or_selection": False, "video_count": len(reports),
        "videos": reports, "fast_manifest_oracle": fast_oracle(expressions),
        "elapsed_sec": time.time() - start,
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    public_cases = cases[:24]
    (OUT / "representative_cases.json").write_text(
        json.dumps({"cases": public_cases}, indent=2) + "\n")
    vis = OUT / "visualizations"
    vis.mkdir(exist_ok=True)
    paths = []
    for index, case in enumerate(public_cases):
        path = vis / f"case_{index:03d}_{case['relation'].replace('-', '_')}.svg"
        draw_case(case, path)
        paths.append(path)
    lines = ["# L28 identity-bank upper-bound audit", "",
             f"Videos: {len(reports)}", f"Bank: {BANK_ROOT.resolve()}", "",
             "GT is used only for oracle audit labels; no training or model selection is performed.",
             "", "## Representative cases", ""]
    lines.extend(f"- {path.name}: {case['relation']} {case['video']}/{case['frame']}"
                 for path, case in zip(paths, public_cases))
    lines.append("\nSee audit.json for per-video similarity, continuation and inactive-continuity statistics.")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "videos": len(reports),
                      "cases": len(public_cases), "elapsed_sec": payload["elapsed_sec"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
