"""Independent quality audit for L18 crude versus L19 reserve identity."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
EVAL_RUN = ROOT / "references/l8/TrackEval_rmot/scripts/run_mot_challenge.py"


def load_record(video: str) -> dict:
    path = ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
    if not path.exists():
        path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
    return pickle.load(path.open("rb"))


def frame_rows(bank: dict, labels: list):
    tensors = bank["tensors"]
    ptr = tensors["frame_ptr"].tolist()
    frames = tensors["frame_ids"].tolist()
    pool = tensors.get("pool_id", torch.zeros(len(tensors["track_id"]), dtype=torch.long)).tolist()
    for fi, frame in enumerate(frames):
        start, end = int(ptr[fi]), int(ptr[fi + 1])
        yield int(frame), start, end, pool[start:end], labels[start:end]


def pairwise_association(match_by_gt: dict[str, list[tuple[int, int]]]) -> float:
    tp = fp = fn = 0
    by_track = defaultdict(list)
    for gt_id, values in match_by_gt.items():
        ordered = sorted(values)
        for frame, track_id in ordered:
            by_track[track_id].append((frame, gt_id))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                fn += int(ordered[i][1] != ordered[j][1])
                tp += int(ordered[i][1] == ordered[j][1])
    for values in by_track.values():
        ordered = sorted(values)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                fp += int(ordered[i][1] != ordered[j][1])
    return 100.0 * tp / max(1, tp + 0.5 * (fp + fn))


def custom_metrics(bank_path: Path) -> dict:
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    labels_path = bank_path.with_suffix(".labels.json")
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    labels = json.loads(labels_path.read_text())["candidate_gt"]
    tensors = bank["tensors"]
    track_ids = tensors["track_id"].tolist()
    pool = tensors.get("pool_id", torch.zeros(len(track_ids), dtype=torch.long)).tolist()
    records = load_record(bank["metadata"]["video_id"])
    gt_by_frame = {int(fr["frame"]): {str(k): v for k, v in
                                       fr.get("gt_boxes", {}).items()}
                   for fr in records["frames"]}
    reserve_rows = 0
    matched = 0
    visible = sum(len(value) for value in gt_by_frame.values())
    by_gt = defaultdict(list)
    track_labels = defaultdict(list)
    duplicate_gt_frames = 0
    duplicate_id_frames = 0
    ptr = tensors["frame_ptr"].tolist()
    for fi, frame in enumerate(tensors["frame_ids"].tolist()):
        start, end = int(ptr[fi]), int(ptr[fi + 1])
        current = [(index, track_ids[index], labels[index])
                   for index in range(start, end) if pool[index] == 1]
        reserve_rows += len(current)
        by_label = defaultdict(list)
        ids = Counter()
        for index, track_id, label in current:
            ids[int(track_id)] += 1
            if label is not None:
                by_label[str(label)].append((index, int(track_id)))
                track_labels[int(track_id)].append(str(label))
        duplicate_id_frames += int(any(value > 1 for value in ids.values()))
        duplicate_gt_frames += sum(len(value) > 1 for value in by_label.values())
        for gt_id, values in by_label.items():
            # The bank label is produced by one-to-one IoU matching; choosing
            # its first row is deterministic for the identity audit.
            by_gt[gt_id].append((int(frame), values[0][1]))
            matched += 1
    matched_track_counts = Counter()
    for values in by_gt.values():
        matched_track_counts.update(track_id for _frame, track_id in values)
    idtp = sum(max(Counter(track for _frame, track in values).values())
               for values in by_gt.values() if values)
    idf1 = 100.0 * 2.0 * idtp / max(1, visible + reserve_rows)
    switches = 0
    fragments = 0
    for values in by_gt.values():
        ordered = [track for _frame, track in sorted(values)]
        fragments += max(0, len(set(ordered)) - 1)
        switches += sum(left != right for left, right in zip(ordered, ordered[1:]))
    purity_values = []
    for values in track_labels.values():
        counts = Counter(values)
        purity_values.append(max(counts.values()) / max(1, len(values)))
    lengths = Counter()
    for fi in range(len(ptr) - 1):
        start, end = int(ptr[fi]), int(ptr[fi + 1])
        for index in range(start, end):
            if pool[index] == 1:
                lengths[int(track_ids[index])] += 1
    return {
        "video": bank["metadata"]["video_id"],
        "visible_gt_instances": int(visible), "reserve_observations": int(reserve_rows),
        "matched_gt_instances": int(matched),
        "candidate_recall_percent": 100.0 * matched / max(1, visible),
        "approx_idf1_percent": idf1,
        "association_pairwise_f1_percent": pairwise_association(by_gt),
        "fragmentations": int(fragments), "id_switches": int(switches),
        "track_purity_mean": float(np.mean(purity_values)) if purity_values else 0.0,
        "track_purity_median": float(np.median(purity_values)) if purity_values else 0.0,
        "mean_track_length_frames": float(np.mean(list(lengths.values()))) if lengths else 0.0,
        "unique_reserve_tracks": int(len(lengths)),
        "duplicate_id_frame_events": int(duplicate_id_frames),
        "duplicate_gt_frame_events": int(duplicate_gt_frames),
        "duplicate_rate_per_reserve_row": duplicate_gt_frames / max(1, matched),
    }


def write_trackeval_tree(bank_root: Path, root: Path) -> Path:
    result = root / "uidm19"
    result.mkdir(parents=True, exist_ok=True)
    seq_lines = []
    for bank_path in sorted(bank_root.glob("*.pt")):
        labels_path = bank_path.with_suffix(".labels.json")
        if not labels_path.exists():
            continue
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        video = bank["metadata"]["video_id"]
        expression = "all"
        directory = result / video / expression
        directory.mkdir(parents=True, exist_ok=True)
        record = load_record(video)
        gt_lines = []
        for frame in record["frames"]:
            number = int(frame["frame"]) + 1
            for gt_id, box in frame.get("gt_boxes", {}).items():
                x1, y1, x2, y2 = [float(value) for value in box]
                gt_lines.append(f"{number},{gt_id},{x1:.3f},{y1:.3f},"
                                f"{x2-x1:.3f},{y2-y1:.3f},1,1,1\n")
        tensors = bank["tensors"]
        ptr = tensors["frame_ptr"].tolist()
        pool = tensors.get("pool_id", torch.zeros(len(tensors["track_id"]), dtype=torch.long)).tolist()
        pred_lines = []
        for fi, frame in enumerate(tensors["frame_ids"].tolist()):
            start, end = int(ptr[fi]), int(ptr[fi + 1])
            for index in range(start, end):
                if pool[index] != 1:
                    continue
                x1, y1, x2, y2 = [float(value) for value in tensors["box"][index]]
                score = float(tensors["objectness"][index])
                pred_lines.append(f"{int(frame)+1},{int(tensors['track_id'][index])},"
                                  f"{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},"
                                  f"{score:.6f},-1,-1,-1\n")
        (directory / "gt.txt").write_text("".join(gt_lines))
        (directory / "predict.txt").write_text("".join(pred_lines))
        seq_lines.append(f"{video}+{expression}")
    (root / "seqmap.txt").write_text("\n".join(seq_lines) + "\n")
    command = [PY, str(EVAL_RUN), "--METRICS", "HOTA", "CLEAR", "Identity",
               "--SEQMAP_FILE", str((root / "seqmap.txt").resolve()),
               "--SKIP_SPLIT_FOL", "True", "--GT_FOLDER", str(result.resolve()),
               "--TRACKERS_FOLDER", str(result.resolve()),
               "--TRACKERS_TO_EVAL", str(result.resolve()),
               "--GT_LOC_FORMAT", "{gt_folder}{video_id}/{expression_id}/gt.txt",
               "--USE_PARALLEL", "False", "--PRINT_ONLY_COMBINED", "False",
               "--PLOT_CURVES", "False"]
    with (root / "trackeval.log").open("w") as handle:
        subprocess.run(command, cwd=str(EVAL_RUN.parent), stdout=handle,
                       stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "RMOT_IMG_ROOT": str(
                           ROOT / "data/kitti_tracking_training/image_02")})
    return result


def read_combined(root: Path) -> dict:
    path = root / "pedestrian_detailed.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("seq") == "COMBINED":
                return {key: float(row[key]) * 100.0 for key in (
                    "HOTA___AUC", "DetA___AUC", "AssA___AUC",
                    "DetRe___AUC", "DetPr___AUC", "IDF1") if key in row}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crude-root", default="outputs/l18/dual_banks/kitti")
    parser.add_argument("--long-root", default="outputs/l19/dual_banks/kitti")
    parser.add_argument("--out", default="outputs/l19/eval/reserve_identity.json")
    parser.add_argument("--skip-trackeval", action="store_true")
    args = parser.parse_args()
    payload = {"matching": "bank candidate_gt at IoU>=.50; GT only for train-val audit",
               "crude": {"videos": [], "aggregate": {}},
               "long": {"videos": [], "aggregate": {}}}
    for name, root in (("crude", Path(args.crude_root)), ("long", Path(args.long_root))):
        rows = []
        for bank_path in sorted(root.glob("*.pt")):
            if bank_path.with_suffix(".labels.json").exists():
                rows.append(custom_metrics(bank_path))
        payload[name]["videos"] = rows
        numeric = [key for key, value in rows[0].items()
                   if isinstance(value, (int, float))] if rows else []
        sum_keys = {
            "visible_gt_instances", "reserve_observations", "matched_gt_instances",
            "fragmentations", "id_switches", "unique_reserve_tracks",
            "duplicate_id_frame_events", "duplicate_gt_frame_events",
        }
        payload[name]["aggregate"] = {
            key: (int(sum(row[key] for row in rows)) if key in sum_keys else
                  float(np.mean([row[key] for row in rows])))
            for key in numeric}
        if not args.skip_trackeval:
            eval_root = write_trackeval_tree(root, Path(args.out).with_name(f"{name}_trackeval"))
            payload[name]["trackeval_combined_percent"] = read_combined(eval_root)
    output = (ROOT / args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({name: payload[name]["aggregate"] for name in ("crude", "long")}, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
