"""Lightweight, grouped Stage L20 KITTI evaluator.

This evaluator is deliberately separate from the L19 row-level evaluator:
one result row represents one observation group, so a main/reserve duplicate
cannot be emitted twice.  Outputs are atomic per query and contain only the
fields required for threshold/ranking/TrackEval analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402
from tools.eval_l18_carr import (  # noqa: E402
    load_model, metadata, trainval_queries,
)
from tools.l20_common import (  # noqa: E402
    BankStore, TextStore, l20_frame_features, l20_frame_targets,
)
from tools.train_l19 import l19_track_membership_index  # noqa: E402


def load_gt_boxes(video: str) -> dict[int, dict[str, list[float]]]:
    path = ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
    if not path.exists():
        path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
    record = pickle.load(path.open("rb"))
    return {
        int(frame["frame"]): {
            str(key): value for key, value in frame.get("gt_boxes", {}).items()
        }
        for frame in record["frames"]
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_manifest(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text())
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists() and sidecar.read_text().split()[0] != digest:
        raise ValueError(f"manifest SHA sidecar mismatch: {path}")
    if payload.get("selection_uses_model_scores", True):
        raise ValueError("L20 manifest may not depend on model scores")
    return payload, digest


def auc(scores, labels) -> float | None:
    scores = np.asarray(scores, np.float64).reshape(-1)
    labels = np.asarray(labels, bool).reshape(-1)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) /
                 (positive * negative))


def stats(values) -> dict:
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90))}


def ranking_summary(data: dict[str, np.ndarray]) -> dict:
    frame = data["frame"]
    score = data["score"]
    label = data["current_match"].astype(bool)
    source = data["source"]
    ranks, first_ranks, reciprocal, aps = [], [], [], []
    source_scores = {"main_pos": [], "main_neg": [],
                     "reserve_pos": [], "reserve_neg": [],
                     "cross_pos": [], "cross_neg": []}
    for frame_id in np.unique(frame):
        indices = np.flatnonzero(frame == frame_id)
        order = indices[np.argsort(-score[indices], kind="stable")]
        ordered = label[order]
        positions = np.flatnonzero(ordered)
        if len(positions):
            first_ranks.append(int(positions[0]) + 1)
            reciprocal.append(1.0 / float(positions[0] + 1))
            aps.append(float(np.mean([
                ordered[:position + 1].mean() for position in positions
            ])))
            rank_by_index = {int(index): rank
                             for rank, index in enumerate(order, 1)}
            ranks.extend([rank_by_index[int(index)] for index in
                          indices[label[indices]]])
        for source_id, name in ((0, "main"), (1, "reserve"), (2, "cross")):
            pool = source[indices] == source_id
            source_scores[f"{name}_pos"].extend(
                score[indices[pool & label[indices]]].tolist())
            source_scores[f"{name}_neg"].extend(
                score[indices[pool & ~label[indices]]].tolist())
    return {
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "frame_ap": float(np.mean(aps)) if aps else None,
        "positive_frame_rank": stats(ranks),
        "first_positive_rank": stats(first_ranks),
        "main_positive_vs_main_negative_auc": auc(
            source_scores["main_pos"] + source_scores["main_neg"],
            [True] * len(source_scores["main_pos"]) +
            [False] * len(source_scores["main_neg"])),
        "reserve_positive_vs_reserve_negative_auc": auc(
            source_scores["reserve_pos"] + source_scores["reserve_neg"],
            [True] * len(source_scores["reserve_pos"]) +
            [False] * len(source_scores["reserve_neg"])),
        "reserve_positive_vs_main_negative_auc": auc(
            source_scores["reserve_pos"] + source_scores["main_neg"],
            [True] * len(source_scores["reserve_pos"]) +
            [False] * len(source_scores["main_neg"])),
        "main_positive_vs_reserve_negative_auc": auc(
            source_scores["main_pos"] + source_scores["reserve_neg"],
            [True] * len(source_scores["main_pos"]) +
            [False] * len(source_scores["reserve_neg"])),
        "main_positive_score": stats(source_scores["main_pos"]),
        "main_negative_score": stats(source_scores["main_neg"]),
        "reserve_positive_score": stats(source_scores["reserve_pos"]),
        "reserve_negative_score": stats(source_scores["reserve_neg"]),
    }


def threshold_summary(data: dict[str, np.ndarray], threshold: float) -> dict:
    selected = data["score"] >= float(threshold)
    labels = data["current_match"].astype(bool)
    tp = int(np.count_nonzero(selected & labels))
    fp = int(np.count_nonzero(selected & ~labels))
    fn = int(np.count_nonzero(~selected & labels))
    result = {
        "threshold": float(threshold), "selected": int(selected.sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
    }
    source = data["source"]
    for source_id, name in ((0, "main"), (1, "reserve"), (2, "cross_pool")):
        pool = source == source_id
        positives = pool & labels
        chosen = pool & selected
        pool_tp = int(np.count_nonzero(chosen & labels))
        result[f"{name}_selected"] = int(chosen.sum())
        result[f"{name}_positive"] = int(positives.sum())
        result[f"{name}_selected_recall"] = float(
            pool_tp / max(1, int(positives.sum())))
        result[f"{name}_selected_precision"] = float(
            pool_tp / max(1, int(chosen.sum())))
    frames = data["frame"]
    null_target = data["null_target"].astype(bool)
    per_frame = []
    absent, uncovered = [], []
    for frame_id in np.unique(frames):
        indices = np.flatnonzero(frames == frame_id)
        frame_selected = bool(selected[indices].any())
        frame_null = bool(null_target[indices][0]) if len(indices) else False
        per_frame.append(int(selected[indices].sum()))
        if frame_null:
            absent.append(frame_selected)
            if data["state"][indices][0] == 3:
                uncovered.append(frame_selected)
    result["absent_or_uncovered_frame_fpr"] = float(
        np.mean(absent)) if absent else None
    result["present_uncovered_frame_fpr"] = float(
        np.mean(uncovered)) if uncovered else None
    result["prediction_count_per_frame"] = stats(per_frame)
    result["predictions_per_positive_group"] = float(
        selected.sum() / max(1, labels.sum()))
    result["cross_pool_duplicate_double_selection_rate"] = 0.0
    result["null_frames"] = int(len(absent))
    return result


def threshold_grid(data: dict[str, np.ndarray]) -> float:
    labels = data["current_match"].astype(bool)
    scores = data["score"]
    if not len(scores) or not labels.any():
        return 0.0
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.995, 96)))
    best = None
    for threshold in candidates.tolist() + [float(scores.min()), float(scores.max())]:
        summary = threshold_summary(data, float(threshold))
        precision, recall = summary["precision"], summary["recall"]
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        # Prefer precision once recall is in the useful range; this prevents
        # calibration from selecting the L19-style million-FP operating point.
        useful = recall >= 0.45
        key = (int(useful), f1, precision, recall, -abs(float(threshold)))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def run_query(model, bank, entry, text_store, device, gt_by_frame) -> tuple[dict, dict]:
    if "l19_track_membership" not in bank:
        bank["l19_track_membership"] = l19_track_membership_index(bank)
    text = str(entry.get("sentence", entry.get("expression", "")))
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    family = expression_family_vector(text).to(device)
    tokens, mask = text_store.get(text, device)
    with torch.no_grad():
        context = model.query_context(tokens, query, family, mask)
    tensors = bank["tensors"]
    state = {}
    arrays = {key: [] for key in (
        "frame", "track_id", "box", "score", "raw_logit", "null_logit",
        "source", "group_id", "group_size", "cross_pool", "current_match",
        "membership", "observation", "gt_iou", "null_target", "state",
    )}
    with torch.no_grad():
        for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
            features, track_ids, begin, end = l20_frame_features(
                bank, frame_index, device)
            if end <= begin:
                continue
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model(features, query, family, track_ids, state,
                               query_tokens=tokens, query_mask=mask,
                               query_context=context)
            state = output["state"]
            raw = output["logits"].float().cpu().numpy()
            null = float(output["null_logit"].float().cpu())
            score = raw - null if output.get("grouping_enabled", True) and \
                model.use_null else raw
            selected = output["group_row_indices"].detach().cpu().numpy()
            group_ids = output["group_ids"].detach().cpu().numpy()
            group_source = output["group_source"].detach().cpu().numpy()
            group_sizes = output["group_sizes"].detach().cpu().numpy()
            group_membership = output["membership_logits"].float().cpu().numpy()
            group_observation = output["observation_logits"].float().cpu().numpy()
            target = l20_frame_targets(
                bank, begin, end, entry, int(frame_id),
                bank["l19_track_membership"])
            member_rows = output.get("group_member_rows")
            if member_rows is None:
                member_rows = [torch.as_tensor([value]) for value in selected.tolist()]
            target_values = {}
            for name, row_name in (("membership", "row_membership"),
                                   ("observation", "row_match")):
                row_values = np.asarray(target[row_name], np.float32)
                target_values[name] = [
                    float(row_values[members.detach().cpu().numpy()].max())
                    if len(members) else 0.0 for members in member_rows]
            target_values["group_target"] = target_values["observation"]
            frame_gt = gt_by_frame.get(int(frame_id), {})
            boxes = tensors["box"][begin:end].numpy().astype(np.float32)
            for index, local_row in enumerate(selected.tolist()):
                box = boxes[local_row]
                target_ids = target["target_ids"]
                iou = max((
                    max(0.0, min(float(box[2]), float(gbox[2])) -
                             max(float(box[0]), float(gbox[0]))) *
                    max(0.0, min(float(box[3]), float(gbox[3])) -
                             max(float(box[1]), float(gbox[1]))) /
                    max(1e-6, (max(0.0, float(box[2] - box[0])) *
                               max(0.0, float(box[3] - box[1])) +
                               max(0.0, float(gbox[2] - gbox[0])) *
                               max(0.0, float(gbox[3] - gbox[1])) -
                               max(0.0, min(float(box[2]), float(gbox[2])) -
                                    max(float(box[0]), float(gbox[0]))) *
                               max(0.0, min(float(box[3]), float(gbox[3])) -
                                    max(float(box[1]), float(gbox[1])))))
                    for gid, gbox in frame_gt.items() if str(gid) in target_ids
                ), default=0.0)
                arrays["frame"].append(int(frame_id))
                arrays["track_id"].append(int(track_ids[local_row]))
                arrays["box"].append(box.tolist())
                arrays["score"].append(float(score[index]))
                arrays["raw_logit"].append(float(raw[index]))
                arrays["null_logit"].append(null)
                arrays["source"].append(int(group_source[index]))
                arrays["group_id"].append(int(group_ids[index]))
                arrays["group_size"].append(int(group_sizes[index]))
                arrays["cross_pool"].append(int(group_source[index] == 2))
                arrays["current_match"].append(int(target_values["group_target"][index]))
                arrays["membership"].append(int(target_values["membership"][index]))
                arrays["observation"].append(int(target_values["observation"][index]))
                arrays["gt_iou"].append(float(iou))
                arrays["null_target"].append(int(target["null_target"]))
                arrays["state"].append(int(target["state"]))
    np_arrays = {
        "frame": np.asarray(arrays["frame"], np.int32),
        "track_id": np.asarray(arrays["track_id"], np.int64),
        "box": np.asarray(arrays["box"], np.float32).reshape(-1, 4),
        "score": np.asarray(arrays["score"], np.float32),
        "raw_logit": np.asarray(arrays["raw_logit"], np.float32),
        "null_logit": np.asarray(arrays["null_logit"], np.float32),
        "source": np.asarray(arrays["source"], np.int8),
        "group_id": np.asarray(arrays["group_id"], np.int64),
        "group_size": np.asarray(arrays["group_size"], np.int16),
        "cross_pool": np.asarray(arrays["cross_pool"], np.uint8),
        "current_match": np.asarray(arrays["current_match"], np.uint8),
        "membership": np.asarray(arrays["membership"], np.uint8),
        "observation": np.asarray(arrays["observation"], np.uint8),
        "gt_iou": np.asarray(arrays["gt_iou"], np.float32),
        "null_target": np.asarray(arrays["null_target"], np.uint8),
        "state": np.asarray(arrays["state"], np.int8),
    }
    return np_arrays, {
        "rows": int(len(np_arrays["score"])),
        "ranking": ranking_summary(np_arrays),
    }


def write_full_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2",
                         "score", "raw_logit", "null_logit", "source",
                         "group_id", "current_match", "gt_iou"])
        for index in range(len(data["score"])):
            writer.writerow([
                int(data["frame"][index]), int(data["track_id"][index]),
                *[float(v) for v in data["box"][index]],
                float(data["score"][index]), float(data["raw_logit"][index]),
                float(data["null_logit"][index]), int(data["source"][index]),
                int(data["group_id"][index]),
                int(data["current_match"][index]), float(data["gt_iou"][index]),
            ])
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-manifest", default=
                        "outputs/l19/protocol/kitti_fast_eval_manifest.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--text-root", default="outputs/l18/data/text_cache")
    parser.add_argument("--split", choices=("all", "calibration", "screening"),
                        default="all")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-full-csv", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard topology")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    manifest_path = (ROOT / args.query_manifest).resolve()
    manifest, manifest_sha = load_manifest(manifest_path)
    all_queries, _gt, _seqmap, _sequences, _protocol = trainval_queries(
        "trainval_kitti")
    lookup = metadata("kitti_v2")
    selected = []
    for row in sorted(manifest["queries"], key=lambda value: value["query_index"]):
        if args.split != "all" and row["split"] != args.split:
            continue
        query_index = int(row["query_index"])
        if query_index < 0 or query_index >= len(all_queries):
            raise ValueError(f"query index outside train-val metadata: {query_index}")
        video, expression, _spec = all_queries[query_index]
        if (video, expression) != (row["video"], row["expression"]):
            raise ValueError(f"manifest query ordering mismatch: {row}")
        selected.append(row)
    if args.max_queries:
        selected = selected[:args.max_queries]
    assigned = [row for position, row in enumerate(selected)
                if position % args.num_shards == args.shard_index]
    base_root = (ROOT / args.out_root).resolve()
    run_root = base_root / f"shard_{args.shard_index}" \
        if args.num_shards > 1 else base_root
    run_root.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model((ROOT / args.checkpoint).resolve(), device)
    model.eval()
    store = BankStore((ROOT / args.bank_root).resolve(), cache_size=1)
    text_store = TextStore((ROOT / args.text_root).resolve())
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    cfg_sha = hashlib.sha256(json.dumps(
        checkpoint.get("cfg", {}), sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    gt_cache = {}
    completed = []
    started = time.time()
    for position, row in enumerate(assigned):
        index = int(row["query_index"])
        result_path = run_root / "queries" / f"q{index:05d}.json"
        cache_path = run_root / "scores" / f"q{index:05d}.npz"
        marker = run_root / "complete" / f"q{index:05d}.complete"
        if args.resume and result_path.exists() and cache_path.exists() and marker.exists():
            completed.append(index)
            continue
        video, expression = row["video"], row["expression"]
        entry = lookup[(video, expression)]
        bank = store.get("kitti", video)
        if video not in gt_cache:
            gt_cache[video] = load_gt_boxes(video)
        data, query_summary = run_query(
            model, bank, entry, text_store, device, gt_cache[video])
        atomic_npz(cache_path, data)
        if args.write_full_csv:
            write_full_csv(run_root / "csv" / f"q{index:05d}.csv", data)
        atomic_json(result_path, {
            "complete": True, "query_index": index, "video": video,
            "expression": expression, "split": row["split"],
            "data_split": "train_val",
            "manifest_sha256": manifest_sha, "checkpoint_sha256": checkpoint_sha,
            "cfg_sha256": cfg_sha, **query_summary,
        })
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("complete\n")
        completed.append(index)
        if position == 0 or (position + 1) % 8 == 0:
            print(f"[l20-fast] shard={args.shard_index} "
                  f"query={position + 1}/{len(assigned)} index={index}", flush=True)
    expected = [int(row["query_index"]) for row in assigned]
    payload = {
        "complete": sorted(completed) == sorted(expected),
        "dataset": "trainval_kitti", "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha, "cfg_sha256": cfg_sha,
        "data_split": "train_val",
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "split": args.split, "expected_query_indices": expected,
        "completed_query_indices": sorted(completed),
        "wall_seconds": time.time() - started,
    }
    atomic_json(run_root / "run_manifest.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
