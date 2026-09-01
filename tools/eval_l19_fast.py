"""Fast, resumable Stage L19 KITTI screening evaluator.

The evaluator intentionally does not export the diagnostic ten-million-row
CSV.  Each query is an independent atomic unit containing only the score,
box, source, and train-val matching labels needed for ranking/threshold
analysis.  Multiple workers write separate ``shard_N`` directories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402
from locatemot.rmot.l19_reserve_identity import box_iou  # noqa: E402
from tools.eval_l18_carr import load_model, metadata, trainval_queries  # noqa: E402
from tools.train_l18_carr import (  # noqa: E402
    BankStore, TextStore, frame_features, l19_frame_targets,
    l19_track_membership_index,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_expression(text: str) -> str:
    return str(text).replace("/", "_")


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
    if sidecar.exists():
        recorded = sidecar.read_text().split()[0]
        if recorded != digest:
            raise ValueError(f"manifest sha256 sidecar mismatch: {path}")
    if payload.get("selection_uses_model_scores", True):
        raise ValueError("fast manifest must be independent of model scores")
    return payload, digest


def load_gt_boxes(video: str) -> dict[int, dict[str, list[float]]]:
    path = ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
    if not path.exists():
        path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
    import pickle
    record = pickle.load(path.open("rb"))
    return {
        int(frame["frame"]): {
            str(key): value for key, value in frame.get("gt_boxes", {}).items()
        }
        for frame in record["frames"]
    }


def ranking_summary(frame: np.ndarray, scores: np.ndarray,
                    labels: np.ndarray, source: np.ndarray) -> dict:
    """Compute threshold-free positive rank/AP statistics for one query."""
    first_ranks, positive_ranks, reciprocal = [], [], []
    frame_aps = []
    reserve_positive_ranks, main_negative_scores = [], []
    reserve_positive_scores = []
    for frame_id in np.unique(frame):
        indices = np.flatnonzero(frame == frame_id)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        ordered_labels = labels[order].astype(bool)
        positives = np.flatnonzero(ordered_labels)
        if not len(positives):
            continue
        rank_by_global = {int(index): int(rank)
                          for rank, index in enumerate(order, 1)}
        values = np.asarray([rank_by_global[int(index)] for index in indices])
        positive_ranks.extend(values[labels[indices].astype(bool)].tolist())
        first = int(positives[0]) + 1
        first_ranks.append(first)
        reciprocal.append(1.0 / first)
        precisions = []
        for position in positives:
            precisions.append(float(ordered_labels[:position + 1].mean()))
        frame_aps.append(float(np.mean(precisions)))
        reserve = (source[indices] == 1) & labels[indices].astype(bool)
        main_negative = (source[indices] == 0) & ~labels[indices].astype(bool)
        reserve_positive_ranks.extend(values[reserve].tolist())
        reserve_positive_scores.extend(scores[indices][reserve].tolist())
        main_negative_scores.extend(scores[indices][main_negative].tolist())
    def stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        values = np.asarray(values, np.float64)
        return {
            "count": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90)),
        }
    return {
        "positive_frame_rank": stats(positive_ranks),
        "first_positive_rank": stats(first_ranks),
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "frame_ap": float(np.mean(frame_aps)) if frame_aps else None,
        "reserve_positive_rank": stats(reserve_positive_ranks),
        "reserve_positive_score": stats(reserve_positive_scores),
        "main_negative_score": stats(main_negative_scores),
    }


def threshold_summary(scores: np.ndarray, labels: np.ndarray,
                      source: np.ndarray, threshold: float | None) -> dict:
    if threshold is None:
        return {"threshold": None}
    selected = scores >= float(threshold)
    tp = int(np.count_nonzero(selected & labels))
    fp = int(np.count_nonzero(selected & ~labels))
    fn = int(np.count_nonzero(~selected & labels))
    result = {
        "threshold": float(threshold), "selected": int(selected.sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
    }
    for source_id, name in ((0, "main"), (1, "reserve")):
        pool = source == source_id
        pool_positive = pool & labels
        pool_selected = pool & selected
        pool_tp = int(np.count_nonzero(pool_selected & labels))
        result[f"{name}_selected"] = int(pool_selected.sum())
        result[f"{name}_positive"] = int(pool_positive.sum())
        result[f"{name}_selected_recall"] = float(
            pool_tp / max(1, int(pool_positive.sum())))
        result[f"{name}_selected_precision"] = float(
            pool_tp / max(1, int(pool_selected.sum())))
    return result


def run_query(model, bank: dict, entry: dict, text_store: TextStore,
              threshold: float | None, device: torch.device,
              gt_by_frame: dict[int, dict[str, list[float]]]) -> tuple[dict, dict]:
    if "l19_track_membership" not in bank:
        bank["l19_track_membership"] = l19_track_membership_index(bank)
    text = str(entry.get("sentence", entry.get("expression", "")))
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    family = expression_family_vector(text).to(device)
    tokens, mask = text_store.get(text, device)
    with torch.no_grad():
        context = model.query_context(tokens, query, family, mask)
    state = {}
    frames, track_ids_all, boxes_all = [], [], []
    scores_all, sources_all, labels_all, ious_all = [], [], [], []
    tensors = bank["tensors"]
    with torch.no_grad():
        for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
            features, track_ids, begin, end = frame_features(
                bank, frame_index, device)
            if end <= begin:
                continue
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model(
                    features, query, family, track_ids, state,
                    query_tokens=tokens, query_mask=mask,
                    query_context=context,
                )
            state = output["state"]
            scores = output["logits"].float().detach().cpu().numpy()
            source = features.get("pool_id", torch.zeros(
                len(track_ids), dtype=torch.long, device=device)).long()
            source = source.detach().cpu().numpy().astype(np.int8)
            candidate_gt = bank.get("candidate_gt", [None] *
                                    len(tensors["track_id"]))[begin:end]
            target = l19_frame_targets(
                bank, begin, end, entry, int(frame_id),
                bank["l19_track_membership"])
            target_ids = set(target["target_ids"])
            frame_gt = gt_by_frame.get(int(frame_id), {})
            boxes = tensors["box"][begin:end].numpy().astype(np.float32)
            labels, ious = [], []
            for local, box in enumerate(boxes):
                labels.append(float(
                    candidate_gt[local] is not None and
                    str(candidate_gt[local]) in target_ids))
                ious.append(max((box_iou(box, frame_gt[gt_id])
                                 for gt_id in target_ids if gt_id in frame_gt),
                                default=0.0))
            count = len(scores)
            frames.extend([int(frame_id)] * count)
            track_ids_all.extend(track_ids.detach().cpu().tolist())
            boxes_all.extend(boxes.tolist())
            scores_all.extend(scores.tolist())
            sources_all.extend(source.tolist())
            labels_all.extend(labels)
            ious_all.extend(ious)
    arrays = {
        "frame": np.asarray(frames, np.int32),
        "track_id": np.asarray(track_ids_all, np.int64),
        "box": np.asarray(boxes_all, np.float32).reshape(-1, 4),
        "score": np.asarray(scores_all, np.float32),
        "source": np.asarray(sources_all, np.int8),
        "current_match": np.asarray(labels_all, np.uint8),
        "gt_iou": np.asarray(ious_all, np.float32),
    }
    labels = arrays["current_match"].astype(bool)
    summary = {
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "source_rows": {
            "main": int(np.count_nonzero(arrays["source"] == 0)),
            "reserve": int(np.count_nonzero(arrays["source"] == 1)),
        },
        "threshold": threshold_summary(
            arrays["score"], labels, arrays["source"], threshold),
        "ranking": ranking_summary(
            arrays["frame"], arrays["score"], labels, arrays["source"]),
    }
    return arrays, summary


def write_full_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2",
                         "score", "source", "current_match", "gt_iou"])
        for index in range(len(arrays["score"])):
            writer.writerow([
                int(arrays["frame"][index]), int(arrays["track_id"][index]),
                *[float(value) for value in arrays["box"][index]],
                float(arrays["score"][index]), int(arrays["source"][index]),
                int(arrays["current_match"][index]),
                float(arrays["gt_iou"][index]),
            ])
    os.replace(temporary, path)


def main() -> None:
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
    parser.add_argument("--minimal-output", action="store_true", default=True)
    parser.add_argument("--write-full-csv", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")
    manifest_path = (ROOT / args.query_manifest).resolve()
    manifest, manifest_sha = load_manifest(manifest_path)
    all_queries, _gt_root, _seqmap, _sequences, _protocol = trainval_queries(
        "trainval_kitti")
    query_lookup = {(video, expression): (index, spec)
                    for index, (video, expression, spec) in enumerate(all_queries)}
    selected = []
    for row in sorted(manifest["queries"], key=lambda value: value["query_index"]):
        key = (row["video"], row["expression"])
        if row["query_index"] not in range(len(all_queries)) or \
                key not in query_lookup:
            raise ValueError(f"manifest query is not in current train-val metadata: {row}")
        if args.split != "all" and row["split"] != args.split:
            continue
        selected.append(row)
    if args.max_queries:
        selected = selected[:args.max_queries]
    assigned = [row for position, row in enumerate(selected)
                if position % args.num_shards == args.shard_index]
    base_root = (ROOT / args.out_root).resolve()
    run_root = base_root / f"shard_{args.shard_index}" \
        if args.num_shards > 1 else base_root
    run_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    model.eval()
    store = BankStore((ROOT / args.bank_root).resolve(), cache_size=1)
    text_store = TextStore((ROOT / args.text_root).resolve())
    lookup = metadata("kitti_v2")
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    cfg_sha = hashlib.sha256(json.dumps(
        checkpoint.get("cfg", {}), sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    started = time.time()
    completed = []
    for position, row in enumerate(assigned):
        index = int(row["query_index"])
        cache_path = run_root / "scores" / f"q{index:05d}.npz"
        result_path = run_root / "queries" / f"q{index:05d}.json"
        marker = run_root / "complete" / f"q{index:05d}.complete"
        if args.resume and marker.exists() and cache_path.exists() and result_path.exists():
            completed.append(index)
            continue
        video, expression = row["video"], row["expression"]
        entry = dict(lookup[(video, expression)])
        entry["spec"] = query_lookup[(video, expression)][1].tolist()
        bank = store.get("kitti", video)
        arrays, summary = run_query(
            model, bank, entry, text_store, args.threshold, device,
            load_gt_boxes(video))
        atomic_npz(cache_path, arrays)
        if args.write_full_csv:
            write_full_csv(run_root / "full_csv" / f"q{index:05d}.csv", arrays)
        result = {
            "complete": True, "query_index": index, "video": video,
            "expression": expression, "split": row["split"],
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": manifest_sha, "cfg_sha256": cfg_sha,
            **summary,
        }
        atomic_json(result_path, result)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker_temp = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
        marker_temp.write_text("complete\n")
        os.replace(marker_temp, marker)
        completed.append(index)
        print(f"[l19-fast] {args.split} {position + 1}/{len(assigned)} "
              f"query={index} rows={summary['rows']}", flush=True)
    run_payload = {
        "complete": len(completed) == len(assigned),
        "dataset": "trainval_kitti", "split": args.split,
        "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "model_name": checkpoint.get("model_name"), "cfg_sha256": cfg_sha,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "query_indices": [int(row["query_index"]) for row in assigned],
        "completed_query_indices": sorted(completed),
        "threshold": args.threshold, "minimal_output": True,
        "write_full_csv": bool(args.write_full_csv),
        "wall_seconds": time.time() - started,
    }
    atomic_json(run_root / "run_manifest.json", run_payload)
    print(json.dumps(run_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
