"""Train-validation and official TrackEval runner for L18."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.flexhook_bank_port import FlexHookBankPort  # noqa: E402
from locatemot.models.l18_coverage_retrieve_repair import L18CARRRetriever  # noqa: E402
from locatemot.models.l19_ungated_retriever import L19UngatedRetriever  # noqa: E402
from locatemot.models.l19_flexhook_correspondence import (  # noqa: E402
    L19FlexHookCorrespondence,
)
from locatemot.models.l20_source_invariant_set_correspondence import (  # noqa: E402
    L20SourceInvariantSetCorrespondence,
)
from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402
from locatemot.rmot.track_bank import load_track_bank  # noqa: E402
from tools.eval_l13_rmot import (  # noqa: E402
    DANCE_GT, DANCE_SEQMAP, EVAL_RUN, PY, V1_DATA, V1_GT, V1_SEQMAP,
    V2_GT, V2_SEQMAP, load_queries,
)
from tools.train_l18_carr import (  # noqa: E402
    BankStore, TextStore, frame_features, load_items,
)


def load_record(video: str) -> dict:
    path = ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
    if not path.exists():
        path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
    return pickle.load(path.open("rb"))


def metadata(dataset: str) -> dict[tuple[str, str], dict]:
    if dataset == "kitti_v1":
        paths = (V1_DATA / "expressions.json",)
    elif dataset == "kitti_v2":
        paths = (ROOT / "outputs/l11/data/rmot_kitti/expressions.json",
                 ROOT / "outputs/l16/data/kitti_missing/records/expressions.json")
    else:
        paths = (ROOT / "outputs/l16/data/protocol/refer_dance_expressions.json",
                 ROOT / "outputs/l8/data/rmot_eval/expressions.json")
    out = {}
    for path in paths:
        if not path.exists():
            continue
        for video, entries in json.loads(path.read_text()).items():
            for entry in entries:
                expression = str(entry.get("expression", entry.get("sentence", "")))
                out[(video, expression)] = entry
    return out


def trainval_queries(kind: str) -> tuple[list[tuple[str, str, np.ndarray]], Path, Path, set[str], str]:
    protocol = json.loads((ROOT / "outputs/l16/data/protocol/split_manifest.json").read_text())
    if kind == "trainval_kitti":
        dataset = "kitti_v2"
        sequences = set(protocol["kitti_v2"]["train_val"])
        gt_root = ROOT / "outputs/l18/data/trainval_gt/kitti"
        seqmap = gt_root / "seqmap.txt"
    elif kind == "trainval_dance":
        dataset = "dance"
        sequences = set(protocol["refer_dance"]["train_val"])
        gt_root = ROOT / "outputs/l18/data/trainval_gt/dance"
        seqmap = gt_root / "seqmap.txt"
    else:
        raise ValueError(kind)
    lookup = metadata(dataset)
    queries = []
    for (video, expression), entry in lookup.items():
        if video not in sequences:
            continue
        labels = entry.get("label", {})
        if not any(bool(values) for values in labels.values()):
            continue
        queries.append((video, expression,
                        np.asarray(entry["spec"], np.float32)))
    queries.sort(key=lambda x: (x[0], x[1]))
    return queries, gt_root, seqmap, sequences, dataset


def official_queries(kind: str):
    if kind == "kitti_v1":
        queries, gt, seqmap = load_queries(kind)
        return queries, gt, seqmap, {x[0] for x in queries}, kind
    if kind == "kitti_v2":
        queries, gt, seqmap = load_queries(kind)
        return queries, gt, seqmap, {x[0] for x in queries}, kind
    if kind == "dance":
        queries, gt, seqmap = load_queries(kind)
        return queries, gt, seqmap, {x[0] for x in queries}, kind
    raise ValueError(kind)


def write_trainval_gt(kind: str, queries, gt_root: Path):
    dataset = "kitti_v2" if kind == "trainval_kitti" else "dance"
    lookup = metadata(dataset)
    for video, expression, _spec in queries:
        destination = gt_root / video / expression / "gt.txt"
        if destination.exists():
            if destination.is_symlink():
                continue
            lines = destination.read_text().splitlines()
            if not lines or len(lines[0].split(",")) == 9:
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if dataset == "dance":
            source = ROOT / "data/refer_dance/gt_template" / video / expression / "gt.txt"
            if source.exists():
                destination.symlink_to(source.resolve())
                continue
        entry = lookup[(video, expression)]
        labels = entry.get("label", {})
        record = load_record(video) if dataset == "kitti_v2" else None
        frame_map = ({int(fr["frame"]): fr for fr in record["frames"]}
                     if record is not None else {})
        lines = []
        if dataset == "dance":
            path = ROOT / "outputs/l8/data/rmot_train" / f"{video}.pkl"
            record = pickle.load(path.open("rb"))
            frame_map = {int(fr["frame"]): fr for fr in record["frames"]}
        for frame, fr in sorted(frame_map.items()):
            ids = labels.get(str(frame), labels.get(frame, []))
            boxes = fr.get("gt_boxes", {})
            for gid in ids:
                box = boxes.get(str(gid))
                if box is None:
                    continue
                x1, y1, x2, y2 = [float(x) for x in box]
                frame_number = frame + (1 if dataset == "kitti_v2" else 0)
                lines.append(f"{frame_number},{gid},{x1:.3f},{y1:.3f},"
                             f"{x2-x1:.3f},{y2-y1:.3f},1,1,1\n")
        destination.write_text("".join(lines))
    gt_root.mkdir(parents=True, exist_ok=True)
    (gt_root / "seqmap.txt").write_text("\n".join(
        f"{video}+{expression}" for video, expression, _ in queries) + "\n")


def link_official_gt(kind: str, queries, out_root: Path):
    if kind == "kitti_v1":
        gt_root = V1_GT
    elif kind == "kitti_v2":
        gt_root = V2_GT
    else:
        gt_root = DANCE_GT
    result_root = out_root / "uidm18"
    for video, expression, _spec in queries:
        src = gt_root / video / expression / "gt.txt"
        dst = result_root / video / expression / "gt.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    if checkpoint["model_name"] == "flexhook":
        model = FlexHookBankPort(
            cfg["hidden"], cfg["heads"], use_slots=not cfg.get("no_slots", False),
            holistic_only=cfg.get("holistic_only", False))
    elif checkpoint["model_name"] == "carr":
        model = L18CARRRetriever(
            cfg["hidden"], cfg["heads"], use_slots=not cfg.get("no_slots", False),
            holistic_only=cfg.get("holistic_only", False),
            use_coverage=not cfg.get("no_coverage", False))
    elif checkpoint["model_name"] == "l19":
        model = L19UngatedRetriever(
            cfg["hidden"], cfg["heads"],
            use_slots=cfg.get("use_slots", not cfg.get("no_slots", True)),
            holistic_only=cfg.get("holistic_only", True),
            coverage_mode=cfg.get("coverage_mode", "aux_only"))
    elif checkpoint["model_name"] == "l19_flexhook_correspondence":
        model = L19FlexHookCorrespondence(
            cfg["hidden"], cfg["heads"],
            dropout=cfg.get("dropout", 0.10),
            token_dim=cfg.get("token_dim", 512),
            temporal_points=cfg.get("temporal_points", 8),
            hook_points=cfg.get("hook_points", 10))
    elif checkpoint["model_name"] == "l20_sint_set":
        model = L20SourceInvariantSetCorrespondence(
            cfg["hidden"], cfg["heads"],
            dropout=cfg.get("dropout", 0.10),
            token_dim=cfg.get("token_dim", 512),
            temporal_points=cfg.get("temporal_points", 8),
            hook_points=cfg.get("hook_points", 10),
            use_source_adapters=cfg.get("use_source_adapters", True),
            use_grouping=cfg.get("use_grouping", True),
            use_null=cfg.get("use_null", True))
    else:
        raise ValueError(checkpoint["model_name"])
    state_dict = dict(checkpoint["model"])
    # The first 100-step smoke checkpoint predates the reliability-gated
    # holistic fusion (512 -> 519 inputs).  Preserve its learned weights and
    # initialize only the seven newly introduced reliability columns to zero.
    key = "slots.holistic_fusion.0.weight"
    expected = model.state_dict().get(key)
    legacy = state_dict.get(key)
    if expected is not None and legacy is not None and expected.shape != legacy.shape:
        if (expected.ndim == legacy.ndim == 2 and
                expected.shape[0] == legacy.shape[0] and
                expected.shape[1] > legacy.shape[1]):
            upgraded = expected.detach().clone()
            upgraded[:, :legacy.shape[1]] = legacy
            state_dict[key] = upgraded
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def predict_query(model, model_name: str, bank: dict, entry: dict,
                  text_store: TextStore, device: torch.device,
                  threshold: float, hysteresis: float = 0.0,
                  max_per_frame: int = 0, dance_frames: bool = False,
                  calibrator: str = "raw"):
    if calibrator not in {"raw", "query_zscore"}:
        raise ValueError(calibrator)
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    family = expression_family_vector(
        entry.get("sentence", entry.get("expression", ""))).to(device)
    text = str(entry.get("sentence", entry.get("expression", "")))
    tokens, mask = text_store.get(text, device)
    context = model.query_context(tokens, query, family, mask)
    tensors = bank["tensors"]
    state = {}
    selected_state = {}
    rows = []
    score_rows = []
    coverage_counts = []
    logit_min = float("inf")
    logit_max = float("-inf")
    logit_sum = 0.0
    logit_count = 0
    for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
        features, track_ids, begin, end = frame_features(bank, frame_index, device)
        if end <= begin:
            continue
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(features, query, family, track_ids, state,
                           query_tokens=tokens, query_mask=mask,
                           query_context=context)
        state = output["state"]
        logits = output["logits"].float()
        if len(logits):
            logit_min = min(logit_min, float(logits.min()))
            logit_max = max(logit_max, float(logits.max()))
            logit_sum += float(logits.sum())
            logit_count += int(logits.numel())
        probability = torch.sigmoid(logits)
        frame_number = int(frame_id) if dance_frames else int(frame_id) + 1
        keep_values = []
        for index, raw_id in enumerate(track_ids.detach().cpu().tolist()):
            absolute = begin + index
            x1, y1, x2, y2 = [float(x) for x in tensors["box"][absolute]]
            score_rows.append([
                frame_number, int(raw_id), x1, y1, x2 - x1, y2 - y1,
                float(probability[index]), float(logits[index]),
            ])
            if calibrator == "raw":
                was_selected = selected_state.get(int(raw_id), False)
                boundary = threshold - hysteresis if was_selected else threshold
                selected = bool(float(logits[index]) >= boundary)
                selected_state[int(raw_id)] = selected
                if selected:
                    keep_values.append(index)
        keep = torch.as_tensor(keep_values, dtype=torch.long, device=device)
        if calibrator == "raw" and max_per_frame > 0 and len(keep) > max_per_frame:
            order = torch.argsort(probability[keep], descending=True)
            keep = keep[order[:max_per_frame]]
        if "state_probabilities" in output:
            coverage_counts.append(
                torch.argmax(output["state_probabilities"]).item())
        if calibrator == "raw":
            for local in keep.detach().cpu().tolist():
                index = begin + local
                x1, y1, x2, y2 = [float(x) for x in tensors["box"][index]]
                rows.append([frame_number, int(tensors["track_id"][index]),
                             x1, y1, x2 - x1, y2 - y1,
                             float(probability[local]), -1, -1, -1])
    calibration = {"name": calibrator}
    if calibrator == "query_zscore":
        values = np.asarray(score_rows, dtype=np.float32).reshape(-1, 8)
        logits = values[:, 7] if len(values) else np.zeros(0, np.float32)
        mean = float(logits.mean()) if len(logits) else 0.0
        std = float(logits.std()) if len(logits) else 1.0
        std = max(std, 1e-6)
        calibration.update({"mean": mean, "std": std})
        calibrated = (logits - mean) / std
        for frame in np.unique(values[:, 0]).tolist() if len(values) else []:
            indices = np.flatnonzero((values[:, 0] == frame) &
                                     (calibrated >= float(threshold)))
            if max_per_frame > 0 and len(indices) > max_per_frame:
                order = np.argsort(-values[indices, 6], kind="stable")
                indices = indices[order[:max_per_frame]]
            for index in indices.tolist():
                row = values[index]
                rows.append([int(row[0]), int(row[1]), float(row[2]),
                             float(row[3]), float(row[4]), float(row[5]),
                             float(row[6]), -1, -1, -1])
    return rows, {
        "selected": len(rows), "coverage_argmax": coverage_counts,
        "logit_min": logit_min if logit_count else None,
        "logit_max": logit_max if logit_count else None,
        "logit_mean": logit_sum / logit_count if logit_count else None,
        "calibrator": calibration,
    }, score_rows


def write_prediction(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(",".join(
        f"{value:.6f}" if isinstance(value, float) else str(value)
        for value in row) + "\n" for row in rows))


def score_path(root: Path, dataset: str, video: str, expression: str) -> Path:
    safe_expression = expression.replace("/", "_")
    return root / dataset / video / f"{safe_expression}.npz"


def save_score_cache(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(rows, dtype=np.float32).reshape(-1, 8)
    np.savez_compressed(path, rows=values)


def materialize_score_cache(path: Path, prediction: Path, threshold: float,
                            max_per_frame: int = 0,
                            calibrator: str = "raw"):
    values = np.asarray(np.load(path, allow_pickle=False)["rows"],
                        dtype=np.float32).reshape(-1, 8)
    selected = []
    logits = values[:, 7] if len(values) else np.zeros(0, np.float32)
    if calibrator == "query_zscore":
        mean = float(logits.mean()) if len(logits) else 0.0
        std = max(1e-6, float(logits.std()) if len(logits) else 1.0)
        scores = (logits - mean) / std
    else:
        mean, std, scores = 0.0, 1.0, logits
    for frame in np.unique(values[:, 0]).tolist() if len(values) else []:
        frame_indices = np.flatnonzero(values[:, 0] == frame)
        frame_rows = values[frame_indices]
        keep = scores[frame_indices] >= float(threshold)
        frame_rows = frame_rows[keep]
        if max_per_frame > 0 and len(frame_rows) > max_per_frame:
            order = np.argsort(-frame_rows[:, 6], kind="stable")
            frame_rows = frame_rows[order[:max_per_frame]]
        selected.extend(frame_rows.tolist())
    rows = [[int(row[0]), int(row[1]), float(row[2]), float(row[3]),
             float(row[4]), float(row[5]), float(row[6]), -1, -1, -1]
            for row in selected]
    write_prediction(prediction, rows)
    logits = values[:, 7] if len(values) else np.zeros(0, np.float32)
    return {
        "selected": len(rows), "coverage_argmax": [],
        "logit_min": float(logits.min()) if len(logits) else None,
        "logit_max": float(logits.max()) if len(logits) else None,
        "logit_mean": float(logits.mean()) if len(logits) else None,
        "calibrator": {"name": calibrator, "mean": mean, "std": std},
    }


def run_trackeval(kind: str, out_root: Path, seqmap: Path, sequences: set[str],
                  allowed_pairs: set[tuple[str, str]] | None = None):
    result_root = (out_root / "uidm18").resolve()
    if kind in ("kitti_v1", "kitti_v2", "trainval_kitti"):
        image_root = ROOT / "data/kitti_tracking_training/image_02"
    else:
        image_root = ROOT / "data/refer_dance/DanceTrack/training/image_02"
    map_path = out_root / "seqmap_l18.txt"
    lines = []
    for line in seqmap.read_text().splitlines():
        if not line.strip():
            continue
        pair = tuple(line.strip().split("+", 1))
        if pair[0] not in sequences:
            continue
        if allowed_pairs is not None and pair not in allowed_pairs:
            continue
        lines.append(line.strip())
    map_path.write_text("\n".join(lines) + "\n")
    command = [PY, str(EVAL_RUN), "--METRICS", "HOTA", "CLEAR", "Identity",
               "--SEQMAP_FILE", str(map_path.resolve()), "--SKIP_SPLIT_FOL", "True",
               "--GT_FOLDER", str(result_root), "--TRACKERS_FOLDER", str(result_root),
               "--TRACKERS_TO_EVAL", str(result_root), "--GT_LOC_FORMAT",
               "{gt_folder}{video_id}/{expression_id}/gt.txt", "--USE_PARALLEL", "False",
               "--PRINT_ONLY_COMBINED", "False", "--PLOT_CURVES", "False"]
    env = dict(os.environ)
    env["RMOT_IMG_ROOT"] = str(image_root)
    log = out_root / "trackeval_l18.log"
    with log.open("w") as handle:
        subprocess.run(command, cwd=str(EVAL_RUN.parent), env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    detailed = result_root / "pedestrian_detailed.csv"
    metrics = {}
    if detailed.exists():
        with detailed.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("seq") == "COMBINED":
                    for name in ("HOTA___AUC", "DetA___AUC", "AssA___AUC",
                                 "DetRe___AUC", "DetPr___AUC", "IDF1"):
                        if name in row:
                            metrics[name] = float(row[name]) * 100.0
    return metrics, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("kitti_v1", "kitti_v2", "dance",
                                               "trainval_kitti", "trainval_dance"),
                        required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bank-root", default="outputs/l18/dual_banks")
    parser.add_argument("--text-root", default="outputs/l18/data/text_cache")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--calibrator", choices=("raw", "query_zscore"),
                        default="raw")
    parser.add_argument("--hysteresis", type=float, default=0.0)
    parser.add_argument("--max-per-frame", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--score-cache-root", default="")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = (ROOT / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if args.dataset.startswith("trainval"):
        queries, gt_root, seqmap, sequences, protocol_kind = trainval_queries(args.dataset)
        write_trainval_gt(args.dataset, queries, gt_root)
    else:
        queries, gt_root, seqmap, sequences, protocol_kind = official_queries(args.dataset)
    if args.max_queries:
        queries = queries[:args.max_queries]
        sequences = {x[0] for x in queries}
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")
    if args.num_shards > 1 and not args.eval_only:
        queries = [query for query in queries if int(hashlib.md5(
            (query[0] + "+" + query[1]).encode()).hexdigest(), 16)
            % args.num_shards == args.shard]
    score_root = ((ROOT / args.score_cache_root).resolve()
                  if args.score_cache_root else None)
    if args.materialize_only and score_root is None:
        raise ValueError("--materialize-only requires --score-cache-root")
    if args.dataset.startswith("trainval"):
        result_root = out_root / "uidm18"
        for video, expression, _ in queries:
            src = gt_root / video / expression / "gt.txt"
            dst = result_root / video / expression / "gt.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())
    else:
        link_official_gt(args.dataset, queries, out_root)
    if args.eval_only:
        allowed = {(video, expression) for video, expression, _ in queries}
        metrics, log = run_trackeval(args.dataset, out_root, seqmap, sequences,
                                     allowed)
        print(json.dumps({"dataset": args.dataset, "metrics": metrics,
                          "log": str(log)}, indent=2), flush=True)
        return
    model = None
    checkpoint = None
    if not args.materialize_only:
        model, checkpoint = load_model(Path(args.checkpoint), device)
    bank_root = (ROOT / args.bank_root).resolve()
    store = BankStore(bank_root, cache_size=1)
    text_store = TextStore((ROOT / args.text_root).resolve())
    model_name = checkpoint["model_name"] if checkpoint is not None else "score_cache"
    grouped = {}
    for query in queries:
        grouped.setdefault(query[0], []).append(query)
    timings = []
    started = time.time()
    entry_lookup = metadata("dance" if args.dataset in
                            ("dance", "trainval_dance") else "kitti_v2")
    for video_index, (video, entries) in enumerate(sorted(grouped.items())):
        dataset_name = "dance_eval" if args.dataset in ("dance", "trainval_dance") \
            else "kitti"
        bank = store.get(dataset_name, video)
        for _video, expression, spec in entries:
            entry = dict(entry_lookup.get((video, expression), {
                "expression": expression, "sentence": expression,
                "spec": spec.tolist(),
            }))
            entry["spec"] = spec.tolist()
            prediction = out_root / "uidm18" / video / expression / "predict.txt"
            cached_scores = (score_path(score_root, args.dataset, video, expression)
                             if score_root is not None else None)
            if args.materialize_only:
                if cached_scores is None or not cached_scores.exists():
                    raise FileNotFoundError(cached_scores)
                diag = materialize_score_cache(
                    cached_scores, prediction, args.threshold,
                    args.max_per_frame, args.calibrator)
                timings.append({"video": video, "expression": expression,
                                **diag, "source": "score_cache"})
                continue
            if prediction.exists():
                continue
            rows, diag, score_rows = predict_query(
                model, model_name, bank, entry, text_store, device,
                args.threshold, args.hysteresis, args.max_per_frame,
                dance_frames=args.dataset in ("dance", "trainval_dance"),
                calibrator=args.calibrator)
            if cached_scores is not None:
                save_score_cache(cached_scores, score_rows)
            if not args.score_only:
                write_prediction(prediction, rows)
            timings.append({"video": video, "expression": expression, **diag})
        print(f"[l18-eval] {args.dataset} video={video} "
              f"{video_index + 1}/{len(grouped)} elapsed={time.time()-started:.1f}",
              flush=True)
    manifest_name = ("prediction_manifest.json" if args.num_shards == 1 else
                     f"prediction_manifest_shard{args.shard}of{args.num_shards}.json")
    manifest = out_root / manifest_name
    manifest.write_text(json.dumps({
        "dataset": args.dataset, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "threshold": args.threshold, "hysteresis": args.hysteresis,
        "calibrator": args.calibrator,
        "max_per_frame": args.max_per_frame, "queries": timings,
        "wall_seconds": time.time() - started,
    }, indent=2) + "\n")
    if not args.predict_only and not args.score_only:
        allowed = {(video, expression) for video, expression, _ in queries}
        metrics, log = run_trackeval(args.dataset, out_root, seqmap, sequences,
                                     allowed)
        print(json.dumps({"dataset": args.dataset, "metrics": metrics,
                          "log": str(log)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
