#!/usr/bin/env python3
"""L89 full-video inference over the frozen L85 Z1 semantic cache.

This tool is intentionally separate from both training and TrackEval.  It
scores every cached legal L69 row with the selected L89 candidate set and
materializes only the tracker text files needed by the local TrackEval
adapter.  Labels are loaded only after all candidate scores/predictions for a
candidate and rule are frozen.  No detector, CLIP, LoRA, top-k, NMS, or new
feature cache is used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
Z1_CACHE = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
LANG_CACHE = WORK_ROOT / "outputs/l89/cache/language_tokens_retry1"
RULES = ("B", "R", "P")
INTERNAL_VIDEOS = {
    "refer_kitti_v1": ("0004", "0018"),
    "refer_kitti_v2": ("0016", "0017", "0020"),
}

for path in (WORK_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import locatemot.models as _models_package  # noqa: E402
import locatemot.rmot as _rmot_package  # noqa: E402
for package, name in ((_models_package, "models"), (_rmot_package, "rmot")):
    value = str(WORK_ROOT / "locatemot" / name)
    if value not in [str(x) for x in package.__path__]:
        package.__path__.append(value)

from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore  # noqa: E402
from locatemot.rmot.l85_runtime import (  # noqa: E402
    build_groups,
    load_fit_train_dev_groups,
    load_internal_eval_groups,
)
from locatemot.rmot.l89_language_cache import L89LanguageTokenCache  # noqa: E402
from locatemot.rmot.l49_data import load_l49_queries  # noqa: E402
from l88c_eval_metrics import corrected_emission_mask  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def key_only(row: dict[str, Any], frame_id: int | None = None) -> dict[str, Any]:
    sentence = str(row.get("sentence") or row.get("expression") or "")
    if not sentence:
        raise AssertionError(f"empty L89 full-video sentence: {row.get('unit_key')}")
    frame = int(row["frame_id"] if frame_id is None else frame_id)
    return {
        "unit_key": f"{row['dataset']}|{row['video']}|{int(row['query_id'])}|{frame}",
        "dataset": str(row["dataset"]), "video": str(row["video"]),
        "query_id": int(row["query_id"]), "frame_id": frame,
        "sentence": sentence, "expression": sentence,
    }


class L85CacheIndex:
    """Path index for compact L85 items; tensors are loaded one group at a time."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        summary = json.loads((self.root / "summary.json").read_text())
        if summary.get("status") != "complete" or summary.get("labels_in_cache"):
            raise AssertionError("L89 requires a complete label-free L85 cache")
        if summary.get("candidate_deletion") or summary.get("candidate_truncation"):
            raise AssertionError("L85 cache candidate contract is invalid")
        self.summary_sha256 = sha256_file(self.root / "summary.json")
        self.paths: dict[str, Path] = {}
        for path in sorted(self.root.rglob("*.pt")):
            item = torch.load(path, map_location="cpu", weights_only=False)
            group_key = str(item.get("group_key", ""))
            if not group_key or group_key in self.paths:
                raise AssertionError(f"L85 cache group key collision/missing: {path}")
            if item.get("labels_in_cache") or item.get("candidate_deletion") or item.get("candidate_truncation"):
                raise AssertionError(f"invalid label-free L85 item: {group_key}")
            self.paths[group_key] = path
            del item
        if not self.paths:
            raise AssertionError(f"empty L85 cache: {self.root}")

    def read(self, group_key: str) -> dict[str, Any]:
        key = str(group_key)
        path = self.paths.get(key)
        if path is None:
            raise KeyError(f"missing L85 cache group: {key}")
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("labels_in_cache") or item.get("candidate_deletion") or item.get("candidate_truncation"):
            raise AssertionError(f"invalid L85 cache item at read: {key}")
        return item


def scope_groups(scope: str) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, tuple[str, ...]]]:
    if scope == "dev":
        groups, _train_keys, dev_keys = load_fit_train_dev_groups()
        split = json.loads((ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json").read_text())
        dev_videos: dict[str, list[str]] = defaultdict(list)
        for value in split.get("dev_videos", []):
            dataset, video = str(value).split("|", 1)
            dev_videos[dataset].append(video)
        keys = [key for key in sorted(dev_keys) if str(groups[key]["dataset"]) in dev_videos and
                str(groups[key]["video"]) in set(dev_videos[str(groups[key]["dataset"])])]
        if len(keys) != 138:
            raise AssertionError(f"L89 dev group count drift: {len(keys)}")
        return groups, keys, {key: tuple(sorted(set(values))) for key, values in dev_videos.items()}
    if scope == "internal":
        groups, calibration_keys, validation_keys = load_internal_eval_groups()
        # The final internal TrackEval pass is the frozen validation split.
        # Calibration keys are used only by the fixed semantic replay and
        # must not add V1/0016 or V2/0015 sequences to the internal result.
        keys = sorted(set(validation_keys))
        expected = {(dataset, video) for dataset, videos in INTERNAL_VIDEOS.items() for video in videos}
        actual = {(str(groups[key]["dataset"]), str(groups[key]["video"])) for key in keys}
        if actual != expected:
            raise AssertionError(f"L89 internal video set drift: {sorted(actual)}")
        return groups, keys, {dataset: tuple(videos) for dataset, videos in INTERNAL_VIDEOS.items()}
    raise ValueError(scope)


def unique_queries(groups: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for key in keys:
        group = groups[key]
        for raw in group["queries"]:
            row = key_only(raw)
            qid = int(row["query_id"])
            current = result[str(row["dataset"])][str(row["video"])].get(qid)
            if current is not None and str(current["sentence"]) != str(row["sentence"]):
                raise AssertionError(f"sentence drift: {row['unit_key']}")
            result[str(row["dataset"])][str(row["video"])].setdefault(qid, row)
    return {dataset: {video: [values[qid] for qid in sorted(values)] for video, values in videos.items()}
            for dataset, videos in result.items()}


def prepare_strategy(root: Path, datasets: list[str], query_map: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for dataset in datasets:
        ds = root / dataset
        paths = {
            "root": ds, "gt": ds / "gt", "trackers": ds / "trackers",
            "tracker_data": ds / "trackers" / "l89" / "data", "seqmap": ds / "seqmap.txt",
        }
        for path in paths.values():
            if path.suffix != ".txt":
                path.mkdir(parents=True, exist_ok=True)
        result[dataset] = paths
    return result


def sequence_id(video: str, query_id: int) -> str:
    return f"{str(video)}__q{int(query_id):05d}"


def sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def load_record(video: str) -> tuple[Path, dict[str, Any]]:
    candidates = [ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl",
                 ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"]
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no legal train-pool record for {video}")
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with path.open("rb") as handle:
        return path, pickle.load(handle)


def native_frame_ids(store: L80BankStore, video: str) -> list[int]:
    store._store.load_video(str(video))
    values = [int(value) for value in store._store.tensors["frame_ids"].tolist()]
    if values != sorted(values) or len(values) != len(set(values)):
        raise AssertionError(f"L69 frame order drift: {video}")
    return values


def materialize_gt(scope: str, dataset: str, queries_by_video: dict[str, list[dict[str, Any]]],
                   paths: dict[str, Path], store: L80BankStore) -> dict[str, Any]:
    """Attach only the legal fit/dev or internal labels after predictions freeze."""
    allowed_videos = set(queries_by_video)
    metadata = {int(row["query_id"]): row for row in load_l49_queries(dataset)
                if str(row.get("video")) in allowed_videos}
    sequences: list[str] = []
    query_audits: list[dict[str, Any]] = []
    record_audits: list[dict[str, Any]] = []
    for video in sorted(queries_by_video):
        record_path, record = load_record(video)
        frames = {int(value["frame"]): value for value in record["frames"]}
        bank_frames = native_frame_ids(store, video)
        if set(frames) != set(bank_frames):
            raise AssertionError(f"bank/GT frame drift: {dataset}|{video}")
        record_audits.append({"video": video, "record_path": str(record_path.resolve()),
                              "record_sha256": sha256_file(record_path), "frame_count": len(frames),
                              "bank_frame_count": len(bank_frames), "scope": scope})
        width, height = [int(x) for x in record.get("image_size", [0, 0])]
        if width <= 0 or height <= 0:
            width, height = [int(x) for x in store._store.bank["metadata"]["image_size"]]
        for query in queries_by_video[video]:
            qid = int(query["query_id"])
            if qid not in metadata:
                raise KeyError(f"legal query metadata missing: {dataset}|{video}|{qid}")
            entry = metadata[qid]
            if str(entry["sentence"]) != str(query["sentence"]):
                raise AssertionError(f"sentence mismatch: {dataset}|{video}|{qid}")
            target = entry.get("target", {})
            seq = sequence_id(video, qid); sequences.append(seq)
            gt_dir = paths["gt"] / seq; gt_dir.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []; gt_rows = 0; target_frames = 0
            for frame in bank_frames:
                ids = target.get(int(frame), target.get(str(frame), set()))
                ids = set(ids or [])
                if ids:
                    target_frames += 1
                boxes = frames[int(frame)].get("gt_boxes", {})
                for raw_id in sorted(str(value) for value in ids):
                    box = boxes.get(raw_id, boxes.get(int(raw_id)) if raw_id.isdigit() else None)
                    if box is None:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box]
                    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
                        raise AssertionError(f"invalid GT box: {dataset}|{video}|{qid}|{frame}|{raw_id}")
                    lines.append(f"{int(frame) + 1},{int(raw_id)},{x1:.6f},{y1:.6f},"
                                 f"{x2-x1:.6f},{y2-y1:.6f},1,1,1\n")
                    gt_rows += 1
            (gt_dir / "gt.txt").write_text("".join(lines), encoding="utf-8")
            (gt_dir / "seqinfo.ini").write_text(
                "[Sequence]\n" f"name={seq}\n" "imDir=img1\n" "frameRate=10\n"
                f"seqLength={max(bank_frames) + 1}\n" f"imWidth={width}\n" f"imHeight={height}\n" "imExt=.png\n", encoding="utf-8")
            query_audits.append({"dataset": dataset, "video": video, "query_id": qid,
                                 "sequence": seq, "gt_rows": gt_rows, "target_present_frames": target_frames,
                                 "labels_attached_after_predictions": True,
                                 "label_source": str(entry.get("label_source", "L49 legal fit/dev/internal"))})
    paths["seqmap"].write_text("name\n" + "\n".join(sequences) + "\n", encoding="utf-8")
    return {"sequence_count": len(sequences), "sequences": sequences,
            "query_gt_audits": query_audits, "record_audits": record_audits,
            "labels_attached_after_predictions": True,
            "label_scope": "fit/dev" if scope == "dev" else "internal validation"}


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def emission_descriptor(paths: dict[str, Path], queries: list[dict[str, Any]]) -> dict[str, float]:
    """Compact post-hoc descriptors used only as the registered dev tie fields."""
    matched_ids: set[str] = set(); all_ids: set[str] = set(); gt_rows = matched = 0
    empty_frames = accepted_empty_frames = 0
    for query in queries:
        seq = sequence_id(str(query["video"]), int(query["query_id"]))
        gt_path = paths["gt"] / seq / "gt.txt"
        pred_path = paths["tracker_data"] / f"{seq}.txt"
        gt_by_frame: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
        pred_by_frame: dict[int, list[list[float]]] = defaultdict(list)
        if gt_path.is_file():
            for line in gt_path.read_text().splitlines():
                if not line.strip():
                    continue
                values = line.split(",")
                frame, target = int(values[0]), str(values[1])
                x, y, w, h = [float(v) for v in values[2:6]]
                gt_by_frame[frame].append((target, [x, y, x + w, y + h]))
        if pred_path.is_file():
            for line in pred_path.read_text().splitlines():
                if not line.strip():
                    continue
                values = line.split(",")
                frame = int(values[0]); x, y, w, h = [float(v) for v in values[2:6]]
                pred_by_frame[frame].append([x, y, x + w, y + h])
        frames = set(gt_by_frame) | set(pred_by_frame)
        for frame in frames:
            targets = gt_by_frame.get(frame, [])
            predictions = pred_by_frame.get(frame, [])
            if not targets:
                empty_frames += 1
                accepted_empty_frames += int(bool(predictions))
            used: set[int] = set()
            for target, box in targets:
                all_ids.add(target); gt_rows += 1
                candidates = sorted(((iou(box, pred), index) for index, pred in enumerate(predictions)
                                     if index not in used), reverse=True)
                if candidates and candidates[0][0] >= 0.5:
                    used.add(candidates[0][1]); matched += 1; matched_ids.add(target)
    return {
        "distinct_target_recall": float(len(matched_ids) / max(1, len(all_ids))),
        "target_row_recall": float(matched / max(1, gt_rows)),
        "inactive_false_acceptance": float(accepted_empty_frames / max(1, empty_frames)),
        "gt_rows": float(gt_rows), "matched_gt_rows": float(matched),
    }


def load_model(checkpoint_info: dict[str, Any], device: torch.device) -> tuple[L89FullRMOT, dict[str, Any]]:
    path = Path(str(checkpoint_info["path"])).resolve()
    if sha256_file(path) != str(checkpoint_info["sha256"]):
        raise AssertionError(f"checkpoint SHA drift: {path}")
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != "locatemot-l89-checkpoint-v1" or int(package.get("seed", -1)) != SEED:
        raise AssertionError(f"invalid L89 package: {path}")
    if str(package.get("manifest_sha256")) != MANIFEST_SHA:
        raise AssertionError("L89 checkpoint manifest drift")
    model = L89FullRMOT(L89Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict L89 reload failed: {loaded}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"path": str(path), "sha256": sha256_file(path), "epoch": int(package["epoch"]),
                   "optimizer_step": int(package["optimizer_step"]), "phase": str(package["phase"]),
                   "model_config": package["model_config"], "strict_reload": True}


def load_frozen_selection_candidate(
    selection_path: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    selection_path = selection_path.resolve()
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "complete":
        raise AssertionError("L89C selection is incomplete")
    if not bool(selection.get("selection_frozen_before_fixed_validation")):
        raise AssertionError("L89C selection was not frozen before fixed validation")
    final = selection.get("final_selection")
    if not isinstance(final, dict):
        raise AssertionError("L89C final_selection missing")
    checkpoint_info = dict(final["checkpoint_info"])
    rule_name = str(final["rule"])
    rule_object = dict(final["rule_object"])
    if rule_name not in {"B", "R", "P"}:
        raise AssertionError(f"invalid L89C rule: {rule_name}")
    for key in ("candidate_threshold", "presence_threshold", "null_margin"):
        if key not in rule_object:
            raise AssertionError(f"L89C frozen rule missing {key}")
        value = float(rule_object[key])
        if not np.isfinite(value):
            raise AssertionError(f"nonfinite L89C threshold {key}")
    checkpoint_path = Path(str(checkpoint_info["path"])).resolve()
    if sha256_file(checkpoint_path) != str(checkpoint_info["sha256"]):
        raise AssertionError("L89C selected checkpoint SHA drift")
    candidate = {
        "shortlist_index": 0,
        "reason": "frozen_corrected_l89c_selection",
        "checkpoint_info": checkpoint_info,
        "rule_fits": {rule_name: rule_object},
        "selection_source": str(selection_path),
        "selection_source_sha256": sha256_file(selection_path),
        "selection_frozen_before_fixed_validation": True,
    }
    return candidate, (rule_name,)


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 full-video output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    store = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89 worktree cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if args.scope not in {"dev", "internal"}:
            raise ValueError(args.scope)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        if args.selection is not None:
            candidate, rule_names = load_frozen_selection_candidate(args.selection)
            candidates = [candidate]
            strategy_source = args.selection.resolve()
            frozen_selection_mode = True
            if args.candidate_index != -1:
                raise ValueError("--candidate-index is invalid with --selection")
        else:
            shortlist_path = args.shortlist.resolve()
            shortlist = json.loads(shortlist_path.read_text())
            if shortlist.get("status") != "complete" or not shortlist.get("shortlist"):
                raise AssertionError("L89 shortlist is incomplete")
            candidates = shortlist["shortlist"]
            if args.candidate_index >= 0:
                if args.candidate_index >= len(candidates):
                    raise IndexError("shortlist candidate index out of range")
                candidates = [candidates[int(args.candidate_index)]]
            rule_names = RULES
            strategy_source = shortlist_path
            frozen_selection_mode = False
        groups, group_keys, videos_by_dataset = scope_groups(args.scope)
        query_map = unique_queries(groups, group_keys)
        datasets = sorted(query_map)
        cache = L85CacheIndex(args.z1_cache.resolve())
        language = L89LanguageTokenCache(args.language_cache.resolve())
        store = L80BankStore(max_history=8)
        summaries: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            checkpoint_info = dict(candidate["checkpoint_info"])
            model, loaded_info = load_model(checkpoint_info, device)
            epoch = int(loaded_info["epoch"])
            candidate_root = out / f"candidate_epoch{epoch:03d}"
            strategy_paths = {rule: prepare_strategy(candidate_root / rule, datasets, query_map) for rule in rule_names}
            audit_handle = (candidate_root / "prediction_audits.jsonl").open("w", encoding="utf-8")
            counters = {rule: {"frames": 0, "queries": 0, "candidate_rows": 0, "selected_rows": 0,
                               "key_digest": hashlib.sha256()} for rule in rule_names}
            seen_group_keys: set[str] = set()
            try:
                for group_index, group_key in enumerate(sorted(group_keys, key=lambda key: (
                        str(groups[key]["dataset"]), str(groups[key]["video"]), int(groups[key]["frame_id"]))), 1):
                    group = groups[group_key]
                    query_rows = [key_only(row) for row in group["queries"]]
                    if not query_rows:
                        raise AssertionError(f"empty L89 full-video group: {group_key}")
                    batch = store.build_unit(query_rows[0])
                    item = cache.read(group_key)
                    if int(item.get("candidate_count", -1)) != batch.candidate_count:
                        raise AssertionError(f"L85/L69 candidate count drift: {group_key}")
                    if [int(x) for x in item.get("row_offsets", [])] != [int(x) for x in batch.row_offsets]:
                        raise AssertionError(f"L85/L69 row offset drift: {group_key}")
                    qids = [int(x) for x in item["query_ids"]]
                    requested_qids = [int(row["query_id"]) for row in query_rows]
                    if qids != requested_qids:
                        raise AssertionError(f"L85/group query order drift: {group_key}")
                    text_tokens, text_mask = language.get_batch(
                        str(batch.dataset), str(batch.video), requested_qids,
                        [str(row["sentence"]) for row in query_rows], device)
                    z1 = item["z1"].float().to(device)
                    text_global = item["text_global"].float().to(device)
                    frame_global = item["frame_global"].float().to(device)
                    if tuple(z1.shape) != (len(query_rows), batch.candidate_count, 256):
                        raise AssertionError(f"L85 z1 shape drift: {group_key}: {tuple(z1.shape)}")
                    if tuple(text_tokens.shape[:2]) != tuple(text_mask.shape) or text_tokens.shape[-1] != 256:
                        raise AssertionError(f"L89 language shape drift: {group_key}")
                    with torch.inference_mode():
                        output = model(
                            z1, text_tokens, text_mask, text_global, frame_global,
                            batch.observations.float().to(device),
                            batch.history_observations.float().to(device),
                            batch.history_mask.to(device),
                            batch.history_frame_ids.to(device),
                            int(batch.frame_id), temporal_enabled=True,
                        )
                    values = output["candidate_energy"].float().detach().cpu().numpy()
                    presence = output["presence_logit"].float().detach().cpu().numpy()
                    null = output["null_logit"].float().detach().cpu().numpy()
                    if values.shape != (len(query_rows), batch.candidate_count) or not np.isfinite(values).all():
                        raise AssertionError(f"L89 full-video score shape/finite drift: {group_key}")
                    if not (np.isfinite(presence).all() and np.isfinite(null).all()):
                        raise AssertionError(f"L89 full-video presence/null finite drift: {group_key}")
                    for local, query in enumerate(query_rows):
                        row_keys = [(str(batch.dataset), str(batch.video), int(query["query_id"]), int(batch.frame_id),
                                     str(batch.bank_path), int(offset)) for offset in batch.row_offsets]
                        if len(row_keys) != batch.candidate_count:
                            raise AssertionError(f"L89 full-video row-key count drift: {group_key}")
                        score = values[local]
                        for rule in rule_names:
                            rule_object = candidate["rule_fits"][rule]
                            candidate_threshold = float(rule_object["candidate_threshold"])
                            presence_threshold = float(rule_object["presence_threshold"])
                            null_margin = float(rule_object["null_margin"])
                            presence_gate = float(presence[local]) >= presence_threshold
                            selected_mask = corrected_emission_mask(
                                score, float(presence[local]), float(null[local]),
                                candidate_threshold, presence_threshold, null_margin)
                            selected = np.flatnonzero(selected_mask)
                            path = strategy_paths[rule][str(batch.dataset)]["tracker_data"] / f"{sequence_id(batch.video, int(query['query_id']))}.txt"
                            with path.open("a", encoding="utf-8") as handle:
                                for index in selected.tolist():
                                    x1, y1, x2, y2 = [float(v) for v in batch.boxes[index].tolist()]
                                    handle.write(f"{int(batch.frame_id)+1},{int(batch.track_ids[index])},{x1:.6f},{y1:.6f},"
                                                 f"{x2-x1:.6f},{y2-y1:.6f},{sigmoid(score[index]):.8f},1,1,1\n")
                            counter = counters[rule]
                            counter["frames"] += 1; counter["queries"] += 1
                            counter["candidate_rows"] += int(batch.candidate_count); counter["selected_rows"] += int(selected.size)
                            counter["key_digest"].update(json.dumps({"unit_key": query["unit_key"], "row_keys": row_keys}, sort_keys=False).encode())
                            audit_handle.write(json.dumps({
                                "format": "locatemot-l89-fullvideo-audit-v1", "scope": args.scope,
                                "rule": rule, "dataset": str(batch.dataset), "video": str(batch.video),
                                "query_id": int(query["query_id"]), "frame_id": int(batch.frame_id),
                                "unit_key": str(query["unit_key"]), "candidate_rows_scored": int(batch.candidate_count),
                                "selected_rows": int(selected.size), "presence_gate": bool(presence_gate),
                                "candidate_vs_null_applied": True,
                                "corrected_candidate_vs_null": True,
                                "candidate_threshold": candidate_threshold,
                                "presence_threshold": presence_threshold,
                                "null_margin": null_margin,
                                "emission_contract": "candidate>=threshold & candidate-null>=margin & presence>=threshold",
                                "candidate_rows_retained": True, "candidate_deletion": False,
                                "candidate_truncation": False, "labels_attached": False,
                            }, ensure_ascii=False) + "\n")
                    seen_group_keys.add(group_key)
                    del item, batch, text_tokens, text_mask, z1, text_global, frame_global, output, values, presence, null
                    store._store._bank = store._store._bank
                    gc.collect()
                    if device.type == "cuda" and group_index % 25 == 0:
                        torch.cuda.empty_cache()
                    if group_index % 50 == 0 or group_index == len(group_keys):
                        print(f"[l89-fullvideo] scope={args.scope} epoch={epoch} group={group_index}/{len(group_keys)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            finally:
                audit_handle.close()
            if seen_group_keys != set(group_keys):
                raise AssertionError("L89 full-video group coverage drift")
            # Ensure every legal query has an explicit (possibly empty) tracker file.
            for rule in rule_names:
                for dataset, videos in query_map.items():
                    for video, queries in videos.items():
                        tracker_dir = strategy_paths[rule][dataset]["tracker_data"]
                        for query in queries:
                            (tracker_dir / f"{sequence_id(video, int(query['query_id']))}.txt").touch(exist_ok=True)
            eval_summary: dict[str, Any] = {}
            emission: dict[str, Any] = {}
            # This is the first point at which legal labels are read.
            for rule in rule_names:
                eval_summary[rule] = {}
                emission[rule] = {}
                for dataset in datasets:
                    eval_summary[rule][dataset] = materialize_gt(
                        args.scope, dataset, query_map.get(dataset, {}), strategy_paths[rule][dataset], store)
                    flat_queries = [query for video in sorted(query_map.get(dataset, {}))
                                    for query in query_map[dataset][video]]
                    emission[rule][dataset] = emission_descriptor(strategy_paths[rule][dataset], flat_queries)
            summaries.append({
                "candidate_index": candidate_index, "checkpoint_info": loaded_info,
                "rules": {rule: {"counters": {**counters[rule], "key_digest": counters[rule]["key_digest"].hexdigest()},
                                 "strategy_paths": {dataset: {name: str(path.resolve()) for name, path in paths.items()}
                                                    for dataset, paths in strategy_paths[rule].items()},
                                 "eval_summary": eval_summary[rule], "emission_descriptor": emission[rule],
                                 "candidate_rows_retained": True, "candidate_deletion": False,
                                 "candidate_truncation": False}
                           for rule in rule_names},
                "prediction_strategy_frozen_before_gt": True,
            })
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        full = True
        payload = {
            "format": "locatemot-l89-fullvideo-v1", "status": "complete", "scope_key": args.scope,
            "scope": "full-video internal fit/dev selection" if args.scope == "dev" else "full-video internal V1/V2 validation",
            "full_video": full, "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "datasets": datasets, "videos": videos_by_dataset, "group_count": len(group_keys),
            "query_count": sum(len(values) for videos in query_map.values() for values in videos.values()),
            "strategy_source": str(strategy_source),
            "strategy_source_sha256": sha256_file(strategy_source),
            "frozen_selection_mode": bool(frozen_selection_mode),
            "shortlist_source": str(strategy_source) if not frozen_selection_mode else None,
            "selection_source": str(strategy_source) if frozen_selection_mode else None,
            "corrected_candidate_vs_null": True,
            "zero_training": True,
            "emission_contract": "candidate>=threshold & candidate-null>=margin & presence>=threshold",
            "candidates": summaries, "candidate_index_filter": int(args.candidate_index),
            "z1_cache": str(args.z1_cache.resolve()), "z1_cache_summary_sha256": cache.summary_sha256,
            "language_cache": str(args.language_cache.resolve()),
            "all_candidate_rows_scored": True, "candidate_deletion": False, "candidate_truncation": False,
            "prediction_strategy_frozen_before_gt": True, "labels_attached_after_predictions": True,
            "fit_dev_labels_only": args.scope == "dev", "internal_validation_labels_only": args.scope == "internal",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "trackeval_run": False,
            "no_hota_or_trackeval": True, "groundingdino_lora_used": False, "groundingdino_trainable": False,
            "bert_trainable": False, "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "manifest_sha256": MANIFEST_SHA, "persistent_dense_cache_written": False,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started, "failure_root_cause": None,
            "next_action": "run L89 TrackEval matrix on frozen strategy outputs",
        }
        write_json(out / "summary.json", payload); write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": "locatemot-l89-fullvideo-v1", "status": "complete",
                                          "corrected_candidate_vs_null": True, "zero_training": True,
                                          "scope": args.scope, "full_video": True, "candidate_count": len(summaries),
                                          "group_count": len(group_keys), "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": False, "no_hota_or_trackeval": True})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89 full-video inference — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {"format": "locatemot-l89-fullvideo-v1", "status": "incomplete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise
    finally:
        if store is not None:
            store._store._bank = None
            store._store._text_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shortlist", type=Path)
    source.add_argument("--selection", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--z1-cache", type=Path, default=Z1_CACHE)
    parser.add_argument("--language-cache", type=Path, default=LANG_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-index", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
