#!/usr/bin/env python3
"""Score every legal native-frame dev query for each R0 checkpoint.

Dev labels are used only for the registered development diagnostics and
shortlist.  Model inputs are prepared without labels; the label attachment
occurs after the model has produced the complete candidate-row scores.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.r0_track_grounding import R0Config, R0TrackGroundingHead  # noqa: E402
from tools.r0_common import (  # noqa: E402
    DEFAULT_LANGUAGE_ROOTS,
    EvalIndex,
    MergedLanguageCache,
    R0RuntimeData,
    VisualCacheIndex,
    check_manifest,
    file_meta,
    model_forward,
    sha256_file,
    standard_flags,
    write_json,
)
from tools.r0_metric_utils import ZeroLogitMetrics  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
EXPECTED_EPOCHS = (2, 4, 6, 8, 10, 12)


def checkpoint_info(path: Path, package: dict[str, Any]) -> dict[str, Any]:
    if package.get("format") != "locatemot-r0-tcgh-checkpoint-v2":
        raise AssertionError(f"invalid R0 checkpoint format: {path}")
    if package.get("driver") != "grouped_ddp_v2" or package.get("grouped_query_training") is not True:
        raise AssertionError(f"R0 checkpoint is not grouped training: {path}")
    if package.get("detector_state_included") or package.get("tracker_state_included"):
        raise AssertionError(f"R0 checkpoint contains forbidden state: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "dataset": str(package["dataset"]), "epoch": int(package["epoch"]),
            "global_step": int(package["global_step"]), "model_config": package["model_config"],
            "format": package["format"], "grouped_query_training": True,
            "primary_selection_eligible": bool(package.get("primary_selection_eligible", False)),
            "strict_reload": True}


def load_model(path: Path, device: torch.device) -> tuple[R0TrackGroundingHead, dict[str, Any]]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    info = checkpoint_info(path, package)
    model = R0TrackGroundingHead(R0Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict R0 dev checkpoint load failed: {loaded}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, info


def make_score_record(record: dict[str, Any], sample: dict[str, Any], output: dict[str, torch.Tensor],
                      query_index: int, dense: EvalIndex) -> dict[str, Any]:
    runtime_batch = sample["batch"]
    query_batch = sample["query_batches"][query_index]
    supervision = sample["supervisions"][query_index]
    values = output["membership_logit"][query_index].detach().float().cpu()
    presence = output["coverage_presence_logit"][query_index].detach().float().cpu()
    scores = [float(value) for value in values.tolist()]
    if len(scores) != runtime_batch.candidate_count or not bool(torch.isfinite(values).all()):
        raise AssertionError(f"R0 dev score shape/finite drift: {record['unit_key']}")
    if not bool(torch.isfinite(presence).all()):
        raise FloatingPointError(f"R0 dev presence nonfinite: {record['unit_key']}")
    labels = [bool(value) for value in supervision["labels"].tolist()]
    candidate_gt = [None if value is None else str(value) for value in supervision["candidate_gt"]]
    if len(labels) != len(scores) or len(candidate_gt) != len(scores):
        raise AssertionError(f"R0 dev label/score length drift: {record['unit_key']}")
    row_keys = [list(value) for value in query_batch.row_keys]
    if len(row_keys) != len(scores) or len({tuple(value) for value in row_keys}) != len(row_keys):
        raise AssertionError(f"R0 dev row key drift: {record['unit_key']}")
    target_ids = [str(value) for value in supervision["target_ids"]]
    return {
        "format": "locatemot-r0-dev-score-record-v1",
        "unit_key": str(record["unit_key"]), "group_key": f"{record['dataset']}|{record['video']}|{record['frame_id']}",
        "dataset": str(record["dataset"]), "video": str(record["video"]),
        "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]),
        "sentence": str(record["sentence"]), "candidate_count": len(scores),
        "bank_path": str(query_batch.bank_path), "row_offsets": [int(value) for value in query_batch.row_offsets],
        "row_keys": row_keys, "candidate_indices": [int(value) for value in query_batch.candidate_indices],
        "track_ids": [int(value) for value in query_batch.track_ids],
        "pool_ids": [int(value) for value in query_batch.pool_ids],
        "boxes_xyxy": query_batch.boxes.float().tolist(), "image_size_wh": list(query_batch.image_size),
        "score": scores, "membership_logit": scores,
        "presence_logit": float(presence), "coverage_presence_logit": float(presence),
        "null_logit": 0.0, "registered_rule": "membership_logit>=0 and coverage_presence_logit>=0",
        "labels": labels, "candidate_gt": candidate_gt, "target_ids": target_ids,
        "positive_row_offsets": [int(value) for value, label in zip(query_batch.row_offsets, labels) if label],
        "positive_count": int(sum(labels)), "target_present": bool(supervision["target_present"]),
        "candidate_present": bool(supervision["candidate_present"]),
        "present_uncovered": bool(supervision["present_uncovered"]),
        "partially_covered": bool(supervision["partially_covered"]),
        "visible_target_count": int(supervision["visible_target_count"]),
        "covered_target_count": int(supervision["covered_target_count"]),
        "coverage_fraction": float(supervision["coverage_fraction"]),
        "coverage_mask": [bool(value) for value in supervision["membership_mask"].tolist()],
        "category": str(supervision["category"]), "label_source": str(record.get("bank_path", "L69 sidecar")),
        "future_history_count": 0, "labels_attached_after_feature_construction": True,
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        "finite_scores": True,
    }


def prepare_group(runtime: R0RuntimeData, dense: EvalIndex, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Prepare without labels, then attach legal dev labels after scoring."""
    sample = runtime.prepare_group(records, attach_labels=False)
    sample["query_batches"] = []
    sample["supervisions"] = []
    return sample


def attach_after_score(runtime: R0RuntimeData, dense: EvalIndex, sample: dict[str, Any],
                       records: list[dict[str, Any]]) -> None:
    dataset = str(records[0]["dataset"]); video = str(records[0]["video"]); frame = int(records[0]["frame_id"])
    for record in records:
        raw_label = dense.get_label(record)
        batch = runtime.store.build_frame(dataset, video, int(record["query_id"]), frame, str(record["sentence"]))
        supervision = runtime.store.attach_frame_labels(batch, raw_label["target_ids"])
        if str(supervision["category"]) != str(raw_label["category"]):
            raise AssertionError(f"R0 dev category drift: {record['unit_key']}")
        if int(supervision["positive_row_count"]) != int(raw_label["positive_row_count"]):
            raise AssertionError(f"R0 dev positive count drift: {record['unit_key']}")
        sample["query_batches"].append(batch)
        sample["supervisions"].append(supervision)


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 dev score output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable, *sys.argv])
    base = {
        "format": "locatemot-r0-dev-scoring-v1", "status": "incomplete", "command": command,
        "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD, "seed": SEED,
        "dataset": str(args.dataset), "dense_root": str(args.dense_root.resolve()),
        "visual_cache": str(args.visual_cache.resolve()), "query_batch_size": int(args.query_batch_size),
        "outputs": {"root": str(out)}, "manifest_sha256": check_manifest(),
        "registered_rule": "membership_logit>=0 and coverage_presence_logit>=0; zero-logit no threshold fitting",
        "preview_only": bool(args.checkpoint is not None),
        "checkpoint_selection_run": bool(args.checkpoint is None),
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "failure_root_cause": None, "next_action": "build legal dev shortlist",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 dev scoring cwd: {Path.cwd()}")
        dense = EvalIndex(args.dense_root)
        visual = VisualCacheIndex(args.visual_cache)
        language = MergedLanguageCache([Path(value) for value in args.language_root])
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for record in dense.label_records:
            grouped[(str(record["video"]), int(record["frame_id"]))].append(record)
        groups = [sorted(values, key=lambda value: int(value["query_id"])) for _key, values in sorted(grouped.items())]
        if not groups:
            raise AssertionError("R0 dev index contains no query/frame groups")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("R0 dev scoring requested CUDA but it is unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        if args.checkpoint is not None:
            checkpoint_paths = [args.checkpoint.resolve()]
            if not checkpoint_paths[0].is_file():
                raise FileNotFoundError(checkpoint_paths[0])
        else:
            checkpoint_paths = sorted(
                (path for path in args.checkpoint_dir.resolve().glob(f"checkpoint_r0_{args.dataset}_epoch*.pt")
                 if "_preview" not in path.stem),
                key=lambda path: int(path.stem.split("epoch")[-1]),
            )
            actual_epochs = tuple(int(path.stem.split("epoch")[-1]) for path in checkpoint_paths)
            if actual_epochs != EXPECTED_EPOCHS:
                raise AssertionError(f"R0 checkpoint epochs drift: {actual_epochs} != {EXPECTED_EPOCHS}")
        checkpoint_summaries: list[dict[str, Any]] = []
        runtime = R0RuntimeData(dense, visual, language, device)
        for checkpoint_path in checkpoint_paths:
            model, info = load_model(checkpoint_path, device)
            epoch_out = out / f"epoch{int(info['epoch']):02d}"
            epoch_out.mkdir(parents=True, exist_ok=False)
            records_path = epoch_out / "score_records.jsonl"
            accumulator = ZeroLogitMetrics()
            record_count = 0
            candidate_rows = 0
            with records_path.open("w", encoding="utf-8") as handle:
                with torch.inference_mode():
                    for group_index, group in enumerate(groups):
                        for start in range(0, len(group), int(args.query_batch_size)):
                            chunk = group[start:start + int(args.query_batch_size)]
                            sample = prepare_group(runtime, dense, chunk)
                            output = model_forward(model, sample)
                            attach_after_score(runtime, dense, sample, chunk)
                            if output["membership_logit"].shape != (len(chunk), int(sample["candidate_count"])):
                                raise AssertionError(f"R0 dev output shape drift at group {group_index}")
                            for query_index, record in enumerate(chunk):
                                scored = make_score_record(record, sample, output, query_index, dense)
                                handle.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")
                                accumulator.add(scored)
                                record_count += 1
                                candidate_rows += int(scored["candidate_count"])
                            del sample, output
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        if (group_index + 1) % 20 == 0:
                            handle.flush()
                handle.flush()
            metrics = accumulator.finish()
            if record_count != len(dense.label_records):
                raise AssertionError(f"R0 dev score record count drift epoch={info['epoch']}: {record_count}")
            summary = {
                "format": "locatemot-r0-dev-score-summary-v1", "status": "complete",
                "dataset": args.dataset, "checkpoint_info": info, "record_count": record_count,
                "candidate_rows": candidate_rows, "score_records": str(records_path.resolve()),
                "score_records_sha256": sha256_file(records_path), "metrics": metrics,
                "dense_summary_sha256": sha256_file(args.dense_root / "summary.json"),
                "visual_manifest_sha256": sha256_file(args.visual_cache / "manifest.jsonl"),
                "language_cache_entries": language.entry_count, "wall_seconds": time.perf_counter() - started,
                "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "no_persistent_raw_dense_cache_created": True,
                "preview_only": bool(args.checkpoint is not None),
                "checkpoint_selection_run": bool(args.checkpoint is None),
                "next_action": "inspect preview only" if args.checkpoint is not None else "include this checkpoint in the legal dev shortlist",
            }
            write_json(epoch_out / "metrics.json", summary)
            write_json(epoch_out / "provenance.json", summary | {"format": "locatemot-r0-dev-score-provenance-v1"})
            write_json(epoch_out / "status.json", summary)
            checkpoint_summaries.append(summary)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        payload = {**base, "status": "complete", "checkpoint_summaries": checkpoint_summaries,
                   "group_count": len(groups), "record_count_per_checkpoint": len(dense.label_records),
                   "wall_seconds": time.perf_counter() - started,
                   "next_action": "inspect preview only" if args.checkpoint is not None else "build a maximum-three legal dev shortlist per benchmark"}
        write_json(out / "score_manifest.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-r0-dev-scoring-provenance-v1"})
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 dev scoring — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=False, default=None)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--language-root", type=Path, action="append", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="score exactly one grouped v2 checkpoint in preview mode")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.language_root is None:
        args.language_root = [Path(value) for value in DEFAULT_LANGUAGE_ROOTS]
    if args.checkpoint is not None and args.checkpoint_dir is None:
        args.checkpoint_dir = args.checkpoint.parent
    if args.checkpoint is None and args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required in formal scoring mode")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
