#!/usr/bin/env python3
"""Aggregate completed per-video R1 predictions, then run legal-dev TrackEval.

The per-video replay is prediction-only. This process verifies all six
complete legal-development video attempts, builds a no-copy symlink view, and
only then opens the legal development targets. It never reads screening or
official-test labels and does not run a detector or a model.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r0_common import EvalIndex  # noqa: E402
from tools import r0_infer_true_fullvideo as _r0_full  # noqa: E402
from tools.r0_infer_true_fullvideo import (  # noqa: E402
    materialize_gt,
    prepare_paths,
    sequence_id,
)
from tools.r0_trackeval_matrix import run_dataset as run_trackeval_dataset  # noqa: E402
from tools import r1_infer_legal_dev as _r1_replay  # noqa: E402
from tools.r1_common import (  # noqa: E402
    DEV_ROOTS,
    MANIFEST,
    MANIFEST_SHA,
    THREAD,
    check_manifest,
    file_meta,
    sha256_file,
    write_json,
)
from tools.r1_infer_legal_dev import (  # noqa: E402
    EPOCHS,
    ELIGIBLE_EPOCHS,
    LEGAL_VIDEOS,
    L89E_ANCHOR,
    L89E_SELECTION,
    RULE,
    RULE_NAME,
    SAFE_TARGET_ROOT,
    _legal_descriptor,
    _load_group_records,
    _selection_key,
)


def _parse_source(value: str) -> tuple[str, str, Path, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"source must be dataset:video:large_root:compact_root, got {value!r}")
    dataset, video, root, compact = parts
    if dataset not in LEGAL_VIDEOS or video not in LEGAL_VIDEOS[dataset]:
        raise ValueError(f"source outside legal scope: {dataset}|{video}")
    return dataset, video, Path(root).resolve(), Path(compact).resolve()


def _line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _line in handle:
            count += 1
    return count


class _NumpyNamespaceUnpickler(pickle.Unpickler):
    """Read legacy records without rewriting their source pickle."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


_SAFE_RECORD_CACHE: dict[str, tuple[Path, dict[str, Any]]] = {}


def _safe_internal_record_with_namespace_remap(video: str) -> tuple[Path, dict[str, Any]]:
    if str(video) in _SAFE_RECORD_CACHE:
        return _SAFE_RECORD_CACHE[str(video)]
    candidates = [
        _r0_full.ASSET_ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl",
        _r0_full.ASSET_ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl",
    ]
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"legal per-video GT record missing for {video}: {candidates}")
    with path.open("rb") as handle:
        value = _NumpyNamespaceUnpickler(handle).load()
    if not isinstance(value, dict) or not isinstance(value.get("frames"), list):
        raise AssertionError(f"invalid legal per-video GT record: {path}")
    result = (path, value)
    _SAFE_RECORD_CACHE[str(video)] = result
    return result


# ``materialize_gt`` resolves this name in the imported module's globals.  The
# replacement is process-local and only changes how the legacy pickle is read.
_r0_full.safe_internal_record = _safe_internal_record_with_namespace_remap
_r1_replay.safe_internal_record = _safe_internal_record_with_namespace_remap


def _index_contract(dataset: str) -> tuple[EvalIndex, dict[str, list[int]], dict[str, list[dict[str, Any]]]]:
    index = EvalIndex(DEV_ROOTS[dataset])
    grouped = _load_group_records(index)
    frames: dict[str, list[int]] = {}
    queries: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for video in LEGAL_VIDEOS[dataset]:
        frames[video] = sorted({int(frame) for (found_video, frame) in grouped if found_video == video})
        if not frames[video]:
            raise AssertionError(f"missing legal-dev frames: {dataset}|{video}")
    for record in index.records:
        video = str(record["video"])
        if video not in LEGAL_VIDEOS[dataset]:
            continue
        query_id = int(record["query_id"])
        sentence = str(record["sentence"])
        previous = queries[video].get(query_id)
        if previous is not None and previous["sentence"] != sentence:
            raise AssertionError(f"legal sentence drift: {dataset}|{video}|{query_id}")
        queries[video][query_id] = {"dataset": dataset, "video": video,
                                     "query_id": query_id, "sentence": sentence}
    normalized = {video: [queries[video][qid] for qid in sorted(queries[video])]
                  for video in LEGAL_VIDEOS[dataset]}
    return index, frames, normalized


def _verify_source(dataset: str, video: str, compact: Path, large_root: Path,
                   expected_records: int, expected_query_ids: list[int], expected_groups: int) -> dict[str, Any]:
    status_path = compact / "status.json"
    provenance_path = compact / "provenance.json"
    if not status_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"prediction-only metadata missing: {compact}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if status.get("status") != "prediction_complete" or provenance.get("status") != "prediction_complete":
        raise AssertionError(f"source is not prediction_complete: {compact}")
    if provenance.get("screening_gt_used") or provenance.get("official_test_labels_read"):
        raise AssertionError(f"forbidden labels read by source: {compact}")
    if provenance.get("labels_attached_after_predictions") is not False:
        raise AssertionError(f"source label-order drift: {compact}")
    scope = provenance.get("selected_scope") or {}
    if scope.get("datasets") != [dataset] or scope.get("video") != video:
        raise AssertionError(f"source scope drift: {compact}: {scope}")
    summaries = provenance.get("prediction_summaries")
    if not isinstance(summaries, list) or len(summaries) != 1:
        raise AssertionError(f"source summary drift: {compact}")
    summary = summaries[0]
    if int(summary.get("group_count_expected", -1)) != expected_groups or \
            int(summary.get("group_count_visited", -1)) != expected_groups:
        raise AssertionError(f"source group completeness drift: {compact}")
    if summary.get("candidate_deletion") or summary.get("candidate_truncation"):
        raise AssertionError(f"source candidate-retention drift: {compact}")
    for epoch in EPOCHS:
        epoch_root = large_root / f"candidate_epoch{epoch:03d}_zero" / dataset
        audit = epoch_root / "prediction_audits.jsonl"
        data_dir = epoch_root / "trackers" / "r0" / "data"
        if not audit.is_file() or not data_dir.is_dir():
            raise FileNotFoundError(f"source epoch output missing: {large_root} epoch={epoch}")
        if _line_count(audit) != expected_records:
            raise AssertionError(f"source audit row count drift: {audit}")
        actual_names = {path.name for path in data_dir.glob("*.txt")}
        expected_names = {f"{sequence_id(video, query_id)}.txt" for query_id in expected_query_ids}
        if actual_names != expected_names:
            raise AssertionError(f"source query prediction-file drift: {data_dir}")
    return {"dataset": dataset, "video": video, "compact_output": str(compact.resolve()),
            "large_prediction_root": str(large_root), "status": "prediction_complete",
            "groups": expected_groups, "query_count": len(expected_query_ids),
            "query_frame_records": expected_records, "candidate_deletion": False,
            "candidate_truncation": False, "labels_used_for_prediction": False,
            "source_status_sha256": sha256_file(status_path)}


def _link_prediction_views(aggregate_root: Path, dataset: str,
                           source_roots: dict[str, Path]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        epoch_root = aggregate_root / f"candidate_epoch{epoch:03d}_zero"
        tracker_data = epoch_root / dataset / "trackers" / "r0" / "data"
        if tracker_data.exists() or tracker_data.is_symlink():
            raise FileExistsError(f"aggregate prediction collision: {tracker_data}")
        paths = prepare_paths(epoch_root, dataset)
        tracker_data = paths["tracker_data"]
        link_count = 0
        for video in LEGAL_VIDEOS[dataset]:
            source = source_roots[video] / f"candidate_epoch{epoch:03d}_zero" / dataset / "trackers" / "r0" / "data"
            for file_path in sorted(source.glob("*.txt")):
                destination = tracker_data / file_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(file_path.resolve())
                link_count += 1
        audits.append({"dataset": dataset, "epoch": epoch, "prediction_link_count": link_count,
                       "prediction_link_mode": "symlink", "aggregate_root": str((epoch_root / dataset).resolve())})
    return audits


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    aggregate_root = args.aggregate_root.resolve()
    trackeval_root = args.trackeval_root.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty aggregate output: {out}")
    if aggregate_root.exists() and any(aggregate_root.iterdir()):
        raise FileExistsError(f"refusing nonempty aggregate root: {aggregate_root}")
    if trackeval_root.exists() and any(trackeval_root.iterdir()):
        raise FileExistsError(f"refusing nonempty aggregate TrackEval root: {trackeval_root}")
    out.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)
    trackeval_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([str(sys.executable), *sys.argv])
    base: dict[str, Any] = {
        "format": "locatemot-r1-legal-dev-aggregate-v1", "status": "incomplete",
        "command": command, "cwd": str(Path.cwd().resolve()), "thread": THREAD,
        "scope": "legal_dev_only", "seed": 20260829, "manifest_sha256": None,
        "rule": {"name": RULE_NAME, **RULE},
        "anchor": {"path": str(L89E_ANCHOR), "sha256": sha256_file(L89E_ANCHOR)},
        "selection_rule": "(HOTA, DetA, AssA, distinct_target_recall, -inactive_false_acceptance, -epoch)",
        "legal_video_scope": {key: list(value) for key, value in LEGAL_VIDEOS.items()},
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "prediction_before_gt": True, "labels_attached_after_predictions": False,
        "prediction_link_mode": "symlink", "failure_root_cause": None,
        "next_action": "freeze selected legal-dev epochs and run fixed semantic diagnostic",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 aggregate cwd: {Path.cwd()}")
        manifest_sha = check_manifest()
        if manifest_sha != MANIFEST_SHA:
            raise AssertionError(f"manifest SHA drift: {manifest_sha}")
        base["manifest_sha256"] = manifest_sha
        parsed = [_parse_source(value) for value in args.source]
        expected_pairs = {(dataset, video) for dataset, videos in LEGAL_VIDEOS.items() for video in videos}
        if {(dataset, video) for dataset, video, _root, _compact in parsed} != expected_pairs:
            raise AssertionError("aggregate source set is not exactly the six legal-dev videos")
        source_roots: dict[str, dict[str, Path]] = defaultdict(dict)
        source_audits: list[dict[str, Any]] = []
        index_data: dict[str, tuple[EvalIndex, dict[str, list[int]], dict[str, list[dict[str, Any]]]]] = {}
        for dataset in LEGAL_VIDEOS:
            index_data[dataset] = _index_contract(dataset)
        for dataset, video, large_root, compact_root in parsed:
            index, frames, queries = index_data[dataset]
            expected_records = sum(1 for record in index.records if str(record["video"]) == video)
            expected_query_ids = [int(value["query_id"]) for value in queries[video]]
            source_audits.append(_verify_source(dataset, video,
                                                compact_root,
                                                large_root, expected_records, expected_query_ids, len(frames[video])))
            source_roots[dataset][video] = large_root
        # All prediction-only sources have now been verified before any legal
        # GT is opened.
        link_audits: list[dict[str, Any]] = []
        for dataset in LEGAL_VIDEOS:
            link_audits.extend(_link_prediction_views(aggregate_root, dataset, source_roots[dataset]))
        trackeval_results: list[dict[str, Any]] = []
        selections: dict[str, Any] = {}
        for dataset in LEGAL_VIDEOS:
            _index, frame_map, queries_by_video = index_data[dataset]
            safe_root = SAFE_TARGET_ROOT / ("v1_dev" if dataset == "refer_kitti_v1" else "v2_dev")
            results: list[dict[str, Any]] = []
            for epoch in EPOCHS:
                epoch_root = aggregate_root / f"candidate_epoch{epoch:03d}_zero"
                paths = prepare_paths(epoch_root, dataset)
                gt_audit = materialize_gt(dataset, queries_by_video, safe_root, paths, frame_map)
                descriptor = _legal_descriptor(dataset, epoch_root, queries_by_video, frame_map, safe_root)
                destination = trackeval_root / f"candidate_epoch{epoch:03d}" / dataset
                trackeval = run_trackeval_dataset(epoch_root / dataset, destination, dataset, tracker_name="r0")
                checkpoint = WORK_ROOT / "outputs/r1/train/formal_fit_attempt1" / f"checkpoint_r1_epoch{epoch:03d}_{dataset}.pt"
                results.append({"dataset": dataset, "epoch": epoch, "rule": RULE_NAME,
                                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                                "gt_audit": gt_audit, "descriptor": descriptor, "trackeval": trackeval,
                                "prediction_before_gt": True, "labels_used_for_prediction": False})
            eligible = [item for item in results if int(item["epoch"]) in ELIGIBLE_EPOCHS]
            if len(eligible) != len(ELIGIBLE_EPOCHS):
                raise AssertionError(f"eligible epoch count drift: {dataset}")
            chosen = max(eligible, key=_selection_key)
            selections[dataset] = {"selection_key": list(_selection_key(chosen)),
                                   "selection_tuple": base["selection_rule"],
                                   "selected_epoch": int(chosen["epoch"]),
                                   "selected_checkpoint": chosen["checkpoint"],
                                   "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
                                   "rule": RULE_NAME, "eligible_epochs": list(ELIGIBLE_EPOCHS),
                                   "legal_dev_only": True}
            trackeval_results.extend(results)
        payload = {**base, "status": "complete", "prediction_sources": source_audits,
                   "prediction_link_audits": link_audits, "trackeval_results": trackeval_results,
                   "selection": selections, "labels_attached_after_predictions": True,
                   "hota_trackeval_run": True, "no_hota_or_trackeval": False,
                   "aggregate_root": str(aggregate_root), "trackeval_root": str(trackeval_root),
                   "wall_seconds": time.perf_counter() - started,
                   "inputs": {"manifest": file_meta(MANIFEST), "anchor": file_meta(L89E_ANCHOR),
                              "selection": file_meta(L89E_SELECTION),
                              "dev_indexes": {dataset: file_meta(DEV_ROOTS[dataset] / "summary.json") for dataset in LEGAL_VIDEOS},
                              "safe_targets": {dataset: file_meta(SAFE_TARGET_ROOT / ("v1_dev" if dataset == "refer_kitti_v1" else "v2_dev") / "manifest.json") for dataset in LEGAL_VIDEOS}},
                   "next_action": "freeze selected V1/V2 epochs and run fixed semantic diagnostic"}
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-r1-legal-dev-aggregate-provenance-v1"})
        write_json(out / "status.json", {"format": base["format"], "status": "complete", "output": str(out),
                                          "selection": selections, "trackeval_result_count": len(trackeval_results),
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
                                          "candidate_deletion": False, "candidate_truncation": False,
                                          "failure_root_cause": None, "next_action": payload["next_action"]})
        return 0
    except BaseException as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R1 legal-dev aggregate — INCOMPLETE\n\n```text\n" + trace + "```\n", encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": base["format"], "status": "incomplete", "command": command,
                                          "output": str(out), "failure_root_cause": payload["failure_root_cause"],
                                          "traceback_path": payload["traceback_path"],
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "next_action": "repair the first aggregation contract issue in a new attempt"})
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True,
                        help="dataset:video:large_prediction_root:compact_prediction_root, exactly six legal videos")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
