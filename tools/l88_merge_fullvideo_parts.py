#!/usr/bin/env python3
"""Merge completed one-video L88 dev matrices into one candidate root.

The merge copies only compact GT/tracker text and audit files.  It never
copies detector, encoder, image, or dense feature data.  Each input part must
contain one complete candidate, one dataset, and one video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any


WORK_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l19/protocol/kitti_fast_eval_manifest.json").resolve()
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
EXPECTED_BY_SCOPE = {
    "dev": {
        "refer_kitti_v1": {"0008", "0010", "0020"},
        "refer_kitti_v2": {"0000", "0008", "0009"},
    },
    "internal": {
        "refer_kitti_v1": {"0004", "0018"},
        "refer_kitti_v2": {"0016", "0017", "0020"},
    },
}
SCOPE_LABEL = {
    "dev": "internal full-video dev selection",
    "internal": "internal V1/V2 full-video validation",
}
RULES = ("B", "R", "P")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    # Several legal parts share one dataset destination but contain distinct
    # videos.  Merge those per-video trees without overwriting an existing
    # sibling; a duplicate file is still rejected by the source-part checks.
    shutil.copytree(source, destination, copy_function=shutil.copy2, dirs_exist_ok=True)


def _max_timestep(path: Path) -> int:
    maximum = 0
    if not path.is_file():
        return maximum
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            maximum = max(maximum, int(str(line).split(",", 1)[0]))
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"invalid TrackEval timestep in {path}: {line!r}") from exc
    return maximum


def repair_sparse_seqinfo(gt_root: Path, tracker_root: Path) -> None:
    """Make merged metadata cover sparse original-frame timesteps.

    The completed video parts predate the sparse-frame fix in the inference
    writer.  This repair only writes new merge outputs and uses the existing
    GT/tracker text to raise seqLength to the largest legal 1-based timestep;
    it never changes boxes, IDs, or prediction rows.
    """
    if not gt_root.is_dir():
        raise FileNotFoundError(gt_root)
    for sequence_dir in sorted(gt_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        seqinfo = sequence_dir / "seqinfo.ini"
        if not seqinfo.is_file():
            raise FileNotFoundError(seqinfo)
        maximum = max(_max_timestep(sequence_dir / "gt.txt"),
                      _max_timestep(tracker_root / f"{sequence_dir.name}.txt"))
        if maximum <= 0:
            continue
        lines = seqinfo.read_text().splitlines()
        replaced = False
        for index, line in enumerate(lines):
            if line.startswith("seqLength="):
                current = int(line.split("=", 1)[1])
                lines[index] = f"seqLength={max(current, maximum)}"
                replaced = True
                break
        if not replaced:
            raise AssertionError(f"seqLength missing from {seqinfo}")
        seqinfo.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 merge output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    try:
        expected = EXPECTED_BY_SCOPE[str(args.scope)]
        expected_part_count = sum(len(values) for values in expected.values())
        if len(args.part_roots) != expected_part_count:
            raise AssertionError(
                f"the registered {args.scope} merge requires {expected_part_count} video parts"
            )
        parts = [path.resolve() for path in args.part_roots]
        candidates: list[dict[str, Any]] = []
        identities: list[dict[str, str]] = []
        for part in parts:
            summary_path = part / "summary.json"
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete" or not summary.get("full_video"):
                raise AssertionError(f"part is not complete full video: {part}")
            if summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
                raise AssertionError(f"forbidden labels in part: {part}")
            if len(summary.get("candidates", [])) != 1:
                raise AssertionError(f"part must contain one candidate: {part}")
            videos = summary.get("videos", {})
            pairs = [(str(dataset), str(video)) for dataset, values in videos.items()
                     for video in values]
            if len(pairs) != 1:
                raise AssertionError(f"part must contain one dataset/video: {part}")
            candidate = summary["candidates"][0]
            candidates.append(candidate)
            identities.append({"dataset": pairs[0][0], "video": pairs[0][1], "part": str(part)})
        first_epoch = int(candidates[0]["checkpoint"]["epoch"])
        first_sha = str(candidates[0]["checkpoint"]["sha256"])
        if any(int(value["checkpoint"]["epoch"]) != first_epoch or
               str(value["checkpoint"]["sha256"]) != first_sha for value in candidates):
            raise AssertionError("part checkpoint identity drift")
        by_dataset: dict[str, set[str]] = {dataset: set() for dataset in expected}
        for identity in identities:
            dataset, video = identity["dataset"], identity["video"]
            if dataset not in expected or video in by_dataset[dataset]:
                raise AssertionError(f"duplicate/unregistered part: {identity}")
            by_dataset[dataset].add(video)
        if by_dataset != expected:
            raise AssertionError(f"part video set drift: {by_dataset}")

        aggregate_rules: dict[str, dict[str, Any]] = {}
        strategy_paths: dict[str, dict[str, dict[str, str]]] = {}
        eval_summary: dict[str, dict[str, dict[str, Any]]] = {rule: {} for rule in RULES}
        for rule in RULES:
            strategy_paths[rule] = {}
            aggregate_rules[rule] = {
                "frames": 0, "queries": 0, "candidate_rows": 0, "selected_rows": 0,
                "candidate_rows_retained": True, "candidate_deletion": False,
                "candidate_truncation": False, "part_count": len(parts),
            }
        for part, candidate, identity in zip(parts, candidates, identities):
            dataset, video = identity["dataset"], identity["video"]
            epoch_root = part / f"candidate_epoch{first_epoch:03d}"
            out_epoch_root = out / f"candidate_epoch{first_epoch:03d}"
            for rule in RULES:
                source = epoch_root / rule / dataset
                destination = out_epoch_root / rule / dataset
                destination.mkdir(parents=True, exist_ok=True)
                source_gt = source / "gt"
                source_trackers = source / "trackers"
                dest_gt = destination / "gt"
                dest_trackers = destination / "trackers"
                copy_tree(source_gt, dest_gt)
                copy_tree(source_trackers, dest_trackers)
                repair_sparse_seqinfo(dest_gt, dest_trackers / "l88/data")
                source_audit = source / "prediction_audits.jsonl"
                if source_audit.is_file():
                    with (destination / "prediction_audits.jsonl").open("a") as handle:
                        handle.write(source_audit.read_text())
                source_seqmap = source / "seqmap.txt"
                sequences = [line.strip() for line in source_seqmap.read_text().splitlines()[1:] if line.strip()]
                existing = []
                seqmap = destination / "seqmap.txt"
                if seqmap.is_file():
                    existing = [line.strip() for line in seqmap.read_text().splitlines()[1:] if line.strip()]
                combined = existing + [value for value in sequences if value not in existing]
                seqmap.write_text("name\n" + "\n".join(combined) + "\n")
                counter = candidate["rules"][rule]
                for key in ("frames", "queries", "candidate_rows", "selected_rows"):
                    aggregate_rules[rule][key] += int(counter[key])
                aggregate_rules[rule].setdefault("source_parts", []).append(str(source.resolve()))
                eval_summary[rule].setdefault(dataset, {
                    "label_scope": "fit/dev only" if args.scope == "dev" else "internal validation only",
                    "labels_attached_after_predictions": True,
                    "query_gt_audits": [], "record_audits": [], "sequence_count": 0, "sequences": [],
                })
                part_eval = candidate.get("eval_summary", {}).get(rule, {}).get(dataset, {})
                target = eval_summary[rule][dataset]
                target["query_gt_audits"].extend(part_eval.get("query_gt_audits", []))
                target["record_audits"].extend(part_eval.get("record_audits", []))
                target["sequences"].extend(value for value in part_eval.get("sequences", [])
                                            if value not in target["sequences"])
                target["sequence_count"] = len(target["sequences"])
            
        for rule in RULES:
            for dataset in expected:
                destination = out / f"candidate_epoch{first_epoch:03d}" / rule / dataset
                strategy_paths[rule][dataset] = {
                    "root": str(destination.resolve()), "gt": str((destination / "gt").resolve()),
                    "tracker_data": str((destination / "trackers/l88/data").resolve()),
                    "seqmap": str((destination / "seqmap.txt").resolve()),
                }
        merged_candidate = {
            "candidate_index": 0, "checkpoint": candidates[0]["checkpoint"],
            "rules": aggregate_rules, "strategy_paths": strategy_paths,
            "eval_summary": eval_summary, "prediction_strategy_frozen_before_gt": True,
            "source_parts": identities,
        }
        payload = {
            "format": "locatemot-l88-fullvideo-matrix-v1", "status": "complete",
            "scope": SCOPE_LABEL[str(args.scope)], "scope_key": str(args.scope), "full_video": True,
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "datasets": list(expected), "videos": {key: sorted(value) for key, value in expected.items()},
            "candidates": [merged_candidate], "source_parts": identities,
            "candidate_merge": True, "candidate_index_filter": 0, "video_filter": None,
            "all_candidate_rows_scored": True, "candidate_deletion": False,
            "candidate_truncation": False, "model_selection_used_validation": False,
            "labels_attached_after_predictions": True,
            "fit_dev_labels_only": str(args.scope) == "dev",
            "internal_validation_labels_only": str(args.scope) == "internal",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "groundingdino_lora_used": True,
            "groundingdino_backbone_trainable": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "manifest_sha256": MANIFEST_SHA, "persistent_dense_cache_written": False,
            "failure_root_cause": None,
            "next_action": ("run L88 dev TrackEval matrix on merged candidate"
                             if args.scope == "dev" else
                             "run L88 internal TrackEval matrix on frozen candidate"),
        }
        write_json(out / "summary.json", payload); write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": payload["format"], "status": "complete",
                                          "full_video": True, "candidate_count": 1,
                                          "source_part_count": expected_part_count, "screening_gt_used": False,
                                          "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": False, "no_hota_or_trackeval": True})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 full-video merge — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l88-fullvideo-matrix-v1", "status": "incomplete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "no_hota_or_trackeval": True})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=tuple(EXPECTED_BY_SCOPE), default="dev")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
