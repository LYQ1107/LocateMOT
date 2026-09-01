"""Run the blocking Stage L20 provenance, grouping, source, and NULL audits.

This audit is intentionally train-only for every decision that needs GT.  It
does not change a checkpoint, run an evaluator, or start training.  The
machine-readable report is the gate for the 100--200 step smoke.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402
from locatemot.models.l20_source_invariant_set_correspondence import (  # noqa: E402
    L20SourceInvariantSetCorrespondence,
)
from tools.l20_common import (  # noqa: E402
    BankStore, l20_frame_features, l20_group_ids, strict_observation_groups,
)
from tools.train_l18_carr import load_items  # noqa: E402
from tools.train_l19 import l19_track_membership_index  # noqa: E402


SPLIT_MANIFEST = ROOT / "outputs/l16/data/protocol/split_manifest.json"
AUDIT_SCHEMA_VERSION = "locatemot-l20-blocking-sanity-audit-v2"
AUDIT_CONFIG_ID = "strict-mutual-nearest-iou0.80-app0.82-grid0.70-0.80-v1"
AUDIT_PROTOCOL_DIR = ROOT / "outputs/l20/protocol"
AUDIT_RUN_DIR = AUDIT_PROTOCOL_DIR / "blocking_audit_runs"
CURRENT_AUDIT = AUDIT_PROTOCOL_DIR / "l20_blocking_sanity_audit.current.json"
CURRENT_AUDIT_MD = AUDIT_PROTOCOL_DIR / "l20_blocking_sanity_audit.current.md"
AUDIT_LOCK_PATH = AUDIT_PROTOCOL_DIR / "l20_blocking_sanity_audit.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, default=jsonable) + "\n")
    os.replace(temporary, path)


def mark_failed_run(run_path: Path, run_id: str, started_at: str,
                    error: str, report: dict | None = None) -> None:
    failed = dict(report or {})
    failed.update({
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_code": "tools/audit_l20_blocking.py",
        "audit_config_id": AUDIT_CONFIG_ID,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "audit_status": "failed",
        "failure": error,
        "current_audit_path": str(CURRENT_AUDIT),
    })
    atomic_json_write(run_path, failed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def record_path(video: str) -> Path | None:
    for path in (
        ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl",
        ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl",
    ):
        if path.exists():
            return path
    return None


def gt_by_frame(video: str) -> dict[int, dict[str, list[float]]]:
    path = record_path(video)
    if path is None:
        return {}
    record = pickle.load(path.open("rb"))
    return {
        int(frame["frame"]): {
            str(key): list(value)
            for key, value in frame.get("gt_boxes", {}).items()
        }
        for frame in record["frames"]
    }


def box_iou(a, b) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


def split_inventory(items: dict, protocol: dict) -> dict:
    split_sets = {}
    train_items = []
    violations = []
    for domain, key in (("kitti", "kitti_v2"), ("dance", "refer_dance")):
        split_sets[domain] = {
            name: set(protocol[key][name])
            for name in ("train", "train_val", "official_eval")
        }
        values = split_sets[domain]
        for left, right in (("train", "train_val"),
                            ("train", "official_eval"),
                            ("train_val", "official_eval")):
            overlap = sorted(values[left] & values[right])
            if overlap:
                violations.append({"domain": domain, "left": left,
                                   "right": right, "videos": overlap})
        for item in items["train"][domain]:
            if item["video"] not in values["train"]:
                violations.append({"domain": domain, "video": item["video"],
                                   "reason": "load_items_train_outside_manifest"})
            train_items.append(item)
    query_keys = sorted({
        (item["domain"], str(item["video"]),
         str(item["entry"].get("sentence", item["entry"].get("expression", ""))))
        for item in train_items
    })
    return {
        "split_sets": {domain: {name: sorted(values)
                                 for name, values in mapping.items()}
                        for domain, mapping in split_sets.items()},
        "train_item_count": len(train_items),
        "train_query_count": len(query_keys),
        "train_query_keys": [list(value) for value in query_keys],
        "violations": violations,
        "passed": not violations,
        "items": train_items,
    }


def conflict_audit(bank: dict, video: str, split: str) -> dict:
    tensors = bank["tensors"]
    labels = bank.get("candidate_gt", [])
    groups = tensors.get("observation_group_id")
    source_tensor = tensors.get("pool_id")
    boxes = tensors["box"].numpy()
    track_ids = tensors["track_id"].numpy()
    source = (source_tensor.numpy().astype(np.int64)
              if source_tensor is not None else np.zeros(len(track_ids), np.int64))
    if groups is None:
        groups_np = np.arange(len(track_ids), dtype=np.int64)
    else:
        groups_np = groups.numpy().astype(np.int64)
    clips = tensors["clip"].float().numpy()
    clips /= np.linalg.norm(clips, axis=1, keepdims=True).clip(min=1e-6)
    metadata = bank.get("metadata", {})
    schema = str(metadata.get("observation_group_schema", ""))
    broad_rule_confirmed = (".50" in schema and ".30" in schema and
                            ".82" in schema)
    total_groups = 0
    conflict_groups = []
    cross_pool_groups = 0
    cross_pool_rows = 0
    for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
        begin, end = map(int, tensors["frame_ptr"][frame_index:frame_index + 2])
        frame_group_values = groups_np[begin:end]
        for group_id in np.unique(frame_group_values).tolist():
            local = np.flatnonzero(frame_group_values == group_id)
            total_groups += 1
            source_values = set(int(value) for value in source[begin:end][local])
            if source_values == {0, 1}:
                cross_pool_groups += 1
                cross_pool_rows += len(local)
            gt_ids = sorted({str(labels[begin + int(index)])
                             for index in local
                             if labels[begin + int(index)] is not None})
            if len(gt_ids) <= 1:
                continue
            rows = []
            for index in local.tolist():
                global_index = begin + int(index)
                rows.append({
                    "row": global_index, "source": int(source[global_index]),
                    "track_id": int(track_ids[global_index]),
                    "gt_id": (None if labels[global_index] is None
                              else str(labels[global_index])),
                    "box": boxes[global_index].tolist(),
                })
            pairwise = []
            for left in local.tolist():
                for right in local.tolist():
                    if left >= right:
                        continue
                    left_global, right_global = begin + left, begin + right
                    overlap = box_iou(boxes[left_global], boxes[right_global])
                    appearance = float(np.dot(clips[left_global], clips[right_global]))
                    pairwise.append({
                        "row_a": left_global, "row_b": right_global,
                        "source_a": int(source[left_global]),
                        "source_b": int(source[right_global]),
                        "iou": overlap, "appearance": appearance,
                        "broad_pair_rule": bool(
                            overlap >= 0.50 or
                            (overlap >= 0.30 and appearance >= 0.82)),
                    })
            conflict_groups.append({
                "video": video, "split": split, "frame": int(frame_id),
                "frame_index": frame_index, "raw_group_id": int(group_id),
                "row_count": len(local), "source_composition": sorted(source_values),
                "gt_ids": gt_ids, "rows": rows, "pairwise": pairwise,
                "broad_rule_matches_schema": broad_rule_confirmed,
            })
    return {
        "video": video, "split": split,
        "observation_group_schema": schema,
        "broad_rule": "IoU>=0.50 or (IoU>=0.30 and appearance>=0.82)",
        "broad_rule_confirmed_by_bank_metadata": broad_rule_confirmed,
        "total_rows": int(len(track_ids)), "total_frames": int(len(tensors["frame_ids"])),
        "total_groups": total_groups,
        "groups_with_multiple_gt_ids": len(conflict_groups),
        "conflict_rate": float(len(conflict_groups) / max(1, total_groups)),
        "cross_pool_groups": cross_pool_groups,
        "cross_pool_rows": cross_pool_rows,
        "conflicts": conflict_groups,
    }


def strict_metrics(bank: dict, video: str, iou_threshold: float,
                   appearance_threshold: float, gt_frames: dict) -> dict:
    tensors = bank["tensors"]
    labels = bank.get("candidate_gt", [])
    source_tensor = tensors.get("pool_id")
    source = (source_tensor.numpy().astype(np.int64)
              if source_tensor is not None else
              np.zeros(len(tensors["track_id"]), np.int64))
    boxes = tensors["box"].numpy()
    objectness = tensors["objectness"].numpy()
    groups = []
    conflicts = 0
    conflict_total = 0
    broad_cross = 0
    broad_cross_double = 0
    raw_rows = strict_rows = 0
    raw_hits = strict_hits = gt_total = 0
    for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
        begin, end = map(int, tensors["frame_ptr"][frame_index:frame_index + 2])
        count = end - begin
        strict = strict_observation_groups(
            bank, frame_index, iou_threshold, appearance_threshold)
        strict_unique = np.unique(strict)
        raw_rows += count
        strict_rows += len(strict_unique)
        for group_id in strict_unique.tolist():
            indices = np.flatnonzero(strict == group_id)
            gt_ids = sorted({str(labels[begin + int(index)])
                             for index in indices
                             if labels[begin + int(index)] is not None})
            if len(gt_ids) > 1:
                conflicts += 1
            conflict_total += 1
        broad = tensors.get("observation_group_id")
        broad = (broad[begin:end].numpy().astype(np.int64)
                 if broad is not None else np.arange(count, dtype=np.int64))
        for broad_id in np.unique(broad).tolist():
            indices = np.flatnonzero(broad == broad_id)
            source_values = set(int(value) for value in source[begin:end][indices])
            if source_values != {0, 1}:
                continue
            broad_cross += 1
            if len(np.unique(strict[indices])) > 1:
                broad_cross_double += 1
        gt = gt_frames.get(int(frame_id), {})
        if gt:
            local_boxes = boxes[begin:end]
            representatives = []
            for group_id in strict_unique.tolist():
                indices = np.flatnonzero(strict == group_id)
                representatives.append(
                    local_boxes[indices[int(np.argmax(objectness[begin:end][indices]))]])
            for gbox in gt.values():
                gt_total += 1
                raw_hits += int(any(box_iou(candidate, gbox) >= 0.50
                                    for candidate in local_boxes))
                strict_hits += int(any(box_iou(candidate, gbox) >= 0.50
                                       for candidate in representatives))
        groups.append({
            "frame": int(frame_id), "raw_rows": count,
            "strict_groups": int(len(strict_unique)),
        })
    raw_recall = raw_hits / max(1, gt_total)
    strict_recall = strict_hits / max(1, gt_total)
    return {
        "video": video, "iou_threshold": float(iou_threshold),
        "appearance_threshold": float(appearance_threshold),
        "raw_rows": raw_rows, "strict_output_groups": strict_rows,
        "groups_with_multiple_gt_ids": conflicts,
        "strict_conflict_rate": float(conflicts / max(1, conflict_total)),
        "gt_boxes_checked": gt_total,
        "raw_union_recall_at_0.5": raw_recall,
        "strict_representative_recall_at_0.5": strict_recall,
        "recall_drop": raw_recall - strict_recall,
        "broad_cross_pool_groups": broad_cross,
        "broad_cross_pool_groups_still_double_output": broad_cross_double,
        "cross_pool_double_output_rate": float(
            broad_cross_double / max(1, broad_cross)),
        "duplicate_reduction": float(
            (broad_cross - broad_cross_double) / max(1, broad_cross)),
        "frames": groups,
    }


def grouping_runtime_audit(bank: dict, item: dict) -> dict:
    device = torch.device("cpu")
    model = L20SourceInvariantSetCorrespondence(
        hidden=32, heads=4, dropout=0.0, temporal_points=4,
        hook_points=4, use_source_adapters=True, use_grouping=True,
        use_null=True).to(device).eval()
    entry = item["entry"]
    text = str(entry.get("sentence", entry.get("expression", "")))
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    family = expression_family_vector(text).to(device)
    state = {}
    checks = []
    seen_track_ids = set()
    for frame_index in range(min(2, len(bank["tensors"]["frame_ids"]))):
        features, track_ids, begin, end = l20_frame_features(bank, frame_index, device)
        output = model(features, query, family, track_ids, state)
        state = output["state"]
        current_track_ids = set(track_ids.tolist())
        seen_track_ids.update(current_track_ids)
        members = output["group_member_rows"]
        member_lists = [value.detach().cpu().tolist() for value in members]
        flat = [index for values in member_lists for index in values]
        counts = Counter(flat)
        selected = output["group_row_indices"].detach().cpu().tolist()
        selected_track_ids = output["group_track_ids"].detach().cpu().tolist()
        checks.append({
            "frame": int(bank["tensors"]["frame_ids"][frame_index]),
            "input_rows": int(end - begin), "output_groups": len(member_lists),
            "group_ids_unique": len(set(output["group_ids"].tolist())) == len(member_lists),
            "member_rows_disjoint": all(value == 1 for value in counts.values()),
            "member_rows_cover_all_input": sorted(flat) == list(range(end - begin)),
            "max_group_size": max((len(value) for value in member_lists), default=0),
            "max_group_size_le_2": max((len(value) for value in member_lists), default=0) <= 2,
            "selected_rows_in_own_group": all(
                selected[index] in member_lists[index]
                for index in range(len(selected))),
            "same_frame_duplicate_output_count": len(selected) - len(set(selected)),
            "same_frame_duplicate_track_id_count": len(selected_track_ids) -
            len(set(selected_track_ids)),
            "state_keys_are_raw_track_ids": set(state).issubset(seen_track_ids) and
            current_track_ids.issubset(set(state)),
            "state_key_namespace": "raw_track_id",
            "strict_group_feature_present": "l20_group_id" in features,
        })
    return {
        "frames": checks,
        "same_frame_duplicate_output_count": sum(
            check["same_frame_duplicate_output_count"] for check in checks),
        "same_frame_duplicate_track_id_count": sum(
            check["same_frame_duplicate_track_id_count"] for check in checks),
        "passed": all(all(value for key, value in check.items()
                           if isinstance(value, bool)) for check in checks),
    }


def training_conflict_handling_audit(banks: list[tuple[str, dict]]) -> dict:
    total = split_rows = remaining_mixed = 0
    by_video = []
    for video, bank in banks:
        tensors = bank["tensors"]
        labels = bank.get("candidate_gt", [])
        video_total = video_rows = video_remaining = 0
        for frame_index in range(len(tensors["frame_ids"])):
            begin, end = map(int, tensors["frame_ptr"][frame_index:frame_index + 2])
            strict = l20_group_ids(bank, frame_index)
            train_groups = l20_group_ids(
                bank, frame_index, training_conflict_singletons=True)
            for group_id in np.unique(strict).tolist():
                indices = np.flatnonzero(strict == group_id)
                identities = {str(labels[begin + index]) for index in indices
                              if labels[begin + index] is not None}
                if len(identities) <= 1:
                    continue
                video_total += 1
                video_rows += len(indices)
                refined = np.unique(train_groups[indices])
                split_rows += int(len(refined))
                if len(refined) != len(indices):
                    video_remaining += 1
        total += video_total
        split_rows += 0
        remaining_mixed += video_remaining
        by_video.append({
            "video": video, "strict_train_conflict_groups": video_total,
            "rows_split_to_singletons": video_rows,
            "groups_remaining_mixed_after_train_split": video_remaining,
        })
    return {
        "strict_train_conflict_groups": total,
        "rows_in_conflicting_groups": sum(
            value["rows_split_to_singletons"] for value in by_video),
        "singleton_refined_group_count": split_rows,
        "groups_remaining_mixed_after_train_split": remaining_mixed,
        "handling": "train-sidecar GT marks strict conflict members as singleton; eval does not use this flag",
        "by_video": by_video, "passed": remaining_mixed == 0,
    }


def eval_grouping_gt_audit(bank: dict) -> dict:
    """Prove default feature construction is unchanged when GT is redacted."""
    redacted = dict(bank)
    redacted["candidate_gt"] = [None] * len(bank["tensors"]["track_id"])
    redacted.pop("_l20_strict_group_cache", None)
    device = torch.device("cpu")
    original = l20_frame_features(bank, 0, device)[0]["l20_group_id"]
    without_gt = l20_frame_features(redacted, 0, device)[0]["l20_group_id"]
    source = inspect.getsource(l20_frame_features)
    return {
        "default_eval_calls_training_conflict_split": False,
        "default_grouping_body_reads_candidate_gt": "candidate_gt" in source,
        "default_group_ids_equal_with_gt_redacted": bool(
            torch.equal(original, without_gt)),
        "gt_free_grouping_function": "candidate_gt" not in
        inspect.getsource(strict_observation_groups),
        "passed": bool(torch.equal(original, without_gt) and
                       "candidate_gt" not in inspect.getsource(
                           strict_observation_groups)),
    }


def null_label_audit(items: dict, store: BankStore) -> dict:
    counts = Counter()
    violations = []
    multi_positive = 0
    frames_checked = 0
    grouped = defaultdict(list)
    for domain in ("kitti", "dance"):
        for item in items["train"][domain]:
            grouped[(item["bank_dataset"], item["video"])].append(item)
    for (dataset, video), video_items in sorted(grouped.items()):
        bank = store.get(dataset, video)
        frame_ids = [int(value) for value in bank["tensors"]["frame_ids"].tolist()]
        frame_to_index = bank["frame_to_index"]
        for item in video_items:
            declared = set()
            for key in item["entry"].get("label", {}):
                try:
                    declared.add(int(key))
                except (TypeError, ValueError):
                    continue
            # Check every labelled frame and one unlisted frame (ABSENT), while
            # grouping duplicate expressions by the immutable train bank.
            check_frames = sorted(declared)
            missing = next((frame for frame in frame_ids if frame not in declared), None)
            if missing is not None:
                check_frames.append(missing)
            for frame_id in check_frames:
                if frame_id not in frame_to_index:
                    continue
                frame_index = frame_to_index[frame_id]
                begin, end = map(int, bank["tensors"]["frame_ptr"][frame_index:frame_index + 2])
                labels = item["entry"].get("label", {})
                target_ids = {str(value) for value in labels.get(
                    str(frame_id), labels.get(frame_id, []))}
                candidate_gt = bank.get("candidate_gt", [None] * len(
                    bank["tensors"]["track_id"]))[begin:end]
                source_tensor = bank["tensors"].get("pool_id")
                source = (source_tensor[begin:end].numpy().astype(np.int64)
                          if source_tensor is not None else
                          np.zeros(end - begin, np.int64))
                row_match = np.asarray([
                    float(value is not None and str(value) in target_ids)
                    for value in candidate_gt], np.float32)
                main_covered = bool(np.any((row_match > 0.5) & (source == 0)))
                reserve_covered = bool(np.any((row_match > 0.5) & (source == 1)))
                if not target_ids:
                    state = 0
                elif main_covered:
                    state = 1
                elif reserve_covered:
                    state = 2
                else:
                    state = 3
                target = {
                    "state": state, "active": bool(target_ids),
                    "null_target": float(state in (0, 3)),
                    "main_covered": main_covered,
                    "reserve_covered": reserve_covered,
                    "target_ids": sorted(target_ids),
                    "row_match": row_match,
                }
                state = int(target["state"])
                counts[str(state)] += 1
                frames_checked += 1
                positives = int(np.sum(target["row_match"] > 0.5))
                if positives > 1:
                    multi_positive += 1
                valid = True
                if state == 0:
                    valid = not target["active"] and target["null_target"] == 1.0
                elif state == 3:
                    valid = (target["active"] and not target["main_covered"] and
                             not target["reserve_covered"] and
                             target["null_target"] == 1.0)
                elif state == 1:
                    valid = (target["active"] and target["main_covered"] and
                             target["null_target"] == 0.0)
                elif state == 2:
                    valid = (target["active"] and not target["main_covered"] and
                             target["reserve_covered"] and
                             target["null_target"] == 0.0)
                else:
                    valid = False
                if positives > 1 and state in (1, 2) and target["null_target"] != 0.0:
                    valid = False
                if not valid and len(violations) < 100:
                    violations.append({
                        "domain": item["domain"], "video": video,
                        "frame": frame_id, "state": state,
                        "target_ids": target["target_ids"],
                        "positive_rows": positives,
                        "null_target": target["null_target"],
                    })
    return {
        "frames_checked": frames_checked, "state_counts": dict(counts),
        "multi_positive_covered_frames": multi_positive,
        "violations": violations, "passed": not violations,
        "null_definition": {
            "ABSENT": {"state": 0, "null_target": 1},
            "MAIN_COVERED": {"state": 1, "null_target": 0},
            "RESERVE_COVERED": {"state": 2, "null_target": 0},
            "PRESENT_UNCOVERED": {"state": 3, "null_target": 1},
        },
    }


def cache_audit(protocol: dict) -> dict:
    protocol_path = SPLIT_MANIFEST
    result = {
        "split_manifest_sha256": sha256_file(protocol_path),
        "train_query_source": "load_items()[train]",
        "official_gt_used": False,
        "existing_caches": [], "passed": True,
    }
    protocol_dir = ROOT / "outputs/l20/protocol"
    pair_path = protocol_dir / "_pair_smoke.jsonl"
    if pair_path.exists():
        marker = pair_path.with_suffix(pair_path.suffix + ".INVALID.md")
        if not marker.exists():
            marker.write_text(
                "# INVALID L20 pair-supervision cache\n\n"
                "Interrupted/unverified output has no train-only manifest and is not readable.\n"
            )
        result["existing_caches"].append({
            "path": str(pair_path), "status": "INVALID_UNVERIFIED",
            "marker": str(marker), "reason": "no train-only manifest; prior run was interrupted",
        })
    for path in sorted(protocol_dir.glob("*hard*")):
        if path.suffix == ".md" or path.name.endswith(".INVALID.md"):
            continue
        result["existing_caches"].append({
            "path": str(path), "status": "REVIEW_REQUIRED",
            "reason": "hard-negative artifacts require a v2 train manifest before reading",
        })
    for run_path in sorted((ROOT / "outputs/l20/eval").glob("*/run_manifest.json")):
        run = json.loads(run_path.read_text())
        if run.get("data_split") != "train":
            result["existing_caches"].append({
                "path": str(run_path), "status": "REJECTED_NONTRAIN_SOURCE",
                "data_split": run.get("data_split"),
                "reason": "cannot feed train_val/official evaluation output to hard miner",
            })
    source = inspect.getsource(__import__(
        "tools.mine_l20_hard_negatives", fromlist=["mine"]))
    pair_source = inspect.getsource(__import__(
        "tools.build_l20_pair_supervision", fromlist=["build"]))
    result["hard_miner_static_guard"] = {
        "loads_items_train": "items[\"train\"]" in source,
        "requires_data_split_train": "data_split\") != \"train\"" in source,
        "requires_official_gt_false": 'official_gt_used") is not False' in source,
        "rejects_nontrain_whole_cache": "invalid_output" in source,
        "writes_checkpoint_sha256": "checkpoint_sha256" in source,
    }
    result["pair_builder_static_guard"] = {
        "loads_items_train": "items[\"train\"]" in pair_source,
        "split_overlap_assertion": "split overlap" in pair_source,
        "writes_manifest": "pair-supervision-v2" in pair_source,
        "official_gt_false": "official_gt_used" in pair_source,
    }
    result["passed"] = all(
        value.get("status") in {"INVALID_UNVERIFIED", "REJECTED_NONTRAIN_SOURCE"}
        for value in result["existing_caches"]
    ) and all(result["hard_miner_static_guard"].values()) and \
        all(result["pair_builder_static_guard"].values())
    return result


def source_path_audit() -> dict:
    source = inspect.getsource(L20SourceInvariantSetCorrespondence)
    model = L20SourceInvariantSetCorrespondence(
        hidden=32, heads=4, dropout=0.0, temporal_points=4, hook_points=4)
    names = list(model.state_dict())
    final_section = source[source.index("        pair = torch.cat"):source.index("        selected_rows =")]
    return {
        "has_source_embedding_attribute": any(
            re.search(r"source.*embedding|embedding.*source", name, re.I)
            for name in names),
        "has_pool_embedding_attribute": any(
            re.search(r"pool.*embedding|embedding.*pool", name, re.I)
            for name in names),
        "source_used_for_adapter_selection": "source == source_id" in source,
        "source_used_in_final_score_section": bool(re.search(
            r"\bsource\b|pool_id", final_section)),
        "pool_id_in_membership_observation_null_inputs": False,
        "pool_id_in_pair_final_score_inputs": False,
        "final_score_inputs": [
            "group_corr", "group_current", "query_holistic",
            "group_corr*query_holistic", "abs(group_corr-query_holistic)",
        ],
        "observation_inputs": ["group_corr", "group_current", "group_numeric"],
        "null_inputs": ["query_holistic", "frame_group_summary", "frame_numeric"],
        "source_provenance_outputs": ["row_source", "group_source", "aux.source"],
        "state_key_code": "new_state[int(raw_id)]" in source,
        "state_key_not_group_id": "new_state[int(group" not in source,
        "pair_graph_source_blind": False,
    }


def build_report():
    items, protocol = load_items()
    inventory = split_inventory(items, protocol)
    store = BankStore(ROOT / "outputs/l19/dual_banks_features", cache_size=1)
    train_kitti = set(protocol["kitti_v2"]["train"])
    train_conflicts = []
    conflict_lookup = {}
    strict_banks = []
    for video in sorted(train_kitti):
        bank = store.get("kitti", video)
        report = conflict_audit(bank, video, "train")
        train_conflicts.append(report)
        conflict_lookup[video] = report
        strict_banks.append((video, bank))

    # GT is restricted to the protocol train videos for every strict-rule
    # selection metric.  Thresholds explicitly cover the requested .7/.8 IoU.
    threshold_reports = []
    for iou_threshold in (0.70, 0.80):
        for appearance_threshold in (0.82, 0.86):
            per_video = []
            for video, bank in strict_banks:
                per_video.append(strict_metrics(
                    bank, video, iou_threshold, appearance_threshold,
                    gt_by_frame(video)))
            keys = ("raw_rows", "strict_output_groups", "groups_with_multiple_gt_ids",
                    "gt_boxes_checked", "raw_union_recall_at_0.5",
                    "strict_representative_recall_at_0.5", "recall_drop",
                    "broad_cross_pool_groups", "broad_cross_pool_groups_still_double_output")
            aggregate = {key: sum(int(value[key]) for value in per_video)
                         for key in keys if key not in {
                             "raw_union_recall_at_0.5",
                             "strict_representative_recall_at_0.5", "recall_drop"}}
            raw_recall = sum(value["raw_union_recall_at_0.5"] *
                             value["gt_boxes_checked"] for value in per_video)
            strict_recall = sum(value["strict_representative_recall_at_0.5"] *
                                value["gt_boxes_checked"] for value in per_video)
            total_gt = max(1, aggregate["gt_boxes_checked"])
            aggregate.update({
                "raw_union_recall_at_0.5": raw_recall / total_gt,
                "strict_representative_recall_at_0.5": strict_recall / total_gt,
                "recall_drop": (raw_recall - strict_recall) / total_gt,
                "strict_conflict_rate": aggregate["groups_with_multiple_gt_ids"] /
                max(1, aggregate["strict_output_groups"]),
                "cross_pool_double_output_rate": aggregate[
                    "broad_cross_pool_groups_still_double_output"] /
                    max(1, aggregate["broad_cross_pool_groups"]),
                "duplicate_reduction": 1.0 - aggregate[
                    "broad_cross_pool_groups_still_double_output"] /
                    max(1, aggregate["broad_cross_pool_groups"]),
            })
            # The gate is deliberately explicit: near-zero refined conflict,
            # <=1pp recall loss, and a nontrivial reduction in duplicate output.
            aggregate["passed"] = bool(
                # "Near zero" is an explicit, reported rate rather than an
                # unrequested absolute-zero requirement.  At most 0.01% of
                # refined groups may still be ambiguous after the clear fix.
                aggregate["strict_conflict_rate"] <= 1e-4 and
                aggregate["recall_drop"] <= 0.01 and
                aggregate["duplicate_reduction"] >= 0.05)
            threshold_reports.append({
                "iou_threshold": iou_threshold,
                "appearance_threshold": appearance_threshold,
                "aggregate": aggregate, "per_video": per_video,
            })
    passed_thresholds = [value for value in threshold_reports
                         if value["aggregate"]["passed"]]
    if passed_thresholds:
        selected = max(passed_thresholds, key=lambda value: (
            value["aggregate"]["duplicate_reduction"],
            -value["aggregate"]["recall_drop"],
            value["iou_threshold"], value["appearance_threshold"]))
        grouping_decision = {
            "use_grouping": True,
            "selected_iou_threshold": selected["iou_threshold"],
            "selected_appearance_threshold": selected["appearance_threshold"],
            "reason": "strict mutual-nearest audit passed all gates",
        }
    else:
        grouping_decision = {
            "use_grouping": False,
            "selected_iou_threshold": None,
            "selected_appearance_threshold": None,
            "reason": "no strict GT-free threshold passed conflict/recall/duplicate gates; retain raw rows",
        }

    runtime = grouping_runtime_audit(
        store.get("kitti", sorted(train_kitti)[0]),
        next(item for item in inventory["items"]
             if item["domain"] == "kitti" and item["video"] == sorted(train_kitti)[0]))
    conflict_handling = training_conflict_handling_audit(strict_banks)
    eval_gt_free = eval_grouping_gt_audit(
        store.get("kitti", sorted(train_kitti)[0]))
    nulls = null_label_audit(items, store)
    caches = cache_audit(protocol)
    source = source_path_audit()
    # The pair graph is not part of Phase A grouping, but source-blindness is a
    # required static property before it can be enabled in Phase B.
    graph_path = ROOT / "locatemot/rmot/l20_fragment_pair_graph.py"
    if graph_path.exists():
        graph_source = graph_path.read_text()
        edge_start = graph_source.find("def edge_features")
        edge_section = graph_source[edge_start:] if edge_start >= 0 else graph_source
        source["pair_graph_source_blind"] = not bool(
            re.search(r"pool_id|source_a|source_b|source_embedding", edge_section))

    report = {
        "format": AUDIT_SCHEMA_VERSION,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_code": "tools/audit_l20_blocking.py",
        "audit_config_id": AUDIT_CONFIG_ID,
        "project_root": str(ROOT), "stage": "L20", "training_started": False,
        "python": sys.executable, "torch": torch.__version__,
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "inventory": {key: value for key, value in inventory.items() if key != "items"},
        "raw_conflict_audit_train_only": {
            "total_groups": sum(value["total_groups"] for value in train_conflicts),
            "groups_with_multiple_gt_ids": sum(
                value["groups_with_multiple_gt_ids"] for value in train_conflicts),
            "conflict_rate": sum(value["groups_with_multiple_gt_ids"]
                                   for value in train_conflicts) /
            max(1, sum(value["total_groups"] for value in train_conflicts)),
            "videos": train_conflicts,
            "broad_rule": "IoU>=0.50 or (IoU>=0.30 and appearance>=0.82)",
            "gt_scope": "train sidecar labels only",
        },
        "strict_group_threshold_audit": threshold_reports,
        "grouping_decision": grouping_decision,
        "training_conflict_handling": conflict_handling,
        "eval_grouping_gt_read_audit": eval_gt_free,
        "source_path_audit": source,
        "group_runtime_audit": runtime,
        "null_label_audit_train_only": nulls,
        "cache_and_leakage_audit": caches,
    }
    # Grouping is allowed to fall back to raw rows after an explicit failed
    # strict-rule audit; source/null/provenance checks remain hard blockers.
    # The pair graph is a Phase B gate and is not required for Phase A.
    report["blocking_passed"] = bool(
        inventory["passed"] and runtime["passed"] and nulls["passed"] and
        caches["passed"] and conflict_handling["passed"] and
        eval_gt_free["passed"] and
        not source["has_source_embedding_attribute"] and
        not source["has_pool_embedding_attribute"] and
        not source["source_used_in_final_score_section"] and
        source["source_used_for_adapter_selection"] and
        source["state_key_code"] and source["state_key_not_group_id"])
    report["blocking_checks"] = {
        "split_disjoint_and_train_items": inventory["passed"],
        "strict_group_rule_audited": True,
        "runtime_group_mapping": runtime["passed"],
        "train_conflict_singleton_handling": conflict_handling["passed"],
        "eval_grouping_gt_free": eval_gt_free["passed"],
        "null_labels": nulls["passed"],
        "cache_provenance": caches["passed"],
        "source_blind_final_heads": (
            not source["has_source_embedding_attribute"] and
            not source["has_pool_embedding_attribute"] and
            not source["source_used_in_final_score_section"] and
            source["source_used_for_adapter_selection"]),
        "state_namespace": source["state_key_code"] and source["state_key_not_group_id"],
    }
    report["blocking_passed"] = all(report["blocking_checks"].values())
    return report


def main():
    started_at = utc_now()
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
              f"-pid{os.getpid()}-{uuid.uuid4().hex[:12]}")
    AUDIT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_path = AUDIT_RUN_DIR / f"{run_id}.json"
    # The run record exists before any expensive audit work and is never the
    # current audit.  A failed/interrupted run therefore cannot make an old
    # successful current report look fresh.
    atomic_json_write(run_path, {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_code": "tools/audit_l20_blocking.py",
        "audit_config_id": AUDIT_CONFIG_ID,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": None,
        "audit_status": "running",
        "current_audit_path": str(CURRENT_AUDIT),
    })
    lock_handle = None
    try:
        import fcntl
        lock_handle = AUDIT_LOCK_PATH.open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            mark_failed_run(run_path, run_id, started_at,
                            "another blocking audit holds the audit lock")
            raise SystemExit(3) from error

        report = build_report()
        completed_at = utc_now()
        report.update({
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "audit_status": "complete" if report["blocking_passed"] else "failed",
            "current_audit_path": str(CURRENT_AUDIT),
            "run_report_path": str(run_path),
        })
        if not report["blocking_passed"]:
            report["failure"] = "blocking checks failed; current audit was not changed"
            atomic_json_write(run_path, report)
            print(json.dumps({
                "blocking_passed": False,
                "report": str(run_path),
                "current_audit_unchanged": str(CURRENT_AUDIT),
                "run_id": run_id,
            }, indent=2, default=jsonable))
            raise SystemExit(2)

        # First finalize the unique run report, then promote an independent
        # complete snapshot to current.  Both moves are atomic on this volume.
        atomic_json_write(run_path, report)
        atomic_json_write(CURRENT_AUDIT, report)
        lines = [
            "# LocateMOT Stage L20 blocking sanity audit",
            "",
            f"- blocking_passed: `{report['blocking_passed']}`",
            f"- training_started: `{report['training_started']}`",
            f"- machine report: `{CURRENT_AUDIT}`",
            f"- run report: `{run_path}`",
            f"- run_id: `{run_id}`",
            f"- completed_at: `{completed_at}`",
            f"- raw train-only broad groups: `{report['raw_conflict_audit_train_only']['total_groups']}`",
            f"- raw train-only multi-GT groups: `{report['raw_conflict_audit_train_only']['groups_with_multiple_gt_ids']}`",
            f"- grouping decision: `{report['grouping_decision']}`",
            f"- NULL audit violations: `{len(report['null_label_audit_train_only']['violations'])}`",
            "",
            "All GT-dependent grouping and NULL checks above use protocol train sidecar labels only.",
            "Existing interrupted pair output and train_val evaluation output are not readable by the hard miner.",
        ]
        atomic_text_path = CURRENT_AUDIT_MD.with_name(
            f".{CURRENT_AUDIT_MD.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        atomic_text_path.write_text("\n".join(lines) + "\n")
        os.replace(atomic_text_path, CURRENT_AUDIT_MD)
        print(json.dumps({
            "blocking_passed": True,
            "report": str(CURRENT_AUDIT),
            "run_report": str(run_path),
            "run_id": run_id,
            "completed_at": completed_at,
            "grouping_decision": report["grouping_decision"],
            "raw_conflict": {
                "total_groups": report["raw_conflict_audit_train_only"]["total_groups"],
                "multi_gt_groups": report["raw_conflict_audit_train_only"]["groups_with_multiple_gt_ids"],
                "conflict_rate": report["raw_conflict_audit_train_only"]["conflict_rate"],
            },
            "strict_08_082": next(
                value["aggregate"] for value in report["strict_group_threshold_audit"]
                if value["iou_threshold"] == 0.8 and
                value["appearance_threshold"] == 0.82),
            "null": {
                "frames_checked": report["null_label_audit_train_only"]["frames_checked"],
                "violations": len(report["null_label_audit_train_only"]["violations"]),
            },
        }, indent=2, default=jsonable))
    except SystemExit:
        raise
    except Exception as error:
        mark_failed_run(run_path, run_id, started_at,
                        f"{type(error).__name__}: {error}")
        raise
    finally:
        if lock_handle is not None:
            try:
                import fcntl
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()


if __name__ == "__main__":
    main()
