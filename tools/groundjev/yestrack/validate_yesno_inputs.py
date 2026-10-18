#!/usr/bin/env python3
"""CPU-only contract validator for native OneJev Yes/No rows.

This validator intentionally does not open a model or render media.  It checks
the data boundary that must hold before a native render/tokenize preflight:
causal history, prompt/GT separation, task/split provenance, and the stable
``id/split/source/task/state/question/target/media/supervision`` schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from locatemot.groundjev.yes_verification import assert_prompt_has_no_ground_truth
from tools.groundjev.yestrack.common import read_records


REQUIRED = ("id", "split", "source", "task", "state", "question", "target", "media", "supervision")
TASKS = {"mot", "rmot", "ovmot", "identity_pair"}


def validate(records: list[dict[str, Any]], *, require_labels: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    video_splits: dict[str, set[str]] = defaultdict(set)
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for index, row in enumerate(records):
        prefix = f"row{index}"
        missing = [key for key in REQUIRED if key not in row]
        errors.extend(f"{prefix}:missing_{key}" for key in missing)
        row_id = str(row.get("id", ""))
        if not row_id:
            errors.append(f"{prefix}:empty_id")
        elif row_id in seen:
            errors.append(f"{prefix}:duplicate_id:{row_id}")
        seen.add(row_id)
        task = str(row.get("task", ""))
        split = str(row.get("split", ""))
        task_counts[task] += 1
        if task not in TASKS:
            errors.append(f"{prefix}:unknown_task:{task}")
        if str(row.get("state", "")) != "verification":
            errors.append(f"{prefix}:invalid_state")
        media = row.get("media")
        if not isinstance(media, list) or not media or any(item in (None, "") for item in media):
            errors.append(f"{prefix}:empty_media")
        try:
            assert_prompt_has_no_ground_truth(str(row.get("question", "")))
        except ValueError:
            errors.append(f"{prefix}:gt_in_question")
        target = row.get("target")
        if target is not None and target not in ("Yes", "No"):
            errors.append(f"{prefix}:invalid_target")
        if require_labels and split.lower() in {"train", "fit", "calibration"} and target is None:
            errors.append(f"{prefix}:missing_training_label")
        if target is not None:
            label_counts[str(target)] += 1
        supervision = row.get("supervision")
        if not isinstance(supervision, dict):
            errors.append(f"{prefix}:invalid_supervision")
            supervision = {}
        if supervision.get("gt_used_for_prompt") is not False:
            errors.append(f"{prefix}:gt_used_for_prompt")
        if supervision.get("future_evidence") is True:
            errors.append(f"{prefix}:future_evidence")
        try:
            frame = int(row.get("frame"))
        except (TypeError, ValueError):
            errors.append(f"{prefix}:invalid_frame")
            frame = 0
        for key in ("history_frame", "previous_frame"):
            if key in row and int(row[key]) >= frame:
                errors.append(f"{prefix}:noncausal_{key}")
        video = str(row.get("video", row.get("source_video", "unknown")))
        video_splits[video].add(split)
        if "mot20-05" in video.lower():
            errors.append(f"{prefix}:forbidden_MOT20-05")
        if bool(row.get("novel_category")) and split.lower() in {"train", "fit", "calibration"}:
            errors.append(f"{prefix}:novel_training_label")
    leakage = {video: sorted(splits) for video, splits in video_splits.items() if len(splits) > 1}
    errors.extend(f"video_split_leakage:{video}" for video in sorted(leakage))
    return {
        "status": "YESNO_INPUTS_VALID" if not errors else "YESNO_INPUTS_INVALID",
        "schema": "locatemot-yesno-v1",
        "row_count": len(records),
        "task_counts": dict(sorted(task_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "split_leakage": leakage,
        "errors": errors,
        "model_training_started": False,
        "gt_in_model_question": any(error.endswith(":gt_in_question") for error in errors),
        "future_evidence": any("future" in error or "noncausal" in error for error in errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unlabeled", action="store_true")
    args = parser.parse_args()
    report = validate(read_records(args.rows), require_labels=not args.allow_unlabeled)
    report["rows"] = str(args.rows.resolve())
    report["rows_sha256"] = hashlib.sha256(args.rows.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "row_count": report["row_count"], "errors": len(report["errors"])}, ensure_ascii=False))
    return 0 if report["status"] == "YESNO_INPUTS_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
