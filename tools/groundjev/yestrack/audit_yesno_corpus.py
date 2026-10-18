#!/usr/bin/env python3
"""Audit a prepared Yes/No corpus without touching model or GPU state."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from locatemot.groundjev.yes_verification import assert_prompt_has_no_ground_truth
from tools.groundjev.yestrack.common import read_records


def audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    video_splits: dict[str, set[str]] = defaultdict(set)
    task_counts: dict[str, int] = defaultdict(int)
    rejected_rows = 0
    for index, row in enumerate(records):
        row_id = str(row.get("id", ""))
        if row_id in seen_ids:
            duplicate_ids.append(row_id)
        seen_ids.add(row_id)
        task = str(row.get("task", ""))
        split = str(row.get("split", ""))
        video = str(row.get("video", "unknown"))
        task_counts[task] += 1
        video_splits[video].add(split)
        for field in ("id", "split", "source", "task", "question", "media", "supervision"):
            if field not in row:
                errors.append(f"row{index}:missing_{field}")
        try:
            assert_prompt_has_no_ground_truth(str(row.get("question", "")))
        except ValueError as error:
            errors.append(f"row{index}:gt_in_prompt:{error}")
        supervision = row.get("supervision") or {}
        if supervision.get("gt_used_for_prompt") is not False:
            errors.append(f"row{index}:gt_used_for_prompt_not_false")
        if supervision.get("future_evidence") is True:
            errors.append(f"row{index}:future_evidence_flag")
        frame = int(row.get("frame", 0))
        if "history_frame" in row and int(row["history_frame"]) >= frame:
            errors.append(f"row{index}:non_causal_history")
        if "MOT20-05" in video or "mot20-05" in video.lower():
            errors.append(f"row{index}:forbidden_MOT20-05")
        if bool(row.get("novel_category")) and split.lower() in {"train", "fit", "calibration"}:
            errors.append(f"row{index}:novel_training_label")
        if bool(row.get("perception_rejected")) or bool(row.get("rejected_frame")):
            rejected_rows += 1
    leakage = {video: sorted(splits) for video, splits in video_splits.items() if len(splits) > 1}
    return {
        "status": "YESNO_CORPUS_AUDIT_PASS" if not errors and not leakage and not duplicate_ids else "YESNO_CORPUS_AUDIT_FAILED",
        "row_count": len(records),
        "task_counts": dict(sorted(task_counts.items())),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:20],
        "split_leakage": leakage,
        "rejected_frame_rows_preserved": rejected_rows,
        "errors": errors,
        "gt_in_prompt": any("gt_in_prompt" in error for error in errors),
        "future_evidence": any("future" in error or "causal" in error for error in errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(read_records(args.rows))
    report["rows"] = str(args.rows.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "row_count": report["row_count"], "errors": len(report["errors"])}, ensure_ascii=False))
    return 0 if report["status"] == "YESNO_CORPUS_AUDIT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
