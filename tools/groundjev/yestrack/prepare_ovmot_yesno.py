#!/usr/bin/env python3
"""Prepare OVMOT category/membership verification rows without GT prompts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from locatemot.groundjev.yes_verification import build_verification_prompt
from tools.groundjev.yestrack.common import first, infer_frame, infer_video, label_from_record, make_row, read_records, write_jsonl_and_manifest


def prepare(records: list[dict], *, limit: int | None = None, allow_novel_eval: bool = False) -> list[dict]:
    rows = []
    for record in records[:limit] if limit else records:
        video = infer_video(record)
        if "MOT20-05" in video or "mot20-05" in video.lower():
            raise ValueError("MOT20-05 is excluded from the Yes/No training corpus")
        split = str(first(record, "split", "partition", default="train")).lower()
        is_novel = bool(first(record, "novel", "is_novel", "novel_category", default=False))
        if is_novel and split in {"train", "fit", "calibration"} and not allow_novel_eval:
            raise ValueError("novel OVMOT categories cannot provide training labels")
        category = str(first(record, "category", "class_name", "query", "expression", default="")).strip()
        if not category:
            raise ValueError(f"OVMOT record has no category/query: {record.get('id')}")
        question = build_verification_prompt(
            task="ovmot",
            expression=category,
            bbox=first(record, "bbox", "box", "bbox_xyxy", "bbox_xywh"),
            question_type="ovmot_membership",
        )
        row = make_row(
            record=record,
            task="ovmot",
            question=question,
            target=label_from_record(record),
            question_type="ovmot_membership",
            proposal_source=str(first(record, "proposal_source", "source", default="native_ovmot_candidate_export")),
            current_frame=infer_frame(record),
            expression=category,
        )
        row["category"] = category
        row["novel_category"] = is_novel
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-novel-eval", action="store_true")
    args = parser.parse_args()
    manifest = write_jsonl_and_manifest(prepare(read_records(args.input), limit=args.limit, allow_novel_eval=args.allow_novel_eval), args.out, tool="prepare_ovmot_yesno", source=args.input)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
