#!/usr/bin/env python3
"""Prepare native RMOT candidate rows for optional Yes/No verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from locatemot.groundjev.yes_verification import build_verification_prompt
from tools.groundjev.yestrack.common import (
    first,
    infer_frame,
    infer_video,
    label_from_record,
    make_row,
    read_records,
    write_jsonl_and_manifest,
)


def prepare(records: list[dict], *, limit: int | None = None) -> list[dict]:
    rows = []
    for record in records[:limit] if limit else records:
        video = infer_video(record)
        if "MOT20-05" in video or "mot20-05" in video.lower():
            raise ValueError("MOT20-05 is excluded from the Yes/No training corpus")
        expression = str(first(record, "expression", "query", "sentence", default="")).strip()
        if not expression:
            raise ValueError(f"RMOT record has no expression: {record.get('id')}")
        bbox = first(record, "bbox", "box", "bbox_xyxy", "bbox_xywh")
        question = build_verification_prompt(
            task="rmot",
            expression=expression,
            bbox=bbox,
            question_type="membership",
        )
        row = make_row(
            record=record,
            task="rmot",
            question=question,
            target=label_from_record(record),
            question_type="membership",
            proposal_source=str(first(record, "proposal_source", "source", default="native_candidate_export")),
            current_frame=infer_frame(record),
            expression=expression,
        )
        row["expression"] = expression
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = prepare(read_records(args.input), limit=args.limit)
    manifest = write_jsonl_and_manifest(rows, args.out, tool="prepare_rmot_yesno", source=args.input)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
