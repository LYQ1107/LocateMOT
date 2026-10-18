#!/usr/bin/env python3
"""Prepare causal same/different-identity verification pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from locatemot.groundjev.yes_verification import assert_prompt_has_no_ground_truth
from tools.groundjev.yestrack.common import first, infer_frame, infer_video, label_from_record, make_row, read_records, write_jsonl_and_manifest


def prepare(records: list[dict], *, limit: int | None = None) -> list[dict]:
    rows = []
    for record in records[:limit] if limit else records:
        frame = infer_frame(record)
        video = infer_video(record)
        previous_frame = int(first(record, "previous_frame", "history_frame", default=frame - 1))
        if previous_frame >= frame:
            raise ValueError("identity pair must use a strictly earlier frame")
        current_id = first(record, "current_track_id", "track_id", "candidate_id", default=None)
        previous_id = first(record, "previous_track_id", "history_track_id", default=None)
        question = "Do these two visual regions depict the same persistent object? Answer exactly Yes or No."
        assert_prompt_has_no_ground_truth(question)
        media = first(record, "media", "images", default=None)
        if media is None:
            media = [first(record, "previous_image", "history_image", default=None), first(record, "image", "current_image", default=None)]
        row = make_row(
            record=record,
            task="identity_pair",
            question=question,
            target=label_from_record(record),
            question_type="identity",
            proposal_source=str(first(record, "proposal_source", "source", default="native_identity_pair")),
            media=[item for item in media if item is not None],
            current_frame=frame,
        )
        row["history_frame"] = previous_frame
        row["current_track_id"] = current_id
        row["previous_track_id"] = previous_id
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = write_jsonl_and_manifest(prepare(read_records(args.input), limit=args.limit), args.out, tool="prepare_identity_pairs", source=args.input)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
