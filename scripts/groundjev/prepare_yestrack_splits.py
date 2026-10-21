#!/usr/bin/env python3
"""Materialize the registered YesTrack validation split without relabeling rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(
        "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/native_yesno_v2_phaseb.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.source.with_name(args.source.stem + "_val.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "other": 0}
    with args.source.open() as source, out.open("w") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split")
            counts[split if split in counts else "other"] += 1
            if split == "val":
                target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    receipt = {
        "status": "YESTRACK_VALIDATION_SPLIT_MATERIALIZED",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(out),
        "output_sha256": digest,
        "counts": counts,
        "validation_rows": counts["val"],
        "gt_used_for_prompt": False,
        "official_test_labels_read": False,
        "future_evidence": False,
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
