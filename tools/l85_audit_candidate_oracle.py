#!/usr/bin/env python3
"""Post-hoc candidate coverage oracle for legal internal validation videos."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from locatemot.rmot.l85_fullvideo_bank import (  # noqa: E402
    EXPECTED_MANIFEST_SHA, INTERNAL_V1, INTERNAL_V2, MANIFEST, bank_path, file_meta,
    sha256_file,
)

DATA = ROOT / "outputs/l49/data"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def unit_rows(path: Path, allowed_videos: set[str]) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") not in {"refer_kitti_v1", "refer_kitti_v2"} or str(row.get("video")) not in allowed_videos:
            continue
        result.append(row)
    return result


def audit_one_video(video: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = bank_path(video)
    package = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    tensors = package["tensors"]
    sidecar = json.loads(path.with_suffix(".labels.json").read_text())["candidate_gt"]
    frame_to_index = {int(x): i for i, x in enumerate(tensors["frame_ids"].tolist())}
    target_present = covered = target_count = covered_targets = 0
    per_category = Counter(); examples = []
    row_total = 0
    for row in rows:
        frame = int(row["frame_id"])
        if frame not in frame_to_index:
            examples.append({"unit_key": row["unit_key"], "reason": "frame_missing"})
            continue
        index = frame_to_index[frame]
        start, end = int(tensors["frame_ptr"][index]), int(tensors["frame_ptr"][index + 1])
        candidate_values = [sidecar[x] for x in range(start, end)]
        targets = {str(x) for x in row.get("target_ids", [])}
        positive_targets = targets.intersection({str(x) for x in candidate_values if x is not None})
        present = bool(targets); is_covered = bool(positive_targets)
        category = str(row.get("category", "unknown"))
        per_category[category] += 1
        row_total += end - start
        target_present += int(present); covered += int(is_covered)
        target_count += len(targets); covered_targets += len(positive_targets)
        if present and not is_covered and len(examples) < 24:
            examples.append({"unit_key": row["unit_key"], "reason": "present_uncovered",
                             "target_ids": sorted(targets), "candidate_count": end - start,
                             "candidate_gt_ids": sorted({str(x) for x in candidate_values if x is not None})})
    return {"video": str(video), "dataset_rows": len(rows), "target_present_units": target_present,
            "covered_units": covered, "unit_coverage": covered / target_present if target_present else None,
            "target_ids": target_count, "covered_target_ids": covered_targets,
            "target_micro_coverage": covered_targets / target_count if target_count else None,
            "mean_candidate_rows_per_unit": row_total / len(rows) if rows else None,
            "categories": dict(per_category), "examples": examples,
            "oracle_only": True, "no_candidate_deletion": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l85/audit/candidate_oracle/oracle.json")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    rows = unit_rows(DATA / "validation_units.jsonl", set(INTERNAL_V1 + INTERNAL_V2))
    per = []
    for video in INTERNAL_V1 + INTERNAL_V2:
        dataset = "refer_kitti_v1" if video in INTERNAL_V1 else "refer_kitti_v2"
        per.append(audit_one_video(video, [x for x in rows if str(x["video"]) == video and x["dataset"] == dataset]))
    present = sum(x["target_present_units"] for x in per); covered = sum(x["covered_units"] for x in per)
    targets = sum(x["target_ids"] for x in per); covered_targets = sum(x["covered_target_ids"] for x in per)
    value = {"format": "locatemot-l85-candidate-oracle-v1", "status": "complete",
             "command": " ".join([sys.executable, *sys.argv]), "cwd": str(ROOT), "luna_thread": THREAD,
             "scope": "legal internal validation units only; post-hoc ORACLE_DIAGNOSTIC",
             "per_video": per, "unit_count": len(rows), "target_present_units": present, "covered_units": covered,
             "unit_coverage": covered / present if present else None, "target_ids": targets,
             "covered_target_ids": covered_targets, "target_micro_coverage": covered_targets / targets if targets else None,
             "inputs": {"manifest": file_meta(MANIFEST), "validation_units": file_meta(DATA / "validation_units.jsonl")},
             "candidate_bank_gt_conditioned": False, "candidate_bank_query_conditioned": False,
             "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
             "training_run": False, "hota_trackeval_run": False,
             "failure_root_cause": None, "next_action": "run label-free memory contract and semantic-state preparation"}
    write_json(out, value); write_json(out.parent / "provenance.json", value); write_json(out.parent / "status.json", value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
