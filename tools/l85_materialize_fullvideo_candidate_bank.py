#!/usr/bin/env python3
"""Register existing L69 rows as the L85 full-video bank.

All legal internal validation videos already have complete L69 rows.  This
command deliberately writes only a small manifest; it does not copy or alter
the multi-gigabyte frozen bank and does not read expression/GT labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from locatemot.rmot.l85_fullvideo_bank import (  # noqa: E402
    ALL_VIDEOS, EXPECTED_MANIFEST_SHA, INTERNAL_V1, INTERNAL_V2, MANIFEST,
    bank_source_manifest, sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l85/bank/fullvideo_manifest.json")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    records = bank_source_manifest(ALL_VIDEOS)
    missing = [x["video"] for x in records if not x["features"]["exists"] or not x["dual"]["exists"]]
    if missing:
        raise FileNotFoundError(f"missing L69 bank videos: {missing}")
    value = {
        "format": "locatemot-l85-fullvideo-bank-manifest-v1", "status": "complete",
        "materialization_mode": "validated_reuse_of_immutable_L69_budget40",
        "videos": list(ALL_VIDEOS), "internal_v1": list(INTERNAL_V1), "internal_v2": list(INTERNAL_V2),
        "source_manifest": records, "manifest_sha256": manifest_sha,
        "candidate_bank_gt_conditioned": False, "candidate_bank_query_conditioned": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False,
        "failure_root_cause": None, "next_action": "run L85 candidate oracle audit",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
