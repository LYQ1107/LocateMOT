#!/usr/bin/env python3
"""Register the compact, zero-drop v2 source manifest without copying rows.

The native JSONL and the v1 manifest are immutable.  v2 adds an explicit
``compact_competition_v1`` transform to the MOT identity source; the runtime
loader applies it in memory before rendering OneJev examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(
        "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/real_three_task_joint_v1"))
    parser.add_argument("--out", type=Path, default=Path(
        "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/real_three_task_joint_v2_compact"))
    args = parser.parse_args()
    old_manifest = json.loads((args.source / "train.sources.json").read_text())
    manifest = dict(old_manifest)
    manifest["protocol"] = "immutable-native-sources-family-filter-full-exposure-v2"
    manifest["manifest_version"] = "real-three-task-joint-v2-compact-competition-v1"
    sources = [dict(source) for source in old_manifest["sources"]]
    sources[0]["state_transform"] = "compact_competition_v1"
    manifest["sources"] = sources
    manifest["transform_contract"] = {
        "name": "compact_competition_v1",
        "source_rows_unchanged": True,
        "row_ids_targets_media_and_candidate_geometry_unchanged": True,
        "dense_competition_rows_removed": True,
        "positive_iou_edges_retained": True,
        "zero_iou_geometry": "derivable from exact candidate boxes",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    (args.out / "train.sources.json").write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    old_receipt = json.loads((args.source / "manifest_receipt.json").read_text())
    receipt = dict(old_receipt)
    receipt.update({
        "status": "REAL_THREE_TASK_COMPACT_SOURCE_MANIFEST_PREPARED_NOT_TRAINED",
        "manifest": str(args.out / "train.sources.json"),
        "manifest_sha256": digest,
        "manifest_version": manifest["manifest_version"],
        "transform_contract": manifest["transform_contract"],
        "overlength_rows_from_previous_v1": 2099,
        "optimizer_updates": 0,
        "gpu_launched": False,
        "full_goal_complete": False,
    })
    (args.out / "manifest_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(args.out / "train.sources.json"),
                      "manifest_sha256": digest, "rows": manifest["rows"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
