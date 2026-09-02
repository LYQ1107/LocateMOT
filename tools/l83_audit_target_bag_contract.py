#!/usr/bin/env python3
"""Audit L83 target-bag construction on complete L69 frame groups."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=4)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L83 output: {out}")
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != EXPECTED:
        raise AssertionError("fixed manifest SHA drift")
    from tools.l82_train_frozen_rank_probe import build_local_groups, load_groups
    groups, train_keys, dev_keys = load_groups()
    selected = (train_keys + dev_keys)[:max(1, int(args.max_groups))]
    built, build_info = build_local_groups(selected, groups, torch.device("cuda:0"))
    records = []
    for data in built:
        if data.candidate_count <= 0 or len(data.row_offsets) != data.candidate_count:
            raise AssertionError(f"candidate count drift: {data.group_key}")
        for row in data.candidate_gt:
            if len(row) != data.candidate_count:
                raise AssertionError(f"candidate_gt length drift: {data.group_key}")
        if len(data.candidate_indices) != data.candidate_count or len(data.pool_ids) != data.candidate_count:
            raise AssertionError(f"candidate metadata length drift: {data.group_key}")
        if not all(bool(torch.isfinite(value).all()) for value in data.features.values() if torch.is_tensor(value)):
            raise FloatingPointError(f"nonfinite representation: {data.group_key}")
        duplicates = data.candidate_count - len(set(data.candidate_indices))
        target_counts = {}
        for values in data.candidate_gt:
            for target in values:
                if target is not None:
                    target_counts[str(target)] = target_counts.get(str(target), 0) + 1
        records.append({
            "group_key": data.group_key, "dataset": data.dataset, "video": data.video,
            "frame_id": data.frame_id, "query_count": len(data.query_unit_keys),
            "candidate_count": data.candidate_count, "row_offsets": list(data.row_offsets),
            "row_keys_digest": list(data.row_keys_digest),
            "duplicate_candidate_index_count": duplicates,
            "target_bag_row_counts": target_counts,
            "categories": list(data.categories),
            "feature_shapes": {key: list(value.shape) for key, value in data.features.items()},
            "candidate_deletion": False, "candidate_truncation": False, "finite": True,
        })
    payload = {
        "format": "locatemot-l83-target-bag-data-contract-v1", "status": "complete",
        "stage": "phase_2_target_bag_data_contract", "command": " ".join([sys.executable] + sys.argv),
        "cwd": str(ROOT), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "selected_group_keys": selected, "records": records,
        "train_group_count": len(train_keys), "dev_group_count": len(dev_keys),
        "target_bag_contract": {
            "primary_pooling": "max per unique non-null candidate_gt target",
            "background": "one singleton negative bag per candidate_gt=None row",
            "duplicate_candidate_index_legal": True,
            "target_ids_as_model_input": False,
            "old_l49_begin_end_used_for_l69_index": False,
        },
        "feature_build": build_info, "inputs": {"manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST)}},
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
    }
    write_json(out / "contract.json", payload)
    write_json(out / "provenance.json", {"format": "locatemot-l83-target-bag-data-provenance-v1", "status": "complete", "command": " ".join([sys.executable] + sys.argv), "inputs": payload["inputs"], "labels": "fit-only labels attached by the inherited audited L82 builder after feature construction", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False})
    write_json(out / "status.json", {"format": "locatemot-l83-target-bag-data-status-v1", "status": "complete", "failure_root_cause": None, "next_action": "run target-bag metric tests and corrected baseline rescore", "command": " ".join([sys.executable] + sys.argv)})
    (out / "unit_records.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n")
    print(json.dumps({"status": "complete", "groups": len(records), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
