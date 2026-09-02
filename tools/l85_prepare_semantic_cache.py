#!/usr/bin/env python3
"""Build a compact, label-free Z1 semantic-state view for L85.

This is a measured derived-state cache, not a raw/dense detector cache: each
file contains only the selected 256-D Z1 rows, two 256-D presence summaries,
and reversible L69 row provenance.  Labels are deliberately not serialized.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.rmot.l85_runtime import (  # noqa: E402
    build_groups, capture_group_z1, file_meta, load_fit_key_rows,
    load_internal_eval_groups, sha256_file,
)
from locatemot.rmot.l85_fullvideo_bank import (  # noqa: E402
    EXPECTED_MANIFEST_SHA, MANIFEST, L69_FEATURE_ROOT,
)
from locatemot.rmot.l80_data import L80BankStore  # noqa: E402

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-internal-eval", action="store_true")
    parser.add_argument("--max-groups", type=int, default=0)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty semantic cache: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
            raise RuntimeError("L85 Z1 semantic preparation requires the verified GPU runtime")
        device = torch.device(args.device); torch.cuda.set_device(device)
        fit_groups = build_groups(load_fit_key_rows())
        split = json.loads((ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json").read_text())
        ordered = [str(x) for x in split["train_group_keys"] + split["dev_group_keys"]]
        if len(ordered) != 662 or len(set(ordered)) != 662:
            raise AssertionError("L82 fit/dev group count drift")
        groups = {key: fit_groups[key] for key in ordered}
        partitions = {key: ("fit_group" if key in set(split["train_group_keys"]) else "dev_group") for key in ordered}
        if args.include_internal_eval:
            eval_groups, calibration_keys, validation_keys = load_internal_eval_groups()
            for key in calibration_keys:
                groups.setdefault(key, eval_groups[key]); partitions.setdefault(key, "internal_calibration")
            for key in validation_keys:
                groups.setdefault(key, eval_groups[key]); partitions.setdefault(key, "internal_validation")
        selected = list(groups)
        if args.max_groups:
            selected = selected[: int(args.max_groups)]
        manifest_rows = []
        complete = 0
        from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime  # noqa: E402
        runtime = GroundingCandidateReferenceRuntime(device)
        bank_store = L80BankStore(max_history=8)
        try:
            for index, key in enumerate(selected):
                group = groups[key]
                item = capture_group_z1(group, device, runtime=runtime, bank_store=bank_store)
                if item["query_unit_keys"] != [str(row["unit_key"]) for row in group["queries"]]:
                    raise AssertionError(f"query order drift: {key}")
                # Labels are not present in item and are not read by this tool.
                item["partition"] = partitions[key]
                item["command"] = command
                item["labels_in_cache"] = False
                item["candidate_count"] = int(item["z1"].shape[1])
                if item["candidate_count"] != len(item["row_offsets"]):
                    raise AssertionError(f"candidate count drift: {key}")
                if not bool(torch.isfinite(item["z1"].float()).all() and torch.isfinite(item["text_global"].float()).all() and torch.isfinite(item["frame_global"].float()).all()):
                    raise FloatingPointError(f"nonfinite cached state: {key}")
                dataset_dir = out / str(group["dataset"])
                dataset_dir.mkdir(parents=True, exist_ok=True)
                safe = key.replace("|", "__")
                path = dataset_dir / f"{safe}.pt"
                if path.exists():
                    raise FileExistsError(path)
                tmp = path.with_suffix(".pt.tmp")
                torch.save(item, tmp)
                tmp.replace(path)
                manifest_rows.append({"group_key": key, "dataset": group["dataset"], "video": group["video"],
                                      "frame_id": int(group["frame_id"]), "partition": partitions[key],
                                      "path": str(path), "bytes": path.stat().st_size,
                                      "query_count": len(item["query_unit_keys"]), "candidate_count": item["candidate_count"],
                                      "query_unit_keys": item["query_unit_keys"], "row_keys_digest": item["row_keys_digest"],
                                      "candidate_deletion": False, "candidate_truncation": False})
                complete += 1
                del item
                if index % 4 == 0:
                    gc.collect(); torch.cuda.empty_cache()
        finally:
            runtime.close()
            del runtime
            del bank_store
            gc.collect()
            torch.cuda.empty_cache()
        manifest_rows.sort(key=lambda row: str(row["group_key"]))
        total_bytes = sum(int(row["bytes"]) for row in manifest_rows)
        summary = {"format": "locatemot-l85-z1-semantic-cache-v1", "status": "complete",
                   "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                   "group_count": len(manifest_rows), "complete_groups": complete,
                   "fit_dev_group_count": 662, "include_internal_eval": bool(args.include_internal_eval),
                   "partition_counts": {name: sum(row["partition"] == name for row in manifest_rows)
                                         for name in sorted(set(row["partition"] for row in manifest_rows))},
                   "bytes": total_bytes, "path": str(out), "selected_representation": "L84 Z1 fixed-reference decoder state",
                   "z1_shape_contract": "[Q,N,256]", "presence_input_contract": "concat(memory_text_mean,encoder_memory_mean) [Q,512]",
                   "labels_in_cache": False, "raw_pixels_in_cache": False, "dense_detector_maps_in_cache": False,
                   "candidate_deletion": False, "candidate_truncation": False,
                   "inputs": {"manifest": file_meta(MANIFEST), "l69_root": str(L69_FEATURE_ROOT),
                              "l69_root_sha256": sha256_file(L69_FEATURE_ROOT / "manifest.json"),
                              "fit_units": file_meta(ROOT / "outputs/l49/data/train_units.jsonl"),
                              "split": file_meta(ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json")},
                   "screening_gt_used": False, "official_test_labels_read": False,
                   "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                   "failure_root_cause": None, "next_action": "run L85 memory contract and factorized training"}
        (out / "manifest.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True, default=str) for row in manifest_rows) + "\n")
        write_json(out / "summary.json", summary); write_json(out / "provenance.json", summary); write_json(out / "status.json", summary)
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text("# L85 semantic cache — INCOMPLETE\n\n" + traceback.format_exc() + "\n")
        write_json(out / "status.json", {"format": "locatemot-l85-z1-semantic-cache-v1", "status": "incomplete", "command": command,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md", "next_action": "targeted runtime repair",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
