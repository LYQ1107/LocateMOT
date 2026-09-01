#!/usr/bin/env python3
"""Read-only shape/causality audit for the L71 correspondence head."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l71_common import L71Bank, load_text_cache, write_json  # noqa: E402
from locatemot.models.l71_bounded_query_track import L71BoundedQueryTrack  # noqa: E402


INDEX = ROOT / "outputs/l71/audit/data_contract/unit_records.jsonl"
SEED = 20260829
DEFAULT_OUT = ROOT / "outputs/l71/audit/forward_contract"


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choose_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fit = [row for row in records if row.get("index_role") == "fit"]
    selected: list[dict[str, Any]] = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for category in ("positive", "multi_positive", "inactive", "present_uncovered"):
            choices = sorted(
                (row for row in fit if row["dataset"] == dataset and row["category"] == category),
                key=lambda row: (str(row["video"]), int(row["query_id"]), int(row["frame_id"])),
            )
            if not choices:
                raise AssertionError(f"missing fit stratum {dataset}/{category}")
            selected.append(choices[0])
        for category in ("positive", "multi_positive"):
            choices = sorted(
                (row for row in fit if row["dataset"] == dataset and row["category"] == category),
                key=lambda row: (str(row["video"]), int(row["query_id"]), int(row["frame_id"])),
            )
            selected.append(choices[1])
    # The first 20 selected rows are deterministic and contain every required
    # domain/category; the fixed evaluation rows are all retained separately.
    return selected[:20]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    running = {
        "format": "locatemot-l71-forward-contract-v1",
        "status": "running",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "seed": SEED,
        "inputs": {"unit_index": str(args.index), "text_cache": str(ROOT / "outputs/l48/data/text_cache.pt")},
        "outputs": {"checks": str(args.out / "forward_checks.jsonl")},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "raw_dense_feature_cache_written": False,
    }
    write_json(args.out / "status.json", running)
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        records = read_records(args.index)
        fit_sample = choose_records(records)
        fixed_eval = sorted(
            (row for row in records if row.get("index_role") == "fixed_eval"),
            key=lambda row: int(row["fixed_eval_order"]),
        )
        if len(fixed_eval) != 40:
            raise AssertionError(f"expected 40 fixed evaluation records, got {len(fixed_eval)}")
        selected = fit_sample + fixed_eval
        text_cache = load_text_cache()
        device = torch.device(args.device)
        if device.type == "cuda" and (device.index not in (None, 0)):
            raise RuntimeError(f"L71 forward audit requires GPU0, got {device}")
        model = L71BoundedQueryTrack().to(device).eval()
        if any(parameter.requires_grad for parameter in model.parameters()) is False:
            raise AssertionError("audit model unexpectedly has no parameters")
        checks: list[dict[str, Any]] = []
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in selected:
            by_video[str(record["video"])].append(record)
        bank_sha: dict[str, str] = {}
        for video in sorted(by_video):
            bank = L71Bank(video)
            bank_sha[video] = bank.sha256
            try:
                for record in by_video[video]:
                    data = {key: value.to(device) for key, value in __import__("tools.l71_common", fromlist=["unit_tensors"]).unit_tensors(record, bank, text_cache).items()}
                    if any(frame > int(record["frame_id"]) for frames in record["history_frame_ids"] for frame in frames):
                        raise AssertionError(f"future history in {record['unit_key']}")
                    with torch.inference_mode():
                        output = model(data)
                    logits = output["correspondence_logits"]
                    n = int(record["candidate_count"])
                    if tuple(logits.shape) != (n,):
                        raise AssertionError(f"logit shape drift for {record['unit_key']}: {tuple(logits.shape)}")
                    if len(record["row_keys"]) != n or [int(key[-1]) for key in record["row_keys"]] != record["row_offsets"]:
                        raise AssertionError(f"row key/order drift for {record['unit_key']}")
                    if not (torch.isfinite(logits).all() and torch.isfinite(output["query_vector"]).all() and torch.isfinite(output["track_vector"]).all()):
                        raise AssertionError(f"nonfinite output for {record['unit_key']}")
                    checks.append({
                        "unit_key": record["unit_key"],
                        "role": record["index_role"],
                        "dataset": record["dataset"],
                        "video": record["video"],
                        "category": record["category"],
                        "candidate_count": n,
                        "positive_count": int(record["positive_count"]),
                        "candidate_keys_complete": True,
                        "history_causal": True,
                        "history_max": max((len(value) for value in record["history_row_offsets"]), default=0),
                        "logit_shape": list(logits.shape),
                        "query_shape": list(output["query_vector"].shape),
                        "track_shape": list(output["track_vector"].shape),
                        "finite": True,
                    })
                    del data, output
            finally:
                bank.close()
        with (args.out / "forward_checks.jsonl").open("w") as handle:
            for row in checks:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        payload = {
            **running,
            "status": "complete",
            "checks": {
                "candidate_key_drift": 0,
                "candidate_deletion": False,
                "candidate_truncation": False,
                "finite": True,
                "future_history_rows": 0,
                "all_rows_retained": True,
                "temperature": 0.07,
                "token_region_alignment": "UNALIGNED",
            },
            "counts": {
                "fit_sample": len(fit_sample),
                "fixed_eval_units": len(fixed_eval),
                "units": len(checks),
                "candidate_rows": sum(int(row["candidate_count"]) for row in checks),
                "datasets": sorted({str(row["dataset"]) for row in checks}),
                "categories": dict(sorted(Counter(str(row["category"]) for row in checks).items())),
            },
            "model": L71BoundedQueryTrack().parameter_summary(),
            "input_bank_sha256": bank_sha,
            "elapsed_seconds": time.perf_counter() - started,
            "failure_root_cause": None,
            "next_action": "run the independent L71 loss contract audit",
        }
        write_json(args.out / "contract.json", payload)
        write_json(args.out / "provenance.json", payload)
        write_json(args.out / "status.json", payload)
        print(json.dumps({"status": "complete", "units": len(checks), "out": str(args.out)}), flush=True)
        return 0
    except Exception as exc:
        failure = {**running, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}", "next_action": "fix only the first forward-contract root cause and rerun in a new directory", "elapsed_seconds": time.perf_counter() - started}
        write_json(args.out / "status.json", failure)
        (args.out / "INCOMPLETE.md").write_text("# L71 forward contract INCOMPLETE\n\n" + f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n```text\n" + traceback.format_exc() + "```\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
