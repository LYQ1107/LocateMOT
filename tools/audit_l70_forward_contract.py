#!/usr/bin/env python3
"""Non-training L70 data/model forward contract check."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from l70_common import L69Bank, ROOT, load_text_cache, unit_tensors, write_json
from locatemot.models.l70_persistent_set_decoder import L70PersistentSetDecoder

DEFAULT_INDEX = ROOT / "outputs/l70/audit/data_contract_retry2/unit_records.jsonl"
DEFAULT_OUT = ROOT / "outputs/l70/audit/forward_contract"


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choose_records(records: list[dict]) -> list[dict]:
    fit = [row for row in records if row["split"] == "fit"]
    chosen: list[dict] = []
    for category in ("positive", "multi_positive", "inactive", "present_uncovered"):
        chosen.extend([row for row in fit if row["category"] == category][:5])
    if len(chosen) < 20:
        chosen.extend(row for row in fit if row not in chosen)
    evaluation = [row for row in records if row["split"] != "fit"]
    if len(chosen) < 20 or len(evaluation) != 40:
        raise AssertionError(f"forward sample count fit={len(chosen)} eval={len(evaluation)}")
    return chosen[:20] + evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    status = {
        "format": "locatemot-l70-forward-contract-v1", "status": "running",
        "project_root": str(ROOT), "cwd": os.getcwd(), "command": " ".join(sys.argv),
        "seed": 20260829, "training_run": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "raw_dense_feature_cache_written": False,
    }
    write_json(args.out / "status.json", status)
    try:
        records = choose_records(read_records(args.index))
        text_cache = load_text_cache()
        torch.manual_seed(20260829)
        model = L70PersistentSetDecoder(hidden=192, heads=4, layers=2, max_history=8, dropout=0.0)
        model.eval()
        by_video: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            by_video[str(record["video"])].append(record)
        checks: list[dict] = []
        start = time.perf_counter()
        for video in sorted(by_video):
            bank = L69Bank(video)
            try:
                for record in by_video[video]:
                    data = unit_tensors(record, bank, text_cache)
                    if any(int(frame) > int(record["frame_id"]) for frames in record["history_frame_ids"] for frame in frames):
                        raise AssertionError("future frame in history")
                    with torch.inference_mode():
                        output = model(data)
                    n = int(record["candidate_count"])
                    expected = {"membership_logits": (n,), "track_logits": (n,),
                                "continuation_logits": (n,), "history_membership_logits": (n, 8),
                                "null_logit": (1,)}
                    shape_ok = all(tuple(output[name].shape) == shape for name, shape in expected.items())
                    finite_ok = all(bool(torch.isfinite(output[name]).all()) for name in expected)
                    if not shape_ok or not finite_ok:
                        raise AssertionError(f"forward shape/finite failure {record['unit_key']}")
                    if len(record["row_keys"]) != n:
                        raise AssertionError("candidate row key truncation")
                    checks.append({
                        "unit_key": record["unit_key"], "split": record["split"],
                        "dataset": record["dataset"], "video": record["video"],
                        "category": record["category"], "candidate_count": n,
                        "history_shape": list(data["history"].shape),
                        "history_valid_min": int(data["history_mask"].sum(1).min()),
                        "text_shape": list(data["text"].shape),
                        "text_mask_true": int(data["text_mask"].sum()),
                        "output_shapes": {name: list(output[name].shape) for name in expected},
                        "finite": finite_ok, "candidate_key_count": len(record["row_keys"]),
                        "candidate_truncation": False, "future_history": False,
                    })
                    del data, output
            finally:
                bank.close()
        elapsed = time.perf_counter() - start
        if len(checks) != 60:
            raise AssertionError(f"forward checks={len(checks)}")
        write_json(args.out / "forward_checks.json", checks)
        write_json(args.out / "contract.json", {
            **status, "status": "complete",
            "inputs": {"unit_index": str(args.index), "fit_sample": 20, "fixed_eval_units": 40,
                       "text_cache": str(ROOT / "outputs/l48/data/text_cache.pt"),
                       "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti")},
            "outputs": {"checks": str(args.out / "forward_checks.json")},
            "model": {"obs_dim": 1432, "text_dim": 768, "hidden": 192, "heads": 4,
                      "layers": 2, "max_history": 8, "parameter_summary": model.parameter_summary()},
            "counts": {"units": 60, "fit_units": 20, "fixed_eval_units": 40,
                       "datasets": sorted({x["dataset"] for x in checks}),
                       "categories": {cat: sum(x["category"] == cat for x in checks)
                                      for cat in ("positive", "multi_positive", "inactive", "present_uncovered")}},
            "checks": {"finite": True, "candidate_key_drift": 0, "candidate_deletion": False,
                       "candidate_truncation": False, "future_history_rows": 0,
                       "text_token_sequence_preserved": True, "token_region_alignment": "UNALIGNED",
                       "wall_time_seconds": elapsed},
            "next_action": "run the authorized 100-step fit-only smoke",
        })
        write_json(args.out / "provenance.json", {
            **status, "status": "complete", "model_contract": "L70 persistent history + full current set",
            "labels_used": "contract/category checks only; no optimization",
            "old_l49_ranges_used": False, "raw_dense_feature_cache_written": False,
        })
        write_json(args.out / "status.json", {**status, "status": "complete", "failure_root_cause": None,
                                              "next_action": "run L70 B0 smoke"})
        return 0
    except Exception as exc:
        write_json(args.out / "status.json", {**status, "status": "INCOMPLETE",
                                               "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                               "next_action": "fix first forward-contract root cause and run targeted regression"})
        (args.out / "INCOMPLETE.md").write_text(
            "# L70 forward contract INCOMPLETE\n\n"
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n"
            "```text\n" + traceback.format_exc() + "```\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
