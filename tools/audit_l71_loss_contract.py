#!/usr/bin/env python3
"""One-batch gradient audit for the L71 unit-local correspondence loss."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l71_common import L71Bank, load_text_cache, write_json  # noqa: E402
from tools.l71_loss import compute_loss  # noqa: E402
from locatemot.models.l71_bounded_query_track import L71BoundedQueryTrack  # noqa: E402


INDEX = ROOT / "outputs/l71/audit/data_contract/unit_records.jsonl"
SEED = 20260829
DEFAULT_OUT = ROOT / "outputs/l71/audit/loss_contract"


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        "format": "locatemot-l71-loss-contract-v1",
        "status": "running",
        "project_root": str(ROOT),
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
        "seed": SEED,
        "inputs": {"unit_index": str(args.index)},
        "outputs": {"contract": str(args.out / "contract.json")},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "training_run": False,
        "hota_trackeval_run": False,
        "raw_dense_feature_cache_written": False,
    }
    write_json(args.out / "status.json", running)
    try:
        records = read_records(args.index)
        candidates = [row for row in records if row.get("index_role") == "fit" and row.get("category") == "multi_positive" and int(row.get("positive_count", 0)) > 1]
        if not candidates:
            raise AssertionError("no fit multi-positive unit available")
        record = sorted(candidates, key=lambda row: (str(row["dataset"]), str(row["video"]), int(row["query_id"]), int(row["frame_id"]))) [0]
        text_cache = load_text_cache()
        device = torch.device(args.device)
        model = L71BoundedQueryTrack().to(device)
        model.train()
        bank = L71Bank(str(record["video"]))
        try:
            cpu_data = __import__("tools.l71_common", fromlist=["unit_tensors"]).unit_tensors(record, bank, text_cache)
        finally:
            bank.close()
        data = {key: value.to(device) for key, value in cpu_data.items()}
        output = model(data)
        logits = output["correspondence_logits"]
        logits.retain_grad()
        loss, parts = compute_loss(output, data)
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite loss")
        loss.backward()
        parameter_grads = [parameter.grad for parameter in model.parameters()]
        finite_gradients = all(gradient is not None and torch.isfinite(gradient).all() for gradient in parameter_grads)
        if not finite_gradients:
            raise AssertionError("nonfinite or missing parameter gradient")
        target = data["membership_target"]
        coverage = data["coverage_mask"]
        positive = (target > 0.5) & coverage
        negative = (target <= 0.5) & coverage
        grad = logits.grad.detach().abs()
        pos_nonzero = int((grad[positive] > 0).sum())
        neg_nonzero = int((grad[negative] > 0).sum())
        if positive.any() and pos_nonzero != int(positive.sum()):
            raise AssertionError("not every positive logit has gradient")
        if negative.any() and neg_nonzero != int(negative.sum()):
            raise AssertionError("not every negative logit has gradient")
        payload = {
            **running,
            "status": "complete",
            "format": "locatemot-l71-loss-contract-v1",
            "unit_key": record["unit_key"],
            "candidate_count": int(record["candidate_count"]),
            "positive_count": int(positive.sum()),
            "negative_count": int(negative.sum()),
            "masked_missing_count": int((~coverage).sum()),
            "all_positive_gradients_nonzero": True,
            "positive_logit_grad_nonzero": pos_nonzero,
            "negative_logit_grad_nonzero": neg_nonzero,
            "finite_loss": True,
            "finite_gradients": True,
            "nonzero_gradients": True,
            "loss_parts": parts,
            "temperature": 0.07,
            "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
            "history_future_rows": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "failure_root_cause": None,
            "next_action": "run the registered L71 100-step fit-only smoke",
        }
        write_json(args.out / "contract.json", payload)
        write_json(args.out / "status.json", payload)
        print(json.dumps({"status": "complete", "unit_key": record["unit_key"], "out": str(args.out)}), flush=True)
        return 0
    except Exception as exc:
        failure = {**running, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}", "next_action": "fix only the first loss-contract root cause and rerun in a new directory", "elapsed_seconds": time.perf_counter() - started}
        write_json(args.out / "status.json", failure)
        (args.out / "INCOMPLETE.md").write_text("# L71 loss contract INCOMPLETE\n\n" + f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n```text\n" + traceback.format_exc() + "```\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
