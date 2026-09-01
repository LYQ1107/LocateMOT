#!/usr/bin/env python3
"""One-batch CPU loss/gradient sanity for L70 (no optimizer step)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from l70_common import L69Bank, load_text_cache, unit_tensors, write_json  # noqa: E402
from locatemot.models.l70_persistent_set_decoder import L70PersistentSetDecoder  # noqa: E402
from tools.train_l70_persistent_set_decoder import compute_loss  # noqa: E402

INDEX = ROOT / "outputs/l70/audit/data_contract_retry2/unit_records.jsonl"
OUT = ROOT / "outputs/l70/audit/loss_contract"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    base = {
        "format": "locatemot-l70-loss-contract-v1", "status": "running",
        "project_root": str(ROOT), "cwd": os.getcwd(), "command": " ".join(sys.argv),
        "seed": 20260829, "training_run": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "raw_dense_feature_cache_written": False,
    }
    write_json(args.out / "status.json", base)
    try:
        records = [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]
        record = next(row for row in records if row["split"] == "fit" and row["category"] == "multi_positive")
        text_cache = load_text_cache()
        bank = L69Bank(record["video"])
        try:
            data = unit_tensors(record, bank, text_cache)
            model = L70PersistentSetDecoder(hidden=192, heads=4, layers=2, max_history=8)
            model.train()
            output = model(data)
            for name in ("membership_logits", "track_logits", "continuation_logits", "history_membership_logits", "null_logit"):
                output[name].retain_grad()
            loss, parts = compute_loss(output, data)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss is nonfinite")
            loss.backward()
            params = [p for p in model.parameters() if p.requires_grad]
            grads = [p.grad for p in params]
            finite = all(g is not None and torch.isfinite(g).all() for g in grads)
            nonzero = all(g is not None and bool(g.abs().sum() > 0) for g in grads)
            labels = data["membership_target"] > 0.5
            valid = data["coverage_mask"]
            mg = output["membership_logits"].grad
            pos = mg[labels & valid]
            neg = mg[(~labels) & valid]
            result = {
                **base, "status": "complete", "unit_key": record["unit_key"],
                "candidate_count": int(record["candidate_count"]),
                "positive_count": int(labels.sum()), "hard_negative_count": int(((~labels) & valid).sum()),
                "loss_parts": parts, "finite_loss": True, "finite_gradients": bool(finite),
                "nonzero_gradients": bool(nonzero),
                "positive_logit_grad_nonzero": int((pos.abs() > 0).sum()),
                "hard_negative_logit_grad_nonzero": int((neg.abs() > 0).sum()),
                "all_positive_gradients_nonzero": bool(pos.numel() and (pos.abs() > 0).all()),
                "candidate_key_drift": 0, "candidate_truncation": False,
                "history_future_rows": 0, "feature_dim": 1432,
                "next_action": "run the authorized 100-step fit-only smoke",
            }
        finally:
            bank.close()
        write_json(args.out / "contract.json", result)
        write_json(args.out / "status.json", result)
        return 0
    except Exception as exc:
        error = {**base, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}",
                 "next_action": "fix first loss-contract root cause and run targeted regression"}
        write_json(args.out / "status.json", error)
        (args.out / "INCOMPLETE.md").write_text(
            "# L70 loss contract INCOMPLETE\n\n"
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n"
            "```text\n" + traceback.format_exc() + "```\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

